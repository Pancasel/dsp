import json
import time

import httpx
from pydantic import BaseModel, Field, ValidationError

from graphrag.config import Settings
from graphrag.dataset import DecagonGraph, Evidence


class ModelPrediction(BaseModel):
    side_effect: str = Field(min_length=1)
    tier: int = Field(ge=1, le=3)
    evidence_ids: list[str] = Field(min_length=1)
    explanation: str = Field(min_length=1)


class PredictionList(BaseModel):
    predictions: list[ModelPrediction]


class LLMConfigurationError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


def _load_json_value(content: str) -> object:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _complete_prediction_objects(content: str) -> list[ModelPrediction]:
    """Recover finished prediction objects when the model stops mid-JSON."""
    decoder = json.JSONDecoder()
    found: list[ModelPrediction] = []
    index = 0
    while True:
        start = content.find("{", index)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        index = end
        candidates = []
        if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
            candidates = payload["predictions"]
        elif isinstance(payload, dict):
            candidates = [payload]
        elif isinstance(payload, list):
            candidates = payload
        for candidate in candidates:
            try:
                found.append(ModelPrediction.model_validate(candidate))
            except ValidationError:
                continue
    if not found:
        raise ValueError("LLM JSON did not contain a complete prediction")
    return found


def _coerce_predictions(content: str) -> list[ModelPrediction]:
    """Accept the required wrapper, a bare prediction, a bare array, or truncated JSON."""
    try:
        payload = _load_json_value(content)
    except json.JSONDecodeError:
        return _complete_prediction_objects(content)
    if isinstance(payload, dict) and "predictions" in payload:
        try:
            return PredictionList.model_validate(payload).predictions
        except ValidationError:
            return _complete_prediction_objects(content)
    if isinstance(payload, dict):
        return [ModelPrediction.model_validate(payload)]
    if isinstance(payload, list):
        return [ModelPrediction.model_validate(item) for item in payload]
    raise ValueError("LLM JSON must be an object or an array")


def generate_predictions(
    settings: Settings,
    drugs: list[str],
    graph_context: dict[str, object],
) -> list[ModelPrediction]:
    if not settings.llm_api_key:
        raise LLMConfigurationError("LLM_API_KEY is not configured")

    system_message = (
        "You are a cautious biomedical decision-support assistant for Vietnamese "
        "users. Use only the provided Decagon subgraph. Drugs and proteins are "
        "given by name with their source ID. Return a JSON object with a "
        "predictions array; each item must contain side_effect (the English "
        "label exactly as given), tier (1, 2, or 3), evidence_ids (one or more "
        "exact IDs from the supplied evidence), and explanation. Write explanation "
        "in Vietnamese, at most 3 sentences, naming the drugs and, when a path "
        "contains proteins, the protein symbols and how shared or linked targets "
        "could raise the risk. Never advise starting, stopping or changing a drug. "
        "Every cited evidence ID must exist, share that tier, and support the "
        "same side-effect label; a prediction that mixes in an unknown or "
        "mismatched ID is discarded. Tier 1 must reproduce a documented pairwise "
        "side effect. Tier 2 must reproduce a documented pairwise effect supported "
        "by a target shared by at least three queried drugs. Tier 3 may describe "
        "a hypothesis only when a supplied bridge evidence item supports the pair "
        "and that pair has no documented side-effect edge. For Tier 3, side_effect "
        "must be copied exactly from that evidence item's candidate_side_effects. "
        "Prefer explaining the effects listed first in top_side_effects, "
        "especially those marked serious. If evidence_truncated "
        "is true, the subgraph is a bounded sample across the regimen, not the "
        "complete neighborhood. Return at most 15 predictions, preferring Tier 1, "
        "then Tier 2, then at most two Tier 3 hypotheses. Do not invent evidence "
        "IDs, claim certainty, or present Tier 3 hypotheses as established "
        "clinical facts. If evidence is insufficient, return {\"predictions\": []}."
    )
    user_message = json.dumps(
        {"drugs": drugs, "retrieved_subgraph": graph_context},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    url = f"{settings.llm_base_url}/chat/completions"
    timeout = httpx.Timeout(settings.llm_timeout, connect=15.0)
    with httpx.Client(timeout=timeout) as client:
        response = None
        for attempt in range(3):
            try:
                response = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {settings.llm_api_key}",
                        "HTTP-Referer": "http://127.0.0.1:8000",
                        "X-Title": "Decagon Polypharmacy Graph-RAG",
                    },
                    json={
                        "model": settings.llm_model,
                        "response_format": {"type": "json_object"},
                        "reasoning": {"effort": "none", "exclude": True},
                        "max_tokens": 4000,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                        ],
                    },
                )
                break
            except httpx.ConnectError:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (2**attempt))
        if response is None:
            raise LLMResponseError("LLM provider did not return a response")
        response.raise_for_status()
    content = ""
    try:
        message = response.json()["choices"][0]["message"]
        content = message.get("content") or message.get("reasoning") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        return _coerce_predictions(content)
    except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
        raise LLMResponseError(
            "LLM returned a response that does not match the required JSON schema"
        ) from exc


def validate_predictions(
    predictions: list[ModelPrediction],
    evidence_items: list[Evidence],
    graph: DecagonGraph,
) -> list[dict[str, object]]:
    evidence_by_id = {item.evidence_id: item for item in evidence_items}
    validated: list[dict[str, object]] = []
    seen: set[tuple[int, str, tuple[str, ...]]] = set()
    for prediction in predictions:
        ordered_ids: list[str] = []
        invalid = False
        for evidence_id in prediction.evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if (
                item is None
                or item.tier != prediction.tier
                or not graph.validate_evidence(item)
            ):
                invalid = True
                break
            if evidence_id not in ordered_ids:
                ordered_ids.append(evidence_id)
        if invalid or not ordered_ids:
            continue
        selected = [evidence_by_id[evidence_id] for evidence_id in ordered_ids]
        label = prediction.side_effect.casefold().strip()
        if prediction.tier in (1, 2):
            if any(
                (item.side_effect or "").casefold().strip() != label for item in selected
            ):
                continue
            side_effect = selected[0].side_effect or ""
        else:
            canonical = graph.side_effect_names.get(label)
            if canonical is None:
                continue
            # Every cited bridge must list the label among its candidates.
            if any(
                item.candidates
                and label not in {name.casefold() for name in item.candidates}
                for item in selected
            ):
                continue
            side_effect = canonical

        evidence_key = tuple(sorted(item.evidence_id for item in selected))
        dedupe_key = (prediction.tier, side_effect.casefold(), evidence_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        validated.append(
            {
                "side_effect": side_effect,
                "tier": prediction.tier,
                "hypothesis": prediction.tier == 3,
                "explanation": prediction.explanation.strip(),
                "evidence": [item.to_dict() for item in selected],
            }
        )
    return validated
