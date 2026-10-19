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


SYSTEM_PROMPT = (
    "Bạn là trợ lý hỗ trợ tra cứu dữ liệu tác dụng phụ, đang trò chuyện bằng tiếng Việt "
    "với người dùng phổ thông. Hãy nói tự nhiên, ấm áp và rõ ràng như trong một cuộc trò "
    "chuyện; không dùng giọng báo cáo, không mở đầu mọi câu theo cùng một công thức, "
    "và không giả vờ là bác sĩ hay con người. Chỉ dùng thông tin trong đồ thị Decagon "
    "được cung cấp. Tên thuốc và protein đã kèm tên cùng mã nguồn.\n\n"
    "Trả về một JSON object có khóa predictions. Mỗi dự đoán gồm side_effect (nhãn "
    "tiếng Anh, chép chính xác), tier (1, 2 hoặc 3), evidence_ids (một hay nhiều ID "
    "chính xác trong evidence được cung cấp) và explanation. explanation phải bằng "
    "tiếng Việt, ngắn gọn (1–2 câu), giải thích tác dụng phụ bằng lời đời thường. "
    "Hãy diễn đạt linh hoạt theo ngữ cảnh, tránh lặp khuôn kiểu “Đã ghi nhận...” ở "
    "mọi mục. Giao diện đã hiện tên thuốc và cặp thuốc nên không cần kể lại chúng. "
    "Với Tier 2, có thể nhắc một cách tự nhiên rằng dữ liệu cho thấy các thuốc có "
    "protein đích chung; đừng biến điều đó thành khẳng định nguy cơ tăng. Với Tier 3, "
    "nói rõ đây chỉ là giả thuyết chưa được ghi nhận. Chỉ mô tả chung tác dụng phụ, "
    "không tự suy diễn nguyên nhân hay cơ chế ngoài evidence, không đưa liều dùng, "
    "không khuyên bắt đầu, ngừng hay đổi thuốc. Không nhắc ID evidence, mã CID, cạnh, "
    "subgraph, tier hay đồ thị trong lời giải thích.\n\n"
    "Mỗi evidence ID phải tồn tại, thuộc tier được nêu và hỗ trợ đúng nhãn tác dụng "
    "phụ; dự đoán dùng ID lạ hoặc không khớp sẽ bị loại. Tier 1 phải là tác dụng phụ "
    "đã ghi nhận cho cặp thuốc. Tier 2 phải là tác dụng phụ đã ghi nhận cho cặp có "
    "protein đích chung với ít nhất ba thuốc đang xét. Chỉ tạo giả thuyết Tier 3 khi "
    "evidence cầu nối được cung cấp hỗ trợ cặp thuốc đó và cặp chưa có cạnh tác dụng "
    "phụ đã ghi nhận; side_effect của Tier 3 phải chép đúng từ candidate_side_effects. "
    "Ưu tiên các mục đầu trong top_side_effects, nhất là mục được đánh dấu nghiêm trọng. "
    "Nếu evidence_truncated là true, đây chỉ là mẫu giới hạn của phác đồ, không phải "
    "toàn bộ dữ liệu liên quan. Trả tối đa 15 dự đoán, ưu tiên Tier 1 rồi Tier 2 và "
    "không quá hai giả thuyết Tier 3. Không bịa evidence, không khẳng định chắc chắn "
    "hoặc trình bày giả thuyết Tier 3 như sự thật lâm sàng. Nếu evidence không đủ, "
    "trả {\"predictions\": []}."
)


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
                            {"role": "system", "content": SYSTEM_PROMPT},
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
    finish_reason = None
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMResponseError("LLM provider returned a body that is not JSON") from exc
    # OpenRouter reports upstream failures (overload, timeout) as HTTP 200
    # with {"error": {...}} instead of choices.
    if isinstance(body, dict) and body.get("error"):
        detail = body["error"]
        message = detail.get("message") if isinstance(detail, dict) else str(detail)
        raise LLMResponseError(f"LLM provider error: {message}")
    try:
        choice = body["choices"][0]
        finish_reason = choice.get("finish_reason")
        message = choice["message"]
        content = message.get("content") or message.get("reasoning") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        return _coerce_predictions(content)
    except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
        error = LLMResponseError(
            "LLM returned a response that does not match the required JSON schema "
            f"(finish_reason={finish_reason}, {len(content)} characters)"
        )
        error.raw_content = content
        raise error from exc


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
            # Every cited bridge must list the label among its candidates; a
            # bridge without candidates supports no label at all.
            if any(
                label not in {name.casefold() for name in item.candidates}
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
