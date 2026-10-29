"""Read the predictions out of the LLM's text, even when the JSON is wrapped, bare or cut off."""

import json

from pydantic import ValidationError

from graphrag.llm.prediction_data_models import ModelPrediction, PredictionList


def remove_markdown_code_fence(text: str) -> str:
    """"```json\\n{...}\\n```" -> "{...}"."""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def load_json_value(content: str) -> object:
    """Parse the text; on failure, parse what lies between the first "{" and the last "}"."""
    text = content.strip()
    if text.startswith("```"):
        text = remove_markdown_code_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def recover_complete_predictions(content: str) -> list[ModelPrediction]:
    """Keep the prediction objects that are complete when the model stopped mid-JSON."""
    decoder = json.JSONDecoder()
    found: list[ModelPrediction] = []
    search_from = 0
    while True:
        object_start = content.find("{", search_from)
        if object_start < 0:
            break
        try:
            payload, object_end = decoder.raw_decode(content, object_start)
        except json.JSONDecodeError:
            search_from = object_start + 1
            continue
        search_from = object_end
        possible_predictions = []
        if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
            possible_predictions = payload["predictions"]
        elif isinstance(payload, dict):
            possible_predictions = [payload]
        elif isinstance(payload, list):
            possible_predictions = payload
        for possible_prediction in possible_predictions:
            try:
                found.append(ModelPrediction.model_validate(possible_prediction))
            except ValidationError:
                continue
    if not found:
        raise ValueError("LLM JSON did not contain a complete prediction")
    return found


def parse_predictions(content: str) -> list[ModelPrediction]:
    """Accept {"predictions": [...]}, a bare prediction, a bare array, or truncated JSON."""
    try:
        payload = load_json_value(content)
    except json.JSONDecodeError:
        return recover_complete_predictions(content)
    if isinstance(payload, dict) and "predictions" in payload:
        try:
            return PredictionList.model_validate(payload).predictions
        except ValidationError:
            return recover_complete_predictions(content)
    if isinstance(payload, dict):
        return [ModelPrediction.model_validate(payload)]
    if isinstance(payload, list):
        return [ModelPrediction.model_validate(item) for item in payload]
    raise ValueError("LLM JSON must be an object or an array")
