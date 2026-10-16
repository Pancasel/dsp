import pytest

from graphrag.llm import _coerce_predictions


def test_single_prediction_object_is_accepted() -> None:
    raw = (
        '{"side_effect": "abdominal distension", "tier": 1, '
        '"evidence_ids": ["E1"], "explanation": "Documented pairwise side effect."}'
    )
    parsed = _coerce_predictions(raw)
    assert len(parsed) == 1
    assert parsed[0].side_effect == "abdominal distension"
    assert parsed[0].evidence_ids == ["E1"]


def test_predictions_wrapper_and_markdown_fence() -> None:
    raw = """```json
{"predictions": [{"side_effect": "nausea", "tier": 2, "evidence_ids": ["E2"], "explanation": "Shared target."}]}
```"""
    parsed = _coerce_predictions(raw)
    assert parsed[0].tier == 2


def test_truncated_predictions_array_keeps_complete_objects() -> None:
    raw = """{"predictions": [
      {"side_effect": "nausea", "tier": 1, "evidence_ids": ["E1"], "explanation": "Recorded."},
      {"side_effect": "pain", "tier": 1, "evidence_ids": ["E2"], "explanation": "Record"""
    parsed = _coerce_predictions(raw)
    assert len(parsed) == 1
    assert parsed[0].side_effect == "nausea"
