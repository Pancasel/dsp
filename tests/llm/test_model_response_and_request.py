import pytest

from graphrag.llm.model_response_parsing import parse_predictions
from graphrag.llm.prediction_data_models import LLMResponseError
from graphrag.llm.prediction_request import generate_predictions
from graphrag.llm.system_prompt import SYSTEM_PROMPT
from graphrag.settings.application_settings import Settings


def test_single_prediction_object_is_accepted() -> None:
    raw = (
        '{"side_effect": "abdominal distension", "tier": 1, '
        '"evidence_ids": ["E1"], "explanation": "Documented pairwise side effect."}'
    )
    parsed = parse_predictions(raw)
    assert len(parsed) == 1
    assert parsed[0].side_effect == "abdominal distension"
    assert parsed[0].evidence_ids == ["E1"]


def test_predictions_wrapper_and_markdown_fence() -> None:
    raw = """```json
{"predictions": [{"side_effect": "nausea", "tier": 2, "evidence_ids": ["E2"], "explanation": "Shared target."}]}
```"""
    parsed = parse_predictions(raw)
    assert parsed[0].tier == 2


def test_truncated_predictions_array_keeps_complete_objects() -> None:
    raw = """{"predictions": [
      {"side_effect": "nausea", "tier": 1, "evidence_ids": ["E1"], "explanation": "Recorded."},
      {"side_effect": "pain", "tier": 1, "evidence_ids": ["E2"], "explanation": "Record"""
    parsed = parse_predictions(raw)
    assert len(parsed) == 1
    assert parsed[0].side_effect == "nausea"


def _settings():
    return Settings(
        drug_targets_file=None, protein_interactions_file=None,
        drug_pair_side_effects_file=None, cache_database_file=None,
        hub_degree_threshold=500, tier3_one_hop_bridges_per_pair=8,
        tier3_two_hop_bridges_per_pair=4, max_evidence_items=400, llm_api_key="test-key", llm_base_url="https://llm.test/v1",
        llm_model="test/model",
    )


def _respond(monkeypatch: pytest.MonkeyPatch, body: dict) -> None:
    import httpx

    def post(self, url, **kwargs):  # no request leaves the machine
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", post)


def test_provider_error_in_http_200_body_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _respond(monkeypatch, {"error": {"message": "A Timeout Occurred", "code": 524}})
    with pytest.raises(LLMResponseError, match="A Timeout Occurred"):
        generate_predictions(_settings(), ["D1"], {})


def test_empty_content_reports_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _respond(monkeypatch, {"choices": [{"finish_reason": "length", "message": {"content": ""}}]})
    with pytest.raises(LLMResponseError, match="finish_reason=length"):
        generate_predictions(_settings(), ["D1"], {})


def test_request_prompt_encourages_natural_tone_without_weakening_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    payloads = []

    def post(self, url, **kwargs):
        payloads.append(kwargs["json"])
        body = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"predictions": []}',
                    },
                }
            ]
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", post)
    generate_predictions(_settings(), ["D1"], {"evidence": []})

    sent_prompt = payloads[0]["messages"][0]["content"]
    assert sent_prompt == SYSTEM_PROMPT
    assert "nói tự nhiên, ấm áp và rõ ràng" in sent_prompt
    assert "tránh lặp khuôn" in sent_prompt
    assert "không khẳng định chắc chắn" in sent_prompt
    assert "không khuyên bắt đầu, ngừng hay đổi thuốc" in sent_prompt
