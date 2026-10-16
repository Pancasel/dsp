import base64
import io

import httpx
import pytest
import respx
from PIL import Image

from graphrag.config import Settings
from graphrag.vision import ImageError, detected_names, parse_vision_json, prepare_image, read_image


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_prepare_image_downscales_to_jpeg() -> None:
    url = prepare_image(_png(3200, 800), "image/png")
    assert url.startswith("data:image/jpeg;base64,")
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(image.size) == 1600


@pytest.mark.parametrize(
    ("data", "content_type", "status"),
    [(b"x", "text/plain", 415), (b"not an image", "image/png", 415), (b"0" * (9 * 1024 * 1024), "image/jpeg", 413)],
    ids=["wrong-type", "not-an-image", "too-large"],
)
def test_prepare_image_rejects(data: bytes, content_type: str, status: int) -> None:
    with pytest.raises(ImageError) as error:
        prepare_image(data, content_type)
    assert error.value.status_code == status


def test_parse_and_extract_names() -> None:
    payload = parse_vision_json(
        '```json\n{"drugs": [{"brand": "Panadol", "active_ingredients": ["Paracetamol"], '
        '"strength": "500mg"}, {"brand": null, "active_ingredients": "Ibuprofen"}], '
        '"raw_text": "PANADOL"}\n```'
    )
    assert detected_names(payload) == ["Paracetamol", "Panadol", "Ibuprofen"]
    assert detected_names({"drugs": [], "raw_text": ""}) == []


def test_read_image_uses_configured_vision_model() -> None:

    # The HTTP call is intercepted by respx; nothing is sent to a real provider.
    settings = Settings(
        targets_file=None, ppi_file=None, combo_file=None, cache_db=None,
        hub_degree_threshold=500, tier3_one_hop_per_pair=8, tier3_two_hop_per_pair=4,
        max_evidence=400, llm_api_key="test-key", llm_base_url="https://llm.test/v1",
        llm_model="text-model", vision_model="vision-model",
    )
    with respx.mock:
        route = respx.post("https://llm.test/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"drugs": [{"brand": "Brufen", "active_ingredients": ["Ibuprofen"]}], "raw_text": "BRUFEN"}'}}]},
            )
        )
        payload = read_image(settings, "data:image/jpeg;base64,AAAA")
        sent = route.calls[0].request
    assert b'"vision-model"' in sent.content
    assert sent.headers["Authorization"] == "Bearer test-key"
    assert detected_names(payload) == ["Ibuprofen", "Brufen"]


def test_vision_can_use_a_separate_provider() -> None:
    from dataclasses import replace

    base = Settings(
        targets_file=None, ppi_file=None, combo_file=None, cache_db=None,
        hub_degree_threshold=500, tier3_one_hop_per_pair=8, tier3_two_hop_per_pair=4,
        max_evidence=400, llm_api_key="llm-key", llm_base_url="https://llm.test/v1",
        llm_model="text-model", vision_model="vision-model",
    )
    settings = replace(base, vision_base_url="https://vision.test/v1", vision_api_key="vision-key")
    with respx.mock:
        route = respx.post("https://vision.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": '{"drugs": []}'}}]})
        )
        payload = read_image(settings, "data:image/jpeg;base64,AAAA")
        assert route.calls[0].request.headers["Authorization"] == "Bearer vision-key"
    assert payload["drugs"] == []
