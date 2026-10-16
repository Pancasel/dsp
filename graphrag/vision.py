"""Read drug names from a photo of a box, blister or prescription.

The image goes to the vision model configured as VISION_MODEL (OpenAI-compatible
chat completions with an image_url part). The model only transcribes names; the
names are then matched by graphrag.resolver, and the user confirms them in the UI.
"""

import base64
import io
import json

import httpx
from PIL import Image, UnidentifiedImageError

from graphrag.config import Settings
from graphrag.llm import LLMConfigurationError, LLMResponseError


ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 8 * 1024 * 1024
MAX_SIDE = 1600

PROMPT = (
    "Ảnh là hộp, vỉ, chai thuốc hoặc đơn thuốc. Hãy đọc chữ in trên đó và trả về "
    "DUY NHẤT một JSON object dạng "
    '{"drugs": [{"brand": "tên thương mại hoặc null", "active_ingredients": '
    '["tên hoạt chất"], "strength": "hàm lượng hoặc null"}], "raw_text": "chữ đọc được"}. '
    "Mỗi sản phẩm thuốc là một phần tử. Chỉ ghi những gì thực sự đọc được trên ảnh, "
    "không đoán. Nếu không đọc được tên thuốc nào, trả về {\"drugs\": [], \"raw_text\": \"\"}."
)


class ImageError(ValueError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def prepare_image(data: bytes, content_type: str | None) -> str:
    """Validate, downscale and return a JPEG data URL."""
    if content_type not in ALLOWED_TYPES:
        raise ImageError("Chỉ nhận ảnh JPEG, PNG hoặc WebP", 415)
    if len(data) > MAX_BYTES:
        raise ImageError("Ảnh lớn hơn 8MB", 413)
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError("Không đọc được file ảnh", 415) from exc
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_vision_json(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in vision output")
    payload = json.loads(text[start : end + 1])
    drugs = payload.get("drugs") or []
    if not isinstance(drugs, list):
        raise ValueError("drugs must be a list")
    return {"drugs": drugs, "raw_text": str(payload.get("raw_text") or "")}


def detected_names(payload: dict[str, object]) -> list[str]:
    """Names to resolve: active ingredients first, then the brand of each product."""
    names: list[str] = []
    for product in payload["drugs"]:
        if not isinstance(product, dict):
            continue
        ingredients = product.get("active_ingredients") or []
        if isinstance(ingredients, str):
            ingredients = [ingredients]
        found = [str(item).strip() for item in ingredients if str(item).strip()]
        brand = str(product.get("brand") or "").strip()
        if brand and brand.lower() != "null":
            found.append(brand)
        for name in found:
            if name.casefold() not in {item.casefold() for item in names}:
                names.append(name)
    return names


def vision_endpoint(settings: Settings) -> tuple[str, str]:
    """Base URL and key for the vision provider, falling back to the LLM's."""
    return (
        settings.vision_base_url or settings.llm_base_url,
        settings.vision_api_key or settings.llm_api_key,
    )


def read_image(settings: Settings, data_url: str) -> dict[str, object]:
    base_url, api_key = vision_endpoint(settings)
    if not api_key:
        raise LLMConfigurationError("VISION_API_KEY (or LLM_API_KEY) is not configured")
    if not settings.vision_model:
        raise LLMConfigurationError("VISION_MODEL is not configured")
    with httpx.Client(timeout=httpx.Timeout(settings.llm_timeout, connect=15.0)) as client:
        response = client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://127.0.0.1:8000",
                "X-Title": "Decagon Polypharmacy Graph-RAG",
            },
            json={
                "model": settings.vision_model,
                "temperature": 0,
                "max_tokens": 1500,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            },
        )
        response.raise_for_status()
    try:
        content = response.json()["choices"][0]["message"].get("content") or ""
        return parse_vision_json(content)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMResponseError("Vision model returned an unreadable answer") from exc
