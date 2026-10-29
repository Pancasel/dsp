"""Send one request with all the evidence to the LLM provider and return its predictions.

The provider must speak the OpenAI chat completions protocol (OpenRouter by default).
"""

import json
import time

import httpx
from pydantic import ValidationError

from graphrag.llm.model_response_parsing import parse_predictions
from graphrag.llm.prediction_data_models import (
    LLMConfigurationError,
    LLMResponseError,
    ModelPrediction,
)
from graphrag.llm.system_prompt import SYSTEM_PROMPT
from graphrag.settings.application_settings import Settings


CONNECT_TIMEOUT_SECONDS = 15.0
CONNECTION_ATTEMPTS = 3
MAX_ANSWER_TOKENS = 4000


def generate_predictions(
    settings: Settings,
    drugs: list[str],
    llm_context: dict[str, object],
) -> list[ModelPrediction]:
    if not settings.llm_api_key:
        raise LLMConfigurationError("LLM_API_KEY is not configured")

    user_message = json.dumps(
        {"drugs": drugs, "retrieved_subgraph": llm_context},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    response = _post_with_retries(settings, user_message)
    content = ""
    finish_reason = None
    try:
        body = response.json()
    except ValueError as error:
        raise LLMResponseError("LLM provider returned a body that is not JSON") from error
    # OpenRouter reports upstream failures (overload, timeout) as HTTP 200
    # with {"error": {...}} instead of choices.
    if isinstance(body, dict) and body.get("error"):
        error_detail = body["error"]
        if isinstance(error_detail, dict):
            error_message = error_detail.get("message")
        else:
            error_message = str(error_detail)
        raise LLMResponseError(f"LLM provider error: {error_message}")
    try:
        choice = body["choices"][0]
        finish_reason = choice.get("finish_reason")
        message = choice["message"]
        content = message.get("content") or message.get("reasoning") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        return parse_predictions(content)
    except (KeyError, IndexError, TypeError, ValueError, ValidationError) as error:
        response_error = LLMResponseError(
            "LLM returned a response that does not match the required JSON schema "
            f"(finish_reason={finish_reason}, {len(content)} characters)"
        )
        response_error.raw_content = content
        raise response_error from error


def _post_with_retries(settings: Settings, user_message: str) -> httpx.Response:
    """POST the chat request; a failed connection is retried after 0.5 s, then 1 s."""
    url = f"{settings.llm_base_url}/chat/completions"
    timeout = httpx.Timeout(settings.llm_timeout_seconds, connect=CONNECT_TIMEOUT_SECONDS)
    with httpx.Client(timeout=timeout) as client:
        response = None
        for attempt in range(CONNECTION_ATTEMPTS):
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
                        "max_tokens": MAX_ANSWER_TOKENS,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                    },
                )
                break
            except httpx.ConnectError:
                if attempt == CONNECTION_ATTEMPTS - 1:
                    raise
                time.sleep(0.5 * (2**attempt))
        if response is None:
            raise LLMResponseError("LLM provider did not return a response")
        response.raise_for_status()
    return response
