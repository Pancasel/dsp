"""Shape of the JSON the LLM must return, and the errors raised when it does not."""

from pydantic import BaseModel, Field


class ModelPrediction(BaseModel):
    # English side-effect name, copied exactly from the evidence.
    side_effect: str = Field(min_length=1)
    tier: int = Field(ge=1, le=3)
    evidence_ids: list[str] = Field(min_length=1)
    # Short Vietnamese explanation shown to the user.
    explanation: str = Field(min_length=1)


class PredictionList(BaseModel):
    predictions: list[ModelPrediction]


class LLMConfigurationError(RuntimeError):
    """The LLM cannot be called (for example LLM_API_KEY is empty)."""


class LLMResponseError(RuntimeError):
    """The LLM answered, but the answer is an error or does not match the JSON shape."""
