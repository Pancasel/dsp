"""Error raised when a regimen cannot be analysed (unknown drug, fewer than three drugs)."""


class RegimenError(ValueError):
    def __init__(self, message: str, details: list[dict[str, object]] | None = None) -> None:
        super().__init__(message)
        # One DrugNameResolution.to_dict() per input that could not be resolved.
        self.details = details or []
