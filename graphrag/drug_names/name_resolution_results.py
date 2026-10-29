"""What resolving one piece of user text into a drug returns."""

from dataclasses import asdict, dataclass, field


@dataclass
class MatchCandidate:
    """A drug the text may mean, with how well it matched."""

    stitch_id: str
    name: str
    # The known name (brand, generic or synonym) that the text matched.
    matched: str
    # Blend of RapidFuzz WRatio and plain ratio, 0-100.
    score: float
    # Plain character similarity, 0-100.
    ratio: float = 0.0


@dataclass
class DrugNameResolution:
    query: str
    # None when the drug is not found or the text is ambiguous.
    stitch_id: str | None
    name: str | None
    # "stitch_id" | "exact" | "brand_alias" | "fuzzy" | "ambiguous" | "not_found"
    method: str
    score: float
    candidates: list[MatchCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
