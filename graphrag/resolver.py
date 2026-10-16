"""Turn user text (generic name, Vietnamese brand, STITCH ID, typo) into a drug ID.

Order: STITCH ID -> exact normalized name/synonym/brand -> fuzzy match.
A fuzzy match is accepted automatically only at score >= ACCEPT_SCORE; between
SUGGEST_SCORE and ACCEPT_SCORE the caller gets candidates to confirm.
"""

import re
from dataclasses import asdict, dataclass, field

from rapidfuzz import fuzz, process

from graphrag.names import NameIndex, normalize


ACCEPT_SCORE = 90.0
# Plain character similarity required for automatic acceptance. It separates
# typos ("ibuprofn", ratio 94) from different drugs ("esomeprazole" vs
# "omeprazole", ratio 91) and partial words ("vitamin c" vs "vitamin d2").
ACCEPT_RATIO = 92.0
SUGGEST_SCORE = 75.0
MIN_KEY_LENGTH = 3
_SPLIT = re.compile(r"\s*(?:[,;+/\n]|\bva\b|\band\b|\bvà\b)\s*", re.IGNORECASE)


@dataclass
class Candidate:
    stitch_id: str
    name: str
    matched: str
    score: float
    ratio: float = 0.0


@dataclass
class ResolveResult:
    query: str
    stitch_id: str | None
    name: str | None
    method: str  # stitch_id | exact | brand_alias | fuzzy | ambiguous | not_found
    score: float
    candidates: list[Candidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def split_query(text: str) -> list[str]:
    """Split "Paracetamol + Codeine, ibuprofen" into separate drug strings."""
    return [part.strip() for part in _SPLIT.split(text) if part and part.strip()]


class DrugResolver:
    def __init__(self, names: NameIndex, drug_ids: set[str]) -> None:
        self.names = names
        self.drug_ids = drug_ids
        # normalized key -> (stitch_id, original text, method)
        self._exact: dict[str, tuple[str, str, str]] = {}
        for stitch_id in sorted(drug_ids):
            drug = names.drugs.get(stitch_id)
            if drug is None:
                continue
            for text in [drug.name, *drug.synonyms]:
                key = normalize(text)
                if len(key) >= MIN_KEY_LENGTH:
                    self._exact.setdefault(key, (stitch_id, text, "exact"))
            for alias in drug.aliases_vi:
                key = normalize(alias)
                if len(key) >= MIN_KEY_LENGTH:
                    # Curated brands win over PubChem synonyms with the same text.
                    self._exact[key] = (stitch_id, alias, "brand_alias")
        self._keys = list(self._exact)

    def resolve(self, query: str) -> ResolveResult:
        raw = query.strip()
        upper = raw.upper()
        if upper in self.drug_ids:
            return ResolveResult(raw, upper, self.names.drug_name(upper), "stitch_id", 100.0)

        key = normalize(raw)
        if not key:
            return ResolveResult(raw, None, None, "not_found", 0.0)
        hit = self._exact.get(key)
        if hit:
            stitch_id, _, method = hit
            return ResolveResult(raw, stitch_id, self.names.drug_name(stitch_id), method, 100.0)

        candidates = self._fuzzy(key)
        if not candidates or candidates[0].score < SUGGEST_SCORE:
            return ResolveResult(raw, None, None, "not_found", candidates[0].score if candidates else 0.0, candidates[:3])
        best = candidates[0]
        runner_up = candidates[1].score if len(candidates) > 1 else 0.0
        # Accept only a clear winner; two close candidates must be confirmed.
        if (
            best.score >= ACCEPT_SCORE
            and best.ratio >= ACCEPT_RATIO
            and best.score - runner_up >= 3
        ):
            return ResolveResult(raw, best.stitch_id, best.name, "fuzzy", best.score, candidates)
        return ResolveResult(raw, None, None, "ambiguous", best.score, candidates)

    def resolve_many(self, text: str) -> list[ResolveResult]:
        return [self.resolve(part) for part in split_query(text)]

    def _fuzzy(self, key: str, limit: int = 5) -> list[Candidate]:
        matches = process.extract(
            key,
            self._keys,
            scorer=fuzz.WRatio,
            limit=40,
            score_cutoff=60,
        )
        best: dict[str, Candidate] = {}
        for matched_key, _, index in matches:
            # WRatio rewards short substrings; blend with plain ratio so that
            # "aspirin" does not match "aspirin pH8" better than a real typo fix.
            ratio = fuzz.ratio(key, matched_key)
            score = round(0.5 * fuzz.WRatio(key, matched_key) + 0.5 * ratio, 1)
            stitch_id, original, _ = self._exact[self._keys[index]]
            current = best.get(stitch_id)
            if current is None or score > current.score:
                best[stitch_id] = Candidate(
                    stitch_id, self.names.drug_name(stitch_id), original, score, round(ratio, 1)
                )
        return sorted(best.values(), key=lambda item: -item.score)[:limit]
