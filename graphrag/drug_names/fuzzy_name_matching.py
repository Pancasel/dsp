"""Approximate matching of a misspelled name against all known names (RapidFuzz)."""

from rapidfuzz import fuzz, process

from graphrag.drug_names.name_index import NameIndex
from graphrag.drug_names.name_resolution_results import MatchCandidate


# RapidFuzz first returns this many rough matches above the cut-off ...
ROUGH_MATCH_COUNT = 40
ROUGH_MATCH_MINIMUM_SCORE = 60
# ... and the caller gets at most this many drugs, best first.
DEFAULT_CANDIDATE_LIMIT = 5


def find_fuzzy_candidates(
    normalized_query: str,
    known_keys: list[str],
    entry_by_key: dict[str, tuple[str, str, str]],
    names: NameIndex,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[MatchCandidate]:
    """Best match per drug, sorted by score.

    entry_by_key maps a normalized known name to (stitch_id, original text, method).
    """
    rough_matches = process.extract(
        normalized_query,
        known_keys,
        scorer=fuzz.WRatio,
        limit=ROUGH_MATCH_COUNT,
        score_cutoff=ROUGH_MATCH_MINIMUM_SCORE,
    )
    best_by_drug: dict[str, MatchCandidate] = {}
    for matched_key, _rough_score, key_position in rough_matches:
        # WRatio rewards short substrings; blending it with the plain ratio
        # keeps "aspirin" from matching "aspirin pH8" better than a real typo fix.
        plain_ratio = fuzz.ratio(normalized_query, matched_key)
        blended_score = round(
            0.5 * fuzz.WRatio(normalized_query, matched_key) + 0.5 * plain_ratio, 1
        )
        stitch_id, original_text, _method = entry_by_key[known_keys[key_position]]
        current_best = best_by_drug.get(stitch_id)
        if current_best is None or blended_score > current_best.score:
            best_by_drug[stitch_id] = MatchCandidate(
                stitch_id,
                names.drug_name(stitch_id),
                original_text,
                blended_score,
                round(plain_ratio, 1),
            )

    def highest_score_first(candidate: MatchCandidate) -> float:
        return -candidate.score

    return sorted(best_by_drug.values(), key=highest_score_first)[:limit]
