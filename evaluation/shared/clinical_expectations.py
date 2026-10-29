"""Expected outcomes of the clinical benchmark cases (A5, A7, A8) and how to find them in a ranking."""

import re
import statistics


# Decagon side-effect names (regular expressions) that count as the expected
# outcome of a benchmark case; the key is the "expected" column of the case files.
EXPECTED_EFFECT_PATTERNS = {
    "bleeding": r"haemorrhag|hemorrhag|bleed|haematemesis|melaena|haematochezia|coughing blood|haemarthrosis|hematoma|haematoma",
    "kidney": r"kidney failure|renal failure|azotaemia",
    "hyperkalaemia": r"excess potassium|hyperkal",
    "bradycardia": r"bradycardia|bradyarrhythmia|heart block|atrioventricular block|sinus arrest",
    "arrhythmia": r"arrhythmia|tachycardia ventricular|ventricular fibrillation|torsade|\bafib\b|\bfib\b|extrasystoles|bradycardia",
    "rhabdomyolysis": r"rhabdomyolysis|myoglobinuria|muscle inflammation|myopathy",
    "hypoglycaemia": r"hypoglyc",
    "myelosuppression": r"neutropenia|agranulocyt|pancytopenia|bone marrow|thrombocytopenia|leucopenia|aplas",
    "cns_respiratory_depression": r"respiratory failure|hypoventilation|\bcoma\b|apnea|apnoea|loss of consciousness|pulmonary arrest|respiratory depression",
    "seizure_serotonin": r"convulsion|seizure|epilep|serotonin",
    "hypotension": r"hypotension|fainting|syncope|arterial pressure nos decreased",
}
EXPECTED_EFFECT_VIETNAMESE = {
    "bleeding": "chảy máu", "kidney": "suy thận", "hyperkalaemia": "tăng kali máu",
    "bradycardia": "nhịp chậm / block tim", "arrhythmia": "loạn nhịp", "rhabdomyolysis": "tiêu cơ vân",
    "hypoglycaemia": "hạ đường huyết", "myelosuppression": "ức chế tủy",
    "cns_respiratory_depression": "ức chế thần kinh / hô hấp", "seizure_serotonin": "co giật / serotonin",
    "hypotension": "tụt huyết áp",
}
# The interface shows this many serious effects.
SERIOUS_EFFECTS_SHOWN = 10


def expected_effect_pattern(expected_outcome: str) -> re.Pattern[str]:
    return re.compile(EXPECTED_EFFECT_PATTERNS[expected_outcome], re.IGNORECASE)


def rank_of_first_match(effects: list[dict[str, object]], pattern: re.Pattern[str]) -> int | None:
    """1-based position of the first effect whose name matches, None when none does."""
    for position, effect in enumerate(effects, start=1):
        if pattern.search(effect["name"]):
            return position
    return None


def hit_rate(ranks: list[int | None], top: int) -> float:
    """Share of cases whose expected effect is within the first `top` positions."""
    return sum(1 for rank in ranks if rank and rank <= top) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Average of 1/rank, counting 0 for a case whose expected effect was not found."""
    return statistics.mean(1 / rank if rank else 0 for rank in ranks)
