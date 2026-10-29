"""Weights of the side-effect score used to order a regimen's effects.

The score formula is explained in
graphrag/side_effect_ranking/regimen_side_effect_scoring.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RankingWeights:
    # How many drug pairs of the regimen record the effect.
    recurrence: float = 1.0
    # How rare the effect is over all Decagon pairs (IDF, see docs/THUAT_NGU.md).
    effect_rarity: float = 0.5
    # The effect is missing from the single-drug side effects of both drugs.
    absent_from_single_drug_data: float = 1.0
    # The effect is on the EMA Important Medical Event list.
    serious: float = 2.0
    # One pair recording the effect shares a target protein with a third drug (Tier 2).
    amplified: float = 1.0
    # Life-threatening class (bleeding, arrhythmia, ...), only added when serious.
    critical: float = 2.0
