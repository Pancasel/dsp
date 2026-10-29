"""Score and order the documented side effects of a whole regimen.

A Decagon drug pair records 53 side effects at the median (up to 540), so the
raw list is unusable. Each effect gets a transparent additive score:

    score = weight.recurrence                   * recurrence    # pairs in the regimen that record it
          + weight.effect_rarity                * effect_rarity # log(total pairs / pairs with the effect), the "IDF"
          + weight.absent_from_single_drug_data * absent        # not a single-drug effect of either drug
          + weight.serious                      * serious       # EMA Important Medical Event
          + weight.critical                     * critical      # life-threatening class, only when serious
          + weight.amplified                    * amplified     # the pair shares a target with a third drug

The weights live in Settings.ranking_weights so that they can be tested
separately (experiment E5).
"""

import math
from dataclasses import dataclass, field
from typing import Callable

from graphrag.settings.ranking_weights import RankingWeights
from graphrag.source_data.data_records import SideEffect


@dataclass
class RankedEffect:
    code: str
    name: str
    pairs: list[tuple[str, str]] = field(default_factory=list)
    # The pairs that share a target with a third regimen drug (Tier 2).
    amplified_pairs: list[tuple[str, str]] = field(default_factory=list)
    recurrence: int = 0
    effect_rarity: float = 0.0
    absent_from_single_drug_data: bool = False
    serious: bool = False
    critical: bool = False
    amplified: bool = False
    score: float = 0.0

    @property
    def tier(self) -> int:
        if self.amplified:
            return 2
        return 1

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "name": self.name,
            "tier": self.tier,
            "pairs": [list(pair) for pair in self.pairs],
            "amplified_pairs": [list(pair) for pair in self.amplified_pairs],
            "recurrence": self.recurrence,
            "effect_rarity": round(self.effect_rarity, 3),
            "absent_from_single_drug_data": self.absent_from_single_drug_data,
            "serious": self.serious,
            "critical": self.critical,
            "amplified": self.amplified,
            "score": round(self.score, 3),
        }


def rank_effects(
    effects_by_pair: dict[tuple[str, str], list[SideEffect]],
    amplified_pairs: set[tuple[str, str]],
    pair_count_by_effect_code: dict[str, int],
    total_pair_count: int,
    single_drug_effect_codes: Callable[[str], set[str]],
    is_serious: Callable[[str], bool] | None = None,
    weights: RankingWeights | None = None,
    is_critical: Callable[[str], bool] | None = None,
) -> list[RankedEffect]:
    """Merge the effects of every pair into one list, highest score first (then by name)."""
    weights = weights or RankingWeights()
    total_pair_count = max(total_pair_count, 1)
    ranked_by_code: dict[str, RankedEffect] = {}
    for pair, effects in effects_by_pair.items():
        drug_a, drug_b = pair
        single_drug_codes_of_pair = single_drug_effect_codes(drug_a) | single_drug_effect_codes(drug_b)
        for effect in effects:
            ranked_effect = ranked_by_code.get(effect.code)
            if ranked_effect is None:
                pairs_with_effect = max(pair_count_by_effect_code.get(effect.code, 1), 1)
                ranked_effect = RankedEffect(
                    code=effect.code,
                    name=effect.name,
                    effect_rarity=math.log(total_pair_count / pairs_with_effect),
                    serious=bool(is_serious and is_serious(effect.name)),
                    critical=bool(is_critical and is_critical(effect.name)),
                )
                ranked_by_code[effect.code] = ranked_effect
            ranked_effect.pairs.append(pair)
            ranked_effect.recurrence += 1
            if effect.code not in single_drug_codes_of_pair:
                ranked_effect.absent_from_single_drug_data = True
            if pair in amplified_pairs:
                ranked_effect.amplified = True
                ranked_effect.amplified_pairs.append(pair)

    for ranked_effect in ranked_by_code.values():
        ranked_effect.score = (
            weights.recurrence * ranked_effect.recurrence
            + weights.effect_rarity * ranked_effect.effect_rarity
            + weights.absent_from_single_drug_data * ranked_effect.absent_from_single_drug_data
            + weights.serious * ranked_effect.serious
            + weights.critical * (ranked_effect.serious and ranked_effect.critical)
            + weights.amplified * ranked_effect.amplified
        )

    def highest_score_then_name(ranked_effect: RankedEffect) -> tuple[float, str]:
        return (-ranked_effect.score, ranked_effect.name.casefold())

    return sorted(ranked_by_code.values(), key=highest_score_then_name)
