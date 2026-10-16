"""Regimen-level ranking of documented side effects.

A Decagon drug pair has 53 recorded side effects at the median (up to 540), so
the raw list is unusable. Each effect gets a transparent additive score:

    score = w_recurrence * recurrence    # pairs in the regimen that record it
          + w_idf        * idf           # log(total pairs / pairs with effect)
          + w_not_in_mono * not_in_mono  # absent from both drugs' mono lists
          + w_serious    * serious       # matches resources/serious_keywords.csv
          + w_amplified  * amplified     # pair shares a target with a third drug

The weights live in Settings.ranking so they can be ablated (eval E5).
"""

import math
from dataclasses import dataclass, field
from typing import Callable

from graphrag.config import RankingWeights


@dataclass
class RankedEffect:
    code: str
    name: str
    pairs: list[tuple[str, str]] = field(default_factory=list)
    # Subset of pairs that share a target with a third regimen drug (Tier 2).
    amplified_pairs: list[tuple[str, str]] = field(default_factory=list)
    recurrence: int = 0
    idf: float = 0.0
    not_in_mono: bool = False
    serious: bool = False
    amplified: bool = False
    score: float = 0.0

    @property
    def tier(self) -> int:
        return 2 if self.amplified else 1

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "name": self.name,
            "tier": self.tier,
            "pairs": [list(pair) for pair in self.pairs],
            "amplified_pairs": [list(pair) for pair in self.amplified_pairs],
            "recurrence": self.recurrence,
            "idf": round(self.idf, 3),
            "not_in_mono": self.not_in_mono,
            "serious": self.serious,
            "amplified": self.amplified,
            "score": round(self.score, 3),
        }


def rank_effects(
    pair_effects: dict[tuple[str, str], list],
    amplified_pairs: set[tuple[str, str]],
    effect_pair_count: dict[str, int],
    total_pairs: int,
    mono_codes: Callable[[str], set[str]],
    is_serious: Callable[[str], bool] | None = None,
    weights: RankingWeights | None = None,
) -> list[RankedEffect]:
    """Aggregate pairwise effects over the regimen and sort by score.

    pair_effects maps a sorted drug pair to SideEffect objects (code, name).
    """
    weights = weights or RankingWeights()
    total_pairs = max(total_pairs, 1)
    by_code: dict[str, RankedEffect] = {}
    for pair, effects in pair_effects.items():
        mono = mono_codes(pair[0]) | mono_codes(pair[1])
        for effect in effects:
            item = by_code.get(effect.code)
            if item is None:
                count = max(effect_pair_count.get(effect.code, 1), 1)
                item = RankedEffect(
                    code=effect.code,
                    name=effect.name,
                    idf=math.log(total_pairs / count),
                    serious=bool(is_serious and is_serious(effect.name)),
                )
                by_code[effect.code] = item
            item.pairs.append(pair)
            item.recurrence += 1
            item.not_in_mono = item.not_in_mono or effect.code not in mono
            if pair in amplified_pairs:
                item.amplified = True
                item.amplified_pairs.append(pair)

    for item in by_code.values():
        item.score = (
            weights.recurrence * item.recurrence
            + weights.idf * item.idf
            + weights.not_in_mono * item.not_in_mono
            + weights.serious * item.serious
            + weights.amplified * item.amplified
        )
    return sorted(by_code.values(), key=lambda item: (-item.score, item.name.casefold()))
