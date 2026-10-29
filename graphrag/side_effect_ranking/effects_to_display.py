"""Which ranked side effects are shown to the user: the top serious ones, then a few others."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EffectDisplayLimits:
    max_serious_effects: int
    max_other_effects: int


def choose_effects_to_display(
    ranked_effects: list[dict[str, object]], limits: EffectDisplayLimits
) -> list[dict[str, object]]:
    """ranked_effects is already sorted by score (RankedEffect.to_dict()), so each group keeps that order."""
    serious_effects = [effect for effect in ranked_effects if effect["serious"]]
    other_effects = [effect for effect in ranked_effects if not effect["serious"]]
    return (
        serious_effects[: limits.max_serious_effects]
        + other_effects[: limits.max_other_effects]
    )
