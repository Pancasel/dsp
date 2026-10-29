from graphrag.settings.ranking_weights import RankingWeights
from graphrag.side_effect_ranking.effects_to_display import (
    EffectDisplayLimits,
    choose_effects_to_display,
)
from graphrag.side_effect_ranking.regimen_side_effect_scoring import rank_effects
from graphrag.source_data.data_records import SideEffect


def no_single_drug_effects(drug: str) -> set[str]:
    return set()


def test_rare_serious_and_interaction_only_effects_rank_first() -> None:
    pair = ("A", "B")
    effects_by_pair = {
        pair: [
            SideEffect("C_common", "nausea"),
            SideEffect("C_rare", "rhabdomyolysis"),
            SideEffect("C_mono", "rash"),
        ]
    }

    def single_drug_effect_codes(drug: str) -> set[str]:
        if drug == "A":
            return {"C_mono"}
        return set()

    ranked = rank_effects(
        effects_by_pair,
        amplified_pairs=set(),
        pair_count_by_effect_code={"C_common": 1000, "C_rare": 5, "C_mono": 5},
        total_pair_count=1000,
        single_drug_effect_codes=single_drug_effect_codes,
        is_serious=lambda effect_name: effect_name == "rhabdomyolysis",
    )
    assert [effect.name for effect in ranked] == ["rhabdomyolysis", "rash", "nausea"]
    assert ranked[0].serious and ranked[0].absent_from_single_drug_data
    assert ranked[1].absent_from_single_drug_data is False
    assert ranked[2].effect_rarity == 0.0


def test_recurrence_across_pairs_and_weight_ablation() -> None:
    effects_by_pair = {
        ("A", "B"): [SideEffect("C1", "bleeding")],
        ("A", "C"): [SideEffect("C1", "bleeding"), SideEffect("C2", "dizziness")],
    }
    ranked = rank_effects(
        effects_by_pair, {("A", "C")}, {"C1": 10, "C2": 10}, 100, no_single_drug_effects
    )
    assert ranked[0].name == "bleeding"
    assert ranked[0].recurrence == 2
    assert ranked[0].pairs == [("A", "B"), ("A", "C")]
    assert ranked[0].tier == 2  # one of its pairs is amplified
    assert ranked[0].amplified_pairs == [("A", "C")]  # only that pair, not ("A", "B")

    rarity_only = rank_effects(
        effects_by_pair,
        set(),
        {"C1": 50, "C2": 2},
        100,
        no_single_drug_effects,
        weights=RankingWeights(
            recurrence=0, effect_rarity=1, absent_from_single_drug_data=0, serious=0, amplified=0
        ),
    )
    assert rarity_only[0].name == "dizziness"


def test_display_prefers_serious_and_limits_each_group() -> None:
    ranked = [
        {"name": "rash", "serious": False},
        {"name": "bleeding", "serious": True},
        {"name": "nausea", "serious": False},
        {"name": "coma", "serious": True},
        {"name": "pain", "serious": False},
    ]
    chosen = choose_effects_to_display(
        ranked, EffectDisplayLimits(max_serious_effects=1, max_other_effects=2)
    )
    assert [effect["name"] for effect in chosen] == ["bleeding", "rash", "nausea"]
