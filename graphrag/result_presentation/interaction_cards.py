"""One card per side effect shown to the user, with its explanation and evidence.

Tier 1/2 cards come from the ranked effects (top serious + a few others, plus
any effect the LLM chose to explain). The explanation is the LLM's when a
validated prediction exists, otherwise the template. Tier 3 cards come only
from validated LLM hypotheses.
"""

from graphrag.drug_names.name_index import NameIndex
from graphrag.result_presentation.evidence_path_text import drug_names_of_pair, readable_evidence
from graphrag.result_presentation.template_explanation import template_explanation
from graphrag.result_presentation.vietnamese_labels import TIER_LABELS_VIETNAMESE
from graphrag.side_effect_ranking.effects_to_display import (
    EffectDisplayLimits,
    choose_effects_to_display,
)
from graphrag.source_data.data_records import Evidence


# When neither the LLM nor the evidence cites anything, this many pairs are cited.
MAX_CITED_ITEMS_WITHOUT_LLM = 3


def build_interaction_cards(
    drugs: list[str],
    graph_context: dict[str, object],
    evidence_items: list[Evidence],
    validated_predictions: list[dict[str, object]],
    names: NameIndex,
    limits: EffectDisplayLimits,
) -> list[dict[str, object]]:
    drug_set = set(drugs)
    ranked_effects = graph_context["ranked_effects"]
    ranked_effect_by_name = {effect["name"].casefold(): effect for effect in ranked_effects}

    # First validated Tier 1/2 prediction for each side effect.
    prediction_by_name: dict[str, dict[str, object]] = {}
    for prediction in validated_predictions:
        if prediction["tier"] in (1, 2):
            prediction_by_name.setdefault(str(prediction["side_effect"]).casefold(), prediction)

    evidence_by_effect_name: dict[str, list[Evidence]] = {}
    for evidence in evidence_items:
        if evidence.side_effect:
            evidence_by_effect_name.setdefault(evidence.side_effect.casefold(), []).append(evidence)

    chosen_effects = choose_effects_to_display(ranked_effects, limits)
    chosen_names = {effect["name"].casefold() for effect in chosen_effects}
    # Effects the LLM chose to explain outside the top ones are kept as well.
    for effect_name in prediction_by_name:
        if effect_name not in chosen_names and effect_name in ranked_effect_by_name:
            chosen_effects.append(ranked_effect_by_name[effect_name])
            chosen_names.add(effect_name)

    cards = []
    for ranked_effect in chosen_effects:
        prediction = prediction_by_name.get(ranked_effect["name"].casefold())
        cited_evidence = _evidence_to_cite(ranked_effect, prediction, evidence_by_effect_name)
        if prediction:
            explanation = str(prediction["explanation"])
        else:
            explanation = template_explanation(ranked_effect, names, graph_context)
        cards.append(
            {
                "side_effect_en": names.side_effect_english(ranked_effect["name"]),
                "side_effect_vi": names.side_effect_vietnamese(ranked_effect["name"]),
                "tier": ranked_effect["tier"],
                "tier_label": TIER_LABELS_VIETNAMESE[ranked_effect["tier"]],
                "hypothesis": False,
                "serious": ranked_effect["serious"],
                "score": ranked_effect["score"],
                "absent_from_single_drug_data": ranked_effect["absent_from_single_drug_data"],
                "pairs": [drug_names_of_pair(pair, names) for pair in ranked_effect["pairs"]],
                # Tier 2 applies only to these pairs; the others are Tier 1.
                "amplified_pairs": [
                    drug_names_of_pair(pair, names) for pair in ranked_effect["amplified_pairs"]
                ],
                "explanation_vi": explanation,
                "explanation_source": "llm" if prediction else "template",
                "evidence": [
                    readable_evidence(evidence, names, drug_set) for evidence in cited_evidence
                ],
            }
        )

    for prediction in validated_predictions:
        if prediction["tier"] != 3:
            continue
        cards.append(_hypothesis_card(prediction, names, drug_set))
    return cards


def _evidence_to_cite(
    ranked_effect: dict[str, object],
    prediction: dict[str, object] | None,
    evidence_by_effect_name: dict[str, list[Evidence]],
) -> list[dict[str, object]]:
    """The LLM's evidence, else the first retrieved evidence, else plain pair edges."""
    if prediction:
        return prediction["evidence"]
    retrieved = evidence_by_effect_name.get(ranked_effect["name"].casefold(), [])
    if retrieved:
        return [evidence.to_dict() for evidence in retrieved[:MAX_CITED_ITEMS_WITHOUT_LLM]]
    return [
        {
            "evidence_id": None,
            "tier": 1,
            "kind": "known_pairwise_effect",
            "drug_pair": list(pair),
            "path": [
                {
                    "source": pair[0],
                    "relation": "has_side_effect",
                    "target": pair[1],
                    "side_effect_code": ranked_effect["code"],
                    "side_effect_name": ranked_effect["name"],
                }
            ],
        }
        for pair in ranked_effect["pairs"][:MAX_CITED_ITEMS_WITHOUT_LLM]
    ]


def _hypothesis_card(
    prediction: dict[str, object], names: NameIndex, drug_set: set[str]
) -> dict[str, object]:
    cited_evidence = prediction["evidence"]
    effect_name = str(prediction["side_effect"])
    pair_names = [drug_names_of_pair(evidence["drug_pair"], names) for evidence in cited_evidence]
    # The same pair may be cited by several bridges; list it once, in order.
    unique_pairs = list(dict.fromkeys(tuple(pair) for pair in pair_names))
    return {
        "side_effect_en": names.side_effect_english(effect_name),
        "side_effect_vi": names.side_effect_vietnamese(effect_name),
        "tier": 3,
        "tier_label": TIER_LABELS_VIETNAMESE[3],
        "hypothesis": True,
        "serious": names.is_serious(effect_name),
        "score": None,
        "absent_from_single_drug_data": None,
        "pairs": [list(pair) for pair in unique_pairs],
        "amplified_pairs": [],
        "explanation_vi": str(prediction["explanation"]),
        "explanation_source": "llm",
        "evidence": [readable_evidence(evidence, names, drug_set) for evidence in cited_evidence],
    }
