"""The compact JSON sent to the LLM: names instead of IDs, only the evidence it needs.

Evidence IDs stay identical to the graph's, so the answer can be validated.
"""

from graphrag.drug_names.name_index import NameIndex
from graphrag.result_presentation.evidence_path_text import (
    describe_path_in_one_line,
    drug_names_of_pair,
)
from graphrag.side_effect_ranking.effects_to_display import (
    EffectDisplayLimits,
    choose_effects_to_display,
)
from graphrag.source_data.data_records import Evidence


def build_llm_context(
    drugs: list[str],
    graph_context: dict[str, object],
    evidence_items: list[Evidence],
    names: NameIndex,
    limits: EffectDisplayLimits,
) -> dict[str, object]:
    drug_set = set(drugs)
    displayed_effects = choose_effects_to_display(graph_context["ranked_effects"], limits)
    displayed_effect_names = {effect["name"].casefold() for effect in displayed_effects}

    serialized_evidence = []
    for evidence in evidence_items:
        # Tier 1/2 evidence about an effect that is not displayed is not needed.
        is_documented_tier = evidence.tier in (1, 2)
        effect_is_displayed = (evidence.side_effect or "").casefold() in displayed_effect_names
        if is_documented_tier and not effect_is_displayed:
            continue
        entry: dict[str, object] = {
            "evidence_id": evidence.evidence_id,
            "tier": evidence.tier,
            "drug_pair": [names.drug_name_with_id(drug) for drug in evidence.drug_pair],
            "path": describe_path_in_one_line(evidence.path, names, drug_set, in_english=True),
        }
        if evidence.side_effect:
            entry["side_effect"] = evidence.side_effect
        if evidence.candidate_side_effects:
            entry["candidate_side_effects"] = list(evidence.candidate_side_effects)
        serialized_evidence.append(entry)

    return {
        "drugs": [names.drug_name_with_id(drug) for drug in drugs],
        "direct_target_counts": {
            names.drug_name(drug): count
            for drug, count in graph_context["direct_target_counts"].items()
        },
        "shared_targets": [
            {
                "protein": names.gene_symbol(shared["protein"]),
                "drugs": [names.drug_name(drug) for drug in shared["drugs"]],
            }
            for shared in graph_context["shared_targets"]
        ],
        "top_side_effects": [
            {
                "side_effect": effect["name"],
                "tier": effect["tier"],
                "serious": effect["serious"],
                "pairs": [drug_names_of_pair(pair, names) for pair in effect["pairs"]],
            }
            for effect in displayed_effects
        ],
        "evidence": serialized_evidence,
        "evidence_truncated": graph_context["evidence_truncated"],
    }
