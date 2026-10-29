"""Mechanism details shown under "Chi tiết thêm": protein bridges, per-pair counts, shared targets."""

import itertools

from graphrag.drug_names.name_index import NameIndex
from graphrag.result_presentation.evidence_path_text import drug_names_of_pair, readable_evidence
from graphrag.source_data.data_records import Evidence


DEFAULT_BRIDGES_SHOWN_PER_PAIR = 2


def protein_bridge_examples(
    drugs: list[str],
    evidence_items: list[Evidence],
    names: NameIndex,
    limit_per_pair: int = DEFAULT_BRIDGES_SHOWN_PER_PAIR,
) -> list[dict[str, object]]:
    """Tier 3 protein bridges between undocumented pairs, shown even without an LLM."""
    drug_set = set(drugs)
    shown_count_by_pair: dict[tuple[str, str], int] = {}
    bridges = []
    for evidence in evidence_items:
        if evidence.tier != 3:
            continue
        shown_count = shown_count_by_pair.get(evidence.drug_pair, 0)
        if shown_count >= limit_per_pair:
            continue
        shown_count_by_pair[evidence.drug_pair] = shown_count + 1
        bridges.append(readable_evidence(evidence, names, drug_set))
    return bridges


def summarize_each_drug_pair(
    drugs: list[str], graph_context: dict[str, object], names: NameIndex
) -> list[dict[str, object]]:
    """Per drug pair: number of documented effects and how many of them are serious."""
    documented_count_by_pair: dict[tuple[str, str], int] = {}
    serious_count_by_pair: dict[tuple[str, str], int] = {}
    for pair in itertools.combinations(sorted(drugs), 2):
        documented_count_by_pair[pair] = 0
        serious_count_by_pair[pair] = 0
    for ranked_effect in graph_context["ranked_effects"]:
        for pair in ranked_effect["pairs"]:
            pair_key = tuple(pair)
            documented_count_by_pair[pair_key] += 1
            if ranked_effect["serious"]:
                serious_count_by_pair[pair_key] += 1
    return [
        {
            "pair": drug_names_of_pair(pair, names),
            "documented_effects": documented_count_by_pair[pair],
            "serious_effects": serious_count_by_pair[pair],
        }
        for pair in documented_count_by_pair
    ]


def readable_shared_targets(
    graph_context: dict[str, object], names: NameIndex
) -> list[dict[str, object]]:
    """Proteins targeted by three or more regimen drugs (the Tier 2 signal), with names."""
    return [
        {
            "protein": names.gene_symbol(shared["protein"]),
            "drugs": [names.drug_name(drug) for drug in shared["drugs"]],
        }
        for shared in graph_context["shared_targets"]
    ]
