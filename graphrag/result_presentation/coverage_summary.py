"""How much of the regimen the data covers: pairs with data, pairs with only a bridge, pairs with nothing."""

import itertools

from graphrag.drug_names.name_index import NameIndex
from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.result_presentation.evidence_path_text import drug_names_of_pair
from graphrag.source_data.data_records import Evidence


def summarize_data_coverage(
    drugs: list[str],
    graph: DecagonGraph,
    graph_context: dict[str, object],
    evidence_items: list[Evidence],
    names: NameIndex,
) -> dict[str, object]:
    pairs = list(itertools.combinations(sorted(drugs), 2))
    documented_pairs = {tuple(pair) for pair in graph_context["pairs_with_documented_effects"]}
    bridged_pairs = {evidence.drug_pair for evidence in evidence_items if evidence.tier == 3}
    return {
        "pairs_total": len(pairs),
        "pairs_with_data": len(documented_pairs),
        "pairs_with_bridge_only": [
            drug_names_of_pair(pair, names) for pair in pairs if pair in bridged_pairs
        ],
        "pairs_without_any_evidence": [
            drug_names_of_pair(pair, names)
            for pair in pairs
            if pair not in documented_pairs and pair not in bridged_pairs
        ],
        "drugs_without_targets": [
            names.drug_name(drug) for drug in drugs if not graph.target_proteins(drug)
        ],
        "documented_effects_total": len(graph_context["ranked_effects"]),
    }
