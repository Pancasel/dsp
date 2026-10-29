"""Re-check an evidence path against the raw graph: every edge must really exist."""

from typing import TYPE_CHECKING

from graphrag.source_data.data_records import Evidence

if TYPE_CHECKING:
    from graphrag.knowledge_graph.decagon_graph import DecagonGraph


def evidence_path_exists_in_graph(graph: "DecagonGraph", evidence: Evidence) -> bool:
    for edge in evidence.path:
        source = edge["source"]
        relation = edge["relation"]
        target = edge["target"]
        if relation == "targets":
            if target not in graph.target_proteins(source):
                return False
        elif relation == "interacts_with":
            if target not in graph.interaction_partners(source):
                return False
        elif relation == "has_side_effect":
            effect_code = edge.get("side_effect_code", "")
            pair_effect_codes = {effect.code for effect in graph.pair_side_effects(source, target)}
            if effect_code not in pair_effect_codes:
                return False
        else:
            return False
    return True
