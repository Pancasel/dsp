"""Hub proteins: proteins with so many interaction partners that a path through them means little.

A protein is a hub when its number of interaction partners (its "degree") is
above graph.hub_degree_threshold. Hubs are never expanded during the bridge
search, and a path may pass through a hub only when a regimen drug targets it.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphrag.knowledge_graph.decagon_graph import DecagonGraph


PROTEIN_RELATIONS = {"targets", "interacts_with"}


def hub_threshold_from_percentile(
    protein_interactions: dict[str, set[str]], percentile: float
) -> int:
    """Degree at the given percentile (0-100): proteins at or below it are not hubs."""
    degrees = sorted(len(partners) for partners in protein_interactions.values())
    if not degrees:
        return 0
    index = min(len(degrees) - 1, int(len(degrees) * percentile / 100))
    return degrees[index]


def is_hub_protein(graph: "DecagonGraph", protein: str) -> bool:
    return graph.interaction_partner_count(protein) > graph.hub_degree_threshold


def protein_may_appear_on_path(
    graph: "DecagonGraph", protein: str, regimen_target_proteins: set[str]
) -> bool:
    """A regimen drug's own target is always allowed; any other protein must not be a hub."""
    return protein in regimen_target_proteins or not is_hub_protein(graph, protein)


def path_avoids_hub_proteins(
    graph: "DecagonGraph",
    path: tuple[dict[str, str], ...],
    drugs: list[str],
    regimen_target_proteins: set[str],
) -> bool:
    """True when every drug-protein and protein-protein edge ends at a drug or an allowed protein."""
    allowed_proteins = set()
    for edge in path:
        if edge["relation"] not in PROTEIN_RELATIONS:
            continue
        for protein in (edge["source"], edge["target"]):
            if protein_may_appear_on_path(graph, protein, regimen_target_proteins):
                allowed_proteins.add(protein)

    for edge in path:
        if edge["relation"] not in PROTEIN_RELATIONS:
            continue
        source_is_allowed = edge["source"] in drugs or edge["source"] in allowed_proteins
        target_is_allowed = edge["target"] in drugs or edge["target"] in allowed_proteins
        if not (source_is_allowed and target_is_allowed):
            return False
    return True
