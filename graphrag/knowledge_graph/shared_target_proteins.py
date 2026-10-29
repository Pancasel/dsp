"""Proteins that three or more regimen drugs act on: the Tier 2 ("amplified") signal."""

import itertools
from typing import TYPE_CHECKING

from graphrag.knowledge_graph.hub_protein_filter import is_hub_protein

if TYPE_CHECKING:
    from graphrag.knowledge_graph.decagon_graph import DecagonGraph


MINIMUM_DRUGS_SHARING_A_PROTEIN = 3


def all_target_proteins(target_proteins_by_drug: dict[str, set[str]]) -> set[str]:
    """Union of the targets of every drug in the regimen."""
    if not target_proteins_by_drug:
        return set()
    return set().union(*target_proteins_by_drug.values())


def shared_target_proteins(
    target_proteins_by_drug: dict[str, set[str]],
) -> dict[str, list[str]]:
    """Protein targeted directly by at least three drugs -> those drugs, sorted."""
    shared: dict[str, list[str]] = {}
    for protein in all_target_proteins(target_proteins_by_drug):
        drugs_targeting_protein = sorted(
            drug for drug, targets in target_proteins_by_drug.items() if protein in targets
        )
        if len(drugs_targeting_protein) >= MINIMUM_DRUGS_SHARING_A_PROTEIN:
            shared[protein] = drugs_targeting_protein
    return shared


def convergence_proteins(
    graph: "DecagonGraph", target_proteins_by_drug: dict[str, set[str]]
) -> dict[str, list[str]]:
    """Proteins within one interaction step of the targets of at least three drugs.

    A drug "reaches" its direct targets and their non-hub interaction
    partners. Proteins already returned by shared_target_proteins are left
    out. Result: protein -> drugs reaching it, sorted.
    """
    drugs_reaching_protein: dict[str, set[str]] = {}
    for drug, drug_targets in target_proteins_by_drug.items():
        for protein in drug_targets:
            drugs_reaching_protein.setdefault(protein, set()).add(drug)
            for partner in graph.sorted_interaction_partners(protein):
                if not is_hub_protein(graph, partner):
                    drugs_reaching_protein.setdefault(partner, set()).add(drug)
    directly_shared = shared_target_proteins(target_proteins_by_drug)
    result: dict[str, list[str]] = {}
    for protein, drugs in drugs_reaching_protein.items():
        if len(drugs) >= MINIMUM_DRUGS_SHARING_A_PROTEIN and protein not in directly_shared:
            result[protein] = sorted(drugs)
    return result


def shared_proteins_by_drug_pair(
    drugs_by_shared_protein: dict[str, list[str]],
) -> dict[tuple[str, str], list[str]]:
    """Turn protein -> drugs into drug pair -> proteins both drugs share with a third drug."""
    proteins_by_pair: dict[tuple[str, str], list[str]] = {}
    for protein, drugs in drugs_by_shared_protein.items():
        for drug_a, drug_b in itertools.combinations(drugs, 2):
            proteins_by_pair.setdefault((drug_a, drug_b), []).append(protein)
    return proteins_by_pair
