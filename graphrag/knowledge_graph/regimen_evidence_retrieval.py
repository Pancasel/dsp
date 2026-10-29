"""Retrieve the subgraph and the numbered evidence for one regimen (the "retrieval" of Graph-RAG).

Steps:
1. Find proteins shared by three or more drugs (shared_target_proteins.py).
2. Read each pair's documented side effects and rank them for the regimen.
3. Build evidence candidates for every pair (evidence_candidate_collection.py).
4. Keep the candidates that fit the budget (evidence_budget_selection.py).
5. Describe the result as a JSON-ready dictionary, the "graph context".
"""

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from graphrag.knowledge_graph.evidence_budget_selection import select_evidence_within_budget
from graphrag.knowledge_graph.evidence_candidate_collection import collect_evidence_candidates
from graphrag.knowledge_graph.hub_protein_filter import (
    PROTEIN_RELATIONS,
    protein_may_appear_on_path,
)
from graphrag.knowledge_graph.shared_target_proteins import (
    all_target_proteins,
    convergence_proteins,
    shared_proteins_by_drug_pair,
    shared_target_proteins,
)
from graphrag.settings.ranking_weights import RankingWeights
from graphrag.side_effect_ranking.effects_to_display import (
    EffectDisplayLimits,
    choose_effects_to_display,
)
from graphrag.side_effect_ranking.regimen_side_effect_scoring import RankedEffect, rank_effects
from graphrag.source_data.data_records import Evidence, SideEffect

if TYPE_CHECKING:
    from graphrag.knowledge_graph.decagon_graph import DecagonGraph


# Full target lists are included only for small regimens (fewer drug-target
# links than this). Larger regimens keep the targets used by the evidence.
DIRECT_TARGET_LIST_LIMIT = 250


@dataclass
class RegimenEvidence:
    # JSON-ready description of the subgraph (keys listed in describe_subgraph).
    graph_context: dict[str, object]
    evidence_items: list[Evidence]


def rank_regimen_side_effects(
    graph: "DecagonGraph",
    pairs: list[tuple[str, str]],
    effects_by_pair: dict[tuple[str, str], list[SideEffect]],
    shared_proteins_by_pair: dict[tuple[str, str], list[str]],
    is_serious: Callable[[str], bool] | None,
    weights: RankingWeights | None,
    is_critical: Callable[[str], bool] | None,
) -> list[RankedEffect]:
    """Score the regimen's documented effects, and re-order each pair's list by that score.

    The re-ordering matters: the evidence budget keeps the first effects of
    each pair, so they must be the most relevant ones.
    """
    amplified_pairs = set()
    for pair in shared_proteins_by_pair:
        if effects_by_pair.get(pair):
            amplified_pairs.add(pair)
    pairs_with_effects = {pair: effects for pair, effects in effects_by_pair.items() if effects}
    ranked_effects = rank_effects(
        pairs_with_effects,
        amplified_pairs,
        graph.pair_count_by_effect_code,
        graph.total_pair_count,
        graph.single_drug_effect_codes,
        is_serious,
        weights,
        is_critical,
    )
    position_by_code = {
        ranked_effect.code: position for position, ranked_effect in enumerate(ranked_effects)
    }
    unranked_position = len(position_by_code)

    def regimen_rank(effect: SideEffect) -> int:
        return position_by_code.get(effect.code, unranked_position)

    for pair in pairs:
        effects_by_pair[pair].sort(key=regimen_rank)
    return ranked_effects


def retrieve_regimen_evidence(
    graph: "DecagonGraph",
    drugs: list[str],
    is_serious: Callable[[str], bool] | None = None,
    weights: RankingWeights | None = None,
    is_critical: Callable[[str], bool] | None = None,
    display_limits: EffectDisplayLimits | None = None,
) -> RegimenEvidence:
    """Subgraph and numbered evidence for a regimen.

    With display_limits, Tier 1/2 evidence is only built for the effects that
    will be shown, which leaves more of the budget for Tier 2 and Tier 3.
    """
    target_proteins_by_drug = {drug: graph.target_proteins(drug) for drug in drugs}
    regimen_target_proteins = all_target_proteins(target_proteins_by_drug)
    directly_shared_proteins = shared_target_proteins(target_proteins_by_drug)
    nearby_shared_proteins = convergence_proteins(graph, target_proteins_by_drug)
    if graph.count_convergence_proteins_as_amplification:
        amplifying_proteins = {**directly_shared_proteins, **nearby_shared_proteins}
    else:
        amplifying_proteins = directly_shared_proteins

    pairs = list(itertools.combinations(sorted(drugs), 2))
    effects_by_pair = graph.side_effects_of_pairs(pairs)
    shared_proteins_by_pair = shared_proteins_by_drug_pair(amplifying_proteins)
    ranked_effects = rank_regimen_side_effects(
        graph, pairs, effects_by_pair, shared_proteins_by_pair, is_serious, weights, is_critical
    )
    ranked_effect_dictionaries = [ranked_effect.to_dict() for ranked_effect in ranked_effects]

    displayed_effect_codes = None
    if display_limits is not None:
        displayed_effects = choose_effects_to_display(ranked_effect_dictionaries, display_limits)
        displayed_effect_codes = {effect["code"] for effect in displayed_effects}
    candidate_groups = collect_evidence_candidates(
        graph,
        pairs,
        effects_by_pair,
        shared_proteins_by_pair,
        target_proteins_by_drug,
        regimen_target_proteins,
        displayed_effect_codes,
    )
    selection = select_evidence_within_budget(
        graph, candidate_groups, drugs, regimen_target_proteins
    )

    graph_context = describe_subgraph(
        graph,
        drugs=drugs,
        pairs=pairs,
        target_proteins_by_drug=target_proteins_by_drug,
        regimen_target_proteins=regimen_target_proteins,
        directly_shared_proteins=directly_shared_proteins,
        nearby_shared_proteins=nearby_shared_proteins,
        ranked_effect_dictionaries=ranked_effect_dictionaries,
        effects_by_pair=effects_by_pair,
        evidence_items=selection.evidence_items,
        truncated=candidate_groups.truncated or selection.truncated,
        pruned_hub_proteins=candidate_groups.pruned_hub_proteins,
    )
    return RegimenEvidence(graph_context, selection.evidence_items)


def describe_subgraph(
    graph: "DecagonGraph",
    *,
    drugs: list[str],
    pairs: list[tuple[str, str]],
    target_proteins_by_drug: dict[str, set[str]],
    regimen_target_proteins: set[str],
    directly_shared_proteins: dict[str, list[str]],
    nearby_shared_proteins: dict[str, list[str]],
    ranked_effect_dictionaries: list[dict[str, object]],
    effects_by_pair: dict[tuple[str, str], list[SideEffect]],
    evidence_items: list[Evidence],
    truncated: bool,
    pruned_hub_proteins: set[str],
) -> dict[str, object]:
    proteins_on_evidence_paths = set()
    for evidence in evidence_items:
        for edge in evidence.path:
            if edge["relation"] in PROTEIN_RELATIONS:
                proteins_on_evidence_paths.add(edge["source"])
                proteins_on_evidence_paths.add(edge["target"])
    allowed_proteins = {
        protein
        for protein in proteins_on_evidence_paths
        if protein_may_appear_on_path(graph, protein, regimen_target_proteins)
    }
    # Proteins on a path that are neither allowed nor a regimen target are hubs too.
    hubs_on_paths = {
        protein
        for protein in proteins_on_evidence_paths
        if protein not in allowed_proteins and protein not in regimen_target_proteins
    }

    drug_target_link_count = sum(len(targets) for targets in target_proteins_by_drug.values())
    if drug_target_link_count <= DIRECT_TARGET_LIST_LIMIT:
        direct_targets = {drug: sorted(target_proteins_by_drug[drug]) for drug in drugs}
    else:
        direct_targets = {
            drug: sorted(target_proteins_by_drug[drug] & allowed_proteins) for drug in drugs
        }

    return {
        "drugs": drugs,
        "direct_targets": direct_targets,
        "direct_target_counts": {drug: len(target_proteins_by_drug[drug]) for drug in drugs},
        "shared_targets": [
            {"protein": protein, "drugs": sharing_drugs}
            for protein, sharing_drugs in sorted(directly_shared_proteins.items())
        ],
        "convergence_targets": [
            {"protein": protein, "drugs": sharing_drugs}
            for protein, sharing_drugs in sorted(nearby_shared_proteins.items())
        ],
        "ranked_effects": ranked_effect_dictionaries,
        "pairs_with_documented_effects": [list(pair) for pair in pairs if effects_by_pair[pair]],
        "evidence": [evidence.to_dict() for evidence in evidence_items],
        "evidence_truncated": truncated,
        "pruned_hubs": sorted(pruned_hub_proteins | hubs_on_paths),
    }
