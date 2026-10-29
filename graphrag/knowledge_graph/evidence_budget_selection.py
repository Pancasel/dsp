"""Choose which evidence candidates fit in graph.max_evidence_items, then number them E1, E2, ...

The budget is spread over drug pairs in turn ("round robin") so that one
pair with hundreds of effects cannot use it all. Tier 2 and Tier 3 each get a
reserved quarter of the budget so that Tier 1 cannot crowd them out.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from graphrag.knowledge_graph.evidence_candidate_collection import (
    EvidenceCandidate,
    EvidenceCandidateGroups,
)
from graphrag.knowledge_graph.hub_protein_filter import path_avoids_hub_proteins
from graphrag.source_data.data_records import Evidence

if TYPE_CHECKING:
    from graphrag.knowledge_graph.decagon_graph import DecagonGraph


# Share of the budget reserved for each of Tier 2 and Tier 3.
RESERVED_BUDGET_DIVISOR = 4


@dataclass
class RoundRobinSelection:
    chosen: list[EvidenceCandidate]
    # True when at least one group still had candidates left.
    truncated: bool


@dataclass
class EvidenceSelection:
    evidence_items: list[Evidence]
    truncated: bool


def take_round_robin(groups: list[list[EvidenceCandidate]], limit: int) -> RoundRobinSelection:
    """Take the first item of every group, then the second of every group, ... up to limit."""
    if limit <= 0:
        return RoundRobinSelection([], any(groups))
    chosen: list[EvidenceCandidate] = []
    next_position_in_group = [0] * len(groups)
    while len(chosen) < limit:
        took_something = False
        for group_index, group in enumerate(groups):
            position = next_position_in_group[group_index]
            if position >= len(group):
                continue
            chosen.append(group[position])
            next_position_in_group[group_index] = position + 1
            took_something = True
            if len(chosen) >= limit:
                break
        if not took_something:
            break
    truncated = False
    for group_index, group in enumerate(groups):
        if next_position_in_group[group_index] < len(group):
            truncated = True
    return RoundRobinSelection(chosen, truncated)


def remove_paths_through_hubs(
    graph: "DecagonGraph",
    groups: list[list[EvidenceCandidate]],
    drugs: list[str],
    regimen_target_proteins: set[str],
) -> list[list[EvidenceCandidate]]:
    """Drop candidates whose path crosses a hub protein, and groups left empty."""
    kept_groups = []
    for group in groups:
        kept_candidates = [
            candidate
            for candidate in group
            if path_avoids_hub_proteins(graph, candidate.path, drugs, regimen_target_proteins)
        ]
        if kept_candidates:
            kept_groups.append(kept_candidates)
    return kept_groups


def select_evidence_within_budget(
    graph: "DecagonGraph",
    candidate_groups: EvidenceCandidateGroups,
    drugs: list[str],
    regimen_target_proteins: set[str],
) -> EvidenceSelection:
    """Pick candidates tier by tier within the budget and number them in the order Tier 1, 2, 3."""
    tier1_groups = remove_paths_through_hubs(
        graph, candidate_groups.tier1_groups, drugs, regimen_target_proteins
    )
    tier2_groups = remove_paths_through_hubs(
        graph, candidate_groups.tier2_groups, drugs, regimen_target_proteins
    )
    tier3_groups = remove_paths_through_hubs(
        graph, candidate_groups.tier3_groups, drugs, regimen_target_proteins
    )
    budget = graph.max_evidence_items

    # 1. Tier 2 and Tier 3 first take at most their reserved share.
    reserved_share = budget // RESERVED_BUDGET_DIVISOR
    tier2_selection = take_round_robin(tier2_groups, reserved_share)
    tier3_selection = take_round_robin(tier3_groups, reserved_share)
    # 2. Tier 1 takes everything the reserved shares did not use.
    tier1_selection = take_round_robin(
        tier1_groups, budget - len(tier2_selection.chosen) - len(tier3_selection.chosen)
    )
    # 3. What Tier 1 left goes back to Tier 2, then to Tier 3.
    tier2_selection = take_round_robin(
        tier2_groups, budget - len(tier1_selection.chosen) - len(tier3_selection.chosen)
    )
    tier3_selection = take_round_robin(
        tier3_groups, budget - len(tier1_selection.chosen) - len(tier2_selection.chosen)
    )

    selected = tier1_selection.chosen + tier2_selection.chosen + tier3_selection.chosen
    evidence_items = []
    for number, candidate in enumerate(selected, start=1):
        evidence_items.append(
            Evidence(
                evidence_id=f"E{number}",
                tier=candidate.tier,
                kind=candidate.kind,
                drug_pair=candidate.drug_pair,
                path=candidate.path,
                side_effect=candidate.effect.name if candidate.effect else None,
                side_effect_code=candidate.effect.code if candidate.effect else None,
                candidate_side_effects=candidate.candidate_side_effects,
            )
        )
    truncated = (
        tier1_selection.truncated or tier2_selection.truncated or tier3_selection.truncated
    )
    return EvidenceSelection(evidence_items, truncated)
