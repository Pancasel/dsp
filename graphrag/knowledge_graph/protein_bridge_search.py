"""Tier 3: find protein bridges between two drugs that have no documented side effect together.

A bridge links a target of drug A to a target of drug B:
- one hop:  A -> protein_a <-> protein_b <- B
- two hops: A -> protein_a <-> middle_protein <-> protein_b <- B
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from graphrag.knowledge_graph.hub_protein_filter import is_hub_protein

if TYPE_CHECKING:
    from graphrag.knowledge_graph.decagon_graph import DecagonGraph


@dataclass(frozen=True)
class ProteinBridge:
    # Target of the first drug of the pair.
    protein_a: str
    # Target of the second drug of the pair.
    protein_b: str
    # None for a one-hop bridge.
    middle_protein: str | None = None


@dataclass
class BridgeSearchResult:
    # One-hop bridges first, then two-hop bridges.
    bridges: list[ProteinBridge] = field(default_factory=list)
    # True when more bridges exist than the per-pair limits allow.
    truncated: bool = False


def find_protein_bridges(
    graph: "DecagonGraph",
    targets_of_drug_a: set[str],
    targets_of_drug_b: set[str],
    regimen_target_proteins: set[str],
    pruned_hub_proteins: set[str],
) -> BridgeSearchResult:
    """Return a limited, always identical sample of one-hop then two-hop bridges.

    The search stops once both per-pair limits are reached and one extra
    bridge has shown that more exist. Hub proteins are never expanded, even
    when a regimen drug targets them; the hubs that would have led to drug B
    are added to pruned_hub_proteins.
    """
    one_hop_limit = graph.tier3_one_hop_bridges_per_pair
    two_hop_limit = graph.tier3_two_hop_bridges_per_pair
    if one_hop_limit == 0 and two_hop_limit == 0:
        return BridgeSearchResult()
    one_hop_bridges: list[ProteinBridge] = []
    two_hop_bridges: list[ProteinBridge] = []
    truncated = False
    one_hop_full = one_hop_limit == 0
    two_hop_full = two_hop_limit == 0

    for protein_a in sorted(targets_of_drug_a):
        if one_hop_full and two_hop_full:
            truncated = True
            break
        for partner in graph.sorted_interaction_partners(protein_a):
            if one_hop_full and two_hop_full:
                truncated = True
                break

            # One hop: the partner is itself a target of drug B.
            if partner in targets_of_drug_b and partner != protein_a:
                if one_hop_full:
                    truncated = True
                else:
                    one_hop_bridges.append(ProteinBridge(protein_a, partner))
                    one_hop_full = len(one_hop_bridges) >= one_hop_limit

            # Two hops: the partner is a middle protein, never a hub.
            if is_hub_protein(graph, partner):
                hub_leads_to_drug_b = not targets_of_drug_b.isdisjoint(
                    graph.interaction_partners(partner)
                )
                if partner not in regimen_target_proteins and hub_leads_to_drug_b:
                    pruned_hub_proteins.add(partner)
                if truncated and one_hop_full and two_hop_full:
                    break
                continue
            targets_of_b_next_to_partner = targets_of_drug_b & graph.interaction_partners(partner)
            if two_hop_full:
                if targets_of_b_next_to_partner - {protein_a}:
                    truncated = True
                if truncated and one_hop_full:
                    break
                continue
            for protein_b in sorted(targets_of_b_next_to_partner):
                if protein_b == protein_a:
                    continue
                if len(two_hop_bridges) >= two_hop_limit:
                    truncated = True
                    two_hop_full = True
                    break
                two_hop_bridges.append(ProteinBridge(protein_a, protein_b, middle_protein=partner))
            if len(two_hop_bridges) >= two_hop_limit:
                two_hop_full = True
            if truncated and one_hop_full and two_hop_full:
                break
        if truncated and one_hop_full and two_hop_full:
            break
    return BridgeSearchResult(one_hop_bridges + two_hop_bridges, truncated)
