"""Build every possible piece of evidence (a "candidate") for each drug pair, grouped by tier.

- Tier 1: a side effect documented for the pair.
- Tier 2: the same documented effect, plus the paths to a protein the pair
  shares with a third drug.
- Tier 3: a protein bridge between two drugs that have no documented effect.

evidence_budget_selection.py later picks which candidates fit the budget.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from graphrag.knowledge_graph.protein_bridge_search import ProteinBridge, find_protein_bridges
from graphrag.source_data.data_records import SideEffect

if TYPE_CHECKING:
    from graphrag.knowledge_graph.decagon_graph import DecagonGraph


# A Tier 2 path lists at most this many shared proteins. Further proteins
# still support the tier, and the retrieval reports the path as truncated.
MAX_SHARED_PROTEINS_IN_PATH = 3


@dataclass(frozen=True)
class EvidenceCandidate:
    """Evidence before it is selected and numbered (see data_records.Evidence)."""

    tier: int
    kind: str
    drug_pair: tuple[str, str]
    path: tuple[dict[str, str], ...]
    effect: SideEffect | None = None
    candidate_side_effects: tuple[str, ...] = ()


@dataclass
class EvidenceCandidateGroups:
    # Each list holds one group per drug pair, so the budget can be shared fairly.
    tier1_groups: list[list[EvidenceCandidate]] = field(default_factory=list)
    tier2_groups: list[list[EvidenceCandidate]] = field(default_factory=list)
    tier3_groups: list[list[EvidenceCandidate]] = field(default_factory=list)
    # Hub proteins that a Tier 3 bridge would have needed.
    pruned_hub_proteins: set[str] = field(default_factory=set)
    # True when shared proteins or bridges were left out because of the limits.
    truncated: bool = False


def side_effect_edge(drug_a: str, drug_b: str, effect: SideEffect) -> dict[str, str]:
    return {
        "source": drug_a,
        "relation": "has_side_effect",
        "target": drug_b,
        "side_effect_code": effect.code,
        "side_effect_name": effect.name,
    }


def path_from_drug_to_protein(
    graph: "DecagonGraph", drug: str, protein: str
) -> list[dict[str, str]]:
    """Edges from a drug to a protein it targets directly or through one interaction step."""
    drug_targets = graph.target_proteins(drug)
    if protein in drug_targets:
        return [{"source": drug, "relation": "targets", "target": protein}]
    # The first target (in sorted order) that interacts with the protein.
    for target in sorted(drug_targets):
        if protein in graph.interaction_partners(target):
            return [
                {"source": drug, "relation": "targets", "target": target},
                {"source": target, "relation": "interacts_with", "target": protein},
            ]
    raise ValueError(f"{drug} does not reach {protein} in one interaction step")


def documented_pair_candidates(
    graph: "DecagonGraph",
    pair: tuple[str, str],
    effects: list[SideEffect],
    shared_proteins: list[str],
) -> tuple[list[EvidenceCandidate], list[EvidenceCandidate]]:
    """Tier 1 candidates for a pair with documented effects, plus Tier 2 when proteins are shared."""
    drug_a, drug_b = pair
    tier1_candidates = []
    for effect in effects:
        tier1_candidates.append(
            EvidenceCandidate(
                tier=1,
                kind="known_pairwise_effect",
                drug_pair=pair,
                path=(side_effect_edge(drug_a, drug_b, effect),),
                effect=effect,
            )
        )

    tier2_candidates: list[EvidenceCandidate] = []
    if shared_proteins:
        paths_to_shared_proteins = []
        for protein in shared_proteins:
            for drug in (drug_a, drug_b):
                paths_to_shared_proteins.extend(path_from_drug_to_protein(graph, drug, protein))
        for effect in effects:
            full_path = paths_to_shared_proteins + [side_effect_edge(drug_a, drug_b, effect)]
            tier2_candidates.append(
                EvidenceCandidate(
                    tier=2,
                    kind="shared_target_amplification",
                    drug_pair=pair,
                    path=tuple(full_path),
                    effect=effect,
                )
            )
    return tier1_candidates, tier2_candidates


def protein_bridge_candidates(
    graph: "DecagonGraph", pair: tuple[str, str], bridges: list[ProteinBridge]
) -> list[EvidenceCandidate]:
    """Tier 3 candidates, one per bridge, all carrying the same candidate effect names."""
    drug_a, drug_b = pair
    candidate_effect_names = tuple(graph.tier3_candidate_effect_names(drug_a, drug_b))
    candidates: list[EvidenceCandidate] = []
    for bridge in bridges:
        if bridge.middle_protein is None:
            path = [
                {"source": drug_a, "relation": "targets", "target": bridge.protein_a},
                {"source": bridge.protein_a, "relation": "interacts_with", "target": bridge.protein_b},
            ]
        else:
            path = [
                {"source": drug_a, "relation": "targets", "target": bridge.protein_a},
                {
                    "source": bridge.protein_a,
                    "relation": "interacts_with",
                    "target": bridge.middle_protein,
                },
                {
                    "source": bridge.middle_protein,
                    "relation": "interacts_with",
                    "target": bridge.protein_b,
                },
            ]
        path.append({"source": drug_b, "relation": "targets", "target": bridge.protein_b})
        candidates.append(
            EvidenceCandidate(
                tier=3,
                kind="novel_pair_protein_bridge",
                drug_pair=pair,
                path=tuple(path),
                candidate_side_effects=candidate_effect_names,
            )
        )
    return candidates


def collect_evidence_candidates(
    graph: "DecagonGraph",
    pairs: list[tuple[str, str]],
    effects_by_pair: dict[tuple[str, str], list[SideEffect]],
    shared_proteins_by_pair: dict[tuple[str, str], list[str]],
    target_proteins_by_drug: dict[str, set[str]],
    regimen_target_proteins: set[str],
    displayed_effect_codes: set[str] | None = None,
) -> EvidenceCandidateGroups:
    """Candidate groups of every pair: Tier 1/2 for documented pairs, Tier 3 for the others.

    When displayed_effect_codes is given, only those effects get Tier 1/2
    candidates, because only effects shown to the user need citable evidence.
    """
    groups = EvidenceCandidateGroups()
    for pair in pairs:
        drug_a, drug_b = pair
        effects = effects_by_pair[pair]
        if effects:
            shared_proteins = sorted(shared_proteins_by_pair.get(pair, []))
            if len(shared_proteins) > MAX_SHARED_PROTEINS_IN_PATH:
                groups.truncated = True
                shared_proteins = shared_proteins[:MAX_SHARED_PROTEINS_IN_PATH]
            if displayed_effect_codes is not None:
                effects = [effect for effect in effects if effect.code in displayed_effect_codes]
            tier1_candidates, tier2_candidates = documented_pair_candidates(
                graph, pair, effects, shared_proteins
            )
            if tier1_candidates:
                groups.tier1_groups.append(tier1_candidates)
            if tier2_candidates:
                groups.tier2_groups.append(tier2_candidates)
            continue

        search_result = find_protein_bridges(
            graph,
            target_proteins_by_drug[drug_a],
            target_proteins_by_drug[drug_b],
            regimen_target_proteins,
            groups.pruned_hub_proteins,
        )
        groups.truncated = groups.truncated or search_result.truncated
        if search_result.bridges:
            groups.tier3_groups.append(protein_bridge_candidates(graph, pair, search_result.bridges))
    return groups
