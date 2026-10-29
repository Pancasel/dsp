"""Tests of the smaller pieces: cache, hub threshold, convergence proteins, evidence budget."""

from pathlib import Path

from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.knowledge_graph.evidence_budget_selection import select_evidence_within_budget
from graphrag.knowledge_graph.evidence_candidate_collection import (
    EvidenceCandidate,
    EvidenceCandidateGroups,
)
from graphrag.knowledge_graph.hub_protein_filter import hub_threshold_from_percentile
from graphrag.knowledge_graph.shared_target_proteins import convergence_proteins
from tests.csv_test_data import DRUG_PAIR_SIDE_EFFECTS_HEADER, write_csv


def test_cache_is_rebuilt_when_the_source_file_changes(
    small_graph: DecagonGraph, tmp_path: Path
) -> None:
    changed_file = tmp_path / "combo.csv"
    write_csv(changed_file, DRUG_PAIR_SIDE_EFFECTS_HEADER, [["D1", "D3", "C2", "new event"]])
    reloaded = DecagonGraph.load(
        drug_targets_file=tmp_path / "targets.csv",
        protein_interactions_file=tmp_path / "ppi.csv",
        drug_pair_side_effects_file=changed_file,
        cache_database_file=tmp_path / "rebuild.sqlite3",
        hub_degree_threshold=2,
    )
    assert [effect.name for effect in reloaded.pair_side_effects("D3", "D1")] == ["new event"]


def test_hub_threshold_from_percentile() -> None:
    protein_interactions = {
        f"P{size}": {f"N{index}" for index in range(size)} for size in range(1, 101)
    }
    assert hub_threshold_from_percentile(protein_interactions, 50) == 51
    assert hub_threshold_from_percentile(protein_interactions, 100) == 100
    assert hub_threshold_from_percentile({}, 90) == 0


def test_convergence_proteins_use_one_step_neighbourhood_without_hubs(
    small_graph: DecagonGraph,
) -> None:
    small_graph.hub_degree_threshold = 3
    target_proteins_by_drug = {"D1": {"G1"}, "D2": {"MID"}, "D3": {"LOW"}}

    result = convergence_proteins(small_graph, target_proteins_by_drug)
    assert result == {"G1": ["D1", "D2", "D3"]}
    # G1 is reached by D2 and D3 only through partners; HUB (4 partners) is skipped.
    assert "HUB" not in result


def test_tier1_cannot_use_up_the_budget_reserved_for_tier2_and_tier3(
    small_graph: DecagonGraph,
) -> None:
    def candidate_group(tier: int, pair: tuple[str, str], count: int) -> list[EvidenceCandidate]:
        return [
            EvidenceCandidate(tier=tier, kind="k", drug_pair=pair, path=(), effect=None)
            for _ in range(count)
        ]

    small_graph.max_evidence_items = 8
    groups = EvidenceCandidateGroups(
        tier1_groups=[candidate_group(1, ("D1", "D2"), 20)],
        tier2_groups=[candidate_group(2, ("D1", "D2"), 20)],
        tier3_groups=[candidate_group(3, ("D1", "D3"), 1)],
    )
    selection = select_evidence_within_budget(small_graph, groups, ["D1", "D2", "D3"], set())
    tiers = [item.tier for item in selection.evidence_items]
    # Tier 3 uses one of its two reserved places; the spare one goes to Tier 1.
    assert (tiers.count(1), tiers.count(2), tiers.count(3)) == (5, 2, 1)
    assert selection.truncated
