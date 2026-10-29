from pathlib import Path

from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.knowledge_graph.evidence_path_validation import evidence_path_exists_in_graph
from graphrag.knowledge_graph.regimen_evidence_retrieval import retrieve_regimen_evidence
from tests.csv_test_data import (
    DRUG_PAIR_SIDE_EFFECTS_HEADER,
    DRUG_TARGETS_HEADER,
    PROTEIN_INTERACTIONS_HEADER,
    write_csv,
)


def _load_graph(tmp_path: Path, targets: list[list[str]], interactions: list[list[str]], **options) -> DecagonGraph:
    write_csv(tmp_path / "targets.csv", DRUG_TARGETS_HEADER, targets)
    write_csv(tmp_path / "ppi.csv", PROTEIN_INTERACTIONS_HEADER, interactions)
    write_csv(tmp_path / "combo.csv", DRUG_PAIR_SIDE_EFFECTS_HEADER, [])
    return DecagonGraph.load(
        drug_targets_file=tmp_path / "targets.csv",
        protein_interactions_file=tmp_path / "ppi.csv",
        drug_pair_side_effects_file=tmp_path / "combo.csv",
        cache_database_file=tmp_path / "cache.sqlite3",
        **options,
    )


def test_bridge_search_stops_at_the_per_pair_limit(tmp_path: Path) -> None:
    graph = _load_graph(
        tmp_path,
        [["D1", "A"], ["D2", "B"], ["D3", "C"]],
        [["A", "M1"], ["M1", "B"], ["A", "M2"], ["M2", "B"], ["A", "M3"], ["M3", "B"]],
        hub_degree_threshold=50,
        tier3_two_hop_bridges_per_pair=2,
    )
    retrieved = retrieve_regimen_evidence(graph, ["D1", "D2", "D3"])
    tier3_items = [
        item
        for item in retrieved.evidence_items
        if item.tier == 3 and item.drug_pair == ("D1", "D2")
    ]

    assert len(tier3_items) == 2
    assert retrieved.graph_context["evidence_truncated"] is True
    assert all(evidence_path_exists_in_graph(graph, item) for item in retrieved.evidence_items)


def test_hub_target_is_an_endpoint_and_not_a_middle_protein(tmp_path: Path) -> None:
    graph = _load_graph(
        tmp_path,
        [["D1", "G1"], ["D2", "HUB"], ["D2", "G2"], ["D3", "C"]],
        [["G1", "HUB"], ["HUB", "G2"], ["HUB", "X"]],
        hub_degree_threshold=2,
    )
    retrieved = retrieve_regimen_evidence(graph, ["D1", "D2", "D3"])
    tier3_items = [
        item
        for item in retrieved.evidence_items
        if item.tier == 3 and item.drug_pair == ("D1", "D2")
    ]

    assert tier3_items
    assert any(
        edge["relation"] == "interacts_with" and edge["source"] == "G1" and edge["target"] == "HUB"
        for item in tier3_items
        for edge in item.path
    )
    assert all(
        edge["source"] != "HUB"
        for item in tier3_items
        for edge in item.path
        if edge["relation"] == "interacts_with"
    )
