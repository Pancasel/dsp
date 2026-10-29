from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.knowledge_graph.evidence_path_validation import evidence_path_exists_in_graph
from graphrag.knowledge_graph.regimen_evidence_retrieval import retrieve_regimen_evidence
from graphrag.side_effect_ranking.effects_to_display import EffectDisplayLimits


def test_retrieves_known_effect_shared_target_and_prunes_hub(small_graph: DecagonGraph) -> None:
    retrieved = retrieve_regimen_evidence(small_graph, ["D1", "D2", "D3"])
    graph_context = retrieved.graph_context
    evidence_items = retrieved.evidence_items

    assert graph_context["shared_targets"] == [
        {"protein": "SHARED", "drugs": ["D1", "D2", "D3"]}
    ]
    assert "HUB" in graph_context["pruned_hubs"]
    assert any(item.tier == 1 and item.side_effect == "known event" for item in evidence_items)
    assert any(item.tier == 2 and item.side_effect == "known event" for item in evidence_items)
    tier3_pairs = {item.drug_pair for item in evidence_items if item.tier == 3}
    assert ("D1", "D3") in tier3_pairs
    assert ("D1", "D2") not in tier3_pairs
    assert all(evidence_path_exists_in_graph(small_graph, item) for item in evidence_items)


def test_two_hop_bridge_is_pruned_when_middle_protein_is_hub(small_graph: DecagonGraph) -> None:
    retrieved = retrieve_regimen_evidence(small_graph, ["D1", "D2", "D3"])
    d1_d2_tier3 = [
        item
        for item in retrieved.evidence_items
        if item.tier == 3 and item.drug_pair == ("D1", "D2")
    ]
    assert not d1_d2_tier3
    assert "HUB" in retrieved.graph_context["pruned_hubs"]
    for item in retrieved.evidence_items:
        proteins_on_path = {edge["source"] for edge in item.path} | {edge["target"] for edge in item.path}
        assert "HUB" not in proteins_on_path


def test_drug_without_targets_keeps_its_pair_history(small_graph: DecagonGraph) -> None:
    retrieved = retrieve_regimen_evidence(small_graph, ["D1", "D3", "D4"])

    assert retrieved.graph_context["direct_targets"]["D4"] == []
    assert any(
        item.tier == 1
        and item.drug_pair == ("D3", "D4")
        and item.side_effect == "combo-only event"
        for item in retrieved.evidence_items
    )
    assert not any(item.tier == 3 and "D4" in item.drug_pair for item in retrieved.evidence_items)


def test_convergence_amplification_adds_validated_tier2_paths(small_graph: DecagonGraph) -> None:
    small_graph.hub_degree_threshold = 3
    small_graph.target_proteins_by_drug = {"D1": {"G1"}, "D2": {"MID"}, "D3": {"LOW"}}
    small_graph.count_convergence_proteins_as_amplification = False
    retrieved = retrieve_regimen_evidence(small_graph, ["D1", "D2", "D3"])
    assert not any(item.tier == 2 for item in retrieved.evidence_items)

    small_graph.count_convergence_proteins_as_amplification = True
    retrieved = retrieve_regimen_evidence(small_graph, ["D1", "D2", "D3"])
    tier2_items = [item for item in retrieved.evidence_items if item.tier == 2]
    assert tier2_items
    assert all(evidence_path_exists_in_graph(small_graph, item) for item in tier2_items)
    assert any(edge["relation"] == "interacts_with" for edge in tier2_items[0].path)
    assert any(effect["amplified"] for effect in retrieved.graph_context["ranked_effects"])


def test_display_limits_keep_only_shown_effects_as_documented_evidence(
    small_graph: DecagonGraph,
) -> None:
    unlimited = retrieve_regimen_evidence(small_graph, ["D1", "D2", "D3"])
    nothing_displayed = retrieve_regimen_evidence(
        small_graph,
        ["D1", "D2", "D3"],
        display_limits=EffectDisplayLimits(max_serious_effects=0, max_other_effects=0),
    )
    assert any(item.tier == 1 for item in unlimited.evidence_items)
    assert not any(item.tier in (1, 2) for item in nothing_displayed.evidence_items)
    tier3_pairs_unlimited = {item.drug_pair for item in unlimited.evidence_items if item.tier == 3}
    tier3_pairs_limited = {item.drug_pair for item in nothing_displayed.evidence_items if item.tier == 3}
    assert tier3_pairs_limited == tier3_pairs_unlimited
