import csv
import zipfile
from pathlib import Path

import pytest

from graphrag.dataset import DecagonGraph
from graphrag.llm import ModelPrediction, validate_predictions


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


@pytest.fixture
def graph(tmp_path: Path) -> DecagonGraph:
    targets = tmp_path / "targets.csv"
    ppi = tmp_path / "ppi.csv"
    combo_csv = tmp_path / "combo.csv"
    _write_csv(
        targets,
        ["STITCH", "Gene"],
        [
            ["D1", "G1"],
            ["D1", "SHARED"],
            ["D2", "G2"],
            ["D2", "SHARED"],
            ["D3", "G3"],
            ["D3", "SHARED"],
        ],
    )
    _write_csv(
        ppi,
        ["Gene 1", "Gene 2"],
        [
            ["G1", "MID"],
            ["MID", "G2"],
            ["G1", "HUB"],
            ["HUB", "G3"],
            ["SHARED", "HUB"],
            ["HUB", "P1"],
            ["G1", "LOW"],
            ["LOW", "G3"],
        ],
    )
    _write_csv(
        combo_csv,
        ["STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"],
        [
            ["D1", "D2", "C1", "known event"],
            ["D3", "D4", "C2", "combo-only event"],
        ],
    )
    combo_zip = tmp_path / "combo.zip"
    with zipfile.ZipFile(combo_zip, "w") as archive:
        archive.write(combo_csv, "combo/bio-decagon-combo.csv")
    cache = tmp_path / "cache.sqlite3"
    return DecagonGraph.load(targets, ppi, combo_zip, cache, 2)


def test_retrieves_known_effect_shared_target_and_prunes_hub(
    graph: DecagonGraph,
) -> None:
    context, evidence = graph.retrieve(["D1", "D2", "D3"])

    assert context["shared_targets"] == [
        {"protein": "SHARED", "drugs": ["D1", "D2", "D3"]}
    ]
    assert "HUB" in context["pruned_hubs"]
    assert any(item.tier == 1 and item.side_effect == "known event" for item in evidence)
    assert any(item.tier == 2 and item.side_effect == "known event" for item in evidence)
    tier3_pairs = {item.drug_pair for item in evidence if item.tier == 3}
    assert ("D1", "D3") in tier3_pairs
    assert ("D1", "D2") not in tier3_pairs
    assert all(graph.validate_evidence(item) for item in evidence)


def test_two_hop_bridge_is_pruned_when_intermediate_is_hub(
    graph: DecagonGraph,
) -> None:
    context, evidence = graph.retrieve(["D1", "D2", "D3"])
    d1_d2_tier3 = [
        item
        for item in evidence
        if item.tier == 3 and item.drug_pair == ("D1", "D2")
    ]
    assert not d1_d2_tier3
    assert "HUB" in context["pruned_hubs"]
    assert all(
        "HUB" not in {edge["source"] for edge in item.path}
        | {edge["target"] for edge in item.path}
        for item in evidence
    )


def test_combo_cache_rebuilds_when_source_changes(graph: DecagonGraph, tmp_path: Path) -> None:
    combo_csv = tmp_path / "combo.csv"
    _write_csv(
        combo_csv,
        ["STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"],
        [["D1", "D3", "C2", "new event"]],
    )
    context = tmp_path / "rebuild.sqlite3"
    loaded = DecagonGraph.load(
        tmp_path / "targets.csv",
        tmp_path / "ppi.csv",
        combo_csv,
        context,
        2,
    )
    assert [effect.name for effect in loaded.side_effects("D3", "D1")] == [
        "new event"
    ]


def test_prediction_validation_discards_unknown_ids_and_wrong_effects(
    graph: DecagonGraph,
) -> None:
    _, evidence = graph.retrieve(["D1", "D2", "D3"])
    accepted = validate_predictions(
        [
            ModelPrediction(
                side_effect="known event",
                tier=1,
                evidence_ids=["E1", "E2", "untrusted-id"],
                explanation="Recorded in the pairwise dataset.",
            ),
            ModelPrediction(
                side_effect="invented known event",
                tier=1,
                evidence_ids=["E1"],
                explanation="Unsupported label.",
            ),
            ModelPrediction(
                side_effect="unsupported hypothesis",
                tier=3,
                evidence_ids=["untrusted-id"],
                explanation="Unsupported path.",
            ),
            ModelPrediction(
                side_effect="known event",
                tier=1,
                evidence_ids=["E1"],
                explanation="Recorded in the pairwise dataset.",
            ),
        ],
        evidence,
        graph,
    )

    assert len(accepted) == 1
    assert accepted[0]["side_effect"] == "known event"
    assert accepted[0]["hypothesis"] is False
    cited = accepted[0]["evidence"]
    assert isinstance(cited, list)
    assert len(cited) == 1
    assert cited[0]["evidence_id"] == "E1"


def test_tier3_label_must_belong_to_the_decagon_vocabulary(graph: DecagonGraph) -> None:
    _, evidence = graph.retrieve(["D1", "D2", "D3"])
    tier3 = next(item for item in evidence if item.tier == 3)
    rejected = validate_predictions(
        [
            ModelPrediction(
                side_effect="invented syndrome",
                tier=3,
                evidence_ids=[tier3.evidence_id],
                explanation="This label is outside the source vocabulary.",
            )
        ],
        evidence,
        graph,
    )
    accepted = validate_predictions(
        [
            ModelPrediction(
                side_effect="Known Event",
                tier=3,
                evidence_ids=[tier3.evidence_id],
                explanation="A hypothesis using a recorded side-effect name.",
            )
        ],
        evidence,
        graph,
    )

    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0]["side_effect"] == "known event"
    assert accepted[0]["hypothesis"] is True


def test_tier3_bridge_search_stops_at_the_per_pair_cap(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    ppi = tmp_path / "ppi.csv"
    combo = tmp_path / "combo.csv"
    _write_csv(
        targets,
        ["STITCH", "Gene"],
        [["D1", "A"], ["D2", "B"], ["D3", "C"]],
    )
    _write_csv(
        ppi,
        ["Gene 1", "Gene 2"],
        [["A", "M1"], ["M1", "B"], ["A", "M2"], ["M2", "B"], ["A", "M3"], ["M3", "B"]],
    )
    _write_csv(
        combo,
        ["STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"],
        [],
    )
    loaded = DecagonGraph.load(
        targets,
        ppi,
        combo,
        tmp_path / "cache.sqlite3",
        50,
        tier3_two_hop_per_pair=2,
    )
    context, evidence = loaded.retrieve(["D1", "D2", "D3"])
    tier3 = [
        item
        for item in evidence
        if item.tier == 3 and item.drug_pair == ("D1", "D2")
    ]

    assert len(tier3) == 2
    assert context["evidence_truncated"] is True
    assert all(loaded.validate_evidence(item) for item in evidence)


def test_hub_target_is_an_endpoint_and_not_a_bridge_intermediate(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    ppi = tmp_path / "ppi.csv"
    combo = tmp_path / "combo.csv"
    _write_csv(
        targets,
        ["STITCH", "Gene"],
        [["D1", "G1"], ["D2", "HUB"], ["D2", "G2"], ["D3", "C"]],
    )
    _write_csv(
        ppi,
        ["Gene 1", "Gene 2"],
        [["G1", "HUB"], ["HUB", "G2"], ["HUB", "X"]],
    )
    _write_csv(
        combo,
        ["STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"],
        [],
    )
    loaded = DecagonGraph.load(targets, ppi, combo, tmp_path / "cache.sqlite3", 2)
    _, evidence = loaded.retrieve(["D1", "D2", "D3"])
    tier3 = [
        item
        for item in evidence
        if item.tier == 3 and item.drug_pair == ("D1", "D2")
    ]

    assert tier3
    assert any(
        edge["relation"] == "interacts_with"
        and edge["source"] == "G1"
        and edge["target"] == "HUB"
        for item in tier3
        for edge in item.path
    )
    assert all(
        edge["source"] != "HUB"
        for item in tier3
        for edge in item.path
        if edge["relation"] == "interacts_with"
    )


def test_combo_only_drug_keeps_pairwise_history(graph: DecagonGraph) -> None:
    context, evidence = graph.retrieve(["D1", "D3", "D4"])

    assert context["direct_targets"]["D4"] == []
    assert any(
        item.tier == 1
        and item.drug_pair == ("D3", "D4")
        and item.side_effect == "combo-only event"
        for item in evidence
    )
    assert not any(item.tier == 3 and "D4" in item.drug_pair for item in evidence)


def test_degree_percentile_threshold():
    from graphrag.dataset import degree_percentile_threshold

    ppi = {f"P{i}": {f"N{j}" for j in range(i)} for i in range(1, 101)}
    assert degree_percentile_threshold(ppi, 50) == 51
    assert degree_percentile_threshold(ppi, 100) == 100
    assert degree_percentile_threshold({}, 90) == 0


def test_rejections_are_reported_with_reason(graph):
    rejections: list[dict[str, str]] = []
    accepted = validate_predictions(
        [ModelPrediction(side_effect="x", tier=1, evidence_ids=["E999"], explanation="n/a")],
        [],
        graph,
        rejections,
    )
    assert accepted == []
    assert rejections == [{"side_effect": "x", "tier": "1", "reason": "unknown_evidence"}]


def test_convergence_targets_use_one_hop_neighbourhood_without_hubs(graph):
    graph.hub_degree_threshold = 3
    targets_by_drug = {"D1": {"G1"}, "D2": {"MID"}, "D3": {"LOW"}}

    assert graph.convergence_targets(targets_by_drug) == {"G1": ["D1", "D2", "D3"]}
    # G1 is reached by D2 and D3 only through neighbours; HUB (degree 4) is skipped.
    assert "HUB" not in graph.convergence_targets(targets_by_drug)


def test_convergence_amplification_adds_validated_tier2_paths(graph):
    graph.hub_degree_threshold = 3
    graph.targets = {"D1": {"G1"}, "D2": {"MID"}, "D3": {"LOW"}}
    graph.convergence_amplification = False
    _, evidence = graph.retrieve(["D1", "D2", "D3"])
    assert not any(item.tier == 2 for item in evidence)

    graph.convergence_amplification = True
    context, evidence = graph.retrieve(["D1", "D2", "D3"])
    tier2 = [item for item in evidence if item.tier == 2]
    assert tier2 and all(graph.validate_evidence(item) for item in tier2)
    assert any(edge["relation"] == "interacts_with" for edge in tier2[0].path)
    assert any(item["amplified"] for item in context["ranked_effects"])


def test_tier1_cannot_use_up_the_budget_reserved_for_tier2_and_tier3(graph):
    from graphrag.dataset import _Candidate

    def group(tier, pair, count):
        return [
            _Candidate(tier=tier, kind="k", drug_pair=pair, path=(), effect=None)
            for _ in range(count)
        ]

    graph.max_evidence = 8
    evidence, truncated = graph._select_evidence(
        ([group(1, ("D1", "D2"), 20)], [group(2, ("D1", "D2"), 20)], [group(3, ("D1", "D3"), 1)]),
        ["D1", "D2", "D3"],
        set(),
    )
    tiers = [item.tier for item in evidence]
    # Tier 3 uses one of its two reserved slots; the spare one goes to Tier 1.
    assert (tiers.count(1), tiers.count(2), tiers.count(3)) == (5, 2, 1)
    assert truncated


def test_focus_limits_keep_only_shown_effects_as_documented_evidence(graph):
    _, all_evidence = graph.retrieve(["D1", "D2", "D3"])
    _, focused = graph.retrieve(["D1", "D2", "D3"], focus_limits=(0, 0))
    assert any(item.tier == 1 for item in all_evidence)
    assert not any(item.tier in (1, 2) for item in focused)
    assert {item.drug_pair for item in focused if item.tier == 3} == {
        item.drug_pair for item in all_evidence if item.tier == 3
    }
