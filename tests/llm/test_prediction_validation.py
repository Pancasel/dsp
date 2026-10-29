from dataclasses import replace
from pathlib import Path

from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.knowledge_graph.regimen_evidence_retrieval import retrieve_regimen_evidence
from graphrag.llm.prediction_data_models import ModelPrediction
from graphrag.llm.prediction_validation import validate_predictions
from tests.csv_test_data import (
    DRUG_PAIR_SIDE_EFFECTS_HEADER,
    DRUG_TARGETS_HEADER,
    PROTEIN_INTERACTIONS_HEADER,
    SINGLE_DRUG_SIDE_EFFECTS_HEADER,
    write_csv,
)


def test_unknown_ids_and_wrong_effects_are_discarded(small_graph: DecagonGraph) -> None:
    evidence_items = retrieve_regimen_evidence(small_graph, ["D1", "D2", "D3"]).evidence_items
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
        evidence_items,
        small_graph,
    )

    assert len(accepted) == 1
    assert accepted[0]["side_effect"] == "known event"
    assert accepted[0]["hypothesis"] is False
    cited = accepted[0]["evidence"]
    assert isinstance(cited, list)
    assert len(cited) == 1
    assert cited[0]["evidence_id"] == "E1"


def test_tier3_label_must_belong_to_the_decagon_vocabulary(small_graph: DecagonGraph) -> None:
    evidence_items = retrieve_regimen_evidence(small_graph, ["D1", "D2", "D3"]).evidence_items
    tier3_item = next(item for item in evidence_items if item.tier == 3)
    rejected = validate_predictions(
        [
            ModelPrediction(
                side_effect="invented syndrome",
                tier=3,
                evidence_ids=[tier3_item.evidence_id],
                explanation="This label is outside the source vocabulary.",
            )
        ],
        evidence_items,
        small_graph,
    )
    accepted = validate_predictions(
        [
            ModelPrediction(
                side_effect="Known Event",
                tier=3,
                evidence_ids=[tier3_item.evidence_id],
                explanation="A hypothesis using a recorded side-effect name.",
            )
        ],
        evidence_items,
        small_graph,
    )

    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0]["side_effect"] == "known event"
    assert accepted[0]["hypothesis"] is True


def test_rejections_are_reported_with_reason(small_graph: DecagonGraph) -> None:
    rejections: list[dict[str, str]] = []
    accepted = validate_predictions(
        [ModelPrediction(side_effect="x", tier=1, evidence_ids=["E999"], explanation="n/a")],
        [],
        small_graph,
        rejections,
    )
    assert accepted == []
    assert rejections == [{"side_effect": "x", "tier": "1", "reason": "unknown_evidence"}]


def test_tier3_labels_must_come_from_pair_candidates(tmp_path: Path) -> None:
    write_csv(tmp_path / "targets.csv", DRUG_TARGETS_HEADER, [["D1", "G1"], ["D2", "G2"], ["D3", "G3"]])
    write_csv(tmp_path / "ppi.csv", PROTEIN_INTERACTIONS_HEADER, [["G1", "G2"]])
    write_csv(
        tmp_path / "combo.csv",
        DRUG_PAIR_SIDE_EFFECTS_HEADER,
        [
            ["D1", "D3", "C1", "shared event"],
            ["D2", "D3", "C1", "shared event"],
            ["D2", "D3", "C2", "other event"],
            ["D3", "D4", "C3", "unrelated event"],
        ],
    )
    write_csv(tmp_path / "mono.csv", SINGLE_DRUG_SIDE_EFFECTS_HEADER, [["D1", "C1", "shared event"]])
    graph = DecagonGraph.load(
        drug_targets_file=tmp_path / "targets.csv",
        protein_interactions_file=tmp_path / "ppi.csv",
        drug_pair_side_effects_file=tmp_path / "combo.csv",
        cache_database_file=tmp_path / "cache.sqlite3",
        hub_degree_threshold=50,
        single_drug_side_effects_file=tmp_path / "mono.csv",
    )

    assert graph.single_drug_effect_codes("D1") == {"C1"}
    assert graph.total_pair_count == 3
    candidate_names = graph.tier3_candidate_effect_names("D1", "D2")
    assert candidate_names[0] == "shared event"  # seen with both drugs
    assert "unrelated event" not in candidate_names

    retrieved = retrieve_regimen_evidence(graph, ["D1", "D2", "D3"])
    tier3_item = next(item for item in retrieved.evidence_items if item.tier == 3)
    assert tier3_item.drug_pair == ("D1", "D2")
    assert "shared event" in tier3_item.candidate_side_effects
    assert retrieved.graph_context["ranked_effects"][0]["name"] == "shared event"

    def tier3_prediction(side_effect: str, evidence_id: str) -> ModelPrediction:
        return ModelPrediction(side_effect=side_effect, tier=3, evidence_ids=[evidence_id], explanation="x")

    accepted = validate_predictions(
        [tier3_prediction("shared event", tier3_item.evidence_id)], retrieved.evidence_items, graph
    )
    rejected = validate_predictions(
        [tier3_prediction("unrelated event", tier3_item.evidence_id)], retrieved.evidence_items, graph
    )
    assert len(accepted) == 1 and accepted[0]["hypothesis"] is True
    assert rejected == []

    # A bridge without candidate effects supports no Tier 3 label at all.
    bridge_without_candidates = replace(tier3_item, candidate_side_effects=())
    assert validate_predictions(
        [tier3_prediction("shared event", bridge_without_candidates.evidence_id)],
        [bridge_without_candidates],
        graph,
    ) == []
