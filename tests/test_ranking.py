import csv
from pathlib import Path

from graphrag.config import RankingWeights
from graphrag.dataset import DecagonGraph, SideEffect
from graphrag.llm import ModelPrediction, validate_predictions
from graphrag.names import NameIndex
from graphrag.ranking import rank_effects


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def test_rare_serious_and_interaction_only_effects_rank_first() -> None:
    pair = ("A", "B")
    effects = {
        pair: [
            SideEffect("C_common", "nausea"),
            SideEffect("C_rare", "rhabdomyolysis"),
            SideEffect("C_mono", "rash"),
        ]
    }
    ranked = rank_effects(
        effects,
        amplified_pairs=set(),
        effect_pair_count={"C_common": 1000, "C_rare": 5, "C_mono": 5},
        total_pairs=1000,
        mono_codes=lambda drug: {"C_mono"} if drug == "A" else set(),
        is_serious=lambda name: name == "rhabdomyolysis",
    )
    assert [item.name for item in ranked] == ["rhabdomyolysis", "rash", "nausea"]
    assert ranked[0].serious and ranked[0].not_in_mono
    assert ranked[1].not_in_mono is False
    assert ranked[2].idf == 0.0


def test_recurrence_across_pairs_and_weight_ablation() -> None:
    effects = {
        ("A", "B"): [SideEffect("C1", "bleeding")],
        ("A", "C"): [SideEffect("C1", "bleeding"), SideEffect("C2", "dizziness")],
    }
    ranked = rank_effects(effects, {("A", "C")}, {"C1": 10, "C2": 10}, 100, lambda d: set())
    assert ranked[0].name == "bleeding"
    assert ranked[0].recurrence == 2
    assert ranked[0].pairs == [("A", "B"), ("A", "C")]
    assert ranked[0].tier == 2  # one of its pairs is amplified
    assert ranked[0].amplified_pairs == [("A", "C")]  # only that pair, not ("A", "B")
    only_idf = rank_effects(
        effects,
        set(),
        {"C1": 50, "C2": 2},
        100,
        lambda d: set(),
        weights=RankingWeights(recurrence=0, idf=1, not_in_mono=0, serious=0, amplified=0),
    )
    assert only_idf[0].name == "dizziness"


def test_serious_keywords_match_whole_words() -> None:
    names = NameIndex({}, {}, {}, ["coma", "haemorrhage"])
    assert names.is_serious("coma")
    assert names.is_serious("gastrointestinal haemorrhage")
    assert not names.is_serious("glaucoma")


def test_tier3_labels_must_come_from_pair_candidates(tmp_path: Path) -> None:
    targets = tmp_path / "targets.csv"
    ppi = tmp_path / "ppi.csv"
    combo = tmp_path / "combo.csv"
    mono = tmp_path / "mono.csv"
    _write_csv(targets, ["STITCH", "Gene"], [["D1", "G1"], ["D2", "G2"], ["D3", "G3"]])
    _write_csv(ppi, ["Gene 1", "Gene 2"], [["G1", "G2"]])
    _write_csv(
        combo,
        ["STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"],
        [
            ["D1", "D3", "C1", "shared event"],
            ["D2", "D3", "C1", "shared event"],
            ["D2", "D3", "C2", "other event"],
            ["D3", "D4", "C3", "unrelated event"],
        ],
    )
    _write_csv(mono, ["STITCH", "Individual Side Effect", "Side Effect Name"], [["D1", "C1", "shared event"]])
    graph = DecagonGraph.load(targets, ppi, combo, tmp_path / "cache.sqlite3", 50, mono_file=mono)

    assert graph.mono_codes("D1") == {"C1"}
    assert graph.total_pairs == 3
    candidates = graph.tier3_candidates("D1", "D2")
    assert candidates[0] == "shared event"  # seen with both drugs
    assert "unrelated event" not in candidates

    context, evidence = graph.retrieve(["D1", "D2", "D3"])
    tier3 = next(item for item in evidence if item.tier == 3)
    assert tier3.drug_pair == ("D1", "D2")
    assert "shared event" in tier3.candidates
    assert context["ranked_effects"][0]["name"] == "shared event"

    accepted = validate_predictions(
        [ModelPrediction(side_effect="shared event", tier=3, evidence_ids=[tier3.evidence_id], explanation="x")],
        evidence,
        graph,
    )
    rejected = validate_predictions(
        [ModelPrediction(side_effect="unrelated event", tier=3, evidence_ids=[tier3.evidence_id], explanation="x")],
        evidence,
        graph,
    )
    assert len(accepted) == 1 and accepted[0]["hypothesis"] is True
    assert rejected == []

    # A bridge without candidate labels supports no Tier 3 label at all.
    from dataclasses import replace

    bare = replace(tier3, candidates=())
    assert validate_predictions(
        [ModelPrediction(side_effect="shared event", tier=3, evidence_ids=[bare.evidence_id], explanation="x")],
        [bare],
        graph,
    ) == []


def test_select_effects_prefers_serious_and_caps_each_group() -> None:
    from graphrag.presenter import select_effects

    ranked = [
        {"name": "rash", "serious": False},
        {"name": "bleeding", "serious": True},
        {"name": "nausea", "serious": False},
        {"name": "coma", "serious": True},
        {"name": "pain", "serious": False},
    ]
    chosen = select_effects(ranked, max_serious=1, max_other=2)
    assert [item["name"] for item in chosen] == ["bleeding", "rash", "nausea"]
