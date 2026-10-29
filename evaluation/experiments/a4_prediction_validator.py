"""A4: does the validator keep correct LLM predictions and reject every corrupted one?

Real evidence is turned into valid predictions and into deliberately wrong
ones: an invented evidence ID, the wrong tier, the wrong side effect, a Tier 3
label outside the bridge's candidates, a label outside the Decagon vocabulary.
"""

import collections
import random

from graphrag.drug_names.name_index import NameIndex
from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.knowledge_graph.regimen_evidence_retrieval import retrieve_regimen_evidence
from graphrag.llm.prediction_data_models import ModelPrediction
from graphrag.llm.prediction_validation import validate_predictions
from graphrag.settings.application_settings import Settings
from evaluation.shared.result_csv_files import write_result_csv


REGIMEN_SIZES = (3, 4, 5)
EVIDENCE_ITEMS_USED_PER_REGIMEN = 6
VOCABULARY_SAMPLE_SIZE = 50


def prediction(side_effect: str, tier: int, evidence_ids: list[str]) -> ModelPrediction:
    return ModelPrediction(side_effect=side_effect, tier=tier, evidence_ids=evidence_ids, explanation="x")


def run_validator_experiment(
    settings: Settings, graph: DecagonGraph, names: NameIndex, seed: int, regimen_count: int
) -> None:
    print(f"A4 validator robustness ({regimen_count} regimens)")
    random_generator = random.Random(seed + 7)
    drugs_with_targets = sorted(graph.target_proteins_by_drug)
    vocabulary = sorted(graph.canonical_side_effect_names.values())
    prediction_count_by_kind = collections.Counter()
    kept_count_by_kind = collections.Counter()
    for _ in range(regimen_count):
        regimen = random_generator.sample(drugs_with_targets, random_generator.choice(REGIMEN_SIZES))
        evidence_items = retrieve_regimen_evidence(
            graph, regimen, names.is_serious, settings.ranking_weights
        ).evidence_items
        if not evidence_items:
            continue
        generated: list[tuple[str, ModelPrediction]] = []
        sampled_evidence = random_generator.sample(
            evidence_items, min(EVIDENCE_ITEMS_USED_PER_REGIMEN, len(evidence_items))
        )
        for evidence in sampled_evidence:
            if evidence.tier in (1, 2):
                generated += _documented_predictions(evidence, evidence_items)
            else:
                generated += _hypothesis_predictions(evidence, vocabulary, random_generator)
        invented_id = f"E{len(evidence_items) + 1000}"
        generated.append(("invented_id", prediction(vocabulary[0], 1, [invented_id])))
        for kind, generated_prediction in generated:
            prediction_count_by_kind[kind] += 1
            kept_count_by_kind[kind] += bool(validate_predictions([generated_prediction], evidence_items, graph))

    def valid_first_then_by_name(kind: str) -> tuple[bool, str]:
        return (kind != "valid", kind)

    rows = []
    for kind in sorted(prediction_count_by_kind, key=valid_first_then_by_name):
        kept_rate = kept_count_by_kind[kind] / prediction_count_by_kind[kind]
        rows.append(
            {
                "case": kind,
                "predictions": prediction_count_by_kind[kind],
                "kept": kept_count_by_kind[kind],
                "rejected": prediction_count_by_kind[kind] - kept_count_by_kind[kind],
                "expected": "kept" if kind == "valid" else "rejected",
                "correct_rate": round(kept_rate if kind == "valid" else 1 - kept_rate, 4),
            }
        )
        print(f"  {kind}: {rows[-1]}")
    write_result_csv("a4_validator.csv", rows)


def _documented_predictions(evidence, evidence_items) -> list[tuple[str, ModelPrediction]]:
    label = evidence.side_effect
    other_evidence = next(
        (item for item in evidence_items if item.side_effect and item.side_effect != label), None
    )
    generated = [
        ("valid", prediction(label, evidence.tier, [evidence.evidence_id])),
        ("invented_id", prediction(label, evidence.tier, [evidence.evidence_id, "E99999"])),
        ("wrong_tier", prediction(label, 3, [evidence.evidence_id])),
    ]
    if other_evidence is not None:
        generated.append(("wrong_label", prediction(other_evidence.side_effect, evidence.tier, [evidence.evidence_id])))
    return generated


def _hypothesis_predictions(evidence, vocabulary, random_generator) -> list[tuple[str, ModelPrediction]]:
    candidates = evidence.candidate_side_effects
    generated = []
    if candidates:
        generated.append(("valid", prediction(random_generator.choice(candidates), 3, [evidence.evidence_id])))
    labels_outside_candidates = [
        label
        for label in random_generator.sample(vocabulary, VOCABULARY_SAMPLE_SIZE)
        if label not in candidates
    ]
    generated.append(("label_not_candidate", prediction(labels_outside_candidates[0], 3, [evidence.evidence_id])))
    generated.append(("invented_label", prediction("made-up syndrome", 3, [evidence.evidence_id])))
    wrong_tier_label = candidates[0] if candidates else "x"
    generated.append(("wrong_tier", prediction(wrong_tier_label, 1, [evidence.evidence_id])))
    return generated
