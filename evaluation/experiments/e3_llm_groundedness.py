"""E3: how many LLM predictions survive the evidence validator.

THIS EXPERIMENT CALLS THE CONFIGURED LLM (LLM_MODEL in .env) ONCE PER REGIMEN
and uses API credit. Its runner refuses to start without --yes-call-api.
"""

import random
import statistics
import time

from graphrag.knowledge_graph.regimen_evidence_retrieval import retrieve_regimen_evidence
from graphrag.llm.prediction_request import generate_predictions
from graphrag.llm.prediction_validation import validate_predictions
from graphrag.result_presentation.llm_context import build_llm_context
from graphrag.side_effect_ranking.effects_to_display import EffectDisplayLimits
from evaluation.shared.result_csv_files import write_result_csv
from evaluation.shared.system_loading import EvaluationSystem


REGIMEN_SIZES = (3, 4, 5)


def run_llm_groundedness_experiment(system: EvaluationSystem, regimen_count: int, seed: int) -> None:
    settings, graph, names = system.settings, system.graph, system.names
    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY is not configured")
    random_generator = random.Random(seed)
    # Drugs with targets can produce Tier 2/3 evidence.
    drugs_with_targets = sorted(graph.target_proteins_by_drug)
    regimens = [
        random_generator.sample(drugs_with_targets, random_generator.choice(REGIMEN_SIZES))
        for _ in range(regimen_count)
    ]
    limits = EffectDisplayLimits(settings.max_serious_effects, settings.max_other_effects)
    print(f"model={settings.llm_model}, {len(regimens)} calls")
    rows = []
    for number, regimen in enumerate(regimens, start=1):
        retrieved = retrieve_regimen_evidence(
            graph, regimen, names.is_serious, settings.ranking_weights, names.is_critical
        )
        llm_context = build_llm_context(
            regimen, retrieved.graph_context, retrieved.evidence_items, names, limits
        )
        start = time.perf_counter()
        try:
            predictions = generate_predictions(settings, regimen, llm_context)
            error_text = ""
        except Exception as error:  # report every failure kind in the table
            predictions = []
            error_text = f"{error.__class__.__name__}: {error}"
        seconds = time.perf_counter() - start
        if predictions:
            validated = validate_predictions(predictions, retrieved.evidence_items, graph)
        else:
            validated = []
        sent_ids = {evidence.evidence_id for evidence in retrieved.evidence_items}
        invented_id_count = sum(
            1
            for prediction in predictions
            for evidence_id in prediction.evidence_ids
            if evidence_id not in sent_ids
        )
        rows.append(
            {
                "regimen": "|".join(regimen),
                "k": len(regimen),
                "evidence": len(llm_context["evidence"]),
                "raw_predictions": len(predictions),
                "valid_predictions": len(validated),
                "tier3_valid": sum(prediction["tier"] == 3 for prediction in validated),
                "invented_ids": invented_id_count,
                "latency_s": round(seconds, 2),
                "error": error_text,
            }
        )
        print(f"  {number}/{len(regimens)} {rows[-1]}")
    write_result_csv("e3_llm_groundedness.csv", rows)

    raw_total = sum(row["raw_predictions"] for row in rows)
    valid_total = sum(row["valid_predictions"] for row in rows)
    if raw_total:
        print(f"kept {valid_total}/{raw_total} predictions ({valid_total / raw_total:.1%})")
    else:
        print("no predictions returned")
    print(f"median latency {statistics.median(row['latency_s'] for row in rows)} s")
