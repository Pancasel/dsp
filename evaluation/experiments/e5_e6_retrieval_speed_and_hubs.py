"""E5: how the hub threshold changes the evidence and speed. E6: retrieval time by regimen size.

Both use random regimens; the timings depend on the machine, so they differ
slightly between runs.
"""

import math
import random
import statistics
import time

from graphrag.drug_names.name_index import NameIndex
from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.knowledge_graph.regimen_evidence_retrieval import retrieve_regimen_evidence
from evaluation.shared.result_csv_files import write_result_csv
from evaluation.shared.system_loading import load_evaluation_system


HUB_THRESHOLDS_TESTED = (100, 250, 500, 1000, 100000)
# 100000 is above every protein's degree: no hub at all.
NO_HUB_THRESHOLD = 100000
REGIMEN_SIZE_FOR_HUB_TEST = 4
REGIMEN_SIZES_FOR_SPEED_TEST = (3, 5, 8)


def random_regimens(graph: DecagonGraph, regimen_size: int, count: int, seed: int) -> list[list[str]]:
    random_generator = random.Random(seed * 1000 + regimen_size)
    all_drugs = sorted(graph.drug_ids)
    return [random_generator.sample(all_drugs, regimen_size) for _ in range(count)]


def run_hub_threshold_experiment(names: NameIndex, seed: int, regimen_count: int) -> None:
    print("E5 hub threshold ablation (k=4)")
    rows = []
    for threshold in HUB_THRESHOLDS_TESTED:
        graph = load_evaluation_system(hub_degree_threshold=threshold).graph
        regimens = random_regimens(graph, REGIMEN_SIZE_FOR_HUB_TEST, regimen_count, seed)
        tier3_counts, pruned_hub_counts, seconds = [], [], []
        for regimen in regimens:
            start = time.perf_counter()
            retrieved = retrieve_regimen_evidence(graph, regimen, names.is_serious)
            seconds.append(time.perf_counter() - start)
            tier3_counts.append(sum(item.tier == 3 for item in retrieved.evidence_items))
            pruned_hub_counts.append(len(retrieved.graph_context["pruned_hubs"]))
        row = {
            "hub_degree_threshold": threshold if threshold < NO_HUB_THRESHOLD else "none",
            "regimens": len(regimens),
            "tier3_evidence_mean": round(statistics.mean(tier3_counts), 2),
            "pruned_hubs_mean": round(statistics.mean(pruned_hub_counts), 2),
            "retrieve_seconds_mean": round(statistics.mean(seconds), 4),
        }
        rows.append(row)
        print(f"  {row}")
    write_result_csv("e5_hub_threshold.csv", rows)


def run_retrieval_latency_experiment(
    graph: DecagonGraph, names: NameIndex, seed: int, regimen_count: int
) -> None:
    print("E6 retrieval latency by regimen size")
    rows = []
    for regimen_size in REGIMEN_SIZES_FOR_SPEED_TEST:
        seconds, evidence_counts, documented_effect_counts = [], [], []
        for regimen in random_regimens(graph, regimen_size, regimen_count, seed):
            start = time.perf_counter()
            retrieved = retrieve_regimen_evidence(graph, regimen, names.is_serious)
            seconds.append(time.perf_counter() - start)
            evidence_counts.append(len(retrieved.evidence_items))
            documented_effect_counts.append(len(retrieved.graph_context["ranked_effects"]))
        percentile_95_position = int(0.95 * (len(seconds) - 1))
        row = {
            "k": regimen_size,
            "pairs": math.comb(regimen_size, 2),
            "regimens": regimen_count,
            "retrieve_seconds_mean": round(statistics.mean(seconds), 4),
            "retrieve_seconds_p95": round(sorted(seconds)[percentile_95_position], 4),
            "evidence_mean": round(statistics.mean(evidence_counts), 1),
            "documented_effects_mean": round(statistics.mean(documented_effect_counts), 1),
        }
        rows.append(row)
        print(f"  {row}")
    write_result_csv("e6_latency.csv", rows)
