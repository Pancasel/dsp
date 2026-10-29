"""A7: does the known-interaction signal survive larger regimens?

Decagon has no ground truth for regimens of 3+ drugs, so this reuses the known
pairwise interactions of clinical_benchmark.csv and adds extra "bystander"
drugs (drawn, with a fixed seed, from the other benchmark cases, never from
the case itself). The expected effect is still the one of the known pair; the
question is whether it stays near the top of the serious list as the regimen
grows. Bystanders can add their own bleeding/renal/... pairs, which is exactly
what makes larger regimens harder.
"""

import random
import statistics

from graphrag.knowledge_graph.regimen_evidence_retrieval import retrieve_regimen_evidence
from evaluation.shared.clinical_expectations import (
    SERIOUS_EFFECTS_SHOWN,
    expected_effect_pattern,
    hit_rate,
    mean_reciprocal_rank,
    rank_of_first_match,
)
from evaluation.shared.result_csv_files import read_test_cases, write_result_csv
from evaluation.shared.system_loading import EvaluationSystem


REGIMEN_SIZES = (3, 4, 5, 6, 8)
SEEDS = range(5)


def run_noise_drug_scaling_experiment(system: EvaluationSystem) -> None:
    settings, graph, names, resolver = system.settings, system.graph, system.names, system.resolver
    cases = read_test_cases("clinical_benchmark.csv")
    all_benchmark_drug_names = sorted({name for case in cases for name in case["drugs"].split("|")})
    drug_id_by_name = {name: resolver.resolve(name).stitch_id for name in all_benchmark_drug_names}

    rows = []
    for regimen_size in REGIMEN_SIZES:
        ranks, chance_of_hit = [], []
        for case in cases:
            case_drug_names = case["drugs"].split("|")
            case_drug_ids = [drug_id_by_name[name] for name in case_drug_names]
            pair_ids = [drug_id_by_name[name] for name in case["interacting_pair"].split("|")]
            if None in case_drug_ids or None in pair_ids:
                continue  # a drug of this case is not in Decagon
            pattern = expected_effect_pattern(case["expected"])
            seeds = SEEDS if regimen_size > 3 else [0]
            for seed in seeds:
                random_generator = random.Random(f"{case['case']}-{regimen_size}-{seed}")
                bystander_pool = [
                    drug_id_by_name[name]
                    for name in all_benchmark_drug_names
                    if name not in case_drug_names and drug_id_by_name[name]
                ]
                drugs = case_drug_ids + random_generator.sample(bystander_pool, regimen_size - len(case_drug_ids))
                ranked_effects = retrieve_regimen_evidence(
                    graph, drugs, names.is_serious, settings.ranking_weights, names.is_critical
                ).graph_context["ranked_effects"]
                serious_effects = [effect for effect in ranked_effects if effect["serious"]]
                rank = rank_of_first_match(serious_effects, pattern)
                ranks.append(rank)
                # Chance that a random order of the serious list puts it in the shown part.
                chance_of_hit.append(min(1.0, SERIOUS_EFFECTS_SHOWN / len(serious_effects)) if rank else 0.0)
                rows.append(
                    {"size": regimen_size, "case": case["case"], "seed": seed, "rank": rank or "",
                     "drugs": ";".join(drugs)}
                )
        found_ranks = [rank for rank in ranks if rank]
        median_rank = statistics.median(found_ranks) if found_ranks else "-"
        print(
            f"{regimen_size} drugs: runs={len(ranks)} hit@3={hit_rate(ranks, 3):.3f} "
            f"hit@10={hit_rate(ranks, SERIOUS_EFFECTS_SHOWN):.3f} "
            f"chance@10={statistics.mean(chance_of_hit):.3f} "
            f"MRR={mean_reciprocal_rank(ranks):.3f} "
            f"median_rank={median_rank}"
        )
    write_result_csv("a7_scaling.csv", rows)
