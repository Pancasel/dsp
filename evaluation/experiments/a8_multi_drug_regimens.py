"""A8: known interactions inside hand-written regimens of 4-8 drugs.

evaluation/test_cases/clinical_benchmark_multi_drug.csv holds 20 realistic
regimens, each containing one well-known interacting pair. Same scoring as A5
(default weights, rank of the expected effect in the serious list; the
interface shows 10). Cases whose pair has no record of the expected effect in
Decagon are listed but left out of the hit rates, as in A5. The cases are
written from standard pharmacology knowledge and have not been reviewed by a
clinician.
"""

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


# (smallest, largest) regimen size of each summary line.
SIZE_GROUPS = ((4, 5), (6, 8), (4, 8))


def run_multi_drug_regimen_experiment(system: EvaluationSystem) -> None:
    settings, graph, names, resolver = system.settings, system.graph, system.names, system.resolver
    cases = read_test_cases("clinical_benchmark_multi_drug.csv")
    rows = []
    # One (regimen size, rank, chance of a hit) per case whose pair documents the effect.
    scored_cases = []
    for case in cases:
        drugs = [resolver.resolve(name).stitch_id for name in case["drugs"].split("|")]
        pair = tuple(
            sorted(resolver.resolve(name).stitch_id for name in case["interacting_pair"].split("|"))
        )
        assert None not in drugs and None not in pair, case["case"]
        pattern = expected_effect_pattern(case["expected"])
        pair_documents_effect = any(pattern.search(effect.name) for effect in graph.pair_side_effects(*pair))
        ranked_effects = retrieve_regimen_evidence(
            graph, drugs, names.is_serious, settings.ranking_weights, names.is_critical
        ).graph_context["ranked_effects"]
        serious_effects = [effect for effect in ranked_effects if effect["serious"]]
        rank = rank_of_first_match(serious_effects, pattern)
        rows.append(
            {"case": case["case"], "drugs_n": len(drugs), "expected": case["expected"],
             "pair_documented": int(pair_documents_effect), "serious_effects": len(serious_effects),
             "rank": rank or ""}
        )
        if pair_documents_effect:
            chance_of_hit = min(1.0, SERIOUS_EFFECTS_SHOWN / len(serious_effects)) if rank else 0.0
            scored_cases.append((len(drugs), rank, chance_of_hit))
    write_result_csv("a8_multi_drug.csv", rows)

    print(f"{len(cases)} cases, {len(scored_cases)} with the expected effect documented for the pair")
    for smallest, largest in SIZE_GROUPS:
        group = [scored for scored in scored_cases if smallest <= scored[0] <= largest]
        label = f"{smallest}-{largest} drugs" if smallest != largest else f"{smallest} drugs"
        _print_summary(label, group)
    for row in rows:
        print(" ", row)


def _print_summary(label: str, group: list[tuple[int, int | None, float]]) -> None:
    ranks = [rank for _, rank, _ in group]
    print(
        f"{label}: cases={len(group)} hit@3={hit_rate(ranks, 3):.3f} "
        f"hit@10={hit_rate(ranks, SERIOUS_EFFECTS_SHOWN):.3f} "
        f"chance@10={statistics.mean(chance for _, _, chance in group):.3f} "
        f"MRR={mean_reciprocal_rank(ranks):.3f} "
        f"ranks={sorted(rank for rank in ranks if rank)}"
    )
