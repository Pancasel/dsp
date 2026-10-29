"""A5: do well-known drug interactions reach the top of the serious side-effect list?

Test cases: evaluation/test_cases/clinical_benchmark.csv (an interacting pair
plus a third drug, and the expected outcome). Four ranking variants are
compared by the position of the expected effect among the serious effects.
"""

import collections
import dataclasses
import statistics

from graphrag.drug_names.drug_name_resolver import DrugResolver
from graphrag.drug_names.name_index import NameIndex
from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.knowledge_graph.regimen_evidence_retrieval import retrieve_regimen_evidence
from graphrag.settings.application_settings import Settings
from graphrag.side_effect_ranking.effects_to_display import (
    EffectDisplayLimits,
    choose_effects_to_display,
)
from evaluation.shared.clinical_expectations import expected_effect_pattern, rank_of_first_match
from evaluation.shared.figure_drawing import METHOD_COLORS, draw_grouped_bars
from evaluation.shared.result_csv_files import read_csv_file, read_test_cases, write_result_csv
from evaluation.shared.system_loading import RESOURCES_FOLDER


SUMMARY_METRICS = ("hit@3", "hit@10", "mrr")


def run_clinical_benchmark_experiment(
    settings: Settings, graph: DecagonGraph, names: NameIndex, resolver: DrugResolver
) -> None:
    print("A5 clinical benchmark of established drug interactions")
    cases = read_test_cases("clinical_benchmark.csv")
    excluded_keywords = [
        row["keyword"]
        for row in read_csv_file(RESOURCES_FOLDER / "serious_keywords.csv")
        if row["category"] == "exclude"
    ]
    # "Serious" decided by keywords only, without the EMA labels.
    keyword_only_names = NameIndex(
        drugs_by_id=names.drugs_by_id,
        gene_symbols_by_id=names.gene_symbols_by_id,
        side_effect_translations=names.side_effect_translations,
        critical_keywords=names.seriousness.critical_keywords,
        serious_labels={},
        excluded_keywords=excluded_keywords,
    )
    default_weights = settings.ranking_weights
    # Variant label (also used in the CSV columns) -> (is_serious, ranking weights).
    variants = {
        "Chỉ từ khóa": (keyword_only_names.is_serious, dataclasses.replace(default_weights, critical=0.0)),
        "IME": (names.is_serious, dataclasses.replace(default_weights, critical=0.0)),
        "IME + đe dọa tính mạng (mặc định)": (names.is_serious, default_weights),
        "IME + đe dọa tính mạng, bỏ IDF": (
            names.is_serious, dataclasses.replace(default_weights, effect_rarity=0.0)
        ),
    }
    limits = EffectDisplayLimits(settings.max_serious_effects, settings.max_other_effects)
    rows = []
    # Variant -> one (rank among serious effects, shown on screen, found at all) per case.
    results_by_variant = collections.defaultdict(list)
    for case in cases:
        drugs = [resolver.resolve(name).stitch_id for name in case["drugs"].split("|")]
        pair_ids = [resolver.resolve(name).stitch_id for name in case["interacting_pair"].split("|")]
        if None in drugs or None in pair_ids:
            print(f"  case {case['case']}: drug not in Decagon, skipped")
            continue
        pair = tuple(sorted(pair_ids))
        pattern = expected_effect_pattern(case["expected"])
        row = {"case": case["case"], "drugs": case["drugs"], "expected": case["expected"]}
        pair_effects = graph.pair_side_effects(*pair)
        row["pair_documented"] = int(any(pattern.search(effect.name) for effect in pair_effects))
        for label, (is_serious, weights) in variants.items():
            ranked_effects = retrieve_regimen_evidence(
                graph, drugs, is_serious, weights, names.is_critical
            ).graph_context["ranked_effects"]
            serious_effects = [effect for effect in ranked_effects if effect["serious"]]
            rank = rank_of_first_match(serious_effects, pattern)
            shown_effects = choose_effects_to_display(ranked_effects, limits)
            shown_serious = [
                effect for effect in shown_effects[: settings.max_serious_effects] if effect["serious"]
            ]
            is_shown = any(pattern.search(effect["name"]) for effect in shown_serious)
            row[f"{label}|rank"] = rank or ""
            found_at_all = any(pattern.search(effect["name"]) for effect in ranked_effects)
            results_by_variant[label].append((rank, is_shown, found_at_all))
        rows.append(row)
    write_result_csv("a5_clinical_benchmark.csv", rows)

    summary = []
    for label, results in results_by_variant.items():
        documented_results = [result for result in results if result[2]]
        documented_ranks = [rank for rank, _, _ in documented_results]
        line = {
            "variant": label,
            "cases": len(results),
            "documented": len(documented_results),
            "hit@3": round(
                sum(1 for rank in documented_ranks if rank and rank <= 3) / len(documented_results), 4
            ),
            "hit@10": round(
                sum(1 for rank in documented_ranks if rank and rank <= 10) / len(documented_results), 4
            ),
            "mrr": round(statistics.mean(1 / rank if rank else 0 for rank in documented_ranks), 4),
        }
        summary.append(line)
        print(
            f"  {label}: documented {line['documented']}/{line['cases']} hit@3={line['hit@3']} "
            f"hit@10={line['hit@10']} MRR={line['mrr']}"
        )
    write_result_csv("a5_summary.csv", summary)

    variant_colors = [METHOD_COLORS["random"], METHOD_COLORS["popularity"], METHOD_COLORS["graph"], METHOD_COLORS["extra"]]
    draw_grouped_bars(
        f"A5. Benchmark tương tác kinh điển ({summary[0]['documented']} ca có dữ liệu)",
        list(SUMMARY_METRICS),
        {line["variant"]: {metric: line[metric] for metric in SUMMARY_METRICS} for line in summary},
        dict(zip(variants, variant_colors)),
        "a5_clinical_benchmark.png",
    )
