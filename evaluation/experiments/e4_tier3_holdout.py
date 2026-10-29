"""E4: do the Tier 3 candidate effects recover the real effects of hidden drug pairs?

Documented pairs are hidden one at a time. For a hidden pair (A, B) the
candidate ranking is rebuilt from all other pairs of A and of B, exactly as
DecagonGraph.tier3_candidate_effect_names does, but with the hidden pair's own
effects removed (no leakage). Compared with the globally most common effects
and with random effects.
"""

import collections
import random
import sqlite3
import statistics

from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from evaluation.shared.result_csv_files import write_result_csv


METHODS = ("graph_candidates", "global_popularity", "random")
METRICS = ("hit@10", "precision@10", "recall@20")


def run_tier3_holdout_experiment(graph: DecagonGraph, seed: int, hidden_pair_count: int) -> None:
    print("E4 Tier 3 candidate labels on held-out pairs")
    drugs_with_targets = set(graph.target_proteins_by_drug)
    effect_counts_by_drug: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    effects_by_pair: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    effect_name_by_code: dict[str, str] = {}
    with sqlite3.connect(graph.cache_database_file) as connection:
        for drug_a, drug_b, effect_code, effect_name in connection.execute(
            "SELECT drug_a, drug_b, effect_code, effect_name FROM drug_pair_side_effects"
        ):
            effect_counts_by_drug[drug_a][effect_code] += 1
            effect_counts_by_drug[drug_b][effect_code] += 1
            effect_name_by_code[effect_code] = effect_name
            if drug_a in drugs_with_targets and drug_b in drugs_with_targets:
                effects_by_pair[(drug_a, drug_b)].add(effect_code)

    def most_pairs_first(item: tuple[str, int]) -> int:
        return -item[1]

    codes_by_popularity = [
        code for code, _ in sorted(graph.pair_count_by_effect_code.items(), key=most_pairs_first)
    ]
    random_generator = random.Random(seed)
    hidden_pairs = random_generator.sample(
        sorted(effects_by_pair), min(hidden_pair_count, len(effects_by_pair))
    )
    limit = graph.tier3_candidate_effects_per_pair

    def candidate_codes_without_pair(drug_a: str, drug_b: str, hidden_effects: set[str]) -> list[str]:
        counts_a = effect_counts_by_drug[drug_a].copy()
        counts_b = effect_counts_by_drug[drug_b].copy()
        for effect_code in hidden_effects:  # remove the hidden pair's own edges
            counts_a[effect_code] -= 1
            counts_b[effect_code] -= 1
        counts_a = {code: count for code, count in counts_a.items() if count > 0}
        counts_b = {code: count for code, count in counts_b.items() if count > 0}
        sort_keys = []
        for effect_code in set(counts_a) | set(counts_b):
            seen_with_both_drugs = effect_code in counts_a and effect_code in counts_b
            total_partners = counts_a.get(effect_code, 0) + counts_b.get(effect_code, 0)
            sort_keys.append(
                (not seen_with_both_drugs, -total_partners, effect_name_by_code[effect_code].casefold(), effect_code)
            )
        sort_keys.sort()
        return [effect_code for *_, effect_code in sort_keys[:limit]]

    all_effect_codes = list(effect_name_by_code)
    values_by_method_and_metric = collections.defaultdict(list)
    rows = []
    for pair in hidden_pairs:
        true_effects = effects_by_pair[pair]
        rankings = {
            "graph_candidates": candidate_codes_without_pair(pair[0], pair[1], true_effects),
            "global_popularity": codes_by_popularity[:limit],
            "random": random_generator.sample(all_effect_codes, limit),
        }
        row = {"drug_a": pair[0], "drug_b": pair[1], "true_effects": len(true_effects)}
        for method, ranking in rankings.items():
            hit_in_top_10 = any(code in true_effects for code in ranking[:10])
            precision_at_10 = sum(code in true_effects for code in ranking[:10]) / 10
            recall_at_20 = sum(code in true_effects for code in ranking[:20]) / len(true_effects)
            values_by_method_and_metric[(method, "hit@10")].append(hit_in_top_10)
            values_by_method_and_metric[(method, "precision@10")].append(precision_at_10)
            values_by_method_and_metric[(method, "recall@20")].append(recall_at_20)
            row[f"{method}_precision@10"] = round(precision_at_10, 3)
            row[f"{method}_recall@20"] = round(recall_at_20, 3)
        rows.append(row)
    write_result_csv("e4_tier3_holdout.csv", rows)

    summary = []
    for method in METHODS:
        line = {"method": method, "pairs": len(hidden_pairs)}
        for metric in METRICS:
            values = values_by_method_and_metric[(method, metric)]
            line[metric] = round(statistics.mean(values), 4)
            line[f"{metric}_std"] = round(statistics.pstdev(values), 4)
        summary.append(line)
        print(
            f"  {method:18} hit@10={line['hit@10']:.3f} precision@10={line['precision@10']:.3f} "
            f"recall@20={line['recall@20']:.3f}"
        )
    write_result_csv("e4_summary.csv", summary)
