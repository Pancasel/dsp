"""Offline experiments for the report (no LLM/API calls).

E1  name resolution accuracy on eval/resolver_cases.csv
E4  Tier 3 candidate labels vs held-out documented pairs (Hit@10, Recall@20)
E5  hub-degree threshold ablation: evidence size and retrieval latency
E6  retrieval latency for regimens of k = 3, 5, 8 drugs

Usage:
    python eval/run_offline.py [--seed 0] [--holdout 2000] [--regimens 30]
Results are written to eval/results/*.csv and summarised on stdout.
"""

import argparse
import collections
import csv
import itertools
import math
import random
import sqlite3
import statistics
import time

from common import RESULTS, load


def write_csv(name: str, rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


def e1_resolver(resolver, names) -> None:
    print("E1 name resolution")
    rows = []
    with (RESULTS.parent / "resolver_cases.csv").open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    for case in cases:
        result = resolver.resolve(case["query"])
        got = result.name or ""
        expected = case["expected_name"]
        if expected:
            outcome = "correct" if got == expected else ("asked" if result.stitch_id is None else "wrong")
        else:
            outcome = "correct" if result.stitch_id is None else "wrong"
        rows.append(
            {
                "query": case["query"],
                "category": case["category"],
                "expected": expected,
                "got": got,
                "method": result.method,
                "score": result.score,
                "outcome": outcome,
            }
        )
    write_csv("e1_resolver.csv", rows)
    by_category = collections.defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row["outcome"])
    for category, outcomes in by_category.items():
        print(
            f"  {category:14} n={len(outcomes):3} correct={outcomes.count('correct') / len(outcomes):.0%} "
            f"asked={outcomes.count('asked')} wrong={outcomes.count('wrong')}"
        )
    total = [row["outcome"] for row in rows]
    print(f"  overall accuracy {total.count('correct') / len(total):.1%}, wrong {total.count('wrong')}")


def e4_tier3_holdout(graph, seed: int, sample: int) -> None:
    """Hide documented pairs and test whether candidate labels recover them.

    For a held-out pair (A, B) the candidate ranking is rebuilt from all other
    pairs of A and of B, exactly as DecagonGraph.tier3_candidates does, but with
    the held-out pair's own effects removed (no leakage).
    """
    print("E4 Tier 3 candidate labels on held-out pairs")
    targeted = set(graph.targets)
    per_drug: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    per_pair: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    names: dict[str, str] = {}
    with sqlite3.connect(graph.combo_db) as connection:
        for drug_a, drug_b, code, name in connection.execute(
            "SELECT drug_a, drug_b, effect_code, effect_name FROM side_effects"
        ):
            per_drug[drug_a][code] += 1
            per_drug[drug_b][code] += 1
            names[code] = name
            if drug_a in targeted and drug_b in targeted:
                per_pair[(drug_a, drug_b)].add(code)
    popularity = [code for code, _ in sorted(graph.effect_pair_count.items(), key=lambda item: -item[1])]
    rng = random.Random(seed)
    held = rng.sample(sorted(per_pair), min(sample, len(per_pair)))
    limit = graph.tier3_candidates_per_pair

    def candidates(drug_a: str, drug_b: str, truth: set[str]) -> list[str]:
        counts_a = per_drug[drug_a].copy()
        counts_b = per_drug[drug_b].copy()
        for code in truth:  # remove the held-out pair's own edges
            counts_a[code] -= 1
            counts_b[code] -= 1
        counts_a = {code: count for code, count in counts_a.items() if count > 0}
        counts_b = {code: count for code, count in counts_b.items() if count > 0}
        scored = []
        for code in set(counts_a) | set(counts_b):
            in_both = code in counts_a and code in counts_b
            total = counts_a.get(code, 0) + counts_b.get(code, 0)
            scored.append((not in_both, -total, names[code].casefold(), code))
        scored.sort()
        return [item[3] for item in scored[:limit]]

    all_codes = list(names)
    metrics = collections.defaultdict(list)
    rows = []
    for pair in held:
        truth = per_pair[pair]
        rankings = {
            "graph_candidates": candidates(pair[0], pair[1], truth),
            "global_popularity": popularity[:limit],
            "random": rng.sample(all_codes, limit),
        }
        row = {"drug_a": pair[0], "drug_b": pair[1], "true_effects": len(truth)}
        for method, ranking in rankings.items():
            hit10 = any(code in truth for code in ranking[:10])
            precision10 = sum(code in truth for code in ranking[:10]) / 10
            recall20 = sum(code in truth for code in ranking[:20]) / len(truth)
            metrics[(method, "hit@10")].append(hit10)
            metrics[(method, "precision@10")].append(precision10)
            metrics[(method, "recall@20")].append(recall20)
            row[f"{method}_precision@10"] = round(precision10, 3)
            row[f"{method}_recall@20"] = round(recall20, 3)
        rows.append(row)
    write_csv("e4_tier3_holdout.csv", rows)
    summary = []
    for method in ("graph_candidates", "global_popularity", "random"):
        line = {"method": method, "pairs": len(held)}
        for metric in ("hit@10", "precision@10", "recall@20"):
            values = metrics[(method, metric)]
            line[metric] = round(statistics.mean(values), 4)
            line[f"{metric}_std"] = round(statistics.pstdev(values), 4)
        summary.append(line)
        print(
            f"  {method:18} hit@10={line['hit@10']:.3f} precision@10={line['precision@10']:.3f} "
            f"recall@20={line['recall@20']:.3f}"
        )
    write_csv("e4_summary.csv", summary)


def random_regimens(graph, k: int, count: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed * 1000 + k)
    pool = sorted(graph.drug_ids)
    return [rng.sample(pool, k) for _ in range(count)]


def e5_hub_threshold(names, seed: int, count: int) -> None:
    print("E5 hub threshold ablation (k=4)")
    rows = []
    for threshold in (100, 250, 500, 1000, 100000):
        _, graph, _, _ = load(hub_degree_threshold=threshold)
        regimens = random_regimens(graph, 4, count, seed)
        tier3, pruned, latency = [], [], []
        for regimen in regimens:
            start = time.perf_counter()
            context, evidence = graph.retrieve(regimen, names.is_serious)
            latency.append(time.perf_counter() - start)
            tier3.append(sum(item.tier == 3 for item in evidence))
            pruned.append(len(context["pruned_hubs"]))
        row = {
            "hub_degree_threshold": threshold if threshold < 100000 else "none",
            "regimens": len(regimens),
            "tier3_evidence_mean": round(statistics.mean(tier3), 2),
            "pruned_hubs_mean": round(statistics.mean(pruned), 2),
            "retrieve_seconds_mean": round(statistics.mean(latency), 4),
        }
        rows.append(row)
        print(f"  {row}")
    write_csv("e5_hub_threshold.csv", rows)


def e6_latency(graph, names, seed: int, count: int) -> None:
    print("E6 retrieval latency by regimen size")
    rows = []
    for k in (3, 5, 8):
        latency, evidence_sizes, effects = [], [], []
        for regimen in random_regimens(graph, k, count, seed):
            start = time.perf_counter()
            context, evidence = graph.retrieve(regimen, names.is_serious)
            latency.append(time.perf_counter() - start)
            evidence_sizes.append(len(evidence))
            effects.append(len(context["ranked_effects"]))
        row = {
            "k": k,
            "pairs": math.comb(k, 2),
            "regimens": count,
            "retrieve_seconds_mean": round(statistics.mean(latency), 4),
            "retrieve_seconds_p95": round(sorted(latency)[int(0.95 * (len(latency) - 1))], 4),
            "evidence_mean": round(statistics.mean(evidence_sizes), 1),
            "documented_effects_mean": round(statistics.mean(effects), 1),
        }
        rows.append(row)
        print(f"  {row}")
    write_csv("e6_latency.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", type=int, default=2000)
    parser.add_argument("--regimens", type=int, default=30)
    parser.add_argument("--only", nargs="*", choices=["e1", "e4", "e5", "e6"])
    args = parser.parse_args()
    selected = set(args.only or ["e1", "e4", "e5", "e6"])
    _, graph, names, resolver = load()
    if "e1" in selected:
        e1_resolver(resolver, names)
    if "e4" in selected:
        e4_tier3_holdout(graph, args.seed, args.holdout)
    if "e5" in selected:
        e5_hub_threshold(names, args.seed, args.regimens)
    if "e6" in selected:
        e6_latency(graph, names, args.seed, args.regimens)


if __name__ == "__main__":
    main()
