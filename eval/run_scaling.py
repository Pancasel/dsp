"""A7: does the known-interaction signal survive larger regimens?

Decagon has no ground truth for regimens of 3+ drugs, so this reuses the 26
known pairwise interactions of clinical_benchmark.csv and adds k extra
"bystander" drugs (drawn from the other benchmark cases, seeded, never from the
case itself). The expected effect is still the one from the known pair; the
question is whether it stays in the top of the serious list as the regimen
grows. Bystanders can add their own bleeding/renal/... pairs, which is exactly
what makes larger regimens harder.

    python eval/run_scaling.py
"""

import csv
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS, load  # noqa: E402
from run_accuracy import EXPECTED_EFFECTS, write_csv  # noqa: E402

SIZES = (3, 4, 5, 6, 8)
SEEDS = range(5)
TOP = 10


def main() -> None:
    settings, graph, names, resolver = load()
    with (RESULTS.parent / "clinical_benchmark.csv").open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    pool = sorted({name for case in cases for name in case["drugs"].split("|")})
    resolved = {name: resolver.resolve(name).stitch_id for name in pool}

    rows = []
    for size in SIZES:
        ranks, chance = [], []
        for case in cases:
            members = case["drugs"].split("|")
            base = [resolved[name] for name in members]
            pair = tuple(sorted(resolved[name] for name in case["interacting_pair"].split("|")))
            pattern = re.compile(EXPECTED_EFFECTS[case["expected"]], re.IGNORECASE)
            if None in base or None in pair:
                continue
            for seed in (SEEDS if size > 3 else [0]):
                rng = random.Random(f"{case['case']}-{size}-{seed}")
                extra_pool = [resolved[n] for n in pool if n not in members and resolved[n]]
                drugs = base + rng.sample(extra_pool, size - len(base))
                context, _ = graph.retrieve(drugs, names.is_serious, settings.ranking, names.is_critical)
                serious = [i for i in context["ranked_effects"] if i["serious"]]
                rank = next((n for n, i in enumerate(serious, 1) if pattern.search(i["name"])), None)
                ranks.append(rank)
                chance.append(min(1.0, TOP / len(serious)) if rank else 0.0)
                rows.append({"size": size, "case": case["case"], "seed": seed, "rank": rank or "",
                             "drugs": ";".join(drugs)})
        found = [r for r in ranks if r]
        print(f"{size} drugs: runs={len(ranks)} hit@3={sum(1 for r in ranks if r and r <= 3) / len(ranks):.3f} "
              f"hit@10={sum(1 for r in ranks if r and r <= TOP) / len(ranks):.3f} "
              f"chance@10={statistics.mean(chance):.3f} "
              f"MRR={statistics.mean(1 / r if r else 0 for r in ranks):.3f} "
              f"median_rank={statistics.median(found) if found else '-'}")
    write_csv("a7_scaling.csv", rows)


if __name__ == "__main__":
    main()
