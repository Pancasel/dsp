"""A8: known interactions inside hand-written 4-8 drug regimens.

eval/clinical_benchmark_multi.csv holds 20 realistic regimens of 4-8 drugs, each
containing one well-known interacting pair. Same scoring as A5 (default
weights, rank of the expected effect in the serious list; the UI shows 10).
Cases whose pair has no record of the expected effect in Decagon are listed but
left out of the hit rates, as in A5. The cases are written from standard
pharmacology knowledge and have not been reviewed by a clinician.

    python eval/run_multi.py
"""

import csv
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULTS, load  # noqa: E402
from run_accuracy import EXPECTED_EFFECTS, write_csv  # noqa: E402

TOP = 10


def main() -> None:
    settings, graph, names, resolver = load()
    with (RESULTS.parent / "clinical_benchmark_multi.csv").open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    rows, scored = [], []
    for case in cases:
        drugs = [resolver.resolve(n).stitch_id for n in case["drugs"].split("|")]
        pair = tuple(sorted(resolver.resolve(n).stitch_id for n in case["interacting_pair"].split("|")))
        assert None not in drugs and None not in pair, case["case"]
        pattern = re.compile(EXPECTED_EFFECTS[case["expected"]], re.IGNORECASE)
        documented = any(pattern.search(e.name) for e in graph.side_effects(*pair))
        context, _ = graph.retrieve(drugs, names.is_serious, settings.ranking, names.is_critical)
        serious = [i for i in context["ranked_effects"] if i["serious"]]
        rank = next((n for n, i in enumerate(serious, 1) if pattern.search(i["name"])), None)
        rows.append({"case": case["case"], "drugs_n": len(drugs), "expected": case["expected"],
                     "pair_documented": int(documented), "serious_effects": len(serious),
                     "rank": rank or ""})
        if documented:
            scored.append((len(drugs), rank, min(1.0, TOP / len(serious)) if rank else 0.0))
    write_csv("a8_multi_drug.csv", rows)

    def summarise(label, items):
        ranks = [r for _, r, _ in items]
        print(f"{label}: cases={len(items)} hit@3={sum(1 for r in ranks if r and r <= 3) / len(items):.3f} "
              f"hit@10={sum(1 for r in ranks if r and r <= TOP) / len(items):.3f} "
              f"chance@10={statistics.mean(c for _, _, c in items):.3f} "
              f"MRR={statistics.mean(1 / r if r else 0 for r in ranks):.3f} "
              f"ranks={sorted(r for r in ranks if r)}")

    print(f"{len(cases)} cases, {len(scored)} with the expected effect documented for the pair")
    for lo, hi in ((4, 5), (6, 8), (4, 8)):
        items = [x for x in scored if lo <= x[0] <= hi]
        summarise(f"{lo}-{hi} drugs" if lo != hi else f"{lo} drugs", items)
    for r in rows:
        print(" ", r)


if __name__ == "__main__":
    main()
