"""E3: how many LLM predictions survive the evidence validator.

THIS SCRIPT CALLS THE CONFIGURED LLM (LLM_MODEL in .env) ONCE PER REGIMEN and
uses API credit. It refuses to run without --yes-call-api.

Usage:
    python eval/e3_llm_groundedness.py --yes-call-api [--regimens 10] [--seed 0]
"""

import argparse

import csv
import random
import statistics
import time

from common import RESULTS, load

from graphrag.llm import generate_predictions, validate_predictions
from graphrag.presenter import build_llm_context


def run(args: argparse.Namespace) -> None:
    settings, graph, names, _ = load()
    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY is not configured")
    rng = random.Random(args.seed)
    pool = sorted(graph.targets)  # drugs with targets can produce Tier 2/3 evidence
    regimens = [rng.sample(pool, rng.choice((3, 4, 5))) for _ in range(args.regimens)]
    print(f"model={settings.llm_model}, {len(regimens)} calls")
    rows = []
    for index, regimen in enumerate(regimens, start=1):
        context, evidence = graph.retrieve(regimen, names.is_serious, settings.ranking, names.is_critical)
        llm_context = build_llm_context(
            regimen, context, evidence, names,
            settings.max_serious_effects, settings.max_other_effects,
        )
        start = time.perf_counter()
        try:
            raw = generate_predictions(settings, regimen, llm_context)
            error = ""
        except Exception as exc:  # report every failure kind in the table
            raw, error = [], f"{exc.__class__.__name__}: {exc}"
        latency = time.perf_counter() - start
        valid = validate_predictions(raw, evidence, graph) if raw else []
        known_ids = {item.evidence_id for item in evidence}
        invented = sum(
            1 for prediction in raw for evidence_id in prediction.evidence_ids if evidence_id not in known_ids
        )
        rows.append(
            {
                "regimen": "|".join(regimen),
                "k": len(regimen),
                "evidence": len(llm_context["evidence"]),
                "raw_predictions": len(raw),
                "valid_predictions": len(valid),
                "tier3_valid": sum(item["tier"] == 3 for item in valid),
                "invented_ids": invented,
                "latency_s": round(latency, 2),
                "error": error,
            }
        )
        print(f"  {index}/{len(regimens)} {rows[-1]}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "e3_llm_groundedness.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw_total = sum(row["raw_predictions"] for row in rows)
    valid_total = sum(row["valid_predictions"] for row in rows)
    print(
        f"kept {valid_total}/{raw_total} predictions "
        f"({valid_total / raw_total:.1%})" if raw_total else "no predictions returned"
    )
    print(f"median latency {statistics.median(row['latency_s'] for row in rows)} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yes-call-api", action="store_true")
    args = parser.parse_args()
    if not args.yes_call_api:
        raise SystemExit("Refusing to call the LLM API without --yes-call-api")
    run(args)


if __name__ == "__main__":
    main()
