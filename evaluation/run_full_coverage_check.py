"""Coverage of ALL drugs in the Decagon data (offline, no LLM/API calls).

C1  the SQLite cache and the loaded graph equal the raw CSV files, for every drug
C2  every drug has a readable name that resolves back to it (alone and in a chat sentence)
C3  regimens that together contain every drug run through PolypharmacySystem.predict()

Usage (from the project root):
    python -m evaluation.run_full_coverage_check [--seed 0] [--rounds 3]
Writes evaluation/results/c_coverage.csv and evaluation/results/COVERAGE.md.
"""

import argparse

from graphrag.pipeline.polypharmacy_system import PolypharmacySystem
from evaluation.experiments.c1_cache_matches_csv import run_cache_check
from evaluation.experiments.c2_all_drug_names import run_drug_name_check
from evaluation.experiments.c3_end_to_end_all_drugs import run_end_to_end_check
from evaluation.experiments.coverage_raw_source_data import read_raw_source_data
from evaluation.reports.coverage_report import write_coverage_report
from evaluation.shared.system_loading import RESULTS_FOLDER, disable_llm_calls


FAILURES_PRINTED = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3, help="C3: times each drug is analysed")
    arguments = parser.parse_args()
    disable_llm_calls()
    system = PolypharmacySystem.load()
    assert not system.settings.llm_api_key, "LLM must stay disabled"
    print("Reading raw CSV files ...")
    raw = read_raw_source_data(system.settings)
    rows = run_cache_check(system.graph, raw)
    name_rows, failures = run_drug_name_check(system)
    rows += name_rows
    end_to_end_rows, end_to_end_failures, run_details = run_end_to_end_check(
        system, raw, arguments.rounds, arguments.seed
    )
    rows += end_to_end_rows
    failures += end_to_end_failures
    for row in rows:
        print(f"  {row['check']}: {row['ok']}/{row['total']}")
    for line in failures[:FAILURES_PRINTED]:
        print("  !", line)
    write_coverage_report(rows, failures, run_details)
    print(f"  -> {RESULTS_FOLDER / 'COVERAGE.md'}")


if __name__ == "__main__":
    main()
