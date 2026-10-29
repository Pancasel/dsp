"""E3: share of LLM predictions that survive the validator.

THIS CALLS THE CONFIGURED LLM (LLM_MODEL in .env) ONCE PER REGIMEN and uses
API credit. It refuses to run without --yes-call-api.

Usage (from the project root):
    python -m evaluation.run_llm_groundedness_experiment --yes-call-api [--regimens 10] [--seed 0]
"""

import argparse

from evaluation.experiments.e3_llm_groundedness import run_llm_groundedness_experiment
from evaluation.shared.system_loading import load_evaluation_system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yes-call-api", action="store_true")
    arguments = parser.parse_args()
    if not arguments.yes_call_api:
        raise SystemExit("Refusing to call the LLM API without --yes-call-api")
    run_llm_groundedness_experiment(load_evaluation_system(), arguments.regimens, arguments.seed)


if __name__ == "__main__":
    main()
