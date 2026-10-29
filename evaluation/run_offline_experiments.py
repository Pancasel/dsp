"""Offline experiments E1, E4, E5, E6 (no LLM/API calls).

E1  name resolution accuracy on test_cases/name_resolution_cases.csv
E4  Tier 3 candidate effects vs hidden documented pairs (Hit@10, Recall@20)
E5  hub-degree threshold: evidence size and retrieval time
E6  retrieval time for regimens of 3, 5 and 8 drugs

Usage (from the project root):
    python -m evaluation.run_offline_experiments [--seed 0] [--holdout 2000] [--regimens 30] [--only e1 e4]
Results are written to evaluation/results/*.csv and summarised on screen.
"""

import argparse

from evaluation.experiments.e1_name_resolution import run_name_resolution_experiment
from evaluation.experiments.e4_tier3_holdout import run_tier3_holdout_experiment
from evaluation.experiments.e5_e6_retrieval_speed_and_hubs import (
    run_hub_threshold_experiment,
    run_retrieval_latency_experiment,
)
from evaluation.shared.system_loading import load_evaluation_system


ALL_EXPERIMENTS = ["e1", "e4", "e5", "e6"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", type=int, default=2000, help="number of hidden pairs for E4")
    parser.add_argument("--regimens", type=int, default=30, help="random regimens per setting for E5/E6")
    parser.add_argument("--only", nargs="*", choices=ALL_EXPERIMENTS)
    arguments = parser.parse_args()
    selected = set(arguments.only or ALL_EXPERIMENTS)
    system = load_evaluation_system()
    if "e1" in selected:
        run_name_resolution_experiment(system.resolver)
    if "e4" in selected:
        run_tier3_holdout_experiment(system.graph, arguments.seed, arguments.holdout)
    if "e5" in selected:
        run_hub_threshold_experiment(system.names, arguments.seed, arguments.regimens)
    if "e6" in selected:
        run_retrieval_latency_experiment(system.graph, system.names, arguments.seed, arguments.regimens)


if __name__ == "__main__":
    main()
