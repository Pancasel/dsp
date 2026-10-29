"""Accuracy experiments for the report (offline, no LLM/API calls), then ACCURACY.md.

E2  drug extraction from chat messages
A1  faithfulness: every returned effect, Tier 2 flag and evidence path re-checked against the raw CSVs
A2  pair-level interaction prediction from the protein graph only (the premise of Tier 3)
A3  per-side-effect link prediction with the Decagon protocol
A4  validator robustness against deliberately corrupted LLM predictions
A5  clinical benchmark of well-known interactions
A6  serious-effect keywords vs the EMA Important Medical Event list

Usage (from the project root):
    python -m evaluation.run_accuracy_experiments [--seed 0] [--regimens 200] [--only a1 a3]
Writes evaluation/results/a*.csv, evaluation/results/figures/*.png and
evaluation/results/ACCURACY.md. Run run_offline_experiments first so that the
report also contains E1 and E4.
"""

import argparse

from evaluation.experiments.a1_faithfulness_to_source_data import run_faithfulness_experiment
from evaluation.experiments.a2_protein_bridge_pair_prediction import run_pair_prediction_experiment
from evaluation.experiments.a3_per_side_effect_prediction import run_per_side_effect_prediction_experiment
from evaluation.experiments.a4_prediction_validator import run_validator_experiment
from evaluation.experiments.a5_clinical_benchmark import run_clinical_benchmark_experiment
from evaluation.experiments.a6_serious_effect_classifier import run_serious_classifier_experiment
from evaluation.experiments.e2_chat_drug_extraction import run_chat_drug_extraction_experiment
from evaluation.reports.accuracy_report import draw_offline_experiment_figures, write_accuracy_report
from evaluation.shared.system_loading import disable_llm_calls, load_evaluation_system


ALL_EXPERIMENTS = ["e2", "a1", "a2", "a3", "a4", "a5", "a6"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regimens", type=int, default=200, help="random regimens for A1 (A4 uses a quarter)")
    parser.add_argument("--min-pairs", type=int, default=500, help="A3: side effects with at least this many pairs")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="A3: share of pairs hidden")
    parser.add_argument("--only", nargs="*", choices=ALL_EXPERIMENTS)
    arguments = parser.parse_args()
    selected = set(arguments.only or ALL_EXPERIMENTS)
    disable_llm_calls()
    system = load_evaluation_system()
    settings, graph, names, resolver = system.settings, system.graph, system.names, system.resolver
    run_details = {
        "seed": arguments.seed,
        "min_pairs": arguments.min_pairs,
        "test_fraction": arguments.test_fraction,
        "a1_regimens": arguments.regimens,
    }
    if "e2" in selected:
        run_chat_drug_extraction_experiment(resolver)
    if "a1" in selected:
        run_faithfulness_experiment(settings, graph, names, arguments.seed, arguments.regimens)
    if "a2" in selected:
        run_pair_prediction_experiment(graph, arguments.seed)
    if "a3" in selected:
        run_per_side_effect_prediction_experiment(
            graph, arguments.seed, arguments.min_pairs, arguments.test_fraction
        )
    if "a4" in selected:
        run_validator_experiment(settings, graph, names, arguments.seed, max(arguments.regimens // 4, 20))
    if "a5" in selected:
        run_clinical_benchmark_experiment(settings, graph, names, resolver)
    if "a6" in selected:
        run_serious_classifier_experiment()
    draw_offline_experiment_figures()
    write_accuracy_report(run_details)


if __name__ == "__main__":
    main()
