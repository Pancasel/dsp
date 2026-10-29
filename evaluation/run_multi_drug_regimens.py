"""A8: known interactions inside hand-written regimens of 4-8 drugs (see experiments/a8_multi_drug_regimens.py).

Usage (from the project root):
    python -m evaluation.run_multi_drug_regimens
"""

from evaluation.experiments.a8_multi_drug_regimens import run_multi_drug_regimen_experiment
from evaluation.shared.system_loading import load_evaluation_system


if __name__ == "__main__":
    run_multi_drug_regimen_experiment(load_evaluation_system())
