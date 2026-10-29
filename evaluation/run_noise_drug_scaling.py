"""A7: known interactions when extra "bystander" drugs are added (see experiments/a7_noise_drug_scaling.py).

Usage (from the project root):
    python -m evaluation.run_noise_drug_scaling
"""

from evaluation.experiments.a7_noise_drug_scaling import run_noise_drug_scaling_experiment
from evaluation.shared.system_loading import load_evaluation_system


if __name__ == "__main__":
    run_noise_drug_scaling_experiment(load_evaluation_system())
