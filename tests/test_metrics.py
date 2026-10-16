import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from run_accuracy import ap_at_k, auroc, average_precision, roc_points  # noqa: E402


def test_auroc_perfect_inverted_and_tied() -> None:
    labels = np.array([1, 1, 0, 0])
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 1.0
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 0.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5


def test_auroc_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(1)
    scores = rng.integers(0, 5, 60).astype(float)  # many ties
    labels = rng.integers(0, 2, 60)
    pos, neg = scores[labels == 1], scores[labels == 0]
    expected = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert auroc(scores, labels) == pytest.approx(expected)


def test_average_precision_and_ap_at_k() -> None:
    rng = np.random.default_rng(0)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1, 0, 1, 0])
    assert average_precision(scores, labels, rng) == pytest.approx((1 + 2 / 3) / 2)
    assert ap_at_k(scores, labels, rng, k=1) == 1.0
    assert ap_at_k(scores, labels, rng, k=50) == pytest.approx((1 + 2 / 3) / 2)


def test_roc_points_end_at_one() -> None:
    fpr, tpr = roc_points(np.array([0.9, 0.4, 0.4, 0.1]), np.array([1, 0, 1, 0]))
    assert (fpr[0], tpr[0]) == (0, 0)
    assert (fpr[-1], tpr[-1]) == (1, 1)
