import numpy as np
import pytest

from evaluation.shared.ranking_metrics import (
    auroc,
    average_precision,
    average_precision_at_top_k,
    roc_curve_points,
)


def test_auroc_perfect_inverted_and_tied() -> None:
    labels = np.array([1, 1, 0, 0])
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 1.0
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 0.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5


def test_auroc_matches_pairwise_definition() -> None:
    random_generator = np.random.default_rng(1)
    scores = random_generator.integers(0, 5, 60).astype(float)  # many ties
    labels = random_generator.integers(0, 2, 60)
    positive_scores, negative_scores = scores[labels == 1], scores[labels == 0]
    expected = np.mean([
        (positive > negative) + 0.5 * (positive == negative)
        for positive in positive_scores
        for negative in negative_scores
    ])
    assert auroc(scores, labels) == pytest.approx(expected)


def test_average_precision_and_average_precision_at_top_k() -> None:
    random_generator = np.random.default_rng(0)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1, 0, 1, 0])
    assert average_precision(scores, labels, random_generator) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision_at_top_k(scores, labels, random_generator, k=1) == 1.0
    assert average_precision_at_top_k(scores, labels, random_generator, k=50) == pytest.approx((1 + 2 / 3) / 2)


def test_roc_points_end_at_one() -> None:
    false_positive_rates, true_positive_rates = roc_curve_points(np.array([0.9, 0.4, 0.4, 0.1]), np.array([1, 0, 1, 0]))
    assert (false_positive_rates[0], true_positive_rates[0]) == (0, 0)
    assert (false_positive_rates[-1], true_positive_rates[-1]) == (1, 1)
