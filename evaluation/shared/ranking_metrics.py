"""Ranking metrics: AUROC, AUPRC (average precision), AP@K and ROC curve points.

All functions take one score and one 0/1 label per item; a higher score means
"more likely positive". See docs/THUAT_NGU.md for what each metric measures.
"""

import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney statistic, tied scores share their average rank)."""
    positive_count = labels.sum()
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    _, position_of_score, tie_counts = np.unique(scores, return_inverse=True, return_counts=True)
    last_rank_of_tie = np.cumsum(tie_counts)
    first_rank_of_tie = last_rank_of_tie - tie_counts + 1
    average_rank = ((first_rank_of_tie + last_rank_of_tie) / 2.0)[position_of_score]
    positive_rank_sum = average_rank[labels == 1].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def labels_sorted_by_score(
    scores: np.ndarray, labels: np.ndarray, random_generator: np.random.Generator
) -> np.ndarray:
    """Labels sorted by descending score; tied scores are put in random order."""
    order = np.lexsort((random_generator.random(len(scores)), -scores))
    return labels[order]


def average_precision(
    scores: np.ndarray, labels: np.ndarray, random_generator: np.random.Generator
) -> float:
    """Mean of the precision at the rank of every positive (area under the precision-recall curve)."""
    sorted_labels = labels_sorted_by_score(scores, labels, random_generator)
    hits_so_far = np.cumsum(sorted_labels)
    positive_count = sorted_labels.sum()
    if positive_count == 0:
        return float("nan")
    precision_at_rank = hits_so_far / np.arange(1, len(sorted_labels) + 1)
    return float((precision_at_rank * sorted_labels).sum() / positive_count)


def average_precision_at_top_k(
    scores: np.ndarray, labels: np.ndarray, random_generator: np.random.Generator, k: int = 50
) -> float:
    """AP@k as in Decagon's evaluation (ml_metrics.apk over the ranked edges)."""
    top_labels = labels_sorted_by_score(scores, labels, random_generator)[:k]
    positive_count = labels.sum()
    if positive_count == 0:
        return float("nan")
    precision_at_rank = np.cumsum(top_labels) / np.arange(1, len(top_labels) + 1)
    return float((precision_at_rank * top_labels).sum() / min(positive_count, k))


def roc_curve_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(false positive rates, true positive rates), one point per distinct threshold, from (0, 0)."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    is_last_of_threshold = np.r_[np.diff(sorted_scores) != 0, True]
    true_positives = np.cumsum(sorted_labels)[is_last_of_threshold]
    false_positives = np.cumsum(1 - sorted_labels)[is_last_of_threshold]
    false_positive_rates = np.r_[0, false_positives / false_positives[-1]]
    true_positive_rates = np.r_[0, true_positives / true_positives[-1]]
    return false_positive_rates, true_positive_rates
