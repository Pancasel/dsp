"""A3: per-side-effect link prediction with the Decagon protocol.

For each side effect with enough pairs, 10 % of its drug pairs are hidden as
the test set, and each hidden (positive) pair gets one random negative pair
that never has that effect. Scores are computed from the training pairs only.
Reported: AUROC, AUPRC and AP@50 averaged over side effects, for the Tier 3
label scorer and baselines, next to the numbers published by Decagon.
"""

import collections
import math
import sqlite3
import statistics

import numpy as np

from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from evaluation.shared.figure_drawing import (
    METHOD_COLORS,
    draw_grouped_bars,
    import_pyplot_with_report_style,
    save_figure,
)
from evaluation.shared.ranking_metrics import auroc, average_precision, average_precision_at_top_k
from evaluation.shared.result_csv_files import write_result_csv


# Decagon (Zitnik et al., Bioinformatics 2018), Table 2: mean over 964 side effects.
DECAGON_REPORTED = {"auroc": 0.872, "auprc": 0.832, "ap@50": 0.803}
METHOD_LABELS = {
    "graph": "Đồ thị: bộ chấm nhãn Tier 3 (số lần)",
    "graph_normalized": "Đồ thị: tần suất chuẩn hoá",
    "popularity": "Baseline: độ phổ biến thuốc",
    "random": "Ngẫu nhiên",
}
# Retries when drawing negative pairs that turned out to be positives.
NEGATIVE_SAMPLING_ATTEMPTS = 50
MINIMUM_TEST_PAIRS_PER_EFFECT = 5


def run_per_side_effect_prediction_experiment(
    graph: DecagonGraph, seed: int, min_pairs: int, test_fraction: float
) -> None:
    print(
        f"A3 per-side-effect link prediction (effects with >= {min_pairs} pairs, "
        f"{test_fraction:.0%} held out)"
    )
    drugs = sorted(graph.drug_ids)
    drug_index = {drug: index for index, drug in enumerate(drugs)}
    effect_codes = sorted(code for code, count in graph.pair_count_by_effect_code.items() if count >= min_pairs)
    effect_index = {code: index for index, code in enumerate(effect_codes)}
    drug_a_list, drug_b_list, effect_list = [], [], []
    with sqlite3.connect(graph.cache_database_file) as connection:
        for drug_a, drug_b, effect_code in connection.execute(
            "SELECT drug_a, drug_b, effect_code FROM drug_pair_side_effects"
        ):
            effect = effect_index.get(effect_code)
            if effect is not None:
                drug_a_list.append(drug_index[drug_a])
                drug_b_list.append(drug_index[drug_b])
                effect_list.append(effect)
    edge_drug_a = np.array(drug_a_list, dtype=np.int64)
    edge_drug_b = np.array(drug_b_list, dtype=np.int64)
    edge_effect = np.array(effect_list, dtype=np.int64)
    drug_count, effect_count = len(drugs), len(effect_codes)
    random_generator = np.random.default_rng(seed)
    is_test_edge = random_generator.random(len(edge_effect)) < test_fraction
    train = ~is_test_edge

    # Training statistics only (the hidden edges are removed: no leakage).
    partner_count = np.zeros((drug_count, effect_count), dtype=np.float64)
    np.add.at(partner_count, (edge_drug_a[train], edge_effect[train]), 1)
    np.add.at(partner_count, (edge_drug_b[train], edge_effect[train]), 1)
    train_pair_keys = np.unique(edge_drug_a[train] * drug_count + edge_drug_b[train])
    pair_degree = np.bincount(
        np.r_[train_pair_keys // drug_count, train_pair_keys % drug_count], minlength=drug_count
    ).astype(np.float64)
    edges_per_drug = np.maximum(partner_count.sum(axis=1, keepdims=True), 1)
    effect_share = partner_count / edges_per_drug  # share of a drug's training edges with each effect

    positive_a = edge_drug_a[is_test_edge]
    positive_b = edge_drug_b[is_test_edge]
    positive_effect = edge_effect[is_test_edge]
    negative_a, negative_b = _sample_negative_pairs(
        random_generator, edge_drug_a, edge_drug_b, edge_effect, positive_effect, drug_count, effect_count
    )

    def score(method: str, drugs_a: np.ndarray, drugs_b: np.ndarray, effects: np.ndarray) -> np.ndarray:
        if method == "graph":  # Tier 3 label scorer: partners of A and B with the effect
            seen_with_both = (partner_count[drugs_a, effects] > 0) & (partner_count[drugs_b, effects] > 0)
            return seen_with_both * 1e6 + partner_count[drugs_a, effects] + partner_count[drugs_b, effects]
        if method == "graph_normalized":
            return effect_share[drugs_a, effects] + effect_share[drugs_b, effects]
        if method == "popularity":  # pair degree only, the same for every effect
            return pair_degree[drugs_a] * pair_degree[drugs_b]
        return random_generator.random(len(drugs_a))

    positive_scores = {method: score(method, positive_a, positive_b, positive_effect) for method in METHOD_LABELS}
    negative_scores = {method: score(method, negative_a, negative_b, positive_effect) for method in METHOD_LABELS}
    order_by_effect = np.argsort(positive_effect, kind="stable")
    effect_boundaries = np.searchsorted(positive_effect[order_by_effect], np.arange(effect_count + 1))
    values_by_method = {method: collections.defaultdict(list) for method in METHOD_LABELS}
    rows = []
    for effect in range(effect_count):
        test_positions = order_by_effect[effect_boundaries[effect]:effect_boundaries[effect + 1]]
        if len(test_positions) < MINIMUM_TEST_PAIRS_PER_EFFECT:
            continue
        labels = np.r_[
            np.ones(len(test_positions), dtype=np.int64), np.zeros(len(test_positions), dtype=np.int64)
        ]
        row = {"effect_code": effect_codes[effect], "effect_name": "", "test_positives": len(test_positions)}
        for method in METHOD_LABELS:
            scores = np.r_[positive_scores[method][test_positions], negative_scores[method][test_positions]]
            metric_values = {
                "auroc": auroc(scores, labels),
                "auprc": average_precision(scores, labels, random_generator),
                "ap@50": average_precision_at_top_k(scores, labels, random_generator, 50),
            }
            for metric, value in metric_values.items():
                values_by_method[method][metric].append(value)
                row[f"{_column_prefix(method)}_{metric}"] = round(value, 4)
        rows.append(row)
    with sqlite3.connect(graph.cache_database_file) as connection:
        name_by_code = dict(connection.execute("SELECT effect_code, effect_name FROM effect_pair_counts"))
    for row in rows:
        row["effect_name"] = name_by_code.get(row["effect_code"], "")
    write_result_csv("a3_per_side_effect.csv", rows)

    summary = []
    for method, label in METHOD_LABELS.items():
        line = {"method": label, "side_effects": len(values_by_method[method]["auroc"])}
        for metric in ("auroc", "auprc", "ap@50"):
            values = values_by_method[method][metric]
            line[metric] = round(statistics.mean(values), 4)
            line[f"{metric}_std"] = round(statistics.pstdev(values), 4)
        summary.append(line)
        print(f"  {label}: AUROC={line['auroc']} AUPRC={line['auprc']} AP@50={line['ap@50']}")
    summary.append(
        {"method": "Decagon (bài báo, GCN)", "side_effects": 964, **DECAGON_REPORTED,
         "auroc_std": "", "auprc_std": "", "ap@50_std": ""}
    )
    write_result_csv("a3_summary.csv", summary)
    _draw_figures(summary, values_by_method, len(rows))


def _column_prefix(method: str) -> str:
    """CSV column prefix; "graph_norm" is kept so that existing result files stay comparable."""
    if method == "graph_normalized":
        return "graph_norm"
    return method


def _sample_negative_pairs(
    random_generator, edge_drug_a, edge_drug_b, edge_effect, positive_effect, drug_count, effect_count
) -> tuple[np.ndarray, np.ndarray]:
    """One random pair per positive that is not the same drug twice and never has that effect."""
    low_drug = np.minimum(edge_drug_a, edge_drug_b)
    high_drug = np.maximum(edge_drug_a, edge_drug_b)
    known_keys = np.sort((low_drug * drug_count + high_drug) * effect_count + edge_effect)
    negative_a = random_generator.integers(0, drug_count, len(positive_effect))
    negative_b = random_generator.integers(0, drug_count, len(positive_effect))
    for _ in range(NEGATIVE_SAMPLING_ATTEMPTS):
        low = np.minimum(negative_a, negative_b)
        high = np.maximum(negative_a, negative_b)
        keys = (low * drug_count + high) * effect_count + positive_effect
        insert_positions = np.searchsorted(known_keys, keys)
        safe_positions = np.minimum(insert_positions, len(known_keys) - 1)
        is_known_positive = (insert_positions < len(known_keys)) & (known_keys[safe_positions] == keys)
        must_redraw = (negative_a == negative_b) | is_known_positive
        if not must_redraw.any():
            break
        negative_a[must_redraw] = random_generator.integers(0, drug_count, must_redraw.sum())
        negative_b[must_redraw] = random_generator.integers(0, drug_count, must_redraw.sum())
    return negative_a, negative_b


def _draw_figures(summary: list[dict], values_by_method: dict, effect_count: int) -> None:
    metric_titles = {"auroc": "AUROC", "auprc": "AUPRC", "ap@50": "AP@50"}
    graph_line, popularity_line, random_line = summary[0], summary[2], summary[3]
    values_by_series = {
        "Đồ thị (Tier 3)": {metric_titles[key]: graph_line[key] for key in metric_titles},
        "Baseline độ phổ biến": {metric_titles[key]: popularity_line[key] for key in metric_titles},
        "Decagon (bài báo)": {metric_titles[key]: value for key, value in DECAGON_REPORTED.items()},
        "Ngẫu nhiên": {metric_titles[key]: random_line[key] for key in metric_titles},
    }
    colors = {
        "Đồ thị (Tier 3)": METHOD_COLORS["graph"],
        "Baseline độ phổ biến": METHOD_COLORS["popularity"],
        "Decagon (bài báo)": METHOD_COLORS["decagon"],
        "Ngẫu nhiên": METHOD_COLORS["random"],
    }
    draw_grouped_bars(
        f"A3. Dự đoán tác dụng phụ theo từng loại ({effect_count} tác dụng phụ)",
        list(metric_titles.values()), values_by_series, colors, "a3_side_effects.png",
    )

    pyplot = import_pyplot_with_report_style()
    figure, axes = pyplot.subplots(figsize=(6.4, 3.4))
    graph_aurocs = values_by_method["graph"]["auroc"]
    popularity_aurocs = values_by_method["popularity"]["auroc"]
    lowest = min(min(graph_aurocs), min(popularity_aurocs))
    bins = np.linspace(math.floor(lowest * 50) / 50, 1.0, 41)
    axes.hist(graph_aurocs, bins=bins, color=METHOD_COLORS["graph"], label="Đồ thị (Tier 3)",
              edgecolor="white", linewidth=0.8)
    axes.hist(popularity_aurocs, bins=bins, color=METHOD_COLORS["popularity"], histtype="step",
              linewidth=2, label="Baseline độ phổ biến")
    axes.axvline(DECAGON_REPORTED["auroc"], color=METHOD_COLORS["decagon"], linewidth=2, linestyle="--",
                 label=f"Decagon trung bình ({DECAGON_REPORTED['auroc']})")
    axes.set_xlabel("AUROC của một tác dụng phụ")
    axes.set_ylabel("Số tác dụng phụ")
    axes.set_title("A3. Phân bố AUROC theo từng tác dụng phụ", loc="left", fontsize=11)
    axes.legend(frameon=False, fontsize=8, loc="upper left")
    save_figure(figure, "a3_auroc_hist.png")
