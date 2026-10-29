"""A2: can the protein graph alone tell which drug pairs have documented side effects?

This tests the premise of Tier 3. Over all pairs of drugs with targets, each
feature (shared target, one-hop bridge, two-hop bridge, ...) is used as a
score; the label is "the pair has data in bio-decagon-combo". Reported:
AUROC, AUPRC and the lift of "has a bridge", next to a popularity baseline.
"""

import collections
import sqlite3

import numpy as np

from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from evaluation.shared.figure_drawing import METHOD_COLORS, import_pyplot_with_report_style, save_figure
from evaluation.shared.ranking_metrics import auroc, average_precision, roc_curve_points
from evaluation.shared.result_csv_files import write_result_csv


RANDOM_FEATURE = "Ngẫu nhiên"
POPULARITY_FEATURE = "Baseline: độ phổ biến thuốc"
COMBINED_BRIDGE_FEATURE = "Tổng hợp cầu protein (Tier 3)"
SHARED_TARGET_FEATURE = "Protein đích chung (Tier 2)"


def run_pair_prediction_experiment(graph: DecagonGraph, seed: int) -> None:
    print("A2 pair-level prediction from the protein graph (drugs with targets)")
    drugs = sorted(graph.target_proteins_by_drug)
    all_proteins = set(graph.protein_interactions).union(*graph.target_proteins_by_drug.values())
    proteins = sorted(all_proteins)
    protein_column = {protein: index for index, protein in enumerate(proteins)}
    drug_count, protein_count = len(drugs), len(proteins)
    is_hub = np.array(
        [len(graph.protein_interactions.get(protein, ())) > graph.hub_degree_threshold for protein in proteins]
    )

    # targets[d, p] = 1 if drug d targets protein p.
    targets = np.zeros((drug_count, protein_count), dtype=np.float32)
    for row, drug in enumerate(drugs):
        targets[row, [protein_column[gene] for gene in graph.target_proteins_by_drug[drug]]] = 1
    partner_columns_by_column = collections.defaultdict(list)
    for protein, partners in graph.protein_interactions.items():
        for partner in partners:
            partner_columns_by_column[protein_column[protein]].append(protein_column[partner])
    # partners_of_targets[d, p] = 1 if protein p interacts with a target of drug d.
    partners_of_targets = np.zeros((drug_count, protein_count), dtype=np.float32)
    target_rows, target_columns = np.nonzero(targets)
    for drug_row, protein_col in zip(target_rows, target_columns):
        partners_of_targets[drug_row, partner_columns_by_column.get(protein_col, [])] = 1
    non_hub_partners_of_targets = partners_of_targets * (~is_hub)

    shared_target_count = targets @ targets.T
    one_hop_count = targets @ partners_of_targets.T
    one_hop_count = one_hop_count + one_hop_count.T  # target of A next to a target of B, and vice versa
    two_hop_count = non_hub_partners_of_targets @ partners_of_targets.T  # non-hub protein next to both
    upper_triangle = np.triu_indices(drug_count, k=1)

    row_of_drug = {drug: index for index, drug in enumerate(drugs)}
    documented = np.zeros((drug_count, drug_count), dtype=np.int8)
    partner_count_by_drug = collections.Counter()
    with sqlite3.connect(graph.cache_database_file) as connection:
        for drug_a, drug_b in connection.execute(
            "SELECT DISTINCT drug_a, drug_b FROM drug_pair_side_effects"
        ):
            partner_count_by_drug[drug_a] += 1
            partner_count_by_drug[drug_b] += 1
            if drug_a in row_of_drug and drug_b in row_of_drug:
                documented[row_of_drug[drug_a], row_of_drug[drug_b]] = 1
                documented[row_of_drug[drug_b], row_of_drug[drug_a]] = 1
    labels = documented[upper_triangle].astype(np.int64)
    # Popularity baseline: product of each drug's number of documented partners,
    # excluding the scored pair itself (no leakage).
    partner_counts = np.array([partner_count_by_drug[drug] for drug in drugs], dtype=np.float64)
    popularity = (partner_counts[:, None] - documented) * (partner_counts[None, :] - documented)

    features = {
        SHARED_TARGET_FEATURE: shared_target_count[upper_triangle],
        "Cầu 1 bước (đích A ↔ đích B)": one_hop_count[upper_triangle],
        "Cầu 2 bước (qua protein trung gian)": two_hop_count[upper_triangle],
        COMBINED_BRIDGE_FEATURE: (
            np.log1p(shared_target_count[upper_triangle]) * 4
            + np.log1p(one_hop_count[upper_triangle]) * 2
            + np.log1p(two_hop_count[upper_triangle])
        ),
        POPULARITY_FEATURE: popularity[upper_triangle],
    }
    random_generator = np.random.default_rng(seed)
    features[RANDOM_FEATURE] = random_generator.random(len(labels))
    base_rate = labels.mean()

    rows = []
    for feature_name, scores in features.items():
        scores = np.asarray(scores, dtype=np.float64)
        row = {
            "feature": feature_name,
            "pairs": len(labels),
            "positives": int(labels.sum()),
            "auroc": round(auroc(scores, labels), 4),
            "auprc": round(average_precision(scores, labels, random_generator), 4),
        }
        if feature_name != RANDOM_FEATURE and not feature_name.startswith("Baseline"):
            has_feature = scores > 0
            row["coverage"] = round(has_feature.mean(), 4)
            row["precision_if_present"] = round(labels[has_feature].mean(), 4) if has_feature.any() else ""
            row["precision_if_absent"] = round(labels[~has_feature].mean(), 4) if (~has_feature).any() else ""
            row["lift"] = round(labels[has_feature].mean() / base_rate, 3) if has_feature.any() else ""
        else:
            row.update(coverage="", precision_if_present="", precision_if_absent="", lift="")
        rows.append(row)
        print(f"  {feature_name}: AUROC={row['auroc']} AUPRC={row['auprc']} lift={row['lift']}")
    print(f"  base rate (documented pairs) = {base_rate:.4f}")
    write_result_csv("a2_pair_prediction.csv", rows)
    _draw_roc_figure(features, labels, rows)


def _draw_roc_figure(features: dict, labels: np.ndarray, rows: list[dict]) -> None:
    pyplot = import_pyplot_with_report_style()
    figure, axes = pyplot.subplots(figsize=(5.4, 4.6))
    line_styles = {
        COMBINED_BRIDGE_FEATURE: (METHOD_COLORS["graph"], "-"),
        SHARED_TARGET_FEATURE: (METHOD_COLORS["extra"], "-"),
        POPULARITY_FEATURE: (METHOD_COLORS["popularity"], "-"),
        RANDOM_FEATURE: (METHOD_COLORS["random"], "--"),
    }
    for feature_name, (color, style) in line_styles.items():
        false_positive_rates, true_positive_rates = roc_curve_points(
            np.asarray(features[feature_name], dtype=np.float64), labels
        )
        feature_auroc = next(row["auroc"] for row in rows if row["feature"] == feature_name)
        axes.plot(
            false_positive_rates, true_positive_rates, style, color=color, linewidth=2,
            label=f"{feature_name} (AUROC {feature_auroc:.3f})",
        )
    axes.set_xlabel("Tỷ lệ dương tính giả (FPR)")
    axes.set_ylabel("Tỷ lệ dương tính thật (TPR)")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_title(f"A2. ROC dự đoán cặp có tương tác ({len(labels):,} cặp)", loc="left", fontsize=11)
    axes.legend(frameon=False, fontsize=7.5, loc="lower right")
    save_figure(figure, "a2_roc.png")
