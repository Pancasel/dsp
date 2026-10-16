"""Accuracy evaluation for the report (offline, no LLM/API calls).

A1  Faithfulness: every documented effect, Tier 2 flag and Tier 3 path the
    system returns for random regimens is re-checked against the raw Decagon
    CSVs, read independently of the SQLite cache (precision and recall).
A2  Pair-level interaction prediction from the protein graph only (the premise
    of Tier 3): do shared targets / PPI bridges separate documented drug pairs
    from undocumented ones? AUROC, AUPRC, precision and lift of "has a bridge".
A3  Per-side-effect link prediction with the Decagon protocol (10 % of the
    edges of each side effect held out, one random negative pair per positive):
    AUROC, AUPRC and AP@50 averaged over side effects, for the Tier 3 label
    scorer and baselines, next to the numbers reported by Decagon.
A4  Validator robustness: real evidence is turned into valid and deliberately
    corrupted LLM predictions; the validator must keep the former and reject
    every corrupted one.

Usage:
    python eval/run_accuracy.py [--seed 0] [--regimens 200] [--only a1 a3]
Writes eval/results/a*.csv, eval/results/figures/*.png and
eval/results/ACCURACY.md (tables ready for the report).
"""

import argparse
import collections
import csv
import itertools
import math
import os
import random
import re
import sqlite3
import statistics
import time

import numpy as np

os.environ["LLM_API_KEY"] = ""  # never call the LLM from this script

from common import RESULTS, load  # noqa: E402

from graphrag.config import Settings  # noqa: E402
from graphrag.dataset import _rows  # noqa: E402
from graphrag.llm import ModelPrediction, validate_predictions  # noqa: E402

FIGURES = RESULTS / "figures"
# Decagon (Zitnik et al., Bioinformatics 2018), Table 2: mean over 964 side effects.
DECAGON_REPORTED = {"auroc": 0.872, "auprc": 0.832, "ap@50": 0.803}
# Colours of the report figures: one fixed colour per method.
COLORS = {
    "graph": "#2a78d6",
    "popularity": "#eb6834",
    "decagon": "#1baf7a",
    "random": "#a3a29b",
    "extra": "#6250d6",
}
E1_CATEGORIES = {
    "generic": "Tên hoạt chất",
    "brand_vi": "Biệt dược VN",
    "with_strength": "Kèm hàm lượng",
    "typo": "Sai chính tả",
    "case": "Hoa/thường",
    "not_in_data": "Không có trong dữ liệu",
}


# ----------------------------------------------------------------- metrics
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    ranks = ((end - counts + 1 + end) / 2.0)[inverse]
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _ranked_labels(scores: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Labels sorted by descending score; ties are broken at random."""
    order = np.lexsort((rng.random(len(scores)), -scores))
    return labels[order]


def average_precision(scores: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> float:
    ranked = _ranked_labels(scores, labels, rng)
    hits = np.cumsum(ranked)
    total = ranked.sum()
    if total == 0:
        return float("nan")
    precision = hits / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / total)


def ap_at_k(scores: np.ndarray, labels: np.ndarray, rng: np.random.Generator, k: int = 50) -> float:
    """AP@k as in Decagon's evaluation (ml_metrics.apk over the ranked edges)."""
    ranked = _ranked_labels(scores, labels, rng)[:k]
    total = labels.sum()
    if total == 0:
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / min(total, k))


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    # One point per distinct threshold.
    last = np.r_[np.diff(sorted_scores) != 0, True]
    tp = np.cumsum(sorted_labels)[last]
    fp = np.cumsum(1 - sorted_labels)[last]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]


# ------------------------------------------------------------------ output
def write_csv(name: str, rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


def read_csv(name: str) -> list[dict[str, str]]:
    path = RESULTS / name
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#52514e",
            "axes.labelcolor": "#0b0b0b",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "figure.dpi": 150,
        }
    )
    return plt


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    print(f"  -> {FIGURES / name}")


def grouped_bars(title: str, metrics: list[str], series: dict[str, dict[str, float]],
                 colors: dict[str, str], name: str, ylabel: str = "Giá trị") -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.8 / len(series)
    for index, (label, values) in enumerate(series.items()):
        xs = [m + (index - (len(series) - 1) / 2) * width for m in range(len(metrics))]
        heights = [values.get(metric, float("nan")) for metric in metrics]
        bars = ax.bar(xs, heights, width * 0.92, label=label, color=colors[label])
        for bar, height in zip(bars, heights):
            if not math.isnan(height):
                ax.annotate(f"{height:.3f}", (bar.get_x() + bar.get_width() / 2, height),
                            ha="center", va="bottom", fontsize=7.5, color="#52514e",
                            xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncols=min(len(series), 4), loc="upper left",
              bbox_to_anchor=(0, -0.1))
    ax.grid(axis="x", visible=False)
    _save(fig, name)


# --------------------------------------------------------------------- A1
def a1_faithfulness(settings: Settings, graph, names, seed: int, count: int) -> dict:
    print(f"A1 faithfulness vs raw CSV ({count} regimens)")
    rng = random.Random(seed)
    targeted = sorted(graph.targets)
    every = sorted(graph.drug_ids)
    regimens = []
    for index in range(count):
        # Half from drugs with targets, so Tier 2/3 evidence actually appears.
        pool = targeted if index % 2 == 0 else every
        regimens.append(rng.sample(pool, rng.choice((3, 4, 5, 6))))

    results = []
    start = time.perf_counter()
    for regimen in regimens:
        context, evidence = graph.retrieve(regimen, names.is_serious, settings.ranking)
        results.append((regimen, context, evidence))
    retrieve_seconds = time.perf_counter() - start

    # Independent ground truth straight from the CSV files.
    wanted = {pair for regimen, _, _ in results for pair in itertools.combinations(sorted(regimen), 2)}
    raw_effects: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in _rows(settings.combo_file, ("STITCH 1", "STITCH 2", "Polypharmacy Side Effect")):
        pair = tuple(sorted((row["STITCH 1"], row["STITCH 2"])))
        if pair in wanted:
            raw_effects[pair].add(row["Polypharmacy Side Effect"])
    raw_targets: dict[str, set[str]] = collections.defaultdict(set)
    for row in _rows(settings.targets_file, ("STITCH", "Gene")):
        raw_targets[row["STITCH"]].add(row["Gene"])
    raw_ppi: set[tuple[str, str]] = set()
    degree: collections.Counter = collections.Counter()
    for row in _rows(settings.ppi_file, ("Gene 1", "Gene 2")):
        a, b = row["Gene 1"], row["Gene 2"]
        if a != b and (a, b) not in raw_ppi:
            raw_ppi.add((a, b))
            raw_ppi.add((b, a))
            degree[a] += 1
            degree[b] += 1

    tally = collections.Counter()
    for regimen, context, evidence in results:
        pairs = list(itertools.combinations(sorted(regimen), 2))
        returned = {(tuple(pair), item["code"]) for item in context["ranked_effects"] for pair in item["pairs"]}
        truth = {(pair, code) for pair in pairs for code in raw_effects.get(pair, ())}
        tally["t1_returned"] += len(returned)
        tally["t1_supported"] += len(returned & truth)
        tally["t1_truth"] += len(truth)
        tally["t1_recalled"] += len(returned & truth)

        # Tier 2: pair with documented effects + a protein shared by >= 3 regimen drugs.
        owners = collections.Counter(gene for drug in regimen for gene in raw_targets.get(drug, ()))
        shared = {gene for gene, n in owners.items() if n >= 3}
        should_amplify = {
            pair for pair in pairs
            if raw_effects.get(pair) and shared & raw_targets.get(pair[0], set()) & raw_targets.get(pair[1], set())
        }
        flagged = {tuple(pair) for item in context["ranked_effects"] for pair in item["amplified_pairs"]}
        tally["t2_flagged"] += len(flagged)
        tally["t2_flagged_correct"] += len(flagged & should_amplify)
        tally["t2_expected"] += len(should_amplify)
        tally["t2_expected_found"] += len(flagged & should_amplify)

        # Every evidence path, edge by edge.
        for item in evidence:
            tally[f"evidence_tier{item.tier}"] += 1
            ok = True
            for edge in item.path:
                relation = edge["relation"]
                if relation == "targets":
                    ok &= edge["target"] in raw_targets.get(edge["source"], set())
                elif relation == "interacts_with":
                    ok &= (edge["source"], edge["target"]) in raw_ppi
                elif relation == "has_side_effect":
                    pair = tuple(sorted((edge["source"], edge["target"])))
                    ok &= edge["side_effect_code"] in raw_effects.get(pair, set())
                else:
                    ok = False
            if item.tier == 3:
                # Tier 3 is only for undocumented pairs, through non-hub intermediates.
                ok &= not raw_effects.get(item.drug_pair)
                regimen_targets = set().union(*(raw_targets.get(d, set()) for d in regimen))
                for edge in item.path[1:-1]:
                    for protein in (edge["source"], edge["target"]):
                        if protein not in regimen_targets:
                            ok &= degree[protein] <= settings.hub_degree_threshold
            tally[f"evidence_tier{item.tier}_ok"] += ok

    def ratio(a: str, b: str) -> float:
        return tally[a] / tally[b] if tally[b] else float("nan")

    rows = [
        {"check": "Tier 1 – precision: tác dụng phụ trả về có trong combo CSV",
         "numerator": tally["t1_supported"], "denominator": tally["t1_returned"], "value": ratio("t1_supported", "t1_returned")},
        {"check": "Tier 1 – recall: tác dụng phụ trong CSV của các cặp được trả về",
         "numerator": tally["t1_recalled"], "denominator": tally["t1_truth"], "value": ratio("t1_recalled", "t1_truth")},
        {"check": "Tier 2 – precision: cặp gắn Tier 2 thỏa quy tắc protein chung",
         "numerator": tally["t2_flagged_correct"], "denominator": tally["t2_flagged"], "value": ratio("t2_flagged_correct", "t2_flagged")},
        {"check": "Tier 2 – recall: cặp thỏa quy tắc được gắn Tier 2",
         "numerator": tally["t2_expected_found"], "denominator": tally["t2_expected"], "value": ratio("t2_expected_found", "t2_expected")},
    ]
    for tier in (1, 2, 3):
        rows.append({
            "check": f"Tier {tier} – đường dẫn bằng chứng đúng từng cạnh",
            "numerator": tally[f"evidence_tier{tier}_ok"], "denominator": tally[f"evidence_tier{tier}"],
            "value": ratio(f"evidence_tier{tier}_ok", f"evidence_tier{tier}"),
        })
    for row in rows:
        row["value"] = round(row["value"], 4) if not math.isnan(row["value"]) else "n/a"
        print(f"  {row['check']}: {row['numerator']}/{row['denominator']} = {row['value']}")
    write_csv("a1_faithfulness.csv", rows)
    return {"regimens": len(regimens), "retrieve_seconds_mean": retrieve_seconds / len(regimens)}


# --------------------------------------------------------------------- A2
def a2_pair_prediction(graph, seed: int) -> None:
    print("A2 pair-level prediction from the protein graph (drugs with targets)")
    drugs = sorted(graph.targets)
    proteins = sorted(set(graph.ppi) | set().union(*graph.targets.values()))
    column = {protein: index for index, protein in enumerate(proteins)}
    n_drugs, n_proteins = len(drugs), len(proteins)
    hub = np.array([len(graph.ppi.get(p, ())) > graph.hub_degree_threshold for p in proteins])
    targets = np.zeros((n_drugs, n_proteins), dtype=np.float32)
    for row, drug in enumerate(drugs):
        targets[row, [column[g] for g in graph.targets[drug]]] = 1
    adjacency_rows, adjacency_cols = [], []
    for protein, neighbours in graph.ppi.items():
        for neighbour in neighbours:
            adjacency_rows.append(column[protein])
            adjacency_cols.append(column[neighbour])
    # neighbours[d, p] = 1 if protein p interacts with a target of drug d.
    neighbours = np.zeros((n_drugs, n_proteins), dtype=np.float32)
    target_rows, target_cols = np.nonzero(targets)
    by_protein = collections.defaultdict(list)
    for r, c in zip(adjacency_rows, adjacency_cols):
        by_protein[r].append(c)
    for drug_row, protein_col in zip(target_rows, target_cols):
        neighbours[drug_row, by_protein.get(protein_col, [])] = 1
    expandable = neighbours * (~hub)

    shared = targets @ targets.T
    one_hop = targets @ neighbours.T
    one_hop = one_hop + one_hop.T  # target of A adjacent to a target of B, and vice versa
    two_hop = expandable @ neighbours.T  # non-hub protein adjacent to targets of both
    iu = np.triu_indices(n_drugs, k=1)

    index = {drug: i for i, drug in enumerate(drugs)}
    documented = np.zeros((n_drugs, n_drugs), dtype=np.int8)
    partners = collections.Counter()
    with sqlite3.connect(graph.combo_db) as connection:
        for a, b in connection.execute("SELECT DISTINCT drug_a, drug_b FROM side_effects"):
            partners[a] += 1
            partners[b] += 1
            if a in index and b in index:
                documented[index[a], index[b]] = documented[index[b], index[a]] = 1
    labels = documented[iu].astype(np.int64)
    # Popularity baseline: product of each drug's number of documented partners,
    # excluding the scored pair itself (no leakage).
    degree = np.array([partners[d] for d in drugs], dtype=np.float64)
    popularity = (degree[:, None] - documented) * (degree[None, :] - documented)

    features = {
        "Protein đích chung (Tier 2)": shared[iu],
        "Cầu 1 bước (đích A ↔ đích B)": one_hop[iu],
        "Cầu 2 bước (qua protein trung gian)": two_hop[iu],
        "Tổng hợp cầu protein (Tier 3)": np.log1p(shared[iu]) * 4 + np.log1p(one_hop[iu]) * 2 + np.log1p(two_hop[iu]),
        "Baseline: độ phổ biến thuốc": popularity[iu],
    }
    rng = np.random.default_rng(seed)
    features["Ngẫu nhiên"] = rng.random(len(labels))
    base_rate = labels.mean()
    rows = []
    for name, scores in features.items():
        scores = np.asarray(scores, dtype=np.float64)
        row = {
            "feature": name,
            "pairs": len(labels),
            "positives": int(labels.sum()),
            "auroc": round(auroc(scores, labels), 4),
            "auprc": round(average_precision(scores, labels, rng), 4),
        }
        if name != "Ngẫu nhiên" and not name.startswith("Baseline"):
            has = scores > 0
            row["coverage"] = round(has.mean(), 4)
            row["precision_if_present"] = round(labels[has].mean(), 4) if has.any() else ""
            row["precision_if_absent"] = round(labels[~has].mean(), 4) if (~has).any() else ""
            row["lift"] = round(labels[has].mean() / base_rate, 3) if has.any() else ""
        else:
            row.update(coverage="", precision_if_present="", precision_if_absent="", lift="")
        rows.append(row)
        print(f"  {name}: AUROC={row['auroc']} AUPRC={row['auprc']} lift={row['lift']}")
    print(f"  base rate (documented pairs) = {base_rate:.4f}")
    write_csv("a2_pair_prediction.csv", rows)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    styles = {
        "Tổng hợp cầu protein (Tier 3)": (COLORS["graph"], "-"),
        "Protein đích chung (Tier 2)": (COLORS["extra"], "-"),
        "Baseline: độ phổ biến thuốc": (COLORS["popularity"], "-"),
        "Ngẫu nhiên": (COLORS["random"], "--"),
    }
    for name, (color, style) in styles.items():
        fpr, tpr = roc_points(np.asarray(features[name], dtype=np.float64), labels)
        value = next(r["auroc"] for r in rows if r["feature"] == name)
        ax.plot(fpr, tpr, style, color=color, linewidth=2, label=f"{name} (AUROC {value:.3f})")
    ax.set_xlabel("Tỷ lệ dương tính giả (FPR)")
    ax.set_ylabel("Tỷ lệ dương tính thật (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"A2. ROC dự đoán cặp có tương tác ({len(labels):,} cặp)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    _save(fig, "a2_roc.png")


# --------------------------------------------------------------------- A3
def a3_side_effect_prediction(graph, seed: int, min_pairs: int, test_fraction: float) -> None:
    print(f"A3 per-side-effect link prediction (effects with >= {min_pairs} pairs, "
          f"{test_fraction:.0%} held out)")
    drugs = sorted(graph.drug_ids)
    drug_index = {drug: i for i, drug in enumerate(drugs)}
    codes = sorted(code for code, n in graph.effect_pair_count.items() if n >= min_pairs)
    code_index = {code: i for i, code in enumerate(codes)}
    a_list, b_list, r_list = [], [], []
    with sqlite3.connect(graph.combo_db) as connection:
        for a, b, code in connection.execute("SELECT drug_a, drug_b, effect_code FROM side_effects"):
            r = code_index.get(code)
            if r is not None:
                a_list.append(drug_index[a])
                b_list.append(drug_index[b])
                r_list.append(r)
    a = np.array(a_list, dtype=np.int64)
    b = np.array(b_list, dtype=np.int64)
    r = np.array(r_list, dtype=np.int64)
    n_drugs, n_effects = len(drugs), len(codes)
    rng = np.random.default_rng(seed)
    test = rng.random(len(r)) < test_fraction

    # Training statistics only (the held-out edges are removed: no leakage).
    counts = np.zeros((n_drugs, n_effects), dtype=np.float64)
    np.add.at(counts, (a[~test], r[~test]), 1)
    np.add.at(counts, (b[~test], r[~test]), 1)
    train_pairs = np.unique(a[~test] * n_drugs + b[~test])
    pair_degree = np.bincount(np.r_[train_pairs // n_drugs, train_pairs % n_drugs], minlength=n_drugs).astype(np.float64)
    effect_total = np.maximum(counts.sum(axis=1, keepdims=True), 1)
    frequency = counts / effect_total  # share of a drug's training edges with effect r

    # One random negative pair per positive; it must not carry r anywhere in the data.
    known = np.sort((np.minimum(a, b) * n_drugs + np.maximum(a, b)) * n_effects + r)
    pos_a, pos_b, pos_r = a[test], b[test], r[test]
    neg_a = rng.integers(0, n_drugs, len(pos_r))
    neg_b = rng.integers(0, n_drugs, len(pos_r))
    for _ in range(50):
        lo, hi = np.minimum(neg_a, neg_b), np.maximum(neg_a, neg_b)
        key = (lo * n_drugs + hi) * n_effects + pos_r
        found = np.searchsorted(known, key)
        bad = (neg_a == neg_b) | ((found < len(known)) & (known[np.minimum(found, len(known) - 1)] == key))
        if not bad.any():
            break
        neg_a[bad] = rng.integers(0, n_drugs, bad.sum())
        neg_b[bad] = rng.integers(0, n_drugs, bad.sum())

    def score(method: str, x: np.ndarray, y: np.ndarray, effect: np.ndarray) -> np.ndarray:
        if method == "graph":  # Tier 3 label scorer: partners of A and B with effect r
            both = (counts[x, effect] > 0) & (counts[y, effect] > 0)
            return both * 1e6 + counts[x, effect] + counts[y, effect]
        if method == "graph_norm":
            return frequency[x, effect] + frequency[y, effect]
        if method == "popularity":  # pair degree only, the same for every effect
            return pair_degree[x] * pair_degree[y]
        return rng.random(len(x))

    methods = {
        "graph": "Đồ thị: bộ chấm nhãn Tier 3 (số lần)",
        "graph_norm": "Đồ thị: tần suất chuẩn hoá",
        "popularity": "Baseline: độ phổ biến thuốc",
        "random": "Ngẫu nhiên",
    }
    pos_scores = {m: score(m, pos_a, pos_b, pos_r) for m in methods}
    neg_scores = {m: score(m, neg_a, neg_b, pos_r) for m in methods}
    order = np.argsort(pos_r, kind="stable")
    bounds = np.searchsorted(pos_r[order], np.arange(n_effects + 1))
    per_effect = {m: collections.defaultdict(list) for m in methods}
    rows = []
    for effect in range(n_effects):
        idx = order[bounds[effect]:bounds[effect + 1]]
        if len(idx) < 5:
            continue
        labels = np.r_[np.ones(len(idx), dtype=np.int64), np.zeros(len(idx), dtype=np.int64)]
        row = {"effect_code": codes[effect], "effect_name": "", "test_positives": len(idx)}
        for m in methods:
            s = np.r_[pos_scores[m][idx], neg_scores[m][idx]]
            values = {"auroc": auroc(s, labels), "auprc": average_precision(s, labels, rng),
                      "ap@50": ap_at_k(s, labels, rng, 50)}
            for metric, value in values.items():
                per_effect[m][metric].append(value)
                row[f"{m}_{metric}"] = round(value, 4)
        rows.append(row)
    with sqlite3.connect(graph.combo_db) as connection:
        names_by_code = dict(connection.execute("SELECT effect_code, effect_name FROM effect_counts"))
    for row in rows:
        row["effect_name"] = names_by_code.get(row["effect_code"], "")
    write_csv("a3_per_side_effect.csv", rows)

    summary = []
    for m, label in methods.items():
        line = {"method": label, "side_effects": len(per_effect[m]["auroc"])}
        for metric in ("auroc", "auprc", "ap@50"):
            values = per_effect[m][metric]
            line[metric] = round(statistics.mean(values), 4)
            line[f"{metric}_std"] = round(statistics.pstdev(values), 4)
        summary.append(line)
        print(f"  {label}: AUROC={line['auroc']} AUPRC={line['auprc']} AP@50={line['ap@50']}")
    summary.append({"method": "Decagon (bài báo, GCN)", "side_effects": 964, **DECAGON_REPORTED,
                    "auroc_std": "", "auprc_std": "", "ap@50_std": ""})
    write_csv("a3_summary.csv", summary)

    labels_vi = {"auroc": "AUROC", "auprc": "AUPRC", "ap@50": "AP@50"}
    series = {
        "Đồ thị (Tier 3)": {labels_vi[k]: summary[0][k] for k in labels_vi},
        "Baseline độ phổ biến": {labels_vi[k]: summary[2][k] for k in labels_vi},
        "Decagon (bài báo)": {labels_vi[k]: v for k, v in DECAGON_REPORTED.items()},
        "Ngẫu nhiên": {labels_vi[k]: summary[3][k] for k in labels_vi},
    }
    colors = {"Đồ thị (Tier 3)": COLORS["graph"], "Baseline độ phổ biến": COLORS["popularity"],
              "Decagon (bài báo)": COLORS["decagon"], "Ngẫu nhiên": COLORS["random"]}
    grouped_bars(f"A3. Dự đoán tác dụng phụ theo từng loại ({len(rows)} tác dụng phụ)",
                 list(labels_vi.values()), series, colors, "a3_side_effects.png")

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    low = min(min(per_effect["graph"]["auroc"]), min(per_effect["popularity"]["auroc"]))
    bins = np.linspace(math.floor(low * 50) / 50, 1.0, 41)
    ax.hist(per_effect["graph"]["auroc"], bins=bins, color=COLORS["graph"], label="Đồ thị (Tier 3)",
            edgecolor="white", linewidth=0.8)
    ax.hist(per_effect["popularity"]["auroc"], bins=bins, color=COLORS["popularity"], histtype="step",
            linewidth=2, label="Baseline độ phổ biến")
    ax.axvline(DECAGON_REPORTED["auroc"], color=COLORS["decagon"], linewidth=2, linestyle="--",
               label=f"Decagon trung bình ({DECAGON_REPORTED['auroc']})")
    ax.set_xlabel("AUROC của một tác dụng phụ")
    ax.set_ylabel("Số tác dụng phụ")
    ax.set_title("A3. Phân bố AUROC theo từng tác dụng phụ", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, "a3_auroc_hist.png")


# --------------------------------------------------------------------- A4
def a4_validator(settings: Settings, graph, names, seed: int, count: int) -> None:
    print(f"A4 validator robustness ({count} regimens)")
    rng = random.Random(seed + 7)
    pool = sorted(graph.targets)
    vocabulary = sorted(graph.side_effect_names.values())
    cases = collections.Counter()
    kept = collections.Counter()
    for _ in range(count):
        regimen = rng.sample(pool, rng.choice((3, 4, 5)))
        _, evidence = graph.retrieve(regimen, names.is_serious, settings.ranking)
        if not evidence:
            continue
        ids = {item.evidence_id for item in evidence}
        generated: list[tuple[str, ModelPrediction]] = []
        for item in rng.sample(evidence, min(6, len(evidence))):
            if item.tier in (1, 2):
                label = item.side_effect
                other = next((e for e in evidence if e.side_effect and e.side_effect != label), None)
                generated.append(("valid", ModelPrediction(side_effect=label, tier=item.tier, evidence_ids=[item.evidence_id], explanation="x")))
                generated.append(("invented_id", ModelPrediction(side_effect=label, tier=item.tier, evidence_ids=[item.evidence_id, "E99999"], explanation="x")))
                generated.append(("wrong_tier", ModelPrediction(side_effect=label, tier=3, evidence_ids=[item.evidence_id], explanation="x")))
                if other is not None:
                    generated.append(("wrong_label", ModelPrediction(side_effect=other.side_effect, tier=item.tier, evidence_ids=[item.evidence_id], explanation="x")))
            else:
                if item.candidates:
                    generated.append(("valid", ModelPrediction(side_effect=rng.choice(item.candidates), tier=3, evidence_ids=[item.evidence_id], explanation="x")))
                outside = [v for v in rng.sample(vocabulary, 50) if v not in item.candidates]
                generated.append(("label_not_candidate", ModelPrediction(side_effect=outside[0], tier=3, evidence_ids=[item.evidence_id], explanation="x")))
                generated.append(("invented_label", ModelPrediction(side_effect="made-up syndrome", tier=3, evidence_ids=[item.evidence_id], explanation="x")))
                generated.append(("wrong_tier", ModelPrediction(side_effect=item.candidates[0] if item.candidates else "x", tier=1, evidence_ids=[item.evidence_id], explanation="x")))
        generated.append(("invented_id", ModelPrediction(side_effect=vocabulary[0], tier=1, evidence_ids=[f"E{len(ids) + 1000}"], explanation="x")))
        for kind, prediction in generated:
            cases[kind] += 1
            kept[kind] += bool(validate_predictions([prediction], evidence, graph))
    rows = []
    for kind in sorted(cases, key=lambda k: (k != "valid", k)):
        rate = kept[kind] / cases[kind]
        rows.append({
            "case": kind,
            "predictions": cases[kind],
            "kept": kept[kind],
            "rejected": cases[kind] - kept[kind],
            "expected": "kept" if kind == "valid" else "rejected",
            "correct_rate": round(rate if kind == "valid" else 1 - rate, 4),
        })
        print(f"  {kind}: {rows[-1]}")
    write_csv("a4_validator.csv", rows)


# --------------------------------------------------------------------- E2
def e2_extraction(resolver) -> None:
    """Drugs found in whole chat messages (the path the chat UI actually uses)."""
    print("E2 drug extraction from chat messages")
    with (RESULTS.parent / "extract_cases.csv").open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    rows = []
    tp = fp = fn = asked = 0
    for case in cases:
        expected = {resolver.resolve(name).stitch_id for name in case["expected"].split("|") if name}
        found = resolver.extract(case["message"])
        got = {item.stitch_id for item in found if item.stitch_id}
        questions = [item.query for item in found if item.stitch_id is None and item.method == "ambiguous"]
        tp += len(got & expected)
        fp += len(got - expected)
        fn += len(expected - got)
        asked += len(questions)
        rows.append({
            "message": case["message"],
            "category": case["category"],
            "expected": "|".join(sorted(resolver.names.drug_name(d) for d in expected)),
            "got": "|".join(sorted(resolver.names.drug_name(d) for d in got)),
            "asked": "|".join(questions),
            "exact": int(got == expected),
        })
        if got != expected:
            print(f"  MISMATCH: {case['message']!r}: expected {rows[-1]['expected']} got {rows[-1]['got']} asked {questions}")
    write_csv("e2_extraction.csv", rows)
    summary = {
        "messages": len(rows),
        "exact_match": round(sum(r["exact"] for r in rows) / len(rows), 4),
        "drug_precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "drug_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "wrong_drugs": fp,
        "missed_drugs": fn,
        "clarifying_questions": asked,
    }
    write_csv("e2_summary.csv", [summary])
    print(f"  {summary}")


# --------------------------------------------------------------------- A5
# Decagon names that count as the expected outcome of each benchmark case.
EXPECTED_EFFECTS = {
    "bleeding": r"haemorrhag|hemorrhag|bleed|haematemesis|melaena|haematochezia|coughing blood|haemarthrosis|hematoma|haematoma",
    "kidney": r"kidney failure|renal failure|azotaemia",
    "hyperkalaemia": r"excess potassium|hyperkal",
    "bradycardia": r"bradycardia|bradyarrhythmia|heart block|atrioventricular block|sinus arrest",
    "arrhythmia": r"arrhythmia|tachycardia ventricular|ventricular fibrillation|torsade|\bafib\b|\bfib\b|extrasystoles|bradycardia",
    "rhabdomyolysis": r"rhabdomyolysis|myoglobinuria|muscle inflammation|myopathy",
    "hypoglycaemia": r"hypoglyc",
    "myelosuppression": r"neutropenia|agranulocyt|pancytopenia|bone marrow|thrombocytopenia|leucopenia|aplas",
    "cns_respiratory_depression": r"respiratory failure|hypoventilation|\bcoma\b|apnea|apnoea|loss of consciousness|pulmonary arrest|respiratory depression",
    "seizure_serotonin": r"convulsion|seizure|epilep|serotonin",
    "hypotension": r"hypotension|fainting|syncope|arterial pressure nos decreased",
}
EXPECTED_VI = {
    "bleeding": "chảy máu", "kidney": "suy thận", "hyperkalaemia": "tăng kali máu",
    "bradycardia": "nhịp chậm / block tim", "arrhythmia": "loạn nhịp", "rhabdomyolysis": "tiêu cơ vân",
    "hypoglycaemia": "hạ đường huyết", "myelosuppression": "ức chế tủy",
    "cns_respiratory_depression": "ức chế thần kinh / hô hấp", "seizure_serotonin": "co giật / serotonin",
    "hypotension": "tụt huyết áp",
}


def a5_clinical_benchmark(settings: Settings, graph, names, resolver) -> None:
    """Do well-established interactions reach the top of the serious list?"""
    import dataclasses

    from graphrag.names import NameIndex
    from graphrag.presenter import select_effects

    print("A5 clinical benchmark of established drug interactions")
    with (RESULTS.parent / "clinical_benchmark.csv").open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    excluded = [r["keyword"] for r in read_csv("../../resources/serious_keywords.csv") if r["category"] == "exclude"]
    keyword_only = NameIndex(names.drugs, names.genes, names.side_effects_vi, names.serious_keywords, {}, excluded)
    base = settings.ranking
    variants = {
        "Chỉ từ khóa": (keyword_only.is_serious, dataclasses.replace(base, critical=0.0)),
        "IME": (names.is_serious, dataclasses.replace(base, critical=0.0)),
        "IME + đe dọa tính mạng (mặc định)": (names.is_serious, base),
        "IME + đe dọa tính mạng, bỏ IDF": (names.is_serious, dataclasses.replace(base, idf=0.0)),
    }
    rows, summary = [], []
    per_variant = collections.defaultdict(list)
    for case in cases:
        drugs = [resolver.resolve(name).stitch_id for name in case["drugs"].split("|")]
        pair = tuple(sorted(resolver.resolve(name).stitch_id for name in case["interacting_pair"].split("|")))
        if None in drugs or None in pair:
            print(f"  case {case['case']}: drug not in Decagon, skipped")
            continue
        pattern = re.compile(EXPECTED_EFFECTS[case["expected"]], re.IGNORECASE)
        row = {"case": case["case"], "drugs": case["drugs"], "expected": case["expected"]}
        documented = graph.side_effects(*pair)
        row["pair_documented"] = int(any(pattern.search(e.name) for e in documented))
        for label, (is_serious, weights) in variants.items():
            context, _ = graph.retrieve(drugs, is_serious, weights, names.is_critical)
            ranked = context["ranked_effects"]
            serious = [item for item in ranked if item["serious"]]
            rank = next((i for i, item in enumerate(serious, 1) if pattern.search(item["name"])), None)
            shown = select_effects(ranked, settings.max_serious_effects, settings.max_other_effects)
            hit = any(pattern.search(item["name"]) for item in shown[: settings.max_serious_effects] if item["serious"])
            row[f"{label}|rank"] = rank or ""
            per_variant[label].append((rank, hit, any(pattern.search(i["name"]) for i in ranked)))
        rows.append(row)
    write_csv("a5_clinical_benchmark.csv", rows)
    for label, results in per_variant.items():
        documented = [r for r in results if r[2]]
        line = {
            "variant": label,
            "cases": len(results),
            "documented": len(documented),
            "hit@3": round(sum(1 for r in documented if r[0] and r[0] <= 3) / len(documented), 4),
            "hit@10": round(sum(1 for r in documented if r[0] and r[0] <= 10) / len(documented), 4),
            "mrr": round(statistics.mean(1 / r[0] if r[0] else 0 for r in documented), 4),
        }
        summary.append(line)
        print(f"  {label}: documented {line['documented']}/{line['cases']} hit@3={line['hit@3']} "
              f"hit@10={line['hit@10']} MRR={line['mrr']}")
    write_csv("a5_summary.csv", summary)
    colors = dict(zip(variants, [COLORS["random"], COLORS["popularity"], COLORS["graph"], COLORS["extra"]]))
    grouped_bars(f"A5. Benchmark tương tác kinh điển ({summary[0]['documented']} ca có dữ liệu)",
                 ["hit@3", "hit@10", "mrr"],
                 {line["variant"]: {k: line[k] for k in ("hit@3", "hit@10", "mrr")} for line in summary},
                 colors, "a5_clinical_benchmark.png")


# --------------------------------------------------------------------- A6
# serious_keywords.csv before the IME labels were introduced (commit a1f2615).
LEGACY_KEYWORDS = (
    "haemorrhage hemorrhage bleeding haematemesis melaena arrhythmia fibrillation|tachycardia ventricular|"
    "torsade|qt prolonged|cardiac arrest|cardiac failure|heart failure|myocardial infarction|bradycardia|"
    "heart block|hypotension|renal failure|kidney failure|acute kidney|nephropathy|hepatic failure|"
    "liver failure|hepatitis|hepatotoxicity|jaundice|rhabdomyolysis|serotonin syndrome|neuroleptic malignant|"
    "convulsion|seizure|coma|stroke|cerebrovascular accident|respiratory depression|respiratory arrest|"
    "respiratory failure|apnoea|pulmonary embolism|thrombosis|anaphylactic|anaphylaxis|angioedema|"
    "stevens-johnson|toxic epidermal necrolysis|agranulocytosis|pancytopenia|neutropenia|thrombocytopenia|"
    "aplastic anaemia|hypoglycaemia|hyperkalaemia|hyponatraemia|lactic acidosis|pancreatitis|"
    "gastrointestinal perforation|sepsis|death"
)


def a6_serious_classifier() -> None:
    """Keyword matching vs the EMA IME reference, on names with an exact IME match."""
    from graphrag.names import _word_pattern

    print("A6 serious-effect classifier vs EMA IME")
    labels = [row for row in read_csv("../../resources/serious_effects.csv") if row["method"] in {"ime_match", "excluded"}]
    if not labels:
        print("  resources/serious_effects.csv missing; run scripts/build_serious_effects.py")
        return
    legacy_words = [w for part in LEGACY_KEYWORDS.split("|") for w in ([part] if " " in part else part.split())]
    keyword_rows = read_csv("../../resources/serious_keywords.csv")
    current = [r["keyword"] for r in keyword_rows if r["category"] != "exclude"]
    excluded = [r["keyword"] for r in keyword_rows if r["category"] == "exclude"]
    methods = {
        "Từ khóa cũ (khớp nguyên từ)": (_word_pattern(legacy_words), None),
        "Từ khóa mới (số nhiều + loại trừ)": (_word_pattern(current, plural=True), _word_pattern(excluded)),
    }
    rows = []
    for label, (pattern, exclude) in methods.items():
        tp = fp = fn = tn = 0
        for row in labels:
            text = row["name_en"].casefold()
            predicted = bool(pattern.search(text)) and not (exclude and exclude.search(text))
            truth = row["serious"] == "1"
            tp += predicted and truth
            fp += predicted and not truth
            fn += truth and not predicted
            tn += not predicted and not truth
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({
            "method": label, "names": len(labels), "ime_serious": tp + fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
        print(f"  {label}: precision={rows[-1]['precision']} recall={rows[-1]['recall']} f1={rows[-1]['f1']}")
    write_csv("a6_serious_classifier.csv", rows)


# ----------------------------------------------------------------- summary
def figures_from_offline() -> None:
    """Figures for E1 and E4 (computed by run_offline.py) if their CSVs exist."""
    e1 = read_csv("e1_resolver.csv")
    if e1:
        plt = _plt()
        by_category = collections.defaultdict(collections.Counter)
        for row in e1:
            by_category[row["category"]][row["outcome"]] += 1
        categories = list(by_category)
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        left = np.zeros(len(categories))
        labels = {"correct": "Đúng", "asked": "Hỏi lại người dùng", "wrong": "Nhận sai"}
        colors = {"correct": COLORS["graph"], "asked": COLORS["random"], "wrong": "#e34948"}
        for outcome in ("correct", "asked", "wrong"):
            values = np.array([by_category[c][outcome] / sum(by_category[c].values()) for c in categories])
            ax.barh([E1_CATEGORIES.get(c, c) for c in categories], values, left=left,
                    color=colors[outcome], label=labels[outcome], edgecolor="white", linewidth=2)
            left += values
        for i, category in enumerate(categories):
            total = sum(by_category[category].values())
            ax.annotate(f"n={total}", (1.01, i), va="center", fontsize=8, color="#52514e")
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Tỷ lệ")
        ax.set_title("E1. Nhận dạng tên thuốc theo loại đầu vào", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper left", bbox_to_anchor=(0, -0.2))
        ax.grid(axis="y", visible=False)
        _save(fig, "e1_resolver.png")
    e4 = {row["method"]: row for row in read_csv("e4_summary.csv")}
    if e4:
        names = {"graph_candidates": "Đồ thị (Tier 3)", "global_popularity": "Baseline độ phổ biến",
                 "random": "Ngẫu nhiên"}
        colors = {"Đồ thị (Tier 3)": COLORS["graph"], "Baseline độ phổ biến": COLORS["popularity"],
                  "Ngẫu nhiên": COLORS["random"]}
        series = {names[m]: {k: float(e4[m][k]) for k in ("hit@10", "precision@10", "recall@20")}
                  for m in names if m in e4}
        grouped_bars("E4. Nhãn Tier 3 trên cặp thuốc bị ẩn", ["hit@10", "precision@10", "recall@20"],
                     series, colors, "e4_tier3_labels.png")


def _table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(title for _, title in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def _pct(value: str) -> str:
    try:
        return f"{float(value):.1%}"
    except ValueError:
        return value


def write_report(meta: dict) -> None:
    parts = ["# Kết quả đánh giá độ chính xác",
             "",
             "Sinh tự động bởi `python eval/run_offline.py` và `python eval/run_accuracy.py` "
             f"(seed {meta['seed']}). Không có bước nào gọi LLM. Hình nằm trong `figures/`.",
             ""]
    e1 = read_csv("e1_resolver.csv")
    if e1:
        outcomes = collections.Counter(row["outcome"] for row in e1)
        by_category = collections.defaultdict(collections.Counter)
        for row in e1:
            by_category[row["category"]][row["outcome"]] += 1
        rows = [{"category": E1_CATEGORIES.get(c, c), "n": sum(v.values()), "correct": _pct(v["correct"] / sum(v.values())),
                 "asked": v["asked"], "wrong": v["wrong"]} for c, v in by_category.items()]
        parts += ["## E1. Nhận dạng tên thuốc",
                  "",
                  f"{len(e1)} câu nhập (generic, biệt dược VN, kèm hàm lượng, sai chính tả, thuốc không có trong dữ liệu). "
                  f"Độ chính xác **{outcomes['correct'] / len(e1):.1%}**, nhận sai **{outcomes['wrong']}**, "
                  f"hỏi lại {outcomes['asked']}.",
                  "",
                  _table(rows, [("category", "Loại"), ("n", "n"), ("correct", "Đúng"), ("asked", "Hỏi lại"), ("wrong", "Sai")]),
                  "",
                  "![E1](figures/e1_resolver.png)",
                  ""]
    e2 = read_csv("e2_summary.csv")
    if e2:
        row = e2[0]
        misses = [r for r in read_csv("e2_extraction.csv") if r["exact"] != "1"]
        parts += ["## E2. Trích thuốc từ câu chat",
                  "",
                  f"{row['messages']} câu chat thực tế (biệt dược, hàm lượng, sai chính tả, tiếng Anh, có từ thừa như "
                  "\"vitamin C\", \"rượu bia\", triệu chứng). Đây là đường đi thật của giao diện chat.",
                  "",
                  _table([{**row, "exact_match": _pct(row["exact_match"]), "drug_precision": _pct(row["drug_precision"]),
                           "drug_recall": _pct(row["drug_recall"])}],
                         [("exact_match", "Đúng cả câu"), ("drug_precision", "Precision thuốc"),
                          ("drug_recall", "Recall thuốc"), ("wrong_drugs", "Thuốc nhận nhầm"),
                          ("missed_drugs", "Thuốc bỏ sót"), ("clarifying_questions", "Câu hỏi lại")]),
                  ""]
        if misses:
            parts += ["Các câu chưa đúng:", ""] + [
                f"- \"{r['message']}\": mong đợi {r['expected'] or '(không có)'}, nhận {r['got'] or '(không có)'}"
                for r in misses] + [""]
    a1 = read_csv("a1_faithfulness.csv")
    if a1:
        for row in a1:
            row["pct"] = _pct(row["value"])
        parts += ["## A1. Độ trung thực của kết quả so với dữ liệu gốc",
                  "",
                  f"{meta.get('a1_regimens', '')} phác đồ ngẫu nhiên (3–6 thuốc). Mọi kết quả được đối chiếu lại với "
                  "các file CSV gốc, đọc độc lập với cache SQLite.",
                  "",
                  _table(a1, [("check", "Kiểm tra"), ("numerator", "Đạt"), ("denominator", "Tổng"), ("pct", "Tỷ lệ")]),
                  ""]
    a2 = read_csv("a2_pair_prediction.csv")
    if a2:
        parts += ["## A2. Dự đoán cặp thuốc có tương tác từ đồ thị protein",
                  "",
                  f"Tất cả {a2[0]['pairs']} cặp giữa các thuốc có protein đích; nhãn dương = cặp có dữ liệu trong "
                  f"`bio-decagon-combo` ({a2[0]['positives']} cặp). Các đặc trưng chỉ dùng targets + PPI, "
                  "không dùng dữ liệu combo. Lưu ý: cặp \"chưa có dữ liệu\" không chắc là không tương tác.",
                  "",
                  _table(a2, [("feature", "Đặc trưng"), ("auroc", "AUROC"), ("auprc", "AUPRC"),
                              ("coverage", "Tỷ lệ cặp có"), ("precision_if_present", "Tỷ lệ tương tác khi có"),
                              ("precision_if_absent", "Tỷ lệ tương tác khi không"), ("lift", "Lift")]),
                  "",
                  "![A2](figures/a2_roc.png)",
                  ""]
        bridge = next((r for r in a2 if "Tier 3" in r["feature"]), None)
        popular = next((r for r in a2 if r["feature"].startswith("Baseline")), None)
        if bridge and popular:
            parts += [f"**Diễn giải.** Cầu protein có ở {_pct(bridge['coverage'])} số cặp nên gần như không "
                      f"phân biệt được (AUROC {bridge['auroc']}, lift {bridge['lift']}), trong khi chỉ biết thuốc "
                      f"nào được báo cáo nhiều đã đạt AUROC {popular['auroc']}. Vì vậy Tier 3 chỉ được trình bày "
                      "là *giả thuyết cơ chế* cần kiểm chứng, không phải dự báo nguy cơ; hệ thống không dùng sự "
                      "tồn tại của cầu protein để kết luận một cặp thuốc có tương tác.", ""]
    a3 = read_csv("a3_summary.csv")
    if a3:
        parts += ["## A3. Dự đoán từng loại tác dụng phụ (giao thức Decagon)",
                  "",
                  f"Mỗi tác dụng phụ có ≥ {meta.get('min_pairs', 500)} cặp: ẩn {meta.get('test_fraction', 0.1):.0%} số cạnh "
                  "làm tập kiểm tra, mỗi cạnh dương ghép với một cặp âm ngẫu nhiên không có tác dụng phụ đó. "
                  "Điểm của phương pháp chỉ tính trên phần huấn luyện. Dòng Decagon lấy từ bài báo "
                  "(GCN, cùng dữ liệu nhưng cách chia khác) — chỉ để tham chiếu.",
                  "",
                  _table(a3, [("method", "Phương pháp"), ("side_effects", "Số TDP"), ("auroc", "AUROC"),
                              ("auprc", "AUPRC"), ("ap@50", "AP@50")]),
                  "",
                  "![A3](figures/a3_side_effects.png)",
                  "",
                  "![A3 hist](figures/a3_auroc_hist.png)",
                  ""]
        if len(a3) >= 4:
            parts += [f"**Diễn giải.** Bộ chấm nhãn Tier 3 (đếm số thuốc bạn của A và B có cùng tác dụng phụ) "
                      f"đạt AUROC {a3[0]['auroc']}, cao hơn baseline độ phổ biến ({a3[2]['auroc']}) và con số "
                      f"Decagon công bố ({DECAGON_REPORTED['auroc']}). Không nên kết luận là tốt hơn Decagon: "
                      "cặp âm lấy ngẫu nhiên phần lớn là cặp ít được báo cáo, nên chỉ riêng độ phổ biến đã đạt "
                      f"{a3[2]['auroc']}; giao thức chia dữ liệu cũng khác bài báo. Kết quả cho thấy nhãn mà "
                      "Tier 3 gợi ý có cơ sở thống kê trên dữ liệu, đồng thời phản ánh thiên lệch báo cáo "
                      "(reporting bias) của FAERS.", ""]
    e4 = read_csv("e4_summary.csv")
    if e4:
        methods_vi = {"graph_candidates": "Đồ thị (ứng viên Tier 3)",
                      "global_popularity": "Baseline: phổ biến toàn cục", "random": "Ngẫu nhiên"}
        for row in e4:
            row["method"] = methods_vi.get(row["method"], row["method"])
        parts += ["## E4. Nhãn Tier 3 trên cặp thuốc bị ẩn",
                  "",
                  f"{e4[0]['pairs']} cặp có dữ liệu (cả hai thuốc có protein đích) được ẩn đi; danh sách ứng viên "
                  "được dựng lại không dùng chính cặp đó.",
                  "",
                  _table(e4, [("method", "Phương pháp"), ("hit@10", "Hit@10"), ("precision@10", "Precision@10"),
                              ("recall@20", "Recall@20")]),
                  "",
                  "![E4](figures/e4_tier3_labels.png)",
                  ""]
    a4 = read_csv("a4_validator.csv")
    if a4:
        cases_vi = {"valid": "Hợp lệ", "invented_id": "ID bằng chứng bịa",
                    "invented_label": "Nhãn bịa (ngoài từ vựng)",
                    "label_not_candidate": "Nhãn Tier 3 ngoài ứng viên",
                    "wrong_label": "Sai nhãn tác dụng phụ", "wrong_tier": "Sai tier"}
        for row in a4:
            row["pct"] = _pct(row["correct_rate"])
            row["case"] = cases_vi.get(row["case"], row["case"])
            row["expected"] = "giữ" if row["expected"] == "kept" else "loại"
        parts += ["## A4. Độ bền của bộ kiểm tra (validator) đầu ra LLM",
                  "",
                  "Dự đoán hợp lệ và dự đoán bị làm sai có chủ đích (ID bằng chứng bịa, sai tier, sai nhãn, "
                  "nhãn Tier 3 ngoài danh sách ứng viên) được sinh từ evidence thật.",
                  "",
                  _table(a4, [("case", "Trường hợp"), ("predictions", "Số dự đoán"), ("expected", "Mong đợi"),
                              ("kept", "Giữ"), ("rejected", "Loại"), ("pct", "Đúng")]),
                  ""]
    a5 = read_csv("a5_summary.csv")
    if a5:
        detail = read_csv("a5_clinical_benchmark.csv")
        default = next((key for key in detail[0] if key.endswith("(mặc định)|rank")), None) if detail else None
        for row in a5:
            for key in ("hit@3", "hit@10"):
                row[key] = _pct(row[key])
        parts += ["## A5. Benchmark tương tác thuốc kinh điển",
                  "",
                  f"{len(detail)} ca tương tác đã được xác lập lâm sàng (`eval/clinical_benchmark.csv`), mỗi ca gồm cặp "
                  "tương tác và một thuốc thứ ba. Đo vị trí tác dụng phụ mong đợi trong danh sách nghiêm trọng "
                  "(hệ thống hiển thị 10 mục đầu).",
                  "",
                  _table(a5, [("variant", "Cách xếp hạng"), ("documented", "Ca có dữ liệu"), ("hit@3", "Hit@3"),
                              ("hit@10", "Hit@10"), ("mrr", "MRR")]),
                  "",
                  "![A5](figures/a5_clinical_benchmark.png)",
                  ""]
        if default:
            parts += ["| Ca | Phác đồ | Mong đợi | Hạng (mặc định) |", "|---|---|---|---|"] + [
                f"| {r['case']} | {r['drugs'].replace('|', ' + ')} | {EXPECTED_VI.get(r['expected'], r['expected'])} | "
                f"{r[default] or 'không nằm trong nhóm nghiêm trọng'} |" for r in detail] + [""]
        parts += ["**Diễn giải.** Decagon chỉ cho biết một tác dụng phụ *có* được báo cáo cho cặp thuốc hay không, "
                  "không có tần suất hay mức độ, nên điểm xếp hạng không phân biệt được tác dụng phụ đặc trưng của "
                  "tương tác với hàng trăm tác dụng phụ khác của cùng cặp. Khoảng một nửa số ca kinh điển lên được "
                  "top 10. Thử thêm chuẩn hoá theo độ đặc trưng của thuốc (kiểu PRR) cũng không cải thiện ổn định, "
                  "nên trọng số không được chỉnh theo bộ 26 ca này để tránh overfit. Hướng cải thiện: dùng số lượng "
                  "báo cáo FAERS để tính disproportionality thật.", ""]
    a6 = read_csv("a6_serious_classifier.csv")
    if a6:
        labels = read_csv("../../resources/serious_effects.csv")
        combo = [r for r in labels if r["vocabulary"] == "combo"]
        parts += ["## A6. Phân loại tác dụng phụ nghiêm trọng",
                  "",
                  "Chuẩn tham chiếu: danh sách *Important Medical Event* (IME) của EMA, MedDRA 29.1 "
                  f"(`scripts/build_serious_effects.py`). {sum(r['serious'] == '1' for r in combo)} / 1317 tác dụng phụ phối hợp "
                  "được gắn nghiêm trọng; ung thư, dị tật bẩm sinh, thai sản, nhiễm trùng cộng đồng và bệnh mạn tính "
                  "bị loại vì trong FAERS chúng thường là chỉ định hay bệnh nền. Bảng dưới so cách dùng từ khóa với "
                  "IME trên các tên khớp trực tiếp một thuật ngữ IME.",
                  "",
                  _table(a6, [("method", "Phương pháp"), ("names", "Số tên"), ("ime_serious", "IME nghiêm trọng"),
                              ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]),
                  "",
                  "**Diễn giải.** Từ khóa chính xác (precision cao) nhưng chỉ bắt được một phần nhỏ (recall thấp) các "
                  "tác dụng phụ IME, và bỏ sót tên thông dụng như \"heart attack\", \"gastrointestinal bleed\", "
                  "\"deep vein thromboses\". Vì vậy hệ thống dùng nhãn IME làm chính; từ khóa chỉ dùng cho tên "
                  "chưa ánh xạ được và để ưu tiên nhóm đe dọa tính mạng khi xếp hạng.", ""]
    e3 = read_csv("e3_llm_groundedness.csv")
    if e3:
        raw = sum(int(row["raw_predictions"]) for row in e3)
        valid = sum(int(row["valid_predictions"]) for row in e3)
        parts += ["## E3. Dự đoán của LLM qua được validator", "",
                  f"{len(e3)} phác đồ: giữ {valid}/{raw} dự đoán ({valid / raw:.1%})." if raw else "Không có dự đoán.", ""]
    else:
        parts += ["## E3. Dự đoán của LLM qua được validator", "",
                  "Chưa chạy (cần gọi API): `python eval/e3_llm_groundedness.py --yes-call-api --regimens 10`.", ""]
    (RESULTS / "ACCURACY.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"  -> {RESULTS / 'ACCURACY.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regimens", type=int, default=200)
    parser.add_argument("--min-pairs", type=int, default=500)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--only", nargs="*", choices=["e2", "a1", "a2", "a3", "a4", "a5", "a6"])
    args = parser.parse_args()
    selected = set(args.only or ["e2", "a1", "a2", "a3", "a4", "a5", "a6"])
    settings, graph, names, resolver = load()
    meta = {"seed": args.seed, "min_pairs": args.min_pairs, "test_fraction": args.test_fraction,
            "a1_regimens": args.regimens}
    if "e2" in selected:
        e2_extraction(resolver)
    if "a1" in selected:
        a1_faithfulness(settings, graph, names, args.seed, args.regimens)
    if "a2" in selected:
        a2_pair_prediction(graph, args.seed)
    if "a3" in selected:
        a3_side_effect_prediction(graph, args.seed, args.min_pairs, args.test_fraction)
    if "a4" in selected:
        a4_validator(settings, graph, names, args.seed, max(args.regimens // 4, 20))
    if "a5" in selected:
        a5_clinical_benchmark(settings, graph, names, resolver)
    if "a6" in selected:
        a6_serious_classifier()
    figures_from_offline()
    write_report(meta)


if __name__ == "__main__":
    main()
