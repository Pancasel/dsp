"""Write evaluation/results/ACCURACY.md from the result CSV files, and draw the E1/E4 figures.

Every section is written only when its CSV exists, so experiments can be run
separately (E1/E4 come from run_offline_experiments, the rest from
run_accuracy_experiments).
"""

import collections

import numpy as np

from evaluation.experiments.a3_per_side_effect_prediction import DECAGON_REPORTED
from evaluation.shared.clinical_expectations import EXPECTED_EFFECT_VIETNAMESE
from evaluation.shared.figure_drawing import (
    METHOD_COLORS,
    draw_grouped_bars,
    import_pyplot_with_report_style,
    save_figure,
)
from evaluation.shared.markdown_tables import build_markdown_table, format_as_percentage
from evaluation.shared.result_csv_files import read_csv_file, read_result_csv
from evaluation.shared.system_loading import RESOURCES_FOLDER, RESULTS_FOLDER


E1_CATEGORY_TITLES = {
    "generic": "Tên hoạt chất",
    "brand_vi": "Biệt dược VN",
    "with_strength": "Kèm hàm lượng",
    "typo": "Sai chính tả",
    "case": "Hoa/thường",
    "not_in_data": "Không có trong dữ liệu",
}


def draw_offline_experiment_figures() -> None:
    """Figures for E1 and E4 (computed by run_offline_experiments) if their CSVs exist."""
    _draw_e1_figure(read_result_csv("e1_resolver.csv"))
    _draw_e4_figure({row["method"]: row for row in read_result_csv("e4_summary.csv")})


def _draw_e1_figure(e1_rows: list[dict[str, str]]) -> None:
    if not e1_rows:
        return
    pyplot = import_pyplot_with_report_style()
    outcome_counts_by_category = collections.defaultdict(collections.Counter)
    for row in e1_rows:
        outcome_counts_by_category[row["category"]][row["outcome"]] += 1
    categories = list(outcome_counts_by_category)
    figure, axes = pyplot.subplots(figsize=(6.8, 3.2))
    bar_starts = np.zeros(len(categories))
    outcome_titles = {"correct": "Đúng", "asked": "Hỏi lại người dùng", "wrong": "Nhận sai"}
    outcome_colors = {"correct": METHOD_COLORS["graph"], "asked": METHOD_COLORS["random"], "wrong": "#e34948"}
    for outcome in ("correct", "asked", "wrong"):
        shares = np.array(
            [
                outcome_counts_by_category[category][outcome]
                / sum(outcome_counts_by_category[category].values())
                for category in categories
            ]
        )
        axes.barh(
            [E1_CATEGORY_TITLES.get(category, category) for category in categories], shares,
            left=bar_starts, color=outcome_colors[outcome], label=outcome_titles[outcome],
            edgecolor="white", linewidth=2,
        )
        bar_starts += shares
    for position, category in enumerate(categories):
        case_count = sum(outcome_counts_by_category[category].values())
        axes.annotate(f"n={case_count}", (1.01, position), va="center", fontsize=8, color="#52514e")
    axes.set_xlim(0, 1.1)
    axes.set_xlabel("Tỷ lệ")
    axes.set_title("E1. Nhận dạng tên thuốc theo loại đầu vào", loc="left", fontsize=11)
    axes.legend(frameon=False, fontsize=8, ncols=3, loc="upper left", bbox_to_anchor=(0, -0.2))
    axes.grid(axis="y", visible=False)
    save_figure(figure, "e1_resolver.png")


def _draw_e4_figure(e4_summary_by_method: dict[str, dict[str, str]]) -> None:
    if not e4_summary_by_method:
        return
    series_titles = {
        "graph_candidates": "Đồ thị (Tier 3)",
        "global_popularity": "Baseline độ phổ biến",
        "random": "Ngẫu nhiên",
    }
    colors = {
        "Đồ thị (Tier 3)": METHOD_COLORS["graph"],
        "Baseline độ phổ biến": METHOD_COLORS["popularity"],
        "Ngẫu nhiên": METHOD_COLORS["random"],
    }
    metrics = ("hit@10", "precision@10", "recall@20")
    values_by_series = {
        series_titles[method]: {metric: float(e4_summary_by_method[method][metric]) for metric in metrics}
        for method in series_titles
        if method in e4_summary_by_method
    }
    draw_grouped_bars(
        "E4. Nhãn Tier 3 trên cặp thuốc bị ẩn", list(metrics), values_by_series, colors, "e4_tier3_labels.png"
    )


def write_accuracy_report(run_details: dict) -> None:
    """run_details: seed, a1_regimens, min_pairs and test_fraction of this run."""
    parts = [
        "# Kết quả đánh giá độ chính xác",
        "",
        "Sinh tự động bởi `python -m evaluation.run_offline_experiments` và "
        "`python -m evaluation.run_accuracy_experiments` "
        f"(seed {run_details['seed']}). Không có bước nào gọi LLM. Hình nằm trong `figures/`.",
        "",
    ]
    parts += _e1_section()
    parts += _e2_section()
    parts += _a1_section(run_details)
    parts += _a2_section()
    parts += _a3_section(run_details)
    parts += _e4_section()
    parts += _a4_section()
    parts += _a5_section()
    parts += _a6_section()
    parts += _e3_section()
    (RESULTS_FOLDER / "ACCURACY.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"  -> {RESULTS_FOLDER / 'ACCURACY.md'}")


def _e1_section() -> list[str]:
    e1_rows = read_result_csv("e1_resolver.csv")
    if not e1_rows:
        return []
    outcome_counts = collections.Counter(row["outcome"] for row in e1_rows)
    outcome_counts_by_category = collections.defaultdict(collections.Counter)
    for row in e1_rows:
        outcome_counts_by_category[row["category"]][row["outcome"]] += 1
    table_rows = [
        {
            "category": E1_CATEGORY_TITLES.get(category, category),
            "n": sum(counts.values()),
            "correct": format_as_percentage(counts["correct"] / sum(counts.values())),
            "asked": counts["asked"],
            "wrong": counts["wrong"],
        }
        for category, counts in outcome_counts_by_category.items()
    ]
    return [
        "## E1. Nhận dạng tên thuốc",
        "",
        f"{len(e1_rows)} câu nhập (generic, biệt dược VN, kèm hàm lượng, sai chính tả, thuốc không có trong dữ liệu). "
        f"Độ chính xác **{outcome_counts['correct'] / len(e1_rows):.1%}**, nhận sai **{outcome_counts['wrong']}**, "
        f"hỏi lại {outcome_counts['asked']}.",
        "",
        build_markdown_table(
            table_rows,
            [("category", "Loại"), ("n", "n"), ("correct", "Đúng"), ("asked", "Hỏi lại"), ("wrong", "Sai")],
        ),
        "",
        "![E1](figures/e1_resolver.png)",
        "",
    ]


def _e2_section() -> list[str]:
    e2_summary = read_result_csv("e2_summary.csv")
    if not e2_summary:
        return []
    summary = e2_summary[0]
    mismatches = [row for row in read_result_csv("e2_extraction.csv") if row["exact"] != "1"]
    formatted_summary = {
        **summary,
        "exact_match": format_as_percentage(summary["exact_match"]),
        "drug_precision": format_as_percentage(summary["drug_precision"]),
        "drug_recall": format_as_percentage(summary["drug_recall"]),
    }
    parts = [
        "## E2. Trích thuốc từ câu chat",
        "",
        f"{summary['messages']} câu chat thực tế (biệt dược, hàm lượng, sai chính tả, tiếng Anh, có từ thừa như "
        "\"vitamin C\", \"rượu bia\", triệu chứng). Đây là đường đi thật của giao diện chat.",
        "",
        build_markdown_table(
            [formatted_summary],
            [("exact_match", "Đúng cả câu"), ("drug_precision", "Precision thuốc"),
             ("drug_recall", "Recall thuốc"), ("wrong_drugs", "Thuốc nhận nhầm"),
             ("missed_drugs", "Thuốc bỏ sót"), ("clarifying_questions", "Câu hỏi lại")],
        ),
        "",
    ]
    if mismatches:
        parts += ["Các câu chưa đúng:", ""]
        for row in mismatches:
            parts.append(
                f"- \"{row['message']}\": mong đợi {row['expected'] or '(không có)'}, "
                f"nhận {row['got'] or '(không có)'}"
            )
        parts.append("")
    return parts


def _a1_section(run_details: dict) -> list[str]:
    a1_rows = read_result_csv("a1_faithfulness.csv")
    if not a1_rows:
        return []
    for row in a1_rows:
        row["percentage"] = format_as_percentage(row["value"])
    return [
        "## A1. Độ trung thực của kết quả so với dữ liệu gốc",
        "",
        f"{run_details.get('a1_regimens', '')} phác đồ ngẫu nhiên (3–6 thuốc). Mọi kết quả được đối chiếu lại với "
        "các file CSV gốc, đọc độc lập với cache SQLite.",
        "",
        build_markdown_table(
            a1_rows, [("check", "Kiểm tra"), ("numerator", "Đạt"), ("denominator", "Tổng"), ("percentage", "Tỷ lệ")]
        ),
        "",
    ]


def _a2_section() -> list[str]:
    a2_rows = read_result_csv("a2_pair_prediction.csv")
    if not a2_rows:
        return []
    parts = [
        "## A2. Dự đoán cặp thuốc có tương tác từ đồ thị protein",
        "",
        f"Tất cả {a2_rows[0]['pairs']} cặp giữa các thuốc có protein đích; nhãn dương = cặp có dữ liệu trong "
        f"`bio-decagon-combo` ({a2_rows[0]['positives']} cặp). Các đặc trưng chỉ dùng targets + PPI, "
        "không dùng dữ liệu combo. Lưu ý: cặp \"chưa có dữ liệu\" không chắc là không tương tác.",
        "",
        build_markdown_table(
            a2_rows,
            [("feature", "Đặc trưng"), ("auroc", "AUROC"), ("auprc", "AUPRC"),
             ("coverage", "Tỷ lệ cặp có"), ("precision_if_present", "Tỷ lệ tương tác khi có"),
             ("precision_if_absent", "Tỷ lệ tương tác khi không"), ("lift", "Lift")],
        ),
        "",
        "![A2](figures/a2_roc.png)",
        "",
    ]
    bridge_row = next((row for row in a2_rows if "Tier 3" in row["feature"]), None)
    popularity_row = next((row for row in a2_rows if row["feature"].startswith("Baseline")), None)
    if bridge_row and popularity_row:
        parts += [
            f"**Diễn giải.** Cầu protein có ở {format_as_percentage(bridge_row['coverage'])} số cặp nên gần như không "
            f"phân biệt được (AUROC {bridge_row['auroc']}, lift {bridge_row['lift']}), trong khi chỉ biết thuốc "
            f"nào được báo cáo nhiều đã đạt AUROC {popularity_row['auroc']}. Vì vậy Tier 3 chỉ được trình bày "
            "là *giả thuyết cơ chế* cần kiểm chứng, không phải dự báo nguy cơ; hệ thống không dùng sự "
            "tồn tại của cầu protein để kết luận một cặp thuốc có tương tác.",
            "",
        ]
    return parts


def _a3_section(run_details: dict) -> list[str]:
    a3_summary = read_result_csv("a3_summary.csv")
    if not a3_summary:
        return []
    parts = [
        "## A3. Dự đoán từng loại tác dụng phụ (giao thức Decagon)",
        "",
        f"Mỗi tác dụng phụ có ≥ {run_details.get('min_pairs', 500)} cặp: ẩn "
        f"{run_details.get('test_fraction', 0.1):.0%} số cạnh "
        "làm tập kiểm tra, mỗi cạnh dương ghép với một cặp âm ngẫu nhiên không có tác dụng phụ đó. "
        "Điểm của phương pháp chỉ tính trên phần huấn luyện. Dòng Decagon lấy từ bài báo "
        "(GCN, cùng dữ liệu nhưng cách chia khác) — chỉ để tham chiếu.",
        "",
        build_markdown_table(
            a3_summary,
            [("method", "Phương pháp"), ("side_effects", "Số TDP"), ("auroc", "AUROC"),
             ("auprc", "AUPRC"), ("ap@50", "AP@50")],
        ),
        "",
        "![A3](figures/a3_side_effects.png)",
        "",
        "![A3 hist](figures/a3_auroc_hist.png)",
        "",
    ]
    if len(a3_summary) >= 4:
        graph_line, popularity_line = a3_summary[0], a3_summary[2]
        parts += [
            f"**Diễn giải.** Bộ chấm nhãn Tier 3 (đếm số thuốc bạn của A và B có cùng tác dụng phụ) "
            f"đạt AUROC {graph_line['auroc']}, cao hơn baseline độ phổ biến ({popularity_line['auroc']}) và con số "
            f"Decagon công bố ({DECAGON_REPORTED['auroc']}). Không nên kết luận là tốt hơn Decagon: "
            "cặp âm lấy ngẫu nhiên phần lớn là cặp ít được báo cáo, nên chỉ riêng độ phổ biến đã đạt "
            f"{popularity_line['auroc']}; giao thức chia dữ liệu cũng khác bài báo. Kết quả cho thấy nhãn mà "
            "Tier 3 gợi ý có cơ sở thống kê trên dữ liệu, đồng thời phản ánh thiên lệch báo cáo "
            "(reporting bias) của FAERS.",
            "",
        ]
    return parts


def _e4_section() -> list[str]:
    e4_summary = read_result_csv("e4_summary.csv")
    if not e4_summary:
        return []
    method_titles = {
        "graph_candidates": "Đồ thị (ứng viên Tier 3)",
        "global_popularity": "Baseline: phổ biến toàn cục",
        "random": "Ngẫu nhiên",
    }
    for row in e4_summary:
        row["method"] = method_titles.get(row["method"], row["method"])
    return [
        "## E4. Nhãn Tier 3 trên cặp thuốc bị ẩn",
        "",
        f"{e4_summary[0]['pairs']} cặp có dữ liệu (cả hai thuốc có protein đích) được ẩn đi; danh sách ứng viên "
        "được dựng lại không dùng chính cặp đó.",
        "",
        build_markdown_table(
            e4_summary,
            [("method", "Phương pháp"), ("hit@10", "Hit@10"), ("precision@10", "Precision@10"),
             ("recall@20", "Recall@20")],
        ),
        "",
        "![E4](figures/e4_tier3_labels.png)",
        "",
    ]


def _a4_section() -> list[str]:
    a4_rows = read_result_csv("a4_validator.csv")
    if not a4_rows:
        return []
    case_titles = {
        "valid": "Hợp lệ", "invented_id": "ID bằng chứng bịa",
        "invented_label": "Nhãn bịa (ngoài từ vựng)",
        "label_not_candidate": "Nhãn Tier 3 ngoài ứng viên",
        "wrong_label": "Sai nhãn tác dụng phụ", "wrong_tier": "Sai tier",
    }
    for row in a4_rows:
        row["percentage"] = format_as_percentage(row["correct_rate"])
        row["case"] = case_titles.get(row["case"], row["case"])
        row["expected"] = "giữ" if row["expected"] == "kept" else "loại"
    return [
        "## A4. Độ bền của bộ kiểm tra (validator) đầu ra LLM",
        "",
        "Dự đoán hợp lệ và dự đoán bị làm sai có chủ đích (ID bằng chứng bịa, sai tier, sai nhãn, "
        "nhãn Tier 3 ngoài danh sách ứng viên) được sinh từ evidence thật.",
        "",
        build_markdown_table(
            a4_rows,
            [("case", "Trường hợp"), ("predictions", "Số dự đoán"), ("expected", "Mong đợi"),
             ("kept", "Giữ"), ("rejected", "Loại"), ("percentage", "Đúng")],
        ),
        "",
    ]


def _a5_section() -> list[str]:
    a5_summary = read_result_csv("a5_summary.csv")
    if not a5_summary:
        return []
    case_rows = read_result_csv("a5_clinical_benchmark.csv")
    default_rank_column = None
    if case_rows:
        default_rank_column = next(
            (column for column in case_rows[0] if column.endswith("(mặc định)|rank")), None
        )
    for row in a5_summary:
        for metric in ("hit@3", "hit@10"):
            row[metric] = format_as_percentage(row[metric])
    parts = [
        "## A5. Benchmark tương tác thuốc kinh điển",
        "",
        f"{len(case_rows)} ca tương tác phổ biến trong nhãn thuốc và tài liệu dược lý "
        "(`evaluation/test_cases/clinical_benchmark.csv`), "
        "mỗi ca gồm cặp tương tác và một thuốc thứ ba. Đo vị trí tác dụng phụ mong đợi trong danh sách "
        "nghiêm trọng (hệ thống hiển thị 10 mục đầu). Lưu ý: danh sách ca do AI tổng hợp từ kiến thức "
        "dược lý công khai và **chưa được bác sĩ/dược sĩ duyệt**; ca 27 trở đi được thêm sau để kiểm tra "
        "kết quả không phụ thuộc vào 26 ca đầu.",
        "",
        build_markdown_table(
            a5_summary,
            [("variant", "Cách xếp hạng"), ("documented", "Ca có dữ liệu"), ("hit@3", "Hit@3"),
             ("hit@10", "Hit@10"), ("mrr", "MRR")],
        ),
        "",
        "![A5](figures/a5_clinical_benchmark.png)",
        "",
    ]
    if default_rank_column:
        parts += ["| Ca | Phác đồ | Mong đợi | Hạng (mặc định) |", "|---|---|---|---|"]
        for row in case_rows:
            expected_title = EXPECTED_EFFECT_VIETNAMESE.get(row["expected"], row["expected"])
            rank_text = row[default_rank_column] or "không nằm trong nhóm nghiêm trọng"
            parts.append(f"| {row['case']} | {row['drugs'].replace('|', ' + ')} | {expected_title} | {rank_text} |")
        parts.append("")
    parts += [
        "**Diễn giải.** Decagon chỉ cho biết một tác dụng phụ *có* được báo cáo cho cặp thuốc hay không, "
        "không có tần suất hay mức độ, nên điểm xếp hạng không phân biệt được tác dụng phụ đặc trưng của "
        "tương tác với hàng trăm tác dụng phụ khác của cùng cặp. Khoảng một nửa số ca kinh điển lên được "
        "top 10. Thử thêm chuẩn hoá theo độ đặc trưng của thuốc (kiểu PRR) cũng không cải thiện ổn định, "
        "nên trọng số không được chỉnh theo bộ 26 ca này để tránh overfit. Hướng cải thiện: dùng số lượng "
        "báo cáo FAERS để tính disproportionality thật.",
        "",
    ]
    return parts


def _a6_section() -> list[str]:
    a6_rows = read_result_csv("a6_serious_classifier.csv")
    if not a6_rows:
        return []
    serious_labels = read_csv_file(RESOURCES_FOLDER / "serious_effects.csv")
    drug_pair_labels = [row for row in serious_labels if row["vocabulary"] == "combo"]
    serious_pair_effect_count = sum(row["serious"] == "1" for row in drug_pair_labels)
    return [
        "## A6. Phân loại tác dụng phụ nghiêm trọng",
        "",
        "Chuẩn tham chiếu: danh sách *Important Medical Event* (IME) của EMA, MedDRA 29.1 "
        f"(`resource_builders/build_serious_effects.py`). {serious_pair_effect_count} / 1317 tác dụng phụ phối hợp "
        "được gắn nghiêm trọng; ung thư, dị tật bẩm sinh, thai sản, nhiễm trùng cộng đồng và bệnh mạn tính "
        "bị loại vì trong FAERS chúng thường là chỉ định hay bệnh nền. Bảng dưới so cách dùng từ khóa với "
        "IME trên các tên khớp trực tiếp một thuật ngữ IME.",
        "",
        build_markdown_table(
            a6_rows,
            [("method", "Phương pháp"), ("names", "Số tên"), ("ime_serious", "IME nghiêm trọng"),
             ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")],
        ),
        "",
        "**Diễn giải.** Từ khóa chính xác (precision cao) nhưng chỉ bắt được một phần nhỏ (recall thấp) các "
        "tác dụng phụ IME, và bỏ sót tên thông dụng như \"heart attack\", \"gastrointestinal bleed\", "
        "\"deep vein thromboses\". Vì vậy hệ thống dùng nhãn IME làm chính; từ khóa chỉ dùng cho tên "
        "chưa ánh xạ được và để ưu tiên nhóm đe dọa tính mạng khi xếp hạng.",
        "",
    ]


def _e3_section() -> list[str]:
    e3_rows = read_result_csv("e3_llm_groundedness.csv")
    title = ["## E3. Dự đoán của LLM qua được validator", ""]
    if not e3_rows:
        return title + [
            "Chưa chạy (cần gọi API): "
            "`python -m evaluation.run_llm_groundedness_experiment --yes-call-api --regimens 10`.",
            "",
        ]
    raw_total = sum(int(row["raw_predictions"]) for row in e3_rows)
    valid_total = sum(int(row["valid_predictions"]) for row in e3_rows)
    if raw_total:
        sentence = f"{len(e3_rows)} phác đồ: giữ {valid_total}/{raw_total} dự đoán ({valid_total / raw_total:.1%})."
    else:
        sentence = "Không có dự đoán."
    return title + [sentence, ""]
