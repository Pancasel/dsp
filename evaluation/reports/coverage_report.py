"""Write evaluation/results/c_coverage.csv and COVERAGE.md from the C1-C3 check rows."""

from evaluation.shared.system_loading import RESULTS_FOLDER


MAXIMUM_FAILURES_LISTED = 50


def write_coverage_report(rows: list[dict], failures: list[str], run_details: dict) -> None:
    """rows: {"check", "ok", "total"}; run_details: regimens, min_times_per_drug, seconds_per_regimen."""
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["rate"] = round(row["ok"] / row["total"], 4) if row["total"] else "n/a"
    with open(RESULTS_FOLDER / "c_coverage.csv", "w", encoding="utf-8", newline="") as stream:
        stream.write("check,ok,total,rate\n")
        for row in rows:
            stream.write(f"\"{row['check']}\",{row['ok']},{row['total']},{row['rate']}\n")
    lines = [
        "# Độ phủ trên toàn bộ thuốc trong dữ liệu",
        "",
        "Sinh tự động bởi `python -m evaluation.run_full_coverage_check` (seed 0, không gọi LLM). "
        "Mọi kiểm tra đối chiếu với file CSV gốc, đọc độc lập với cache SQLite.",
        "",
        f"C3 chạy {run_details['regimens']} phác đồ phủ mọi thuốc; mỗi thuốc xuất hiện ít nhất "
        f"{run_details['min_times_per_drug']} lần; trung bình {run_details['seconds_per_regimen']} s/phác đồ.",
        "",
        "| Kiểm tra | Đạt | Tổng | Tỷ lệ |",
        "|---|---|---|---|",
    ]
    for row in rows:
        rate = f"{row['rate'] * 100:.1f}%" if row["rate"] != "n/a" else "n/a"
        lines.append(f"| {row['check']} | {row['ok']} | {row['total']} | {rate} |")
    lines += ["", f"Số lỗi chi tiết: {len(failures)}."]
    if failures:
        lines += ["", "```", *failures[:MAXIMUM_FAILURES_LISTED], "```"]
    (RESULTS_FOLDER / "COVERAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
