"""Write experiment results to evaluation/results/*.csv and read them back (for the reports)."""

import csv
from pathlib import Path

from evaluation.shared.system_loading import RESULTS_FOLDER, TEST_CASES_FOLDER


def write_result_csv(file_name: str, rows: list[dict]) -> None:
    """Columns are the keys of the first row, in order."""
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    path = RESULTS_FOLDER / file_name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


def read_csv_file(path: Path) -> list[dict[str, str]]:
    """All rows, or an empty list when the file does not exist yet."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_result_csv(file_name: str) -> list[dict[str, str]]:
    return read_csv_file(RESULTS_FOLDER / file_name)


def read_test_cases(file_name: str) -> list[dict[str, str]]:
    return read_csv_file(TEST_CASES_FOLDER / file_name)
