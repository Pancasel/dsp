"""Read CSV files: the large Decagon files (plain or inside a .zip) and the small resource files."""

import csv
import io
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


@contextmanager
def open_csv_as_text(path: Path) -> Iterator[TextIO]:
    """Open a .csv file, or the CSV inside a .zip archive, as a text stream."""
    if path.suffix.lower() != ".zip":
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")
        stream = path.open("r", encoding="utf-8-sig", newline="")
        try:
            yield stream
        finally:
            stream.close()
        return

    if not path.is_file():
        raise FileNotFoundError(f"Dataset archive does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        csv_member_names = [
            member_name
            for member_name in archive.namelist()
            if member_name.lower().endswith(".csv")
            # "._name" files are macOS metadata, not data.
            and not Path(member_name).name.startswith("._")
        ]
        if not csv_member_names:
            raise ValueError(f"No CSV file found in dataset archive: {path}")
        # Prefer the CSV named like the archive ("combo.zip" -> "combo.csv").
        chosen_member_name = csv_member_names[0]
        for member_name in csv_member_names:
            if Path(member_name).stem.lower() == path.stem.lower():
                chosen_member_name = member_name
                break
        with archive.open(chosen_member_name) as binary_stream:
            text_stream = io.TextIOWrapper(binary_stream, encoding="utf-8-sig", newline="")
            try:
                yield text_stream
            finally:
                text_stream.detach()


def read_csv_rows_with_required_columns(
    path: Path, required_columns: tuple[str, ...]
) -> Iterator[dict[str, str]]:
    """Yield each row with only the required columns, values stripped of spaces.

    Raises ValueError when the header is missing or lacks a required column.
    """
    with open_csv_as_text(path) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset is missing a CSV header: {path}")
        missing_columns = set(required_columns) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Dataset {path} is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        for row in reader:
            yield {column: (row.get(column) or "").strip() for column in required_columns}


def read_all_rows_if_file_exists(path: Path | None) -> list[dict[str, str]]:
    """All rows of a small optional resource file; an empty list when it is missing."""
    if path is None or not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))
