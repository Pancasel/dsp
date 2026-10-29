"""Write the small CSV files that tests build their graphs from."""

import csv
from pathlib import Path


DRUG_TARGETS_HEADER = ["STITCH", "Gene"]
PROTEIN_INTERACTIONS_HEADER = ["Gene 1", "Gene 2"]
DRUG_PAIR_SIDE_EFFECTS_HEADER = ["STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"]
SINGLE_DRUG_SIDE_EFFECTS_HEADER = ["STITCH", "Individual Side Effect", "Side Effect Name"]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
