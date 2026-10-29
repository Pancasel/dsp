"""The raw Decagon CSV files read in full, independently of the SQLite cache (used by C1-C3)."""

import collections
from dataclasses import dataclass

from graphrag.settings.application_settings import Settings
from graphrag.source_data.csv_file_reader import read_csv_rows_with_required_columns


# A drug name that is still an ID ("CID0..."), a formula with brackets or many
# digits, or longer than 40 characters is not readable for a user.
UNREADABLE_NAME_PATTERN = r"^CID\d|^Cid \d|[\[\]]|\d.*\d.*\d|^.{41,}$"


@dataclass
class RawSourceData:
    # Sorted drug pair -> effect codes.
    pair_effect_codes: dict[tuple[str, str], set[str]]
    # Drug -> single-drug effect codes.
    single_drug_effect_codes: dict[str, set[str]]
    # Drug -> target proteins.
    targets_by_drug: dict[str, set[str]]
    # (protein, protein) in both directions.
    interacting_protein_pairs: set[tuple[str, str]]


def read_raw_source_data(settings: Settings) -> RawSourceData:
    pair_effect_codes: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    rows = read_csv_rows_with_required_columns(
        settings.drug_pair_side_effects_file, ("STITCH 1", "STITCH 2", "Polypharmacy Side Effect")
    )
    for row in rows:
        pair = tuple(sorted((row["STITCH 1"], row["STITCH 2"])))
        pair_effect_codes[pair].add(row["Polypharmacy Side Effect"])
    single_drug_effect_codes: dict[str, set[str]] = collections.defaultdict(set)
    if settings.single_drug_side_effects_file:
        rows = read_csv_rows_with_required_columns(
            settings.single_drug_side_effects_file, ("STITCH", "Individual Side Effect")
        )
        for row in rows:
            single_drug_effect_codes[row["STITCH"]].add(row["Individual Side Effect"])
    targets_by_drug: dict[str, set[str]] = collections.defaultdict(set)
    for row in read_csv_rows_with_required_columns(settings.drug_targets_file, ("STITCH", "Gene")):
        targets_by_drug[row["STITCH"]].add(row["Gene"])
    interacting_protein_pairs: set[tuple[str, str]] = set()
    for row in read_csv_rows_with_required_columns(settings.protein_interactions_file, ("Gene 1", "Gene 2")):
        interacting_protein_pairs.add((row["Gene 1"], row["Gene 2"]))
        interacting_protein_pairs.add((row["Gene 2"], row["Gene 1"]))
    return RawSourceData(pair_effect_codes, single_drug_effect_codes, targets_by_drug, interacting_protein_pairs)
