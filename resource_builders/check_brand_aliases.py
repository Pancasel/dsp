"""Validate resources/brand_aliases_vietnamese.csv and fill in missing STITCH IDs.

Every brand must point to a generic name that exists in resources/drug_names.csv
(as the display name or a synonym). Run after build_drug_names:

    python -m resource_builders.check_brand_aliases [--fill]
"""

import argparse
import csv
import sys

from graphrag.settings.environment_variable_readers import PROJECT_ROOT


DRUG_NAMES_FILE = PROJECT_ROOT / "resources" / "drug_names.csv"
BRAND_ALIASES_FILE = PROJECT_ROOT / "resources" / "brand_aliases_vietnamese.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill", action="store_true", help="write resolved stitch_id values")
    arguments = parser.parse_args()

    stitch_id_by_lowercase_name: dict[str, str] = {}
    known_stitch_ids: set[str] = set()
    with DRUG_NAMES_FILE.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            known_stitch_ids.add(row["stitch_id"])
            for name in [row["name"], *row["synonyms"].split("|")]:
                if name:
                    stitch_id_by_lowercase_name.setdefault(name.casefold(), row["stitch_id"])

    with BRAND_ALIASES_FILE.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        column_names = reader.fieldnames or []
        rows = list(reader)

    error_count = 0
    for row in rows:
        expected_stitch_id = stitch_id_by_lowercase_name.get(row["generic_name"].casefold())
        if expected_stitch_id is None:
            print(f"ERROR {row['alias']}: generic '{row['generic_name']}' is not in Decagon")
            error_count += 1
            continue
        if not row["stitch_id"]:
            row["stitch_id"] = expected_stitch_id
        elif row["stitch_id"] not in known_stitch_ids:
            print(f"ERROR {row['alias']}: unknown stitch_id {row['stitch_id']}")
            error_count += 1
        elif row["stitch_id"] != expected_stitch_id:
            print(
                f"ERROR {row['alias']}: stitch_id {row['stitch_id']} does not match "
                f"generic '{row['generic_name']}' ({expected_stitch_id})"
            )
            error_count += 1

    if arguments.fill and error_count == 0:
        with BRAND_ALIASES_FILE.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=column_names)
            writer.writeheader()
            writer.writerows(rows)
        print(f"filled {BRAND_ALIASES_FILE}")
    print(f"{len(rows)} aliases checked, {error_count} errors")
    return 1 if error_count else 0


if __name__ == "__main__":
    sys.exit(main())
