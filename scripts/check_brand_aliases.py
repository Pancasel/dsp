"""Validate resources/brand_aliases_vi.csv and fill in missing STITCH IDs.

Every alias must point to a generic name that exists in resources/drug_names.csv
(as the display name or a synonym). Run after build_drug_names.py:

    python scripts/check_brand_aliases.py [--fill]
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRUG_NAMES = ROOT / "resources" / "drug_names.csv"
ALIASES = ROOT / "resources" / "brand_aliases_vi.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill", action="store_true", help="write resolved stitch_id values")
    args = parser.parse_args()

    by_name: dict[str, str] = {}
    known_ids: set[str] = set()
    with DRUG_NAMES.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            known_ids.add(row["stitch_id"])
            for name in [row["name"], *row["synonyms"].split("|")]:
                if name:
                    by_name.setdefault(name.casefold(), row["stitch_id"])

    with ALIASES.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)

    errors = 0
    for row in rows:
        expected = by_name.get(row["generic_name"].casefold())
        if expected is None:
            print(f"ERROR {row['alias']}: generic '{row['generic_name']}' is not in Decagon")
            errors += 1
            continue
        if not row["stitch_id"]:
            row["stitch_id"] = expected
        elif row["stitch_id"] not in known_ids:
            print(f"ERROR {row['alias']}: unknown stitch_id {row['stitch_id']}")
            errors += 1
        elif row["stitch_id"] != expected:
            print(
                f"ERROR {row['alias']}: stitch_id {row['stitch_id']} does not match "
                f"generic '{row['generic_name']}' ({expected})"
            )
            errors += 1

    if args.fill and errors == 0:
        with ALIASES.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"filled {ALIASES}")
    print(f"{len(rows)} aliases checked, {errors} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
