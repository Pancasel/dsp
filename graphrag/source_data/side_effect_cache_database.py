"""SQLite copy of the Decagon side-effect files, so that lookups do not re-read 4.6 million CSV rows.

Tables:
- drug_pair_side_effects (drug_a, drug_b, effect_code, effect_name), drug_a < drug_b
- single_drug_side_effects (drug, effect_code, effect_name)
- effect_pair_counts (effect_code, effect_name, pair_count): pairs recording each effect
- statistics (key, value): "total_pairs" = number of distinct drug pairs
- metadata (signature): fingerprint of the source files; a different one means rebuild
"""

import hashlib
import sqlite3
from pathlib import Path

from graphrag.source_data.csv_file_reader import read_csv_rows_with_required_columns


# Bump when the tables change so that an old cache is rebuilt.
CACHE_SCHEMA_VERSION = "3"
# Rows written to SQLite per executemany() call while building the cache.
INSERT_BATCH_SIZE = 10_000


def file_fingerprint(path: Path) -> str:
    """Changes when the file is replaced or edited (path, size, modification time)."""
    file_status = path.stat()
    value = f"{path.resolve()}|{file_status.st_size}|{file_status.st_mtime_ns}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_signature(
    drug_pair_side_effects_file: Path, single_drug_side_effects_file: Path | None
) -> str:
    parts = [CACHE_SCHEMA_VERSION, file_fingerprint(drug_pair_side_effects_file)]
    if single_drug_side_effects_file is not None:
        parts.append(file_fingerprint(single_drug_side_effects_file))
    else:
        parts.append("no-single-drug-file")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def cache_is_up_to_date(
    drug_pair_side_effects_file: Path,
    database_file: Path,
    single_drug_side_effects_file: Path | None = None,
) -> bool:
    if not database_file.is_file():
        return False
    try:
        with sqlite3.connect(database_file) as connection:
            row = connection.execute("SELECT signature FROM metadata").fetchone()
    except sqlite3.DatabaseError:
        return False
    expected_signature = cache_signature(
        drug_pair_side_effects_file, single_drug_side_effects_file
    )
    return row is not None and row[0] == expected_signature


def build_cache_database(
    drug_pair_side_effects_file: Path,
    database_file: Path,
    single_drug_side_effects_file: Path | None = None,
) -> None:
    """Write the cache to a temporary file first, then move it into place."""
    database_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = database_file.with_name(database_file.name + ".tmp")
    temporary_file.unlink(missing_ok=True)
    signature = cache_signature(drug_pair_side_effects_file, single_drug_side_effects_file)
    connection = sqlite3.connect(temporary_file)
    try:
        # The file is rebuilt from scratch on any failure, so durability is not needed.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE metadata (signature TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES (?)", (signature,))
        _write_drug_pair_side_effects(connection, drug_pair_side_effects_file)
        _write_single_drug_side_effects(connection, single_drug_side_effects_file)
        connection.commit()
    except Exception:
        connection.close()
        temporary_file.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    temporary_file.replace(database_file)


def _write_drug_pair_side_effects(connection: sqlite3.Connection, source_file: Path) -> None:
    connection.execute(
        """
        CREATE TABLE drug_pair_side_effects (
            drug_a TEXT NOT NULL,
            drug_b TEXT NOT NULL,
            effect_code TEXT NOT NULL,
            effect_name TEXT NOT NULL,
            PRIMARY KEY (drug_a, drug_b, effect_code)
        ) WITHOUT ROWID
        """
    )
    insert_statement = """
        INSERT OR IGNORE INTO drug_pair_side_effects
        (drug_a, drug_b, effect_code, effect_name) VALUES (?, ?, ?, ?)
    """
    pending_rows: list[tuple[str, str, str, str]] = []
    source_rows = read_csv_rows_with_required_columns(
        source_file,
        ("STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"),
    )
    for row in source_rows:
        drug_a, drug_b = sorted((row["STITCH 1"], row["STITCH 2"]))
        effect_code = row["Polypharmacy Side Effect"]
        if drug_a and drug_b and effect_code:
            effect_name = row["Side Effect Name"] or effect_code
            pending_rows.append((drug_a, drug_b, effect_code, effect_name))
        if len(pending_rows) >= INSERT_BATCH_SIZE:
            connection.executemany(insert_statement, pending_rows)
            pending_rows.clear()
    if pending_rows:
        connection.executemany(insert_statement, pending_rows)

    # The primary key already serves lookups by drug_a; this index serves
    # lookups by drug_b (Tier 3 candidate effects of one drug).
    connection.execute(
        "CREATE INDEX drug_pair_side_effects_drug_b ON drug_pair_side_effects (drug_b)"
    )
    # Number of drug pairs that record each effect, for the effect rarity score.
    connection.execute(
        """
        CREATE TABLE effect_pair_counts AS
        SELECT effect_code, MIN(effect_name) AS effect_name, COUNT(*) AS pair_count
        FROM drug_pair_side_effects GROUP BY effect_code
        """
    )
    connection.execute("CREATE TABLE statistics (key TEXT PRIMARY KEY, value INTEGER)")
    connection.execute(
        """
        INSERT INTO statistics
        SELECT 'total_pairs', COUNT(*) FROM
        (SELECT DISTINCT drug_a, drug_b FROM drug_pair_side_effects)
        """
    )


def _write_single_drug_side_effects(
    connection: sqlite3.Connection, source_file: Path | None
) -> None:
    """The table always exists; it stays empty when there is no source file."""
    connection.execute(
        """
        CREATE TABLE single_drug_side_effects (
            drug TEXT NOT NULL,
            effect_code TEXT NOT NULL,
            effect_name TEXT NOT NULL,
            PRIMARY KEY (drug, effect_code)
        ) WITHOUT ROWID
        """
    )
    if source_file is None:
        return
    rows_to_insert = []
    source_rows = read_csv_rows_with_required_columns(
        source_file, ("STITCH", "Individual Side Effect", "Side Effect Name")
    )
    for row in source_rows:
        if row["STITCH"] and row["Individual Side Effect"]:
            rows_to_insert.append(
                (row["STITCH"], row["Individual Side Effect"], row["Side Effect Name"])
            )
    connection.executemany(
        "INSERT OR IGNORE INTO single_drug_side_effects VALUES (?, ?, ?)", rows_to_insert
    )
