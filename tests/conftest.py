import zipfile
from pathlib import Path

import pytest

from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from tests.csv_test_data import (
    DRUG_PAIR_SIDE_EFFECTS_HEADER,
    DRUG_TARGETS_HEADER,
    PROTEIN_INTERACTIONS_HEADER,
    write_csv,
)


@pytest.fixture
def small_graph(tmp_path: Path) -> DecagonGraph:
    """D1, D2, D3 all target SHARED; HUB (4 partners) is above the hub threshold of 2.

    D1-D2 has a documented effect; D3-D4 too, and D4 has no targets.
    The drug pair file is read from a .zip archive.
    """
    write_csv(
        tmp_path / "targets.csv",
        DRUG_TARGETS_HEADER,
        [
            ["D1", "G1"],
            ["D1", "SHARED"],
            ["D2", "G2"],
            ["D2", "SHARED"],
            ["D3", "G3"],
            ["D3", "SHARED"],
        ],
    )
    write_csv(
        tmp_path / "ppi.csv",
        PROTEIN_INTERACTIONS_HEADER,
        [
            ["G1", "MID"],
            ["MID", "G2"],
            ["G1", "HUB"],
            ["HUB", "G3"],
            ["SHARED", "HUB"],
            ["HUB", "P1"],
            ["G1", "LOW"],
            ["LOW", "G3"],
        ],
    )
    pair_effects_csv = tmp_path / "combo.csv"
    write_csv(
        pair_effects_csv,
        DRUG_PAIR_SIDE_EFFECTS_HEADER,
        [
            ["D1", "D2", "C1", "known event"],
            ["D3", "D4", "C2", "combo-only event"],
        ],
    )
    pair_effects_zip = tmp_path / "combo.zip"
    with zipfile.ZipFile(pair_effects_zip, "w") as archive:
        archive.write(pair_effects_csv, "combo/bio-decagon-combo.csv")
    return DecagonGraph.load(
        drug_targets_file=tmp_path / "targets.csv",
        protein_interactions_file=tmp_path / "ppi.csv",
        drug_pair_side_effects_file=pair_effects_zip,
        cache_database_file=tmp_path / "cache.sqlite3",
        hub_degree_threshold=2,
    )
