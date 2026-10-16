"""Shared loading for the evaluation scripts (no LLM calls here)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphrag.config import Settings  # noqa: E402
from graphrag.dataset import DecagonGraph  # noqa: E402
from graphrag.names import NameIndex  # noqa: E402
from graphrag.resolver import DrugResolver  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def load(hub_degree_threshold: int | None = None):
    settings = Settings.from_env()
    graph = DecagonGraph.load(
        settings.targets_file,
        settings.ppi_file,
        settings.combo_file,
        settings.cache_db,
        settings.hub_degree_threshold if hub_degree_threshold is None else hub_degree_threshold,
        tier3_one_hop_per_pair=settings.tier3_one_hop_per_pair,
        tier3_two_hop_per_pair=settings.tier3_two_hop_per_pair,
        max_evidence=settings.max_evidence,
        mono_file=settings.mono_file,
        tier3_candidates_per_pair=settings.tier3_candidates_per_pair,
    )
    names = NameIndex.load(
        settings.drug_names_file,
        settings.brand_aliases_file,
        settings.gene_names_file,
        settings.side_effects_vi_file,
        settings.serious_keywords_file,
        settings.serious_effects_file,
    )
    return settings, graph, names, DrugResolver(names, graph.drug_ids)
