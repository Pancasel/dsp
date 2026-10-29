"""Load the graph, names and resolver the same way the app does, and the folders experiments use."""

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

from graphrag.drug_names.drug_name_resolver import DrugResolver
from graphrag.drug_names.name_index import NameIndex
from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from graphrag.settings.application_settings import Settings


EVALUATION_FOLDER = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVALUATION_FOLDER.parent
RESULTS_FOLDER = EVALUATION_FOLDER / "results"
FIGURES_FOLDER = RESULTS_FOLDER / "figures"
TEST_CASES_FOLDER = EVALUATION_FOLDER / "test_cases"
RESOURCES_FOLDER = PROJECT_ROOT / "resources"


@dataclass
class EvaluationSystem:
    settings: Settings
    graph: DecagonGraph
    names: NameIndex
    resolver: DrugResolver


def disable_llm_calls() -> None:
    """Offline experiments must never spend API credit: an empty key disables the LLM."""
    os.environ["LLM_API_KEY"] = ""


def load_evaluation_system(hub_degree_threshold: int | None = None) -> EvaluationSystem:
    """hub_degree_threshold replaces the value from .env (used by experiment E5)."""
    settings = Settings.from_environment()
    if hub_degree_threshold is not None:
        settings = dataclasses.replace(settings, hub_degree_threshold=hub_degree_threshold)
    graph = DecagonGraph.load_from_settings(settings)
    names = NameIndex.load(
        drug_names_file=settings.drug_names_file,
        brand_aliases_file=settings.brand_aliases_file,
        gene_names_file=settings.gene_names_file,
        side_effect_translations_file=settings.side_effect_translations_file,
        serious_keywords_file=settings.serious_keywords_file,
        serious_effects_file=settings.serious_effects_file,
    )
    return EvaluationSystem(settings, graph, names, DrugResolver(names, graph.drug_ids))
