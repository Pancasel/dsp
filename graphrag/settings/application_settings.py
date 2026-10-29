"""All settings of the application, read once from .env (see .env.example)."""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from graphrag.settings.environment_variable_readers import (
    PROJECT_ROOT,
    read_decimal_number,
    read_integer_at_least,
    read_optional_path,
    read_path,
    read_percentage,
    read_text,
    read_true_or_false,
)
from graphrag.settings.ranking_weights import RankingWeights


@dataclass(frozen=True, kw_only=True)
class Settings:
    # Decagon source files (see docs/THUAT_NGU.md for what each one holds).
    drug_targets_file: Path
    protein_interactions_file: Path
    drug_pair_side_effects_file: Path
    single_drug_side_effects_file: Path | None = None
    # SQLite copy of the drug pair side effects, built on the first run.
    cache_database_file: Path

    # Name and translation files in resources/. None disables the file.
    drug_names_file: Path | None = None
    brand_aliases_file: Path | None = None
    gene_names_file: Path | None = None
    side_effect_translations_file: Path | None = None
    serious_keywords_file: Path | None = None
    serious_effects_file: Path | None = None

    # Proteins with more interaction partners than this are "hubs" and are not
    # used as a step in a protein path.
    hub_degree_threshold: int
    # 0 keeps hub_degree_threshold; otherwise the threshold becomes this
    # percentile (0-100) of the number of interaction partners of all proteins.
    hub_degree_percentile: float = 0.0
    # Also count proteins one interaction step away from the targets of >= 3
    # drugs as Tier 2 amplification, not only proteins all three target directly.
    count_convergence_proteins_as_amplification: bool = False
    tier3_one_hop_bridges_per_pair: int
    tier3_two_hop_bridges_per_pair: int
    tier3_candidate_effects_per_pair: int = 20
    # Total number of evidence items sent to the LLM for one regimen.
    max_evidence_items: int

    # Shown per analysis: the most dangerous effects, plus a few others.
    max_serious_effects: int = 10
    max_other_effects: int = 5
    ranking_weights: RankingWeights = RankingWeights()

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_timeout_seconds: float = 120.0

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        return cls(
            drug_targets_file=read_path(
                "DRUG_TARGETS_FILE", "../bio-decagon-targets/bio-decagon-targets.csv"
            ),
            protein_interactions_file=read_path(
                "PROTEIN_INTERACTIONS_FILE", "../bio-decagon-ppi/bio-decagon-ppi.csv"
            ),
            drug_pair_side_effects_file=read_path(
                "DRUG_PAIR_SIDE_EFFECTS_FILE", "../bio-decagon-combo/bio-decagon-combo.csv"
            ),
            single_drug_side_effects_file=read_optional_path(
                "SINGLE_DRUG_SIDE_EFFECTS_FILE", "../bio-decagon-mono/bio-decagon-mono.csv"
            ),
            cache_database_file=read_path("CACHE_DATABASE_FILE", ".cache/decagon.sqlite3"),
            drug_names_file=read_optional_path("DRUG_NAMES_FILE", "resources/drug_names.csv"),
            brand_aliases_file=read_optional_path(
                "BRAND_ALIASES_FILE", "resources/brand_aliases_vietnamese.csv"
            ),
            gene_names_file=read_optional_path("GENE_NAMES_FILE", "resources/gene_names.csv"),
            side_effect_translations_file=read_optional_path(
                "SIDE_EFFECT_TRANSLATIONS_FILE",
                "resources/side_effect_translations_vietnamese.csv",
            ),
            serious_keywords_file=read_optional_path(
                "SERIOUS_KEYWORDS_FILE", "resources/serious_keywords.csv"
            ),
            serious_effects_file=read_optional_path(
                "SERIOUS_EFFECTS_FILE", "resources/serious_effects.csv"
            ),
            hub_degree_threshold=read_integer_at_least("HUB_DEGREE_THRESHOLD", "500", minimum=0),
            hub_degree_percentile=read_percentage("HUB_DEGREE_PERCENTILE"),
            count_convergence_proteins_as_amplification=read_true_or_false(
                "COUNT_CONVERGENCE_PROTEINS_AS_AMPLIFICATION", "0"
            ),
            tier3_one_hop_bridges_per_pair=read_integer_at_least(
                "TIER3_ONE_HOP_BRIDGES_PER_PAIR", "8", minimum=0
            ),
            tier3_two_hop_bridges_per_pair=read_integer_at_least(
                "TIER3_TWO_HOP_BRIDGES_PER_PAIR", "4", minimum=0
            ),
            tier3_candidate_effects_per_pair=read_integer_at_least(
                "TIER3_CANDIDATE_EFFECTS_PER_PAIR", "20", minimum=1
            ),
            max_evidence_items=read_integer_at_least("MAX_EVIDENCE_ITEMS", "400", minimum=1),
            max_serious_effects=read_integer_at_least("MAX_SERIOUS_EFFECTS", "10", minimum=1),
            max_other_effects=read_integer_at_least("MAX_OTHER_EFFECTS", "5", minimum=0),
            ranking_weights=RankingWeights(
                recurrence=read_decimal_number("RANKING_WEIGHT_RECURRENCE", "1.0"),
                effect_rarity=read_decimal_number("RANKING_WEIGHT_EFFECT_RARITY", "0.5"),
                absent_from_single_drug_data=read_decimal_number(
                    "RANKING_WEIGHT_ABSENT_FROM_SINGLE_DRUG_DATA", "1.0"
                ),
                serious=read_decimal_number("RANKING_WEIGHT_SERIOUS", "2.0"),
                amplified=read_decimal_number("RANKING_WEIGHT_AMPLIFIED", "1.0"),
                critical=read_decimal_number("RANKING_WEIGHT_CRITICAL", "2.0"),
            ),
            llm_api_key=read_text("LLM_API_KEY", ""),
            llm_base_url=read_text("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            llm_model=read_text("LLM_MODEL", "google/gemini-2.5-flash"),
            llm_timeout_seconds=read_decimal_number("LLM_TIMEOUT_SECONDS", "120"),
        )
