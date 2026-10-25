import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: str) -> Path:
    path = Path(os.environ.get(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_path_from_env(name: str, default: str) -> Path | None:
    """Like _path_from_env, but an explicitly empty value disables the file."""
    value = os.environ.get(name, default)
    if not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _bounded_int(name: str, default: str, *, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be {minimum} or greater")
    return value


def _float(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _percentile(name: str) -> float:
    value = _float(name, "0")
    if not 0 <= value <= 100:
        raise ValueError(f"{name} must be between 0 and 100")
    return value


@dataclass(frozen=True)
class RankingWeights:
    """Weights of the regimen-level side-effect score (see app/ranking.py)."""

    recurrence: float = 1.0
    idf: float = 0.5
    not_in_mono: float = 1.0
    serious: float = 2.0
    amplified: float = 1.0
    critical: float = 2.0


@dataclass(frozen=True)
class Settings:
    targets_file: Path
    ppi_file: Path
    combo_file: Path
    cache_db: Path
    hub_degree_threshold: int
    tier3_one_hop_per_pair: int
    tier3_two_hop_per_pair: int
    max_evidence: int
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    mono_file: Path | None = None
    drug_names_file: Path | None = None
    brand_aliases_file: Path | None = None
    gene_names_file: Path | None = None
    side_effects_vi_file: Path | None = None
    serious_keywords_file: Path | None = None
    serious_effects_file: Path | None = None
    llm_timeout: float = 120.0
    # Shown per analysis: the most dangerous effects, plus a few others.
    max_serious_effects: int = 10
    max_other_effects: int = 5
    tier3_candidates_per_pair: int = 20
    # 0 keeps the absolute hub_degree_threshold; otherwise the threshold is
    # that percentile (0-100) of the PPI degree distribution.
    hub_degree_percentile: float = 0.0
    # Count proteins within one PPI hop of >= 3 drugs' targets as Tier 2
    # amplification, not only proteins all three target directly.
    convergence_amplification: bool = False
    ranking: RankingWeights = RankingWeights()

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        llm_model = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")
        return cls(
            targets_file=_path_from_env(
                "TARGETS_FILE", "../bio-decagon-targets/bio-decagon-targets.csv"
            ),
            ppi_file=_path_from_env("PPI_FILE", "../bio-decagon-ppi/bio-decagon-ppi.csv"),
            combo_file=_path_from_env(
                "COMBO_FILE", "../bio-decagon-combo/bio-decagon-combo.csv"
            ),
            cache_db=_path_from_env("CACHE_DB", ".cache/decagon.sqlite3"),
            hub_degree_threshold=_bounded_int(
                "HUB_DEGREE_THRESHOLD", "500", minimum=0
            ),
            tier3_one_hop_per_pair=_bounded_int(
                "TIER3_ONE_HOP_PER_PAIR", "8", minimum=0
            ),
            tier3_two_hop_per_pair=_bounded_int(
                "TIER3_TWO_HOP_PER_PAIR", "4", minimum=0
            ),
            max_evidence=_bounded_int("MAX_EVIDENCE", "400", minimum=1),
            llm_api_key=os.environ.get("LLM_API_KEY", ""),
            llm_base_url=os.environ.get(
                "LLM_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            llm_model=llm_model,
            mono_file=_optional_path_from_env(
                "MONO_FILE", "../bio-decagon-mono/bio-decagon-mono.csv"
            ),
            drug_names_file=_optional_path_from_env(
                "DRUG_NAMES_FILE", "resources/drug_names.csv"
            ),
            brand_aliases_file=_optional_path_from_env(
                "BRAND_ALIASES_FILE", "resources/brand_aliases_vi.csv"
            ),
            gene_names_file=_optional_path_from_env(
                "GENE_NAMES_FILE", "resources/gene_names.csv"
            ),
            side_effects_vi_file=_optional_path_from_env(
                "SIDE_EFFECTS_VI_FILE", "resources/side_effects_vi.csv"
            ),
            serious_keywords_file=_optional_path_from_env(
                "SERIOUS_KEYWORDS_FILE", "resources/serious_keywords.csv"
            ),
            serious_effects_file=_optional_path_from_env(
                "SERIOUS_EFFECTS_FILE", "resources/serious_effects.csv"
            ),
            llm_timeout=_float("LLM_TIMEOUT", "120"),
            max_serious_effects=_bounded_int("MAX_SERIOUS_EFFECTS", "10", minimum=1),
            max_other_effects=_bounded_int("MAX_OTHER_EFFECTS", "5", minimum=0),
            tier3_candidates_per_pair=_bounded_int(
                "TIER3_CANDIDATES_PER_PAIR", "20", minimum=1
            ),
            hub_degree_percentile=_percentile("HUB_DEGREE_PERCENTILE"),
            convergence_amplification=os.environ.get(
                "CONVERGENCE_AMPLIFICATION", "0"
            ).strip().lower() in {"1", "true", "yes"},
            ranking=RankingWeights(
                recurrence=_float("RANK_W_RECURRENCE", "1.0"),
                idf=_float("RANK_W_IDF", "0.5"),
                not_in_mono=_float("RANK_W_NOT_IN_MONO", "1.0"),
                serious=_float("RANK_W_SERIOUS", "2.0"),
                amplified=_float("RANK_W_AMPLIFIED", "1.0"),
                critical=_float("RANK_W_CRITICAL", "2.0"),
            ),
        )
