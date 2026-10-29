"""The Decagon graph in memory: drug -> target proteins, protein <-> protein, and side-effect lookups.

Drug pair and single-drug side effects stay in the SQLite cache
(graphrag/source_data/side_effect_cache_database.py) and are read on demand.
The algorithms that walk the graph live in the other files of this folder.
"""

import sqlite3
from pathlib import Path

from graphrag.knowledge_graph.hub_protein_filter import hub_threshold_from_percentile
from graphrag.settings.application_settings import Settings
from graphrag.source_data.csv_file_reader import read_csv_rows_with_required_columns
from graphrag.source_data.data_records import SideEffect
from graphrag.source_data.side_effect_cache_database import (
    build_cache_database,
    cache_is_up_to_date,
)


class DecagonGraph:
    def __init__(
        self,
        *,
        target_proteins_by_drug: dict[str, set[str]],
        protein_interactions: dict[str, set[str]],
        cache_database_file: Path,
        hub_degree_threshold: int,
        drug_ids: set[str],
        canonical_side_effect_names: dict[str, str],
        tier3_one_hop_bridges_per_pair: int,
        tier3_two_hop_bridges_per_pair: int,
        max_evidence_items: int,
        pair_count_by_effect_code: dict[str, int] | None = None,
        total_pair_count: int = 0,
        tier3_candidate_effects_per_pair: int = 20,
        count_convergence_proteins_as_amplification: bool = False,
    ) -> None:
        # Drug STITCH ID -> Entrez gene IDs of the proteins it acts on.
        self.target_proteins_by_drug = target_proteins_by_drug
        # Protein -> proteins it interacts with (both directions are stored).
        self.protein_interactions = protein_interactions
        self.cache_database_file = cache_database_file
        self.hub_degree_threshold = hub_degree_threshold
        # Every drug that has targets or appears in a drug pair.
        self.drug_ids = drug_ids
        # Lower-case side-effect name -> the name exactly as Decagon writes it.
        self.canonical_side_effect_names = canonical_side_effect_names
        self.tier3_one_hop_bridges_per_pair = tier3_one_hop_bridges_per_pair
        self.tier3_two_hop_bridges_per_pair = tier3_two_hop_bridges_per_pair
        self.max_evidence_items = max_evidence_items
        self.pair_count_by_effect_code = pair_count_by_effect_code or {}
        self.total_pair_count = total_pair_count
        self.tier3_candidate_effects_per_pair = tier3_candidate_effects_per_pair
        self.count_convergence_proteins_as_amplification = (
            count_convergence_proteins_as_amplification
        )
        self._single_drug_effects_cache: dict[str, list[SideEffect]] = {}
        self._single_drug_effect_drug_counts: dict[str, int] | None = None
        self._partner_counts_cache: dict[str, dict[str, tuple[str, int]]] = {}
        self._sorted_partners_cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def load_from_settings(cls, settings: Settings) -> "DecagonGraph":
        single_drug_file = settings.single_drug_side_effects_file
        if single_drug_file is not None and not single_drug_file.exists():
            single_drug_file = None
        return cls.load(
            drug_targets_file=settings.drug_targets_file,
            protein_interactions_file=settings.protein_interactions_file,
            drug_pair_side_effects_file=settings.drug_pair_side_effects_file,
            cache_database_file=settings.cache_database_file,
            hub_degree_threshold=settings.hub_degree_threshold,
            tier3_one_hop_bridges_per_pair=settings.tier3_one_hop_bridges_per_pair,
            tier3_two_hop_bridges_per_pair=settings.tier3_two_hop_bridges_per_pair,
            max_evidence_items=settings.max_evidence_items,
            single_drug_side_effects_file=single_drug_file,
            tier3_candidate_effects_per_pair=settings.tier3_candidate_effects_per_pair,
            hub_degree_percentile=settings.hub_degree_percentile,
            count_convergence_proteins_as_amplification=(
                settings.count_convergence_proteins_as_amplification
            ),
        )

    @classmethod
    def load(
        cls,
        *,
        drug_targets_file: Path,
        protein_interactions_file: Path,
        drug_pair_side_effects_file: Path,
        cache_database_file: Path,
        hub_degree_threshold: int,
        tier3_one_hop_bridges_per_pair: int = 8,
        tier3_two_hop_bridges_per_pair: int = 4,
        max_evidence_items: int = 400,
        single_drug_side_effects_file: Path | None = None,
        tier3_candidate_effects_per_pair: int = 20,
        hub_degree_percentile: float = 0.0,
        count_convergence_proteins_as_amplification: bool = False,
    ) -> "DecagonGraph":
        target_proteins_by_drug = _read_drug_targets(drug_targets_file)
        protein_interactions = _read_protein_interactions(protein_interactions_file)
        if hub_degree_percentile > 0:
            hub_degree_threshold = hub_threshold_from_percentile(
                protein_interactions, hub_degree_percentile
            )

        if not cache_is_up_to_date(
            drug_pair_side_effects_file, cache_database_file, single_drug_side_effects_file
        ):
            build_cache_database(
                drug_pair_side_effects_file, cache_database_file, single_drug_side_effects_file
            )
        with sqlite3.connect(cache_database_file) as connection:
            drugs_in_pairs = set()
            for (drug,) in connection.execute(
                """
                SELECT drug_a FROM drug_pair_side_effects
                UNION
                SELECT drug_b FROM drug_pair_side_effects
                """
            ):
                drugs_in_pairs.add(drug)
            canonical_side_effect_names = {}
            for (effect_name,) in connection.execute("SELECT effect_name FROM effect_pair_counts"):
                if effect_name:
                    canonical_side_effect_names[effect_name.casefold()] = effect_name
            pair_count_by_effect_code = dict(
                connection.execute("SELECT effect_code, pair_count FROM effect_pair_counts")
            )
            total_pairs_row = connection.execute(
                "SELECT value FROM statistics WHERE key = 'total_pairs'"
            ).fetchone()
        return cls(
            target_proteins_by_drug=target_proteins_by_drug,
            protein_interactions=protein_interactions,
            cache_database_file=cache_database_file,
            hub_degree_threshold=hub_degree_threshold,
            drug_ids=set(target_proteins_by_drug) | drugs_in_pairs,
            canonical_side_effect_names=canonical_side_effect_names,
            tier3_one_hop_bridges_per_pair=tier3_one_hop_bridges_per_pair,
            tier3_two_hop_bridges_per_pair=tier3_two_hop_bridges_per_pair,
            max_evidence_items=max_evidence_items,
            pair_count_by_effect_code=pair_count_by_effect_code,
            total_pair_count=total_pairs_row[0] if total_pairs_row else 0,
            tier3_candidate_effects_per_pair=tier3_candidate_effects_per_pair,
            count_convergence_proteins_as_amplification=(
                count_convergence_proteins_as_amplification
            ),
        )

    def target_proteins(self, drug: str) -> set[str]:
        return self.target_proteins_by_drug.get(drug, set())

    def interaction_partners(self, protein: str) -> set[str]:
        return self.protein_interactions.get(protein, set())

    def interaction_partner_count(self, protein: str) -> int:
        """Number of interacting proteins, called the "degree" of the protein."""
        return len(self.protein_interactions.get(protein, ()))

    def sorted_interaction_partners(self, protein: str) -> tuple[str, ...]:
        """Sorted interacting proteins, cached because the bridge search revisits them."""
        partners = self._sorted_partners_cache.get(protein)
        if partners is None:
            partners = tuple(sorted(self.interaction_partners(protein)))
            self._sorted_partners_cache[protein] = partners
        return partners

    def single_drug_side_effects(self, drug: str) -> list[SideEffect]:
        """Side effects of the drug taken alone (bio-decagon-mono), sorted by name."""
        effects = self._single_drug_effects_cache.get(drug)
        if effects is None:
            with sqlite3.connect(self.cache_database_file) as connection:
                rows = connection.execute(
                    "SELECT effect_code, effect_name FROM single_drug_side_effects "
                    "WHERE drug = ? ORDER BY effect_name",
                    (drug,),
                )
                effects = [SideEffect(code, name) for code, name in rows]
            self._single_drug_effects_cache[drug] = effects
        return effects

    def single_drug_effect_codes(self, drug: str) -> set[str]:
        return {effect.code for effect in self.single_drug_side_effects(drug)}

    def single_drug_effect_drug_counts(self) -> dict[str, int]:
        """Effect code -> number of drugs that list it as a single-drug effect (loaded once)."""
        if self._single_drug_effect_drug_counts is None:
            with sqlite3.connect(self.cache_database_file) as connection:
                self._single_drug_effect_drug_counts = dict(
                    connection.execute(
                        "SELECT effect_code, COUNT(*) FROM single_drug_side_effects "
                        "GROUP BY effect_code"
                    ).fetchall()
                )
        return self._single_drug_effect_drug_counts

    def pair_side_effects(self, drug_a: str, drug_b: str) -> list[SideEffect]:
        """Documented side effects of one drug pair, in either order, sorted by code."""
        drug_a, drug_b = sorted((drug_a, drug_b))
        with sqlite3.connect(self.cache_database_file) as connection:
            rows = connection.execute(
                """
                SELECT effect_code, effect_name FROM drug_pair_side_effects
                WHERE drug_a = ? AND drug_b = ? ORDER BY effect_code
                """,
                (drug_a, drug_b),
            ).fetchall()
        return [SideEffect(code, name) for code, name in rows]

    def side_effects_of_pairs(
        self, sorted_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], list[SideEffect]]:
        """Same as pair_side_effects for many pairs (each already sorted) with one connection."""
        effects_by_pair: dict[tuple[str, str], list[SideEffect]] = {}
        for pair in sorted_pairs:
            effects_by_pair[pair] = []
        if not sorted_pairs:
            return effects_by_pair
        query = """
            SELECT effect_code, effect_name FROM drug_pair_side_effects
            WHERE drug_a = ? AND drug_b = ? ORDER BY effect_code
        """
        with sqlite3.connect(self.cache_database_file) as connection:
            for pair in sorted_pairs:
                effects_by_pair[pair] = [
                    SideEffect(code, name) for code, name in connection.execute(query, pair)
                ]
        return effects_by_pair

    def tier3_candidate_effect_names(self, drug_a: str, drug_b: str) -> list[str]:
        """Side-effect names a Tier 3 hypothesis may use for a pair with no documented effect.

        Effects that each drug already shows with other partners are the
        plausible ones. Effects seen with both drugs rank first, then effects
        with more partners in total.
        """
        partner_counts_a = self._partner_counts_by_effect(drug_a)
        partner_counts_b = self._partner_counts_by_effect(drug_b)
        sort_keys = []
        for effect_code in set(partner_counts_a) | set(partner_counts_b):
            if effect_code in partner_counts_a:
                effect_name = partner_counts_a[effect_code][0]
            else:
                effect_name = partner_counts_b[effect_code][0]
            seen_with_both_drugs = (
                effect_code in partner_counts_a and effect_code in partner_counts_b
            )
            total_partners = (
                partner_counts_a.get(effect_code, ("", 0))[1]
                + partner_counts_b.get(effect_code, ("", 0))[1]
            )
            sort_keys.append(
                (not seen_with_both_drugs, -total_partners, effect_name.casefold(), effect_name)
            )
        sort_keys.sort()
        chosen_keys = sort_keys[: self.tier3_candidate_effects_per_pair]
        return [effect_name for *_, effect_name in chosen_keys]

    def _partner_counts_by_effect(self, drug: str) -> dict[str, tuple[str, int]]:
        """Effect code -> (effect name, number of partner drugs) over all pairs of the drug."""
        counts = self._partner_counts_cache.get(drug)
        if counts is None:
            with sqlite3.connect(self.cache_database_file) as connection:
                rows = connection.execute(
                    """
                    SELECT effect_code, MIN(effect_name), COUNT(*) FROM (
                        SELECT effect_code, effect_name FROM drug_pair_side_effects
                        WHERE drug_a = ?
                        UNION ALL
                        SELECT effect_code, effect_name FROM drug_pair_side_effects
                        WHERE drug_b = ?
                    ) GROUP BY effect_code
                    """,
                    (drug, drug),
                )
                counts = {code: (name, count) for code, name, count in rows}
            self._partner_counts_cache[drug] = counts
        return counts


def _read_drug_targets(drug_targets_file: Path) -> dict[str, set[str]]:
    target_proteins_by_drug: dict[str, set[str]] = {}
    for row in read_csv_rows_with_required_columns(drug_targets_file, ("STITCH", "Gene")):
        if row["STITCH"] and row["Gene"]:
            target_proteins_by_drug.setdefault(row["STITCH"], set()).add(row["Gene"])
    if not target_proteins_by_drug:
        raise ValueError(f"No drug-target relationships found in {drug_targets_file}")
    return target_proteins_by_drug


def _read_protein_interactions(protein_interactions_file: Path) -> dict[str, set[str]]:
    """Both directions are stored; a protein interacting with itself is skipped."""
    protein_interactions: dict[str, set[str]] = {}
    rows = read_csv_rows_with_required_columns(protein_interactions_file, ("Gene 1", "Gene 2"))
    for row in rows:
        protein_a, protein_b = row["Gene 1"], row["Gene 2"]
        if protein_a and protein_b and protein_a != protein_b:
            protein_interactions.setdefault(protein_a, set()).add(protein_b)
            protein_interactions.setdefault(protein_b, set()).add(protein_a)
    return protein_interactions
