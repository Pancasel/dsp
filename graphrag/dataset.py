import csv
import hashlib
import io
import itertools
import sqlite3
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, TextIO

from graphrag.config import RankingWeights
from graphrag.ranking import rank_effects


# A Tier 2 path records at most this many shared proteins. Further proteins
# still support the tier, and the retrieval reports that the path was shortened.
MAX_SHARED_PROTEINS_IN_PATH = 3
# Full target lists are included only for small regimens. Larger regimens keep
# the targets that appear in the retrieved evidence.
DIRECT_TARGET_LIST_LIMIT = 250
# Bump when the cache schema changes so old caches are rebuilt.
CACHE_SCHEMA_VERSION = "2"


@dataclass(frozen=True)
class SideEffect:
    code: str
    name: str


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    tier: int
    kind: str
    drug_pair: tuple[str, str]
    path: tuple[dict[str, str], ...]
    side_effect: str | None = None
    side_effect_code: str | None = None
    # Tier 3 only: side-effect labels the LLM may use for this pair.
    candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "evidence_id": self.evidence_id,
            "tier": self.tier,
            "kind": self.kind,
            "drug_pair": list(self.drug_pair),
            "path": list(self.path),
        }
        if self.side_effect is not None:
            result["side_effect"] = self.side_effect
        if self.side_effect_code is not None:
            result["side_effect_code"] = self.side_effect_code
        if self.candidates:
            result["candidate_side_effects"] = list(self.candidates)
        return result


@dataclass(frozen=True)
class _Candidate:
    tier: int
    kind: str
    drug_pair: tuple[str, str]
    path: tuple[dict[str, str], ...]
    effect: SideEffect | None = None
    candidates: tuple[str, ...] = ()


@contextmanager
def _open_csv(path: Path) -> Iterator[TextIO]:
    if path.suffix.lower() != ".zip":
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")
        stream = path.open("r", encoding="utf-8-sig", newline="")
        try:
            yield stream
        finally:
            stream.close()
        return

    if not path.is_file():
        raise FileNotFoundError(f"Dataset archive does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not Path(name).name.startswith("._")
        ]
        if not candidates:
            raise ValueError(f"No CSV file found in dataset archive: {path}")
        preferred = path.stem.lower()
        member = next(
            (name for name in candidates if Path(name).stem.lower() == preferred),
            candidates[0],
        )
        with archive.open(member) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            try:
                yield text
            finally:
                text.detach()


def _rows(path: Path, required: tuple[str, ...]) -> Iterator[dict[str, str]]:
    with _open_csv(path) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset is missing a CSV header: {path}")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Dataset {path} is missing required columns: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            yield {key: (row.get(key) or "").strip() for key in required}


def _source_signature(path: Path) -> str:
    stat = path.stat()
    value = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cache_signature(source: Path, mono: Path | None) -> str:
    parts = [CACHE_SCHEMA_VERSION, _source_signature(source)]
    parts.append(_source_signature(mono) if mono is not None else "no-mono")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _build_combo_cache(source: Path, database: Path, mono: Path | None = None) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_name(database.name + ".tmp")
    temporary.unlink(missing_ok=True)
    signature = _cache_signature(source, mono)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            """
            CREATE TABLE side_effects (
                drug_a TEXT NOT NULL,
                drug_b TEXT NOT NULL,
                effect_code TEXT NOT NULL,
                effect_name TEXT NOT NULL,
                PRIMARY KEY (drug_a, drug_b, effect_code)
            ) WITHOUT ROWID
            """
        )
        connection.execute("CREATE TABLE metadata (signature TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES (?)", (signature,))
        insert = """
            INSERT OR IGNORE INTO side_effects
            (drug_a, drug_b, effect_code, effect_name) VALUES (?, ?, ?, ?)
        """
        batch: list[tuple[str, str, str, str]] = []
        for row in _rows(
            source,
            ("STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"),
        ):
            drug_a, drug_b = sorted((row["STITCH 1"], row["STITCH 2"]))
            code = row["Polypharmacy Side Effect"]
            if drug_a and drug_b and code:
                batch.append(
                    (drug_a, drug_b, code, row["Side Effect Name"] or code)
                )
            if len(batch) >= 10_000:
                connection.executemany(insert, batch)
                batch.clear()
        if batch:
            connection.executemany(insert, batch)
        # Lookups by either drug of a pair (Tier 3 candidate labels).
        connection.execute("CREATE INDEX side_effects_drug_b ON side_effects (drug_b)")
        # Number of drug pairs that record each effect, for the ranking IDF.
        connection.execute(
            """
            CREATE TABLE effect_counts AS
            SELECT effect_code, MIN(effect_name) AS effect_name, COUNT(*) AS pair_count
            FROM side_effects GROUP BY effect_code
            """
        )
        connection.execute("CREATE TABLE stats (key TEXT PRIMARY KEY, value INTEGER)")
        connection.execute(
            """
            INSERT INTO stats
            SELECT 'total_pairs', COUNT(*) FROM
            (SELECT DISTINCT drug_a, drug_b FROM side_effects)
            """
        )
        connection.execute(
            """
            CREATE TABLE mono (
                drug TEXT NOT NULL,
                effect_code TEXT NOT NULL,
                effect_name TEXT NOT NULL,
                PRIMARY KEY (drug, effect_code)
            ) WITHOUT ROWID
            """
        )
        if mono is not None:
            mono_rows = [
                (row["STITCH"], row["Individual Side Effect"], row["Side Effect Name"])
                for row in _rows(
                    mono, ("STITCH", "Individual Side Effect", "Side Effect Name")
                )
                if row["STITCH"] and row["Individual Side Effect"]
            ]
            connection.executemany(
                "INSERT OR IGNORE INTO mono VALUES (?, ?, ?)", mono_rows
            )
        connection.commit()
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    temporary.replace(database)


def _cache_is_current(source: Path, database: Path, mono: Path | None = None) -> bool:
    if not database.is_file():
        return False
    try:
        with sqlite3.connect(database) as connection:
            row = connection.execute("SELECT signature FROM metadata").fetchone()
    except sqlite3.DatabaseError:
        return False
    return row is not None and row[0] == _cache_signature(source, mono)


def _take_round_robin(
    groups: list[list[_Candidate]], limit: int
) -> tuple[list[_Candidate], bool]:
    """Spread a limited evidence budget across drug pairs."""
    if limit <= 0:
        return [], any(groups)
    chosen: list[_Candidate] = []
    cursors = [0] * len(groups)
    while len(chosen) < limit:
        progressed = False
        for index, group in enumerate(groups):
            cursor = cursors[index]
            if cursor >= len(group):
                continue
            chosen.append(group[cursor])
            cursors[index] = cursor + 1
            progressed = True
            if len(chosen) >= limit:
                break
        if not progressed:
            break
    truncated = any(cursors[index] < len(group) for index, group in enumerate(groups))
    return chosen, truncated


class DecagonGraph:
    def __init__(
        self,
        targets: dict[str, set[str]],
        ppi: dict[str, set[str]],
        combo_db: Path,
        hub_degree_threshold: int,
        drug_ids: set[str],
        side_effect_names: dict[str, str],
        tier3_one_hop_per_pair: int,
        tier3_two_hop_per_pair: int,
        max_evidence: int,
        effect_pair_count: dict[str, int] | None = None,
        total_pairs: int = 0,
        tier3_candidates_per_pair: int = 20,
    ) -> None:
        self.targets = targets
        self.ppi = ppi
        self.combo_db = combo_db
        self.hub_degree_threshold = hub_degree_threshold
        self.drug_ids = drug_ids
        self.side_effect_names = side_effect_names
        self.tier3_one_hop_per_pair = tier3_one_hop_per_pair
        self.tier3_two_hop_per_pair = tier3_two_hop_per_pair
        self.max_evidence = max_evidence
        self.effect_pair_count = effect_pair_count or {}
        self.total_pairs = total_pairs
        self.tier3_candidates_per_pair = tier3_candidates_per_pair
        self._mono_cache: dict[str, list[SideEffect]] = {}
        self._drug_effect_cache: dict[str, dict[str, tuple[str, int]]] = {}

    @classmethod
    def load(
        cls,
        targets_file: Path,
        ppi_file: Path,
        combo_file: Path,
        combo_db: Path,
        hub_degree_threshold: int,
        tier3_one_hop_per_pair: int = 8,
        tier3_two_hop_per_pair: int = 4,
        max_evidence: int = 400,
        mono_file: Path | None = None,
        tier3_candidates_per_pair: int = 20,
    ) -> "DecagonGraph":
        targets: dict[str, set[str]] = {}
        for row in _rows(targets_file, ("STITCH", "Gene")):
            if row["STITCH"] and row["Gene"]:
                targets.setdefault(row["STITCH"], set()).add(row["Gene"])
        if not targets:
            raise ValueError(f"No drug-target relationships found in {targets_file}")

        ppi: dict[str, set[str]] = {}
        for row in _rows(ppi_file, ("Gene 1", "Gene 2")):
            gene_a, gene_b = row["Gene 1"], row["Gene 2"]
            if gene_a and gene_b and gene_a != gene_b:
                ppi.setdefault(gene_a, set()).add(gene_b)
                ppi.setdefault(gene_b, set()).add(gene_a)

        if not _cache_is_current(combo_file, combo_db, mono_file):
            _build_combo_cache(combo_file, combo_db, mono_file)
        with sqlite3.connect(combo_db) as connection:
            combo_drugs = {
                drug
                for row in connection.execute(
                    """
                    SELECT drug_a FROM side_effects
                    UNION
                    SELECT drug_b FROM side_effects
                    """
                )
                for drug in row
            }
            side_effect_names = {
                name.casefold(): name
                for (name,) in connection.execute(
                    "SELECT effect_name FROM effect_counts"
                )
                if name
            }
            effect_pair_count = {
                code: count
                for code, count in connection.execute(
                    "SELECT effect_code, pair_count FROM effect_counts"
                )
            }
            stats = connection.execute(
                "SELECT value FROM stats WHERE key = 'total_pairs'"
            ).fetchone()
        return cls(
            targets,
            ppi,
            combo_db,
            hub_degree_threshold,
            set(targets) | combo_drugs,
            side_effect_names,
            tier3_one_hop_per_pair,
            tier3_two_hop_per_pair,
            max_evidence,
            effect_pair_count,
            stats[0] if stats else 0,
            tier3_candidates_per_pair,
        )

    def mono_effects(self, drug: str) -> list[SideEffect]:
        """Individual (single-drug) side effects from bio-decagon-mono."""
        cached = self._mono_cache.get(drug)
        if cached is None:
            with sqlite3.connect(self.combo_db) as connection:
                cached = [
                    SideEffect(code, name)
                    for code, name in connection.execute(
                        "SELECT effect_code, effect_name FROM mono WHERE drug = ? "
                        "ORDER BY effect_name",
                        (drug,),
                    )
                ]
            self._mono_cache[drug] = cached
        return cached

    def mono_effect_drug_count(self) -> dict[str, int]:
        """effect code -> number of drugs listing it individually (loaded once)."""
        if not hasattr(self, "_mono_drug_count"):
            with sqlite3.connect(self.combo_db) as connection:
                self._mono_drug_count = dict(
                    connection.execute(
                        "SELECT effect_code, COUNT(*) FROM mono GROUP BY effect_code"
                    ).fetchall()
                )
        return self._mono_drug_count

    def mono_codes(self, drug: str) -> set[str]:
        return {effect.code for effect in self.mono_effects(drug)}

    def _drug_effect_counts(self, drug: str) -> dict[str, tuple[str, int]]:
        """effect code -> (name, number of partner drugs) over all pairs of drug."""
        cached = self._drug_effect_cache.get(drug)
        if cached is None:
            with sqlite3.connect(self.combo_db) as connection:
                cached = {
                    code: (name, count)
                    for code, name, count in connection.execute(
                        """
                        SELECT effect_code, MIN(effect_name), COUNT(*) FROM (
                            SELECT effect_code, effect_name FROM side_effects
                            WHERE drug_a = ?
                            UNION ALL
                            SELECT effect_code, effect_name FROM side_effects
                            WHERE drug_b = ?
                        ) GROUP BY effect_code
                        """,
                        (drug, drug),
                    )
                }
            self._drug_effect_cache[drug] = cached
        return cached

    def tier3_candidates(self, drug_a: str, drug_b: str) -> list[str]:
        """Labels a Tier 3 hypothesis may use for an undocumented pair.

        Effects that each drug already shows with other partners are the
        plausible ones; effects seen with both drugs rank first, then by the
        total number of partners.
        """
        counts_a = self._drug_effect_counts(drug_a)
        counts_b = self._drug_effect_counts(drug_b)
        scored = []
        for code in set(counts_a) | set(counts_b):
            name = (counts_a.get(code) or counts_b[code])[0]
            in_both = code in counts_a and code in counts_b
            total = counts_a.get(code, ("", 0))[1] + counts_b.get(code, ("", 0))[1]
            scored.append((not in_both, -total, name.casefold(), name))
        scored.sort()
        return [item[3] for item in scored[: self.tier3_candidates_per_pair]]

    def side_effects(self, drug_a: str, drug_b: str) -> list[SideEffect]:
        drug_a, drug_b = sorted((drug_a, drug_b))
        with sqlite3.connect(self.combo_db) as connection:
            rows = connection.execute(
                """
                SELECT effect_code, effect_name FROM side_effects
                WHERE drug_a = ? AND drug_b = ? ORDER BY effect_code
                """,
                (drug_a, drug_b),
            ).fetchall()
        return [SideEffect(code, name) for code, name in rows]

    def _pair_side_effects(
        self, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], list[SideEffect]]:
        found = {pair: [] for pair in pairs}
        if not pairs:
            return found
        query = """
            SELECT effect_code, effect_name FROM side_effects
            WHERE drug_a = ? AND drug_b = ? ORDER BY effect_code
        """
        with sqlite3.connect(self.combo_db) as connection:
            for pair in pairs:
                found[pair] = [
                    SideEffect(code, name)
                    for code, name in connection.execute(query, pair)
                ]
        return found

    def _kept_intermediate(self, protein: str, regimen_targets: set[str]) -> bool:
        return (
            protein in regimen_targets
            or len(self.ppi.get(protein, ())) <= self.hub_degree_threshold
        )

    def _expandable(self, protein: str) -> bool:
        return len(self.ppi.get(protein, ())) <= self.hub_degree_threshold

    def _sorted_neighbors(self, protein: str) -> tuple[str, ...]:
        """Sorted PPI neighbours, cached because bridge search revisits them."""
        cache = self.__dict__.setdefault("_neighbor_cache", {})
        neighbors = cache.get(protein)
        if neighbors is None:
            neighbors = tuple(sorted(self.ppi.get(protein, ())))
            cache[protein] = neighbors
        return neighbors

    def _bridges_for_pair(
        self,
        targets_a: set[str],
        targets_b: set[str],
        regimen_targets: set[str],
        pruned_hubs: set[str],
    ) -> tuple[list[tuple[str, str, str | None]], bool]:
        """Return a bounded, deterministic sample of 1-hop then 2-hop bridges.

        Search stops once both per-pair caps are full and one extra candidate
        has shown that more bridges exist. High-degree proteins are never
        expanded, including when a queried drug targets them.
        """
        one_hop_cap = self.tier3_one_hop_per_pair
        two_hop_cap = self.tier3_two_hop_per_pair
        if one_hop_cap == 0 and two_hop_cap == 0:
            return [], False
        one_hop: list[tuple[str, str, str | None]] = []
        two_hop: list[tuple[str, str, str | None]] = []
        truncated = False
        one_done = one_hop_cap == 0
        two_done = two_hop_cap == 0

        for protein_a in sorted(targets_a):
            if one_done and two_done:
                truncated = True
                break
            for neighbor in self._sorted_neighbors(protein_a):
                if one_done and two_done:
                    truncated = True
                    break
                if neighbor in targets_b and neighbor != protein_a:
                    if one_done:
                        truncated = True
                    else:
                        one_hop.append((protein_a, neighbor, None))
                        one_done = len(one_hop) >= one_hop_cap
                if not self._expandable(neighbor):
                    if neighbor not in regimen_targets and not targets_b.isdisjoint(
                        self.ppi.get(neighbor, ())
                    ):
                        pruned_hubs.add(neighbor)
                    if truncated and one_done and two_done:
                        break
                    continue
                if two_done:
                    if (targets_b & self.ppi.get(neighbor, set())) - {protein_a}:
                        truncated = True
                    if truncated and one_done:
                        break
                    continue
                # Same order as scanning the sorted neighbours and skipping
                # non-targets, but done with a set intersection.
                for protein_b in sorted(targets_b & self.ppi.get(neighbor, set())):
                    if protein_b == protein_a:
                        continue
                    if len(two_hop) >= two_hop_cap:
                        truncated = True
                        two_done = True
                        break
                    two_hop.append((protein_a, protein_b, neighbor))
                if len(two_hop) >= two_hop_cap:
                    two_done = True
                if truncated and one_done and two_done:
                    break
            if truncated and one_done and two_done:
                break
        return one_hop + two_hop, truncated

    def _path_is_kept(
        self,
        path: tuple[dict[str, str], ...],
        drugs: list[str],
        regimen_targets: set[str],
    ) -> bool:
        kept_proteins = {
            protein
            for edge in path
            if edge["relation"] in {"targets", "interacts_with"}
            for protein in (edge["source"], edge["target"])
            if self._kept_intermediate(protein, regimen_targets)
        }
        return all(
            edge["relation"] not in {"targets", "interacts_with"}
            or (
                (edge["source"] in drugs or edge["source"] in kept_proteins)
                and (edge["target"] in drugs or edge["target"] in kept_proteins)
            )
            for edge in path
        )

    def retrieve(
        self,
        drugs: list[str],
        is_serious: Callable[[str], bool] | None = None,
        weights: RankingWeights | None = None,
        is_critical: Callable[[str], bool] | None = None,
    ) -> tuple[dict[str, object], list[Evidence]]:
        targets_by_drug = {drug: self.targets.get(drug, set()) for drug in drugs}
        regimen_targets = set().union(*targets_by_drug.values()) if drugs else set()
        shared_targets: dict[str, list[str]] = {}
        for protein in regimen_targets:
            owners = sorted(
                drug for drug, targets in targets_by_drug.items() if protein in targets
            )
            if len(owners) >= 3:
                shared_targets[protein] = owners

        pairs = list(itertools.combinations(sorted(drugs), 2))
        pair_effects = self._pair_side_effects(pairs)
        proteins_by_pair: dict[tuple[str, str], list[str]] = {}
        for protein, owners in shared_targets.items():
            for drug_a, drug_b in itertools.combinations(owners, 2):
                proteins_by_pair.setdefault((drug_a, drug_b), []).append(protein)

        amplified_pairs = {
            pair for pair in proteins_by_pair if pair_effects.get(pair)
        }
        ranked = rank_effects(
            {pair: effects for pair, effects in pair_effects.items() if effects},
            amplified_pairs,
            self.effect_pair_count,
            self.total_pairs,
            self.mono_codes,
            is_serious,
            weights,
            is_critical,
        )
        # Order each pair's effects by the regimen-level score so that the
        # bounded evidence sample keeps the most relevant effects first.
        rank_of = {item.code: index for index, item in enumerate(ranked)}
        for pair in pairs:
            pair_effects[pair].sort(key=lambda effect: rank_of.get(effect.code, len(rank_of)))

        tier1_groups: list[list[_Candidate]] = []
        tier2_groups: list[list[_Candidate]] = []
        tier3_groups: list[list[_Candidate]] = []
        pruned_hubs: set[str] = set()
        truncated = False

        for drug_a, drug_b in pairs:
            pair = (drug_a, drug_b)
            effects = pair_effects[pair]
            if effects:
                tier1_groups.append(
                    [
                        _Candidate(
                            tier=1,
                            kind="known_pairwise_effect",
                            drug_pair=pair,
                            path=(
                                {
                                    "source": drug_a,
                                    "relation": "has_side_effect",
                                    "target": drug_b,
                                    "side_effect_code": effect.code,
                                    "side_effect_name": effect.name,
                                },
                            ),
                            effect=effect,
                        )
                        for effect in effects
                    ]
                )
                shared_proteins = sorted(proteins_by_pair.get(pair, []))
                if len(shared_proteins) > MAX_SHARED_PROTEINS_IN_PATH:
                    truncated = True
                    shared_proteins = shared_proteins[:MAX_SHARED_PROTEINS_IN_PATH]
                if shared_proteins:
                    tier2_groups.append(
                        [
                            _Candidate(
                                tier=2,
                                kind="shared_target_amplification",
                                drug_pair=pair,
                                path=tuple(
                                    [
                                        {
                                            "source": drug,
                                            "relation": "targets",
                                            "target": protein,
                                        }
                                        for protein in shared_proteins
                                        for drug in (drug_a, drug_b)
                                    ]
                                    + [
                                        {
                                            "source": drug_a,
                                            "relation": "has_side_effect",
                                            "target": drug_b,
                                            "side_effect_code": effect.code,
                                            "side_effect_name": effect.name,
                                        }
                                    ]
                                ),
                                effect=effect,
                            )
                            for effect in effects
                        ]
                    )
                continue

            bridges, bridge_truncated = self._bridges_for_pair(
                targets_by_drug[drug_a],
                targets_by_drug[drug_b],
                regimen_targets,
                pruned_hubs,
            )
            truncated = truncated or bridge_truncated
            if not bridges:
                continue
            candidates = tuple(self.tier3_candidates(drug_a, drug_b))
            group: list[_Candidate] = []
            for protein_a, protein_b, intermediate in bridges:
                path: list[dict[str, str]] = [
                    {"source": drug_a, "relation": "targets", "target": protein_a},
                    {
                        "source": protein_a,
                        "relation": "interacts_with",
                        "target": protein_b if intermediate is None else intermediate,
                    },
                ]
                if intermediate is not None:
                    path.append(
                        {
                            "source": intermediate,
                            "relation": "interacts_with",
                            "target": protein_b,
                        }
                    )
                path.append(
                    {"source": drug_b, "relation": "targets", "target": protein_b}
                )
                group.append(
                    _Candidate(
                        tier=3,
                        kind="novel_pair_protein_bridge",
                        drug_pair=pair,
                        path=tuple(path),
                        candidates=candidates,
                    )
                )
            tier3_groups.append(group)

        selected_tier1, tier1_truncated = _take_round_robin(
            tier1_groups, self.max_evidence
        )
        selected_tier2, tier2_truncated = _take_round_robin(
            tier2_groups, self.max_evidence - len(selected_tier1)
        )
        selected_tier3, tier3_truncated = _take_round_robin(
            tier3_groups,
            self.max_evidence - len(selected_tier1) - len(selected_tier2),
        )
        truncated = truncated or tier1_truncated or tier2_truncated or tier3_truncated
        selected = [
            item
            for item in selected_tier1 + selected_tier2 + selected_tier3
            if self._path_is_kept(item.path, drugs, regimen_targets)
        ]
        evidence = [
            Evidence(
                evidence_id=f"E{index}",
                tier=item.tier,
                kind=item.kind,
                drug_pair=item.drug_pair,
                path=item.path,
                side_effect=item.effect.name if item.effect else None,
                side_effect_code=item.effect.code if item.effect else None,
                candidates=item.candidates,
            )
            for index, item in enumerate(selected, start=1)
        ]

        proteins_in_paths = {
            endpoint
            for item in evidence
            for edge in item.path
            if edge["relation"] in {"targets", "interacts_with"}
            for endpoint in (edge["source"], edge["target"])
        }
        kept_proteins = {
            protein
            for protein in proteins_in_paths
            if self._kept_intermediate(protein, regimen_targets)
        }
        assignments = sum(len(genes) for genes in targets_by_drug.values())
        if assignments <= DIRECT_TARGET_LIST_LIMIT:
            direct_targets = {
                drug: sorted(targets_by_drug[drug]) for drug in drugs
            }
        else:
            direct_targets = {
                drug: sorted(targets_by_drug[drug] & kept_proteins) for drug in drugs
            }
        shared_list = [
            {"protein": protein, "drugs": owners}
            for protein, owners in sorted(shared_targets.items())
        ]

        graph_context: dict[str, object] = {
            "drugs": drugs,
            "direct_targets": direct_targets,
            "direct_target_counts": {
                drug: len(targets_by_drug[drug]) for drug in drugs
            },
            "shared_targets": shared_list,
            "ranked_effects": [item.to_dict() for item in ranked],
            "pairs_with_documented_effects": [
                list(pair) for pair in pairs if pair_effects[pair]
            ],
            "evidence": [item.to_dict() for item in evidence],
            "evidence_truncated": truncated,
            "pruned_hubs": sorted(
                pruned_hubs
                | {
                    protein
                    for protein in proteins_in_paths
                    if protein not in kept_proteins and protein not in regimen_targets
                }
            ),
        }
        return graph_context, evidence

    def validate_evidence(self, evidence: Evidence) -> bool:
        for edge in evidence.path:
            source, relation, target = (
                edge["source"],
                edge["relation"],
                edge["target"],
            )
            if relation == "targets":
                if target not in self.targets.get(source, set()):
                    return False
            elif relation == "interacts_with":
                if target not in self.ppi.get(source, set()):
                    return False
            elif relation == "has_side_effect":
                code = edge.get("side_effect_code", "")
                if not any(
                    effect.code == code for effect in self.side_effects(source, target)
                ):
                    return False
            else:
                return False
        return True
