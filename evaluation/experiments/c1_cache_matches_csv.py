"""C1: does the data the system loads (SQLite cache and graph) equal the raw CSV files, for every drug?"""

import collections
import sqlite3

from graphrag.knowledge_graph.decagon_graph import DecagonGraph
from evaluation.experiments.coverage_raw_source_data import RawSourceData


def run_cache_check(graph: DecagonGraph, raw: RawSourceData) -> list[dict]:
    cached_effect_codes: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    with sqlite3.connect(graph.cache_database_file) as connection:
        for drug_a, drug_b, effect_code in connection.execute(
            "SELECT drug_a, drug_b, effect_code FROM drug_pair_side_effects"
        ):
            cached_effect_codes[(drug_a, drug_b)].add(effect_code)
    raw_pairs = raw.pair_effect_codes
    all_pairs = set(raw_pairs) | set(cached_effect_codes)
    matching_pairs = sum(
        raw_pairs.get(pair, set()) == cached_effect_codes.get(pair, set()) for pair in all_pairs
    )
    drugs_in_data = (
        {drug for pair in raw_pairs for drug in pair}
        | set(raw.single_drug_effect_codes)
        | set(raw.targets_by_drug)
    )
    matching_single_drug_effects = sum(
        graph.single_drug_effect_codes(drug) == raw.single_drug_effect_codes.get(drug, set())
        for drug in drugs_in_data
    )
    matching_targets = sum(
        graph.target_proteins(drug) == raw.targets_by_drug.get(drug, set()) for drug in drugs_in_data
    )
    matching_pair_effects = sum(
        len(raw_pairs.get(pair, set()) & cached_effect_codes.get(pair, set())) for pair in all_pairs
    )
    all_pair_effects = sum(
        len(raw_pairs.get(pair, set()) | cached_effect_codes.get(pair, set())) for pair in all_pairs
    )
    return [
        {"check": "C1 – cặp thuốc: tập TDP trong cache = CSV", "ok": matching_pairs, "total": len(all_pairs)},
        {"check": "C1 – (cặp, TDP) trong cache = CSV", "ok": matching_pair_effects, "total": all_pair_effects},
        {"check": "C1 – thuốc trong dữ liệu được hệ thống nạp",
         "ok": len(drugs_in_data & graph.drug_ids), "total": len(drugs_in_data)},
        {"check": "C1 – TDP đơn lẻ (mono) mỗi thuốc = CSV",
         "ok": matching_single_drug_effects, "total": len(drugs_in_data)},
        {"check": "C1 – protein đích mỗi thuốc = CSV", "ok": matching_targets, "total": len(drugs_in_data)},
    ]
