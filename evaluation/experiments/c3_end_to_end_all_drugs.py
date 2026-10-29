"""C3: run the whole pipeline (LLM disabled) on regimens that together contain every drug.

Each drug appears at least `rounds` times. Counted: crashes, displayed Tier 1/2
effects whose pairs differ from the CSV, Tier 3 bridges for documented pairs,
and evidence edges that are not in the raw files.
"""

import collections
import itertools
import random
import re
import time
import traceback

from graphrag.pipeline.polypharmacy_system import PolypharmacySystem
from evaluation.experiments.coverage_raw_source_data import UNREADABLE_NAME_PATTERN, RawSourceData


MINIMUM_REGIMEN_SIZE = 3


def regimens_covering_every_drug(drugs: list[str], rounds: int, seed: int) -> list[list[str]]:
    """Each round shuffles all drugs and cuts them into regimens of 3, 4, 5 or 6 drugs."""
    random_generator = random.Random(seed)
    regimens: list[list[str]] = []
    for round_index in range(rounds):
        shuffled = drugs[:]
        random_generator.shuffle(shuffled)
        regimen_size = 3 + round_index % 4
        for start in range(0, len(shuffled), regimen_size):
            regimen = shuffled[start : start + regimen_size]
            while len(regimen) < MINIMUM_REGIMEN_SIZE:  # pad the last group
                extra_drug = random_generator.choice(drugs)
                if extra_drug not in regimen:
                    regimen.append(extra_drug)
            regimens.append(regimen)
    return regimens


def run_end_to_end_check(
    system: PolypharmacySystem, raw: RawSourceData, rounds: int, seed: int
) -> tuple[list[dict], list[str], dict]:
    unreadable_pattern = re.compile(UNREADABLE_NAME_PATTERN)
    drugs = sorted(system.graph.drug_ids)
    regimens = regimens_covering_every_drug(drugs, rounds, seed)
    tally = collections.Counter()
    failures: list[str] = []
    analysis_count_by_drug: collections.Counter = collections.Counter()
    start = time.perf_counter()
    for regimen in regimens:
        try:
            result = system.predict(regimen)
        except Exception:  # noqa: BLE001 - every crash is a finding
            tally["crash"] += 1
            failures.append(f"crash {regimen}: {traceback.format_exc(limit=2)}")
            continue
        tally["regimens_ok"] += 1
        analysis_count_by_drug.update(regimen)
        drug_by_name = {system.names.drug_name(drug): drug for drug in regimen}
        pairs = list(itertools.combinations(sorted(regimen), 2))
        for card in result["drugs"]:
            tally["cards"] += 1
            tally["cards_named"] += not unreadable_pattern.search(card["name"])
        for card in result["interactions"]:
            _check_interaction_card(card, regimen, pairs, drug_by_name, raw, tally, failures)
        bridges = result["mechanism_bridges"]
        for bridge in bridges:
            tally["bridges"] += 1
            tally["bridges_undocumented"] += not raw.pair_effect_codes.get(tuple(sorted(bridge["drug_pair"])))
        displayed_evidence = [evidence for card in result["interactions"] for evidence in card["evidence"]]
        for evidence in displayed_evidence + bridges:
            for edge in evidence["path"]:
                tally["edges"] += 1
                edge_is_correct = _edge_exists(edge, raw)
                tally["edges_ok"] += edge_is_correct
                if not edge_is_correct:
                    failures.append(f"bad edge {edge} in {regimen}")
    seconds = time.perf_counter() - start
    rows = [
        {"check": "C3 – phác đồ chạy hết pipeline không lỗi", "ok": tally["regimens_ok"], "total": len(regimens)},
        {"check": "C3 – thuốc được phân tích ít nhất 1 lần", "ok": len(analysis_count_by_drug), "total": len(drugs)},
        {"check": "C3 – thẻ thuốc có tên đọc được", "ok": tally["cards_named"], "total": tally["cards"]},
        {"check": "C3 – TDP Tier 1/2 hiển thị: mọi cặp có trong CSV",
         "ok": tally["shown_t12_supported"], "total": tally["shown_t12"]},
        {"check": "C3 – TDP Tier 1/2 hiển thị: liệt kê đủ mọi cặp trong CSV",
         "ok": tally["shown_t12_complete"], "total": tally["shown_t12"]},
        {"check": "C3 – cầu protein (Tier 3) chỉ cho cặp chưa có dữ liệu",
         "ok": tally["bridges_undocumented"], "total": tally["bridges"]},
        {"check": "C3 – cạnh bằng chứng hiển thị (cả cầu Tier 3) đúng với file gốc",
         "ok": tally["edges_ok"], "total": tally["edges"]},
    ]
    meta = {
        "regimens": len(regimens),
        "min_times_per_drug": min(analysis_count_by_drug.values()) if analysis_count_by_drug else 0,
        "seconds_per_regimen": round(seconds / max(len(regimens), 1), 3),
    }
    return rows, failures, meta


def _check_interaction_card(card, regimen, pairs, drug_by_name, raw, tally, failures) -> None:
    evidence = card["evidence"]
    effect_code = evidence[0]["side_effect_code"] if evidence else None
    if card["tier"] in (1, 2):
        tally["shown_t12"] += 1
        shown_pairs = {tuple(sorted(drug_by_name[name] for name in pair)) for pair in card["pairs"]}
        true_pairs = {pair for pair in pairs if effect_code in raw.pair_effect_codes.get(pair, set())}
        tally["shown_t12_supported"] += bool(shown_pairs) and shown_pairs <= true_pairs
        tally["shown_t12_complete"] += shown_pairs == true_pairs
        if not (shown_pairs and shown_pairs == true_pairs):
            failures.append(f"pairs mismatch {card['side_effect_en']} {regimen}")
    else:
        tally["shown_t3"] += 1
        if card["pairs"]:
            first_pair = tuple(sorted(drug_by_name[name] for name in card["pairs"][0]))
        else:
            first_pair = None
        tally["shown_t3_undocumented"] += first_pair is not None and not raw.pair_effect_codes.get(first_pair)


def _edge_exists(edge: dict, raw: RawSourceData) -> bool:
    relation = edge["relation"]
    if relation == "targets":
        return edge["target"] in raw.targets_by_drug.get(edge["source"], set())
    if relation == "interacts_with":
        return (edge["source"], edge["target"]) in raw.interacting_protein_pairs
    if relation == "has_side_effect":
        pair = tuple(sorted((edge["source"], edge["target"])))
        return edge["side_effect_code"] in raw.pair_effect_codes.get(pair, set())
    return False
