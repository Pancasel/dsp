"""Coverage of ALL drugs in the Decagon data (offline, no LLM/API calls).

C1  Data cache: for every one of the documented drug pairs, the side effects in
    the SQLite cache equal the set read straight from bio-decagon-combo.csv; the
    same for every drug's mono side effects and protein targets.
C2  Names: every drug has a readable name (not a CID / IUPAC string), its name
    resolves back to the same drug alone and inside a chat sentence, and every
    curated Vietnamese brand resolves to its drug.
C3  End to end: regimens that together contain every drug (each drug at least
    COVER_ROUNDS times) go through PolypharmacySystem.predict() with the LLM
    disabled. Counted: crashes, displayed Tier 1 effects not in the CSV,
    documented pairs of a displayed effect missing from its pair list, and
    evidence edges not in the raw files.

Usage:
    python eval/run_coverage.py [--seed 0] [--rounds 3]
Writes eval/results/c_coverage.csv and eval/results/COVERAGE.md.
"""

import argparse
import collections
import itertools
import os
import random
import re
import sqlite3
import sys
import time
import traceback
from pathlib import Path

os.environ["LLM_API_KEY"] = ""  # never call the LLM from this script

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import RESULTS  # noqa: E402

from graphrag.dataset import _rows  # noqa: E402
from graphrag.names import normalize  # noqa: E402
from graphrag.pipeline import PolypharmacySystem  # noqa: E402

UNREADABLE = re.compile(r"^CID\d|^Cid \d|[\[\]]|\d.*\d.*\d|^.{41,}$")


def read_raw(settings):
    combo: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in _rows(settings.combo_file, ("STITCH 1", "STITCH 2", "Polypharmacy Side Effect")):
        pair = tuple(sorted((row["STITCH 1"], row["STITCH 2"])))
        combo[pair].add(row["Polypharmacy Side Effect"])
    mono: dict[str, set[str]] = collections.defaultdict(set)
    if settings.mono_file:
        for row in _rows(settings.mono_file, ("STITCH", "Individual Side Effect")):
            mono[row["STITCH"]].add(row["Individual Side Effect"])
    targets: dict[str, set[str]] = collections.defaultdict(set)
    for row in _rows(settings.targets_file, ("STITCH", "Gene")):
        targets[row["STITCH"]].add(row["Gene"])
    ppi: set[tuple[str, str]] = set()
    for row in _rows(settings.ppi_file, ("Gene 1", "Gene 2")):
        ppi.add((row["Gene 1"], row["Gene 2"]))
        ppi.add((row["Gene 2"], row["Gene 1"]))
    return combo, mono, targets, ppi


def c1_cache(system, combo, mono, targets) -> list[dict]:
    graph = system.graph
    cached: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    with sqlite3.connect(graph.combo_db) as connection:
        for drug_a, drug_b, code in connection.execute(
            "SELECT drug_a, drug_b, effect_code FROM side_effects"
        ):
            cached[(drug_a, drug_b)].add(code)
    pairs = set(combo) | set(cached)
    pair_ok = sum(combo.get(pair, set()) == cached.get(pair, set()) for pair in pairs)
    drugs_in_data = {drug for pair in combo for drug in pair} | set(mono) | set(targets)
    mono_ok = sum(
        {item.code for item in graph.mono_effects(drug)} == mono.get(drug, set())
        for drug in drugs_in_data
    )
    target_ok = sum(graph.targets.get(drug, set()) == targets.get(drug, set()) for drug in drugs_in_data)
    return [
        {"check": "C1 – cặp thuốc: tập TDP trong cache = CSV", "ok": pair_ok, "total": len(pairs)},
        {"check": "C1 – (cặp, TDP) trong cache = CSV",
         "ok": sum(len(combo.get(p, set()) & cached.get(p, set())) for p in pairs),
         "total": sum(len(combo.get(p, set()) | cached.get(p, set())) for p in pairs)},
        {"check": "C1 – thuốc trong dữ liệu được hệ thống nạp", "ok": len(drugs_in_data & graph.drug_ids), "total": len(drugs_in_data)},
        {"check": "C1 – TDP đơn lẻ (mono) mỗi thuốc = CSV", "ok": mono_ok, "total": len(drugs_in_data)},
        {"check": "C1 – protein đích mỗi thuốc = CSV", "ok": target_ok, "total": len(drugs_in_data)},
    ]


def c2_names(system) -> tuple[list[dict], list[str]]:
    resolver, names = system.resolver, system.names
    failures: list[str] = []
    drugs = sorted(system.graph.drug_ids)
    readable = alone = in_chat = 0
    for drug in drugs:
        name = names.drug_name(drug)
        if UNREADABLE.search(name):
            failures.append(f"unreadable name {drug}: {name}")
        else:
            readable += 1
        if resolver.resolve(name).stitch_id == drug:
            alone += 1
        else:
            failures.append(f"name does not resolve {drug}: {name}")
        found = [item.stitch_id for item in resolver.extract(f"tôi đang uống {name} với aspirin")]
        if drug in found:
            in_chat += 1
        else:
            failures.append(f"name not found in chat {drug}: {name} -> {found}")
    brands = brand_ok = 0
    for drug in drugs:
        for alias in names.drugs[drug].aliases_vi:
            if len(normalize(alias)) < 3:
                continue
            brands += 1
            if resolver.resolve(alias).stitch_id == drug:
                brand_ok += 1
            else:
                failures.append(f"brand does not resolve {alias} -> expected {names.drug_name(drug)}")
    rows = [
        {"check": "C2 – thuốc có tên đọc được", "ok": readable, "total": len(drugs)},
        {"check": "C2 – tên thuốc nhận đúng thuốc (gõ riêng)", "ok": alone, "total": len(drugs)},
        {"check": "C2 – tên thuốc nhận đúng thuốc (trong câu chat)", "ok": in_chat, "total": len(drugs)},
        {"check": "C2 – biệt dược VN nhận đúng thuốc", "ok": brand_ok, "total": brands},
    ]
    return rows, failures


def c3_end_to_end(system, combo, targets, ppi, rounds: int, seed: int) -> tuple[list[dict], list[str], dict]:
    rng = random.Random(seed)
    drugs = sorted(system.graph.drug_ids)
    regimens: list[list[str]] = []
    for round_index in range(rounds):
        order = drugs[:]
        rng.shuffle(order)
        size = 3 + round_index % 4  # 3, 4, 5, 6 drugs
        for start in range(0, len(order), size):
            regimen = order[start : start + size]
            while len(regimen) < 3:  # pad the last group
                extra = rng.choice(drugs)
                if extra not in regimen:
                    regimen.append(extra)
            regimens.append(regimen)

    tally = collections.Counter()
    failures: list[str] = []
    seen_drugs: collections.Counter = collections.Counter()
    start = time.perf_counter()
    for regimen in regimens:
        try:
            result = system.predict(regimen)
        except Exception:  # noqa: BLE001 - every crash is a finding
            tally["crash"] += 1
            failures.append(f"crash {regimen}: {traceback.format_exc(limit=2)}")
            continue
        tally["regimens_ok"] += 1
        seen_drugs.update(regimen)
        by_name = {system.names.drug_name(drug): drug for drug in regimen}
        pairs = list(itertools.combinations(sorted(regimen), 2))
        for card in result["drugs"]:
            tally["cards"] += 1
            tally["cards_named"] += not UNREADABLE.search(card["name"])
        for item in result["interactions"]:
            evidence = item["evidence"]
            code = evidence[0]["side_effect_code"] if evidence else None
            if item["tier"] in (1, 2):
                tally["shown_t12"] += 1
                shown = {tuple(sorted(by_name[n] for n in pair)) for pair in item["pairs"]}
                truth = {pair for pair in pairs if code in combo.get(pair, set())}
                tally["shown_t12_supported"] += bool(shown) and shown <= truth
                tally["shown_t12_complete"] += shown == truth
                if not (shown and shown == truth):
                    failures.append(f"pairs mismatch {item['side_effect_en']} {regimen}")
            else:
                tally["shown_t3"] += 1
                pair = tuple(sorted(by_name[n] for n in item["pairs"][0])) if item["pairs"] else None
                tally["shown_t3_undocumented"] += pair is not None and not combo.get(pair)
        bridges = result["mechanism_bridges"]
        for bridge in bridges:
            tally["bridges"] += 1
            tally["bridges_undocumented"] += not combo.get(tuple(sorted(bridge["drug_pair"])))
        for ev in [e for item in result["interactions"] for e in item["evidence"]] + bridges:
            for edge in ev["path"]:
                tally["edges"] += 1
                relation = edge["relation"]
                if relation == "targets":
                    ok = edge["target"] in targets.get(edge["source"], set())
                elif relation == "interacts_with":
                    ok = (edge["source"], edge["target"]) in ppi
                elif relation == "has_side_effect":
                    p = tuple(sorted((edge["source"], edge["target"])))
                    ok = edge["side_effect_code"] in combo.get(p, set())
                else:
                    ok = False
                tally["edges_ok"] += ok
                if not ok:
                    failures.append(f"bad edge {edge} in {regimen}")
    seconds = time.perf_counter() - start
    rows = [
        {"check": "C3 – phác đồ chạy hết pipeline không lỗi", "ok": tally["regimens_ok"], "total": len(regimens)},
        {"check": "C3 – thuốc được phân tích ít nhất 1 lần", "ok": len(seen_drugs), "total": len(drugs)},
        {"check": "C3 – thẻ thuốc có tên đọc được", "ok": tally["cards_named"], "total": tally["cards"]},
        {"check": "C3 – TDP Tier 1/2 hiển thị: mọi cặp có trong CSV", "ok": tally["shown_t12_supported"], "total": tally["shown_t12"]},
        {"check": "C3 – TDP Tier 1/2 hiển thị: liệt kê đủ mọi cặp trong CSV", "ok": tally["shown_t12_complete"], "total": tally["shown_t12"]},
        {"check": "C3 – cầu protein (Tier 3) chỉ cho cặp chưa có dữ liệu", "ok": tally["bridges_undocumented"], "total": tally["bridges"]},
        {"check": "C3 – cạnh bằng chứng hiển thị (cả cầu Tier 3) đúng với file gốc", "ok": tally["edges_ok"], "total": tally["edges"]},
    ]
    meta = {
        "regimens": len(regimens),
        "min_times_per_drug": min(seen_drugs.values()) if seen_drugs else 0,
        "seconds_per_regimen": round(seconds / max(len(regimens), 1), 3),
    }
    return rows, failures, meta


def write(rows: list[dict], failures: list[str], meta: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["rate"] = round(row["ok"] / row["total"], 4) if row["total"] else "n/a"
    with open(RESULTS / "c_coverage.csv", "w", encoding="utf-8", newline="") as stream:
        stream.write("check,ok,total,rate\n")
        for row in rows:
            stream.write(f"\"{row['check']}\",{row['ok']},{row['total']},{row['rate']}\n")
    lines = [
        "# Độ phủ trên toàn bộ thuốc trong dữ liệu",
        "",
        "Sinh tự động bởi `python eval/run_coverage.py` (seed 0, không gọi LLM). "
        "Mọi kiểm tra đối chiếu với file CSV gốc, đọc độc lập với cache SQLite.",
        "",
        f"C3 chạy {meta['regimens']} phác đồ phủ mọi thuốc; mỗi thuốc xuất hiện ít nhất "
        f"{meta['min_times_per_drug']} lần; trung bình {meta['seconds_per_regimen']} s/phác đồ.",
        "",
        "| Kiểm tra | Đạt | Tổng | Tỷ lệ |",
        "|---|---|---|---|",
    ]
    for row in rows:
        rate = f"{row['rate'] * 100:.1f}%" if row["rate"] != "n/a" else "n/a"
        lines.append(f"| {row['check']} | {row['ok']} | {row['total']} | {rate} |")
    lines += ["", f"Số lỗi chi tiết: {len(failures)}."]
    if failures:
        lines += ["", "```", *failures[:50], "```"]
    (RESULTS / "COVERAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    system = PolypharmacySystem.load()
    assert not system.settings.llm_api_key, "LLM must stay disabled"
    print("Reading raw CSV files ...")
    combo, mono, targets, ppi = read_raw(system.settings)
    rows = c1_cache(system, combo, mono, targets)
    name_rows, failures = c2_names(system)
    rows += name_rows
    e2e_rows, e2e_failures, meta = c3_end_to_end(system, combo, targets, ppi, args.rounds, args.seed)
    rows += e2e_rows
    failures += e2e_failures
    for row in rows:
        print(f"  {row['check']}: {row['ok']}/{row['total']}")
    for line in failures[:30]:
        print("  !", line)
    write(rows, failures, meta)
    print(f"  -> {RESULTS / 'COVERAGE.md'}")


if __name__ == "__main__":
    main()
