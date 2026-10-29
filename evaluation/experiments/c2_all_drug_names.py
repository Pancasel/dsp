"""C2: does every drug have a readable name that resolves back to it, alone and inside a chat sentence?

Also checks that every curated Vietnamese brand resolves to its drug.
"""

import re

from graphrag.drug_names.text_normalization import normalize_for_matching
from graphrag.pipeline.polypharmacy_system import PolypharmacySystem
from evaluation.experiments.coverage_raw_source_data import UNREADABLE_NAME_PATTERN


MINIMUM_BRAND_KEY_LENGTH = 3


def run_drug_name_check(system: PolypharmacySystem) -> tuple[list[dict], list[str]]:
    resolver, names = system.resolver, system.names
    unreadable_pattern = re.compile(UNREADABLE_NAME_PATTERN)
    failures: list[str] = []
    drugs = sorted(system.graph.drug_ids)
    readable_count = resolved_alone_count = resolved_in_chat_count = 0
    for drug in drugs:
        name = names.drug_name(drug)
        if unreadable_pattern.search(name):
            failures.append(f"unreadable name {drug}: {name}")
        else:
            readable_count += 1
        if resolver.resolve(name).stitch_id == drug:
            resolved_alone_count += 1
        else:
            failures.append(f"name does not resolve {drug}: {name}")
        found = [resolution.stitch_id for resolution in resolver.extract(f"tôi đang uống {name} với aspirin")]
        if drug in found:
            resolved_in_chat_count += 1
        else:
            failures.append(f"name not found in chat {drug}: {name} -> {found}")

    brand_count = resolved_brand_count = 0
    for drug in drugs:
        for brand in names.drugs_by_id[drug].vietnamese_brand_aliases:
            if len(normalize_for_matching(brand)) < MINIMUM_BRAND_KEY_LENGTH:
                continue
            brand_count += 1
            if resolver.resolve(brand).stitch_id == drug:
                resolved_brand_count += 1
            else:
                failures.append(f"brand does not resolve {brand} -> expected {names.drug_name(drug)}")
    rows = [
        {"check": "C2 – thuốc có tên đọc được", "ok": readable_count, "total": len(drugs)},
        {"check": "C2 – tên thuốc nhận đúng thuốc (gõ riêng)", "ok": resolved_alone_count, "total": len(drugs)},
        {"check": "C2 – tên thuốc nhận đúng thuốc (trong câu chat)", "ok": resolved_in_chat_count, "total": len(drugs)},
        {"check": "C2 – biệt dược VN nhận đúng thuốc", "ok": resolved_brand_count, "total": brand_count},
    ]
    return rows, failures
