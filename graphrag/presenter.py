"""Readable LLM context, template fallback and the final /predict response.

The graph layer works with source IDs (STITCH, Entrez). This module is the only
place that turns them into names, Vietnamese labels and short path texts.
"""

import itertools

from graphrag.dataset import DecagonGraph, Evidence
from graphrag.names import NameIndex


TIER_LABELS_VI = {1: "Đã ghi nhận", 2: "Khuếch đại", 3: "Giả thuyết"}
DISCLAIMER_VI = (
    "Thông tin chỉ mang tính tham khảo ở mức quần thể (dữ liệu Decagon/FAERS), "
    "không có liều dùng hay mức độ nặng, và không thay thế tư vấn của bác sĩ "
    "hoặc dược sĩ. Không tự ý ngưng hay đổi thuốc. Mục 'Giả thuyết' (Tier 3) "
    "chưa được ghi nhận và cần kiểm chứng."
)
MAX_INDIVIDUAL_EFFECTS = 10
MAX_TARGET_GENES = 12


def path_text(
    path: list[dict[str, str]] | tuple[dict[str, str], ...],
    names: NameIndex,
    drugs: set[str],
    english: bool = False,
) -> str:
    """One-line readable path, e.g. "Ibuprofen → PTGS2; PTGS2 ↔ PTGS1"."""
    parts: list[str] = []
    for edge in path:
        source = names.node_label(edge["source"], drugs)
        target = names.node_label(edge["target"], drugs)
        relation = edge["relation"]
        if relation == "has_side_effect":
            label = edge.get("side_effect_name", "")
            effect = label if english else names.side_effect_vi(label)
            parts.append(f"{source} + {target} ⇒ {effect}")
        elif relation == "targets":
            parts.append(f"{source} → {target}")
        else:
            parts.append(f"{source} ↔ {target}")
    return "; ".join(parts)


def readable_evidence(item: Evidence | dict, names: NameIndex, drugs: set[str]) -> dict[str, object]:
    data = item.to_dict() if isinstance(item, Evidence) else dict(item)
    data["drug_pair_names"] = [names.drug_name(drug) for drug in data["drug_pair"]]
    data["path_text"] = path_text(data["path"], names, drugs)
    data["readable_path"] = [
        {
            "source": names.node_label(edge["source"], drugs),
            "relation": edge["relation"],
            "target": names.node_label(edge["target"], drugs),
            "side_effect_vi": names.side_effect_vi(edge["side_effect_name"])
            if edge.get("side_effect_name")
            else None,
        }
        for edge in data["path"]
    ]
    return data


def select_effects(
    ranked: list[dict[str, object]], max_serious: int, max_other: int
) -> list[dict[str, object]]:
    """The effects to show: top serious ones by score, then a few others.

    `ranked` is already sorted by score, so each group keeps that order.
    """
    serious = [item for item in ranked if item["serious"]][:max_serious]
    other = [item for item in ranked if not item["serious"]][:max_other]
    return serious + other


def build_llm_context(
    drugs: list[str],
    graph_context: dict[str, object],
    evidence: list[Evidence],
    names: NameIndex,
    max_serious: int,
    max_other: int,
) -> dict[str, object]:
    """Compact, name-resolved context. Evidence IDs stay identical to the graph's."""
    drug_set = set(drugs)
    top = select_effects(graph_context["ranked_effects"], max_serious, max_other)
    top_names = {item["name"].casefold() for item in top}
    serialized = []
    for item in evidence:
        if item.tier in (1, 2) and (item.side_effect or "").casefold() not in top_names:
            continue
        entry: dict[str, object] = {
            "evidence_id": item.evidence_id,
            "tier": item.tier,
            "drug_pair": [names.drug_label(drug) for drug in item.drug_pair],
            "path": path_text(item.path, names, drug_set, english=True),
        }
        if item.side_effect:
            entry["side_effect"] = item.side_effect
        if item.candidates:
            entry["candidate_side_effects"] = list(item.candidates)
        serialized.append(entry)
    return {
        "drugs": [names.drug_label(drug) for drug in drugs],
        "direct_target_counts": {
            names.drug_name(drug): count
            for drug, count in graph_context["direct_target_counts"].items()
        },
        "shared_targets": [
            {
                "protein": names.gene_symbol(item["protein"]),
                "drugs": [names.drug_name(drug) for drug in item["drugs"]],
            }
            for item in graph_context["shared_targets"]
        ],
        "top_side_effects": [
            {
                "side_effect": item["name"],
                "tier": item["tier"],
                "serious": item["serious"],
                "pairs": [[names.drug_name(d) for d in pair] for pair in item["pairs"]],
            }
            for item in top
        ],
        "evidence": serialized,
        "evidence_truncated": graph_context["evidence_truncated"],
    }


def template_explanation(item: dict[str, object], names: NameIndex, graph_context: dict[str, object]) -> str:
    """Deterministic Vietnamese explanation used when the LLM is unavailable."""
    effect = names.side_effect_vi(str(item["name"]))
    pairs = [
        " và ".join(names.drug_name(drug) for drug in pair) for pair in item["pairs"]
    ]
    text = f"Đã ghi nhận \"{effect}\" khi dùng đồng thời {'; '.join(pairs[:3])}"
    text += f" (thêm {len(pairs) - 3} cặp khác)." if len(pairs) > 3 else "."
    if item["recurrence"] and int(item["recurrence"]) > 1:
        text += f" Tác dụng phụ này xuất hiện ở {item['recurrence']} cặp trong phác đồ."
    if item.get("amplified"):
        shared = [
            entry
            for entry in graph_context["shared_targets"]
            if any(set(pair) <= set(entry["drugs"]) for pair in item["amplified_pairs"])
        ]
        if shared:
            proteins = ", ".join(names.gene_symbol(entry["protein"]) for entry in shared[:3])
            drugs = ", ".join(names.drug_name(d) for d in shared[0]["drugs"])
            text += f" {drugs} cùng tác động lên protein {proteins}, có thể làm tăng nguy cơ."
    if not item.get("not_in_mono"):
        # not_in_mono is False only when, for every pair, at least one of the
        # two drugs already lists this effect on its own.
        text += (
            " Lưu ý: tác dụng phụ này cũng đã được ghi nhận khi dùng riêng ít nhất "
            "một thuốc trong mỗi cặp, nên chưa chắc là do phối hợp."
        )
    return text


def individual_effects(graph: DecagonGraph, names: NameIndex, drug: str) -> list[dict[str, object]]:
    effects = graph.mono_effects(drug)
    # Decagon's mono list has no frequencies. Effects listed for few other
    # drugs are the most characteristic of this one; serious ones come first.
    drug_count = graph.mono_effect_drug_count()
    ordered = sorted(
        effects,
        key=lambda effect: (
            not names.is_serious(effect.name),
            drug_count.get(effect.code, 0),
            effect.name.casefold(),
        ),
    )
    return [
        {
            "name_en": names.side_effect_en(effect.name),
            "name_vi": names.side_effect_vi(effect.name),
            "serious": names.is_serious(effect.name),
        }
        for effect in ordered[:MAX_INDIVIDUAL_EFFECTS]
    ]


def drug_cards(
    graph: DecagonGraph,
    names: NameIndex,
    drugs: list[str],
    inputs: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    cards = []
    for drug in drugs:
        card = names.drug_display(drug)
        targets = sorted(graph.targets.get(drug, set()))
        card.update(
            {
                "input": inputs.get(drug, {}).get("input", drug),
                "match_method": inputs.get(drug, {}).get("method", "stitch_id"),
                "has_targets": bool(targets),
                "target_count": len(targets),
                "target_genes": [names.gene_symbol(gene) for gene in targets[:MAX_TARGET_GENES]],
                "individual_side_effect_count": len(graph.mono_effects(drug)),
                "individual_side_effects": individual_effects(graph, names, drug),
            }
        )
        cards.append(card)
    return cards


def build_interactions(
    drugs: list[str],
    graph_context: dict[str, object],
    evidence: list[Evidence],
    validated: list[dict[str, object]],
    names: NameIndex,
    max_serious: int,
    max_other: int,
) -> list[dict[str, object]]:
    drug_set = set(drugs)
    ranked = graph_context["ranked_effects"]
    by_name = {item["name"].casefold(): item for item in ranked}
    llm_by_name: dict[str, dict[str, object]] = {}
    for prediction in validated:
        if prediction["tier"] in (1, 2):
            llm_by_name.setdefault(str(prediction["side_effect"]).casefold(), prediction)

    evidence_by_effect: dict[str, list[Evidence]] = {}
    for item in evidence:
        if item.side_effect:
            evidence_by_effect.setdefault(item.side_effect.casefold(), []).append(item)

    chosen = select_effects(ranked, max_serious, max_other)
    chosen_names = {item["name"].casefold() for item in chosen}
    # Effects the LLM chose to explain outside the top-k are kept as well.
    for key in llm_by_name:
        if key not in chosen_names and key in by_name:
            chosen.append(by_name[key])
            chosen_names.add(key)

    interactions = []
    for item in chosen:
        key = item["name"].casefold()
        prediction = llm_by_name.get(key)
        cited = (
            prediction["evidence"]
            if prediction
            else [entry.to_dict() for entry in evidence_by_effect.get(key, [])[:3]]
        )
        if not cited:
            cited = [
                {
                    "evidence_id": None,
                    "tier": 1,
                    "kind": "known_pairwise_effect",
                    "drug_pair": list(pair),
                    "path": [
                        {
                            "source": pair[0],
                            "relation": "has_side_effect",
                            "target": pair[1],
                            "side_effect_code": item["code"],
                            "side_effect_name": item["name"],
                        }
                    ],
                }
                for pair in item["pairs"][:3]
            ]
        interactions.append(
            {
                "side_effect_en": names.side_effect_en(item["name"]),
                "side_effect_vi": names.side_effect_vi(item["name"]),
                "tier": item["tier"],
                "tier_label": TIER_LABELS_VI[item["tier"]],
                "hypothesis": False,
                "serious": item["serious"],
                "score": item["score"],
                "not_in_mono": item["not_in_mono"],
                "pairs": [[names.drug_name(d) for d in pair] for pair in item["pairs"]],
                # Tier 2 applies only to these pairs; the others are Tier 1.
                "amplified_pairs": [
                    [names.drug_name(d) for d in pair] for pair in item["amplified_pairs"]
                ],
                "explanation_vi": str(prediction["explanation"])
                if prediction
                else template_explanation(item, names, graph_context),
                "explanation_source": "llm" if prediction else "template",
                "evidence": [readable_evidence(entry, names, drug_set) for entry in cited],
            }
        )

    for prediction in validated:
        if prediction["tier"] != 3:
            continue
        cited = prediction["evidence"]
        pairs = [[names.drug_name(d) for d in entry["drug_pair"]] for entry in cited]
        interactions.append(
            {
                "side_effect_en": names.side_effect_en(str(prediction["side_effect"])),
                "side_effect_vi": names.side_effect_vi(str(prediction["side_effect"])),
                "tier": 3,
                "tier_label": TIER_LABELS_VI[3],
                "hypothesis": True,
                "serious": names.is_serious(str(prediction["side_effect"])),
                "score": None,
                "not_in_mono": None,
                "pairs": [list(pair) for pair in dict.fromkeys(tuple(p) for p in pairs)],
                "amplified_pairs": [],
                "explanation_vi": str(prediction["explanation"]),
                "explanation_source": "llm",
                "evidence": [readable_evidence(entry, names, drug_set) for entry in cited],
            }
        )
    return interactions


def mechanism_bridges(
    drugs: list[str], evidence: list[Evidence], names: NameIndex, limit_per_pair: int = 2
) -> list[dict[str, object]]:
    """Protein bridges between undocumented pairs, shown even without an LLM."""
    drug_set = set(drugs)
    shown: dict[tuple[str, str], int] = {}
    bridges = []
    for item in evidence:
        if item.tier != 3:
            continue
        count = shown.get(item.drug_pair, 0)
        if count >= limit_per_pair:
            continue
        shown[item.drug_pair] = count + 1
        bridges.append(readable_evidence(item, names, drug_set))
    return bridges


def pair_summary(
    drugs: list[str], graph_context: dict[str, object], names: NameIndex
) -> list[dict[str, object]]:
    """Per drug pair: number of documented effects and how many are serious."""
    counts: dict[tuple[str, str], list[int]] = {
        pair: [0, 0] for pair in itertools.combinations(sorted(drugs), 2)
    }
    for item in graph_context["ranked_effects"]:
        for pair in item["pairs"]:
            entry = counts[tuple(pair)]
            entry[0] += 1
            entry[1] += bool(item["serious"])
    return [
        {
            "pair": [names.drug_name(drug) for drug in pair],
            "documented_effects": total,
            "serious_effects": serious,
        }
        for pair, (total, serious) in counts.items()
    ]


def shared_targets(graph_context: dict[str, object], names: NameIndex) -> list[dict[str, object]]:
    """Proteins targeted by three or more regimen drugs (the Tier 2 signal)."""
    return [
        {
            "protein": names.gene_symbol(item["protein"]),
            "drugs": [names.drug_name(drug) for drug in item["drugs"]],
        }
        for item in graph_context["shared_targets"]
    ]


def coverage(
    drugs: list[str], graph: DecagonGraph, graph_context: dict[str, object], evidence: list[Evidence], names: NameIndex
) -> dict[str, object]:
    pairs = list(itertools.combinations(sorted(drugs), 2))
    documented = {tuple(pair) for pair in graph_context["pairs_with_documented_effects"]}
    bridged = {item.drug_pair for item in evidence if item.tier == 3}
    return {
        "pairs_total": len(pairs),
        "pairs_with_data": len(documented),
        "pairs_with_bridge_only": [
            [names.drug_name(d) for d in pair] for pair in pairs if pair in bridged
        ],
        "pairs_without_any_evidence": [
            [names.drug_name(d) for d in pair]
            for pair in pairs
            if pair not in documented and pair not in bridged
        ],
        "drugs_without_targets": [
            names.drug_name(drug) for drug in drugs if not graph.targets.get(drug)
        ],
        "documented_effects_total": len(graph_context["ranked_effects"]),
    }
