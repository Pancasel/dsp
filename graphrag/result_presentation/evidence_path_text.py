"""Turn evidence paths, written with source IDs (STITCH, Entrez), into names people can read."""

from graphrag.drug_names.name_index import NameIndex
from graphrag.source_data.data_records import Evidence


def describe_path_in_one_line(
    path: list[dict[str, str]] | tuple[dict[str, str], ...],
    names: NameIndex,
    drugs: set[str],
    in_english: bool = False,
) -> str:
    """For example "Ibuprofen → PTGS2; PTGS2 ↔ PTGS1; Ibuprofen + Warfarin ⇒ xuất huyết"."""
    parts: list[str] = []
    for edge in path:
        source = names.path_node_label(edge["source"], drugs)
        target = names.path_node_label(edge["target"], drugs)
        relation = edge["relation"]
        if relation == "has_side_effect":
            effect_name = edge.get("side_effect_name", "")
            if not in_english:
                effect_name = names.side_effect_vietnamese(effect_name)
            parts.append(f"{source} + {target} ⇒ {effect_name}")
        elif relation == "targets":
            parts.append(f"{source} → {target}")
        else:
            parts.append(f"{source} ↔ {target}")
    return "; ".join(parts)


def readable_evidence(
    evidence: Evidence | dict, names: NameIndex, drugs: set[str]
) -> dict[str, object]:
    """The evidence dictionary plus drug names, a one-line path and a readable edge list."""
    if isinstance(evidence, Evidence):
        data = evidence.to_dict()
    else:
        data = dict(evidence)
    data["drug_pair_names"] = [names.drug_name(drug) for drug in data["drug_pair"]]
    data["path_text"] = describe_path_in_one_line(data["path"], names, drugs)
    readable_path = []
    for edge in data["path"]:
        if edge.get("side_effect_name"):
            side_effect_vietnamese = names.side_effect_vietnamese(edge["side_effect_name"])
        else:
            side_effect_vietnamese = None
        readable_path.append(
            {
                "source": names.path_node_label(edge["source"], drugs),
                "relation": edge["relation"],
                "target": names.path_node_label(edge["target"], drugs),
                "side_effect_vi": side_effect_vietnamese,
            }
        )
    data["readable_path"] = readable_path
    return data


def drug_names_of_pair(pair: list[str] | tuple[str, str], names: NameIndex) -> list[str]:
    return [names.drug_name(drug) for drug in pair]
