"""Fixed-wording Vietnamese explanation of a side effect, used when the LLM is unavailable."""

from graphrag.drug_names.name_index import NameIndex


# Number of drug pairs named before "... và N cặp khác".
PAIRS_NAMED_IN_EXPLANATION = 3
PROTEINS_NAMED_IN_EXPLANATION = 3


def template_explanation(
    ranked_effect: dict[str, object], names: NameIndex, graph_context: dict[str, object]
) -> str:
    effect_name = names.side_effect_vietnamese(str(ranked_effect["name"]))
    pair_texts = [
        " và ".join(names.drug_name(drug) for drug in pair) for pair in ranked_effect["pairs"]
    ]
    pair_count = int(ranked_effect["recurrence"])

    text = f"Trong dữ liệu, mình thấy {effect_name} được ghi nhận"
    if pair_count > 1:
        text += f" ở {pair_count} cặp trong phác đồ"
    text += f": {'; '.join(pair_texts[:PAIRS_NAMED_IN_EXPLANATION])}"
    if pair_count > PAIRS_NAMED_IN_EXPLANATION:
        text += f" và {pair_count - PAIRS_NAMED_IN_EXPLANATION} cặp khác."
    else:
        text += "."

    if ranked_effect.get("amplified"):
        shared_entries = []
        for shared in graph_context["shared_targets"]:
            for pair in ranked_effect["amplified_pairs"]:
                if set(pair) <= set(shared["drugs"]):
                    shared_entries.append(shared)
                    break
        if shared_entries:
            protein_names = ", ".join(
                names.gene_symbol(shared["protein"])
                for shared in shared_entries[:PROTEINS_NAMED_IN_EXPLANATION]
            )
            drug_names = ", ".join(names.drug_name(drug) for drug in shared_entries[0]["drugs"])
            text += (
                f" Mình cũng thấy {drug_names} có protein đích chung ({protein_names}); "
                "đây là thông tin về cơ chế, chưa đủ để khẳng định nguy cơ tăng."
            )

    if not ranked_effect.get("absent_from_single_drug_data"):
        text += (
            " Có một điểm cần lưu ý: tác dụng phụ này cũng xuất hiện trong dữ liệu "
            "khi dùng riêng ít nhất một thuốc ở mỗi cặp, nên chưa thể quy riêng "
            "cho việc phối hợp."
        )
    return text
