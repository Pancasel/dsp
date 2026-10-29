"""Draw one analysis result: side-effect cards first, then the folded "Chi tiết thêm" tabs."""

import html
import json
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from web_interface.cached_resources import interaction_graph_html
from web_interface.interface_texts import (
    EFFECT_CARDS_SHOWN_FIRST,
    TIER_COLORS,
    TIER_EXPLANATIONS,
)


# Evidence IDs ("E3", "(E1, E2)") and drug IDs ("(CID000003672)") that the LLM
# may write into an explanation; users never see those.
EVIDENCE_REFERENCE_PATTERN = re.compile(
    r"\s*\(CID\d+\)|\s*\((?:E\d+(?:\s*(?:,|;|và|and|&)\s*)?)+\)|\bE\d+\b"
)
GRAPH_FRAME_HEIGHT = 560


def remove_evidence_references(text: str) -> str:
    text = EVIDENCE_REFERENCE_PATTERN.sub("", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def format_drug_pairs(pairs: list) -> str:
    """[["A", "B"], ["A", "C"]] -> "A + B; A + C"."""
    return "; ".join(" + ".join(pair) for pair in pairs)


def render_effect_card(rank: int, card: dict) -> None:
    vietnamese_name = html.escape(card["side_effect_vi"])
    english_name = html.escape(card["side_effect_en"])
    tier_color = TIER_COLORS[card["tier"]]
    warning_icon = "⚠️ " if card["serious"] else ""
    if vietnamese_name != english_name:
        english_html = f"<span class='effect-card-english'>{english_name}</span>"
    else:
        english_html = ""
    with st.container(border=True):
        st.markdown(
            f"<div class='effect-card-head'><span class='effect-card-rank'>{rank}</span>"
            f"<span class='effect-card-name'>{warning_icon}{vietnamese_name}</span>{english_html}"
            f"<span class='tier-badge' style='background:{tier_color}'>Tier {card['tier']} · "
            f"{html.escape(card['tier_label'])}</span></div>",
            unsafe_allow_html=True,
        )
        st.write(remove_evidence_references(card["explanation_vi"]))
        st.markdown(
            "<div class='effect-card-pairs'>Được báo cáo ở: "
            f"{html.escape(format_drug_pairs(card['pairs']))}</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Bằng chứng"):
            st.caption(TIER_EXPLANATIONS[card["tier"]])
            if card["amplified_pairs"]:
                st.caption(
                    "Có protein đích chung với thuốc thứ ba ở: "
                    + format_drug_pairs(card["amplified_pairs"])
                )
            for evidence in card["evidence"]:
                st.code(evidence["path_text"], language=None)


def render_answer(result: dict, widget_key: str) -> None:
    """The bot's answer: the most relevant effects as cards, details folded away.

    widget_key keeps the widgets of several answers on the page apart.
    """
    coverage = result["coverage"]
    serious_cards = [card for card in result["interactions"] if card["serious"]]
    other_cards = [card for card in result["interactions"] if not card["serious"]]

    st.markdown(
        f"<div class='answer-summary'>{coverage['pairs_with_data']}/{coverage['pairs_total']} "
        "cặp thuốc có dữ liệu · "
        f"{coverage['documented_effects_total']} tác dụng phụ đã được ghi nhận · "
        f"hiển thị {len(serious_cards)} mục nghiêm trọng nhất</div>",
        unsafe_allow_html=True,
    )
    if result["llm_status"] != "ok":
        st.warning(f"Phần giải thích đang dùng câu mẫu vì LLM không dùng được: {result['llm_error']}")
    if not serious_cards:
        st.info("Không có tác dụng phụ nghiêm trọng nào được ghi nhận cho phác đồ này.")
    for rank, card in enumerate(serious_cards[:EFFECT_CARDS_SHOWN_FIRST], start=1):
        render_effect_card(rank, card)
    hidden_serious_count = len(serious_cards) - EFFECT_CARDS_SHOWN_FIRST
    if hidden_serious_count > 0:
        with st.expander(f"Xem thêm {hidden_serious_count} tác dụng phụ nghiêm trọng"):
            hidden_cards = serious_cards[EFFECT_CARDS_SHOWN_FIRST:]
            for rank, card in enumerate(hidden_cards, start=EFFECT_CARDS_SHOWN_FIRST + 1):
                render_effect_card(rank, card)
    if other_cards:
        with st.expander(f"Tác dụng phụ khác thường gặp ({len(other_cards)})"):
            for rank, card in enumerate(other_cards, start=len(serious_cards) + 1):
                render_effect_card(rank, card)

    with st.expander("Chi tiết thêm: từng thuốc, protein, đồ thị"):
        render_detail_tabs(result, widget_key)
    st.caption(
        result["disclaimer"]
        + " Phần mô tả tác dụng phụ do AI viết, mang tính chung và không lấy từ dữ liệu."
    )


def render_detail_tabs(result: dict, widget_key: str) -> None:
    """Secondary information: per-drug table, protein mechanisms, relation graph, JSON download."""
    drugs_tab, mechanism_tab, graph_tab = st.tabs(["Từng thuốc", "Cơ chế protein", "Đồ thị quan hệ"])
    with drugs_tab:
        _render_drug_table(result)
    with mechanism_tab:
        _render_protein_mechanisms(result)
    with graph_tab:
        _render_relation_graph(result)
    st.download_button(
        "Tải kết quả JSON",
        json.dumps(result, ensure_ascii=False, indent=2),
        file_name="ket_qua_tuong_tac.json",
        mime="application/json",
        key=f"dl_{widget_key}",
    )


def _render_drug_table(result: dict) -> None:
    rows = []
    for drug in result["drugs"]:
        top_single_drug_effects = [
            ("⚠️ " if effect["serious"] else "") + effect["name_vi"]
            for effect in drug["individual_side_effects"][:4]
        ]
        if drug["has_targets"]:
            target_text = f"{drug['target_count']}: " + ", ".join(drug["target_genes"][:5])
        else:
            target_text = "Không có dữ liệu"
        rows.append(
            {
                "Thuốc": drug["name"],
                "Biệt dược": ", ".join(drug["aliases_vi"][:4]) or "–",
                "Protein đích": target_text,
                "Tác dụng phụ khi dùng riêng": "; ".join(top_single_drug_effects) or "–",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    drugs_without_targets = result["coverage"]["drugs_without_targets"]
    if drugs_without_targets:
        st.caption(
            "Không có dữ liệu protein đích: "
            + ", ".join(drugs_without_targets)
            + ". Với các thuốc này chỉ có bằng chứng Tier 1."
        )


def _render_protein_mechanisms(result: dict) -> None:
    if result["pair_summary"]:
        pair_rows = [
            {
                "Cặp thuốc": " + ".join(pair_summary["pair"]),
                "Đã ghi nhận": pair_summary["documented_effects"],
                "Nghiêm trọng": pair_summary["serious_effects"],
            }
            for pair_summary in result["pair_summary"]
        ]
        st.dataframe(pd.DataFrame(pair_rows), hide_index=True, width="stretch")
    for shared in result["shared_targets"]:
        st.write(f"Protein chung `{shared['protein']}`: " + ", ".join(shared["drugs"]))
    for bridge in result["mechanism_bridges"]:
        st.code(f"{' + '.join(bridge['drug_pair_names'])}: {bridge['path_text']}", language=None)
    if result["mechanism_bridges"]:
        st.caption("Các liên kết protein trên là giả thuyết cho cặp chưa có dữ liệu, chưa phải kết luận.")
    elif not result["shared_targets"]:
        st.caption("Không có protein chung hay cầu nối protein nào cho phác đồ này.")


def _render_relation_graph(result: dict) -> None:
    st.caption(
        "🔵 thuốc · 🔶 protein đích chung · 🟣 protein cầu nối · 🟥 tác dụng phụ nghiêm trọng. "
        "Kéo, phóng to, rê chuột để xem chi tiết."
    )
    try:
        graph_page = interaction_graph_html(json.dumps(result, ensure_ascii=False), True, True)
    except ImportError:
        st.warning(
            "Thiếu thư viện `pyvis` trong môi trường Python đang chạy app. "
            "Cài bằng `python -m pip install -r requirements.txt` (đúng venv của dự án) rồi tải lại trang."
        )
        return
    if hasattr(st, "iframe"):  # Streamlit >= 1.52 replaces components.html
        st.iframe(graph_page, height=GRAPH_FRAME_HEIGHT)
    else:
        components.html(graph_page, height=GRAPH_FRAME_HEIGHT, scrolling=False)
