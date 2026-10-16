"""Local UI: pick or photograph drugs, then show each drug and combined side effects.

    streamlit run app.py

The Graph-RAG pipeline runs in this process (no separate API server).
"""

import json

import httpx
import streamlit as st

from graphrag.llm import LLMConfigurationError, LLMResponseError
from graphrag.pipeline import PolypharmacySystem, RegimenError
from graphrag.vision import ImageError

TIER_COLORS = {1: "#2563eb", 2: "#ea580c", 3: "#7c3aed"}
TIER_HELP = {
    1: "Đã ghi nhận: tác dụng phụ đã được báo cáo cho cặp thuốc này.",
    2: "Khuếch đại: đã ghi nhận, và có protein đích chung của ≥3 thuốc trong phác đồ.",
    3: "Giả thuyết: cặp thuốc chưa có dữ liệu, suy luận qua liên kết protein. Cần kiểm chứng.",
}

st.set_page_config(page_title="Tương tác thuốc", page_icon="💊", layout="wide")


@st.cache_resource(show_spinner="Đang nạp đồ thị Decagon (lần đầu mất khoảng 2 phút để tạo cache)…")
def load_system() -> PolypharmacySystem:
    return PolypharmacySystem.load()


@st.cache_data
def load_drugs() -> list[dict]:
    return load_system().list_drugs()


def drug_option(item: dict) -> str:
    aliases = ", ".join(item["aliases_vi"][:3])
    return f"{item['name']} ({aliases})" if aliases else item["name"]


def add_to_regimen(stitch_id: str, name: str) -> None:
    regimen = st.session_state.setdefault("regimen", {})
    regimen[stitch_id] = name


def badge(tier: int, label: str) -> str:
    return (
        f"<span style='background:{TIER_COLORS[tier]};color:white;padding:2px 8px;"
        f"border-radius:10px;font-size:0.8em'>Tier {tier} · {label}</span>"
    )


def evidence_graph(evidence: list[dict]) -> str:
    """Graphviz DOT for the cited paths, using readable labels."""
    lines = ["graph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#eef2ff"];']
    seen = set()
    for item in evidence:
        for edge in item["readable_path"]:
            source, target = edge["source"], edge["target"]
            relations = {"targets": "tác động", "interacts_with": "tương tác"}
            label = edge["side_effect_vi"] or relations.get(edge["relation"], edge["relation"])
            key = (source, target, label)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f'"{source}" -- "{target}" [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


st.title("💊 Kiểm tra tác dụng phụ khi phối hợp thuốc")
try:
    system = load_system()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Không nạp được dữ liệu: {exc}. Kiểm tra đường dẫn CSV trong .env")
    st.stop()
health = system.status()

with st.sidebar:
    st.subheader("Trạng thái")
    st.write(f"Thuốc trong dữ liệu: **{health['drugs']}** ({health['drugs_with_targets']} có protein đích)")
    st.write(f"Bản dịch tác dụng phụ: **{health['side_effect_translations']}**")
    st.write("LLM: " + (f"`{health['llm_model']}`" if health["llm_configured"] else "chưa cấu hình (dùng giải thích mẫu)"))
    st.caption("Dữ liệu: Decagon/BioSNAP (645 thuốc). Thuốc ngoài danh sách sẽ không tra được.")

drugs = load_drugs()
by_label = {drug_option(item): item for item in drugs}
st.session_state.setdefault("regimen", {})

left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("1. Chọn thuốc")
    picked = st.multiselect("Tìm theo tên hoạt chất hoặc biệt dược", list(by_label), placeholder="Gõ tên thuốc…")
    if st.button("Thêm thuốc đã chọn", disabled=not picked):
        for label in picked:
            add_to_regimen(by_label[label]["stitch_id"], by_label[label]["name"])

    free_text = st.text_input("Hoặc gõ nhiều tên, cách nhau bằng dấu phẩy", placeholder="Panadol, ibuprofen, warfarin")
    if st.button("Nhận diện tên", disabled=not free_text.strip()):
        st.session_state["text_results"] = [
            system.search(part.strip()).to_dict()
            for part in free_text.replace(";", ",").split(",")
            if part.strip()
        ]
    for index, result in enumerate(st.session_state.get("text_results", [])):
        if result["stitch_id"]:
            add_to_regimen(result["stitch_id"], result["name"])
            st.success(f"“{result['query']}” → {result['name']}")
        elif result["candidates"]:
            options = {f"{c['name']} (khớp '{c['matched']}', {c['score']:.0f})": c for c in result["candidates"]}
            choice = st.radio(f"“{result['query']}” là thuốc nào?", list(options), key=f"pick_text_{index}", index=None)
            if choice:
                add_to_regimen(options[choice]["stitch_id"], options[choice]["name"])
        else:
            st.warning(f"“{result['query']}” không có trong dữ liệu Decagon")
    st.session_state["text_results"] = [r for r in st.session_state.get("text_results", []) if not r["stitch_id"]]

    st.markdown("**Hoặc chụp/tải ảnh hộp, vỉ, đơn thuốc**")
    photo = st.camera_input("Chụp ảnh", label_visibility="collapsed")
    upload = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png", "webp"])
    image = photo or upload
    if image is not None and st.button("Đọc thuốc trong ảnh"):
        try:
            with st.spinner("Đang đọc ảnh…"):
                st.session_state["image_result"] = system.identify_image(image.getvalue(), image.type)
        except ImageError as exc:
            st.error(f"Ảnh không hợp lệ: {exc}")
        except LLMConfigurationError as exc:
            st.error(f"Chưa cấu hình model đọc ảnh: {exc}")
        except (LLMResponseError, httpx.HTTPError) as exc:
            st.error(f"Lỗi khi gọi model đọc ảnh: {exc}")
    image_result = st.session_state.get("image_result")
    if image_result:
        if image_result["message"]:
            st.info(image_result["message"])
        for index, item in enumerate(image_result["detected"]):
            if item["stitch_id"]:
                if st.checkbox(f"{item['detected_text']} → **{item['name']}**", value=True, key=f"img_{index}"):
                    add_to_regimen(item["stitch_id"], item["name"])
            elif item["candidates"]:
                options = {c["name"]: c for c in item["candidates"]}
                choice = st.radio(f"“{item['detected_text']}” là thuốc nào?", list(options), key=f"img_pick_{index}", index=None)
                if choice:
                    add_to_regimen(options[choice]["stitch_id"], choice)
            else:
                st.warning(f"“{item['detected_text']}” không có trong dữ liệu")
        with st.expander("Chữ đọc được"):
            st.text(image_result["raw_text"])

    st.subheader("2. Phác đồ hiện tại")
    regimen: dict = st.session_state["regimen"]
    if not regimen:
        st.caption("Chưa có thuốc nào.")
    for stitch_id, name in list(regimen.items()):
        cols = st.columns([4, 1])
        cols[0].write(f"• {name}")
        if cols[1].button("Xóa", key=f"del_{stitch_id}"):
            regimen.pop(stitch_id)
            st.rerun()
    top_k = st.slider("Số tác dụng phụ hiển thị", 5, 50, 15)
    analyze = st.button("🔍 Phân tích", type="primary", disabled=len(regimen) < 3, use_container_width=True)
    if len(regimen) < 3:
        st.caption("Cần ít nhất 3 thuốc.")

if analyze:
    try:
        with right, st.spinner("Đang truy xuất đồ thị và tạo giải thích…"):
            st.session_state["result"] = system.predict(list(st.session_state["regimen"]), top_k)
    except RegimenError as exc:
        st.session_state["result"] = None
        right.error(str(exc))

result = st.session_state.get("result")
with right:
    if not result:
        st.info("Chọn ít nhất 3 thuốc rồi bấm **Phân tích**.")
    else:
        st.warning(result["disclaimer"], icon="⚠️")
        if result["llm_status"] != "ok":
            st.caption(f"Giải thích đang dùng mẫu có sẵn (LLM không dùng được: {result['llm_error']}).")

        st.subheader("Thuốc trong phác đồ")
        cards = st.columns(len(result["drugs"]))
        for column, drug in zip(cards, result["drugs"]):
            with column, st.container(border=True):
                st.markdown(f"**{drug['name']}**")
                if drug["aliases_vi"]:
                    st.caption("Biệt dược: " + ", ".join(drug["aliases_vi"][:4]))
                if drug["input"] != drug["stitch_id"]:
                    st.caption(f"Nhập: “{drug['input']}” · {drug['stitch_id']}")
                else:
                    st.caption(drug["stitch_id"])
                if drug["has_targets"]:
                    st.caption(f"Protein đích ({drug['target_count']}): " + ", ".join(drug["target_genes"][:6]))
                else:
                    st.caption("Không có dữ liệu protein đích")
                with st.expander(f"Tác dụng phụ riêng ({drug['individual_side_effect_count']})"):
                    for effect in drug["individual_side_effects"]:
                        mark = "⚠️ " if effect["serious"] else ""
                        english = f" ({effect['name_en']})" if effect["name_vi"] != effect["name_en"] else ""
                        st.write(f"{mark}{effect['name_vi']}{english}")

        cov = result["coverage"]
        st.subheader("Tác dụng phụ khi phối hợp")
        st.caption(
            f"{cov['pairs_with_data']}/{cov['pairs_total']} cặp thuốc có dữ liệu ghi nhận · "
            f"{cov['documented_effects_total']} tác dụng phụ đã ghi nhận, hiển thị theo mức ưu tiên"
            + (f" · Không có dữ liệu protein: {', '.join(cov['drugs_without_targets'])}" if cov["drugs_without_targets"] else "")
        )
        filters = st.columns(2)
        tiers = filters[0].multiselect("Tier", [1, 2, 3], default=[1, 2, 3], format_func=lambda t: TIER_HELP[t].split(":")[0])
        only_serious = filters[1].toggle("Chỉ hiện tác dụng phụ nghiêm trọng")
        shown = [
            item for item in result["interactions"]
            if item["tier"] in tiers and (item["serious"] or not only_serious)
        ]
        if not shown:
            st.info("Không có tác dụng phụ nào khớp bộ lọc.")
        for index, item in enumerate(shown):
            with st.container(border=True):
                mark = "⚠️ " if item["serious"] else ""
                english = (
                    f" <span style='color:gray'>({item['side_effect_en']})</span>"
                    if item["side_effect_vi"] != item["side_effect_en"]
                    else ""
                )
                st.markdown(
                    f"{badge(item['tier'], item['tier_label'])} &nbsp; **{mark}{item['side_effect_vi']}**{english}",
                    unsafe_allow_html=True,
                )
                st.caption("Cặp thuốc: " + "; ".join(" + ".join(pair) for pair in item["pairs"]))
                if item["amplified_pairs"]:
                    st.caption(
                        "Khuếch đại (protein đích chung với thuốc thứ ba) ở: "
                        + "; ".join(" + ".join(pair) for pair in item["amplified_pairs"])
                    )
                st.write(item["explanation_vi"])
                with st.expander("Bằng chứng"):
                    st.caption(TIER_HELP[item["tier"]])
                    for evidence in item["evidence"]:
                        st.code(evidence["path_text"], language=None)
                    st.graphviz_chart(evidence_graph(item["evidence"]))

        if result["mechanism_bridges"]:
            with st.expander("Liên kết protein giữa các cặp chưa có dữ liệu (chưa phải kết luận)"):
                for bridge in result["mechanism_bridges"]:
                    st.code(f"{' + '.join(bridge['drug_pair_names'])}: {bridge['path_text']}", language=None)
        if cov["pairs_without_any_evidence"]:
            st.caption(
                "Không có dữ liệu nào cho: "
                + "; ".join(" + ".join(pair) for pair in cov["pairs_without_any_evidence"])
            )
        st.download_button(
            "Tải kết quả JSON",
            json.dumps(result, ensure_ascii=False, indent=2),
            file_name="ket_qua_tuong_tac.json",
            mime="application/json",
        )
