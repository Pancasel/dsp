"""Chat UI: type the drugs you take, get the dangerous combined side effects.

    streamlit run app.py

The Graph-RAG pipeline runs in this process (no separate API server). Each
analysis sends at most one LLM request.
"""

import json
import re
import unicodedata

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from graphrag.names import normalize
from graphrag.pipeline import PolypharmacySystem, RegimenError
from graphrag.visualize import build_graph_html  # imports pyvis lazily

TIER_COLORS = {1: "#2563eb", 2: "#ea580c", 3: "#7c3aed"}
TIER_HELP = {
    1: "Đã ghi nhận: tác dụng phụ đã được báo cáo cho cặp thuốc này.",
    2: "Khuếch đại: đã ghi nhận, và cặp thuốc có protein đích chung với thuốc thứ ba.",
    3: "Giả thuyết: cặp thuốc chưa có dữ liệu, suy luận qua liên kết protein. Cần kiểm chứng.",
}
RESET_COMMANDS = {"lam lai", "reset", "xoa het", "phac do moi", "bat dau lai", "clear"}
# Matched on the accented text: without accents "bỏ" (remove) and "bố"
# (father, as in "bố tôi uống ...") would both become "bo".
REMOVE_PREFIXES = ("bỏ ", "bớt ", "xóa ", "xoá ", "xoa ", "remove ", "delete ")
WELCOME = (
    "Chào bạn! Hãy cho mình biết bạn đang dùng những thuốc nào, ví dụ: "
    "*“tôi đang uống Panadol, ibuprofen với warfarin”*. Có thể gõ tên hoạt chất "
    "hoặc biệt dược, mỗi lần một hay nhiều thuốc. Khi có từ 3 thuốc trở lên mình "
    "sẽ phân tích tác dụng phụ khi phối hợp.\n\n"
    "Lệnh: **bỏ <tên thuốc>** để bỏ một thuốc, **làm lại** để bắt đầu phác đồ mới."
)

st.set_page_config(page_title="Tương tác thuốc", page_icon="💊", layout="wide")


@st.cache_resource(show_spinner="Đang nạp đồ thị Decagon (lần đầu mất khoảng 2 phút để tạo cache)…")
def load_system() -> PolypharmacySystem:
    return PolypharmacySystem.load()


@st.cache_data(show_spinner=False, max_entries=8)
def graph_html(result_json: str, show_proteins: bool, show_effects: bool) -> str:
    return build_graph_html(json.loads(result_json), show_proteins, show_effects)


def init_state() -> None:
    st.session_state.setdefault("messages", [{"role": "assistant", "content": WELCOME}])
    st.session_state.setdefault("regimen", {})  # stitch_id -> display name
    st.session_state.setdefault("pending", [])  # ambiguous inputs to confirm
    st.session_state.setdefault("result", None)


def say(content: str, result: dict | None = None) -> None:
    st.session_state["messages"].append({"role": "assistant", "content": content, "result": result})


def analyse_if_ready(system: PolypharmacySystem) -> None:
    regimen: dict = st.session_state["regimen"]
    names = ", ".join(regimen.values())
    if len(regimen) < 3:
        st.session_state["result"] = None  # an older result no longer matches
        need = 3 - len(regimen)
        say(f"Phác đồ hiện có {len(regimen)} thuốc ({names or 'chưa có'}). Cần thêm ít nhất {need} thuốc để phân tích.")
        return
    try:
        result = system.predict(list(regimen))
    except RegimenError as exc:
        say(f"Chưa phân tích được: {exc}")
        return
    st.session_state["result"] = result
    serious = sum(item["serious"] for item in result["interactions"])
    say(f"Kết quả cho **{names}**: {serious} tác dụng phụ nghiêm trọng đáng chú ý nhất.", result)


def add_drugs(found: list) -> list[str]:
    added = []
    for item in found:
        if item.stitch_id and item.stitch_id not in st.session_state["regimen"]:
            st.session_state["regimen"][item.stitch_id] = item.name
            note = f" (từ “{item.query}”)" if normalize(item.query) != normalize(item.name) else ""
            added.append(f"**{item.name}**{note}")
    return added


def handle_message(system: PolypharmacySystem, text: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": text})
    key = normalize(text)
    if key in RESET_COMMANDS:
        st.session_state.update(regimen={}, pending=[], result=None)
        say("Đã xóa phác đồ. Bạn đang dùng những thuốc nào?")
        return
    lowered = unicodedata.normalize("NFC", text).strip().casefold()
    if lowered.startswith(REMOVE_PREFIXES):
        targets = system.resolver.extract(lowered.split(" ", 1)[1])
        removed = [st.session_state["regimen"].pop(item.stitch_id) for item in targets
                   if item.stitch_id in st.session_state["regimen"]]
        if not removed:
            say("Không tìm thấy thuốc đó trong phác đồ hiện tại.")
            return
        say(f"Đã bỏ {', '.join(removed)}.")
        analyse_if_ready(system)
        return

    found = system.resolver.extract(text)
    if not found:
        say("Mình chưa thấy tên thuốc nào trong tin nhắn. Bạn thử gõ tên hoạt chất hoặc biệt dược, ví dụ “Panadol, ibuprofen, warfarin”.")
        return
    added = add_drugs(found)
    pending = [item.to_dict() for item in found if item.stitch_id is None and item.candidates and item.method == "ambiguous"]
    missing = [item.query for item in found
               if item.stitch_id is None and not (item.method == "ambiguous" and item.candidates)]
    st.session_state["pending"].extend(pending)
    lines = []
    if added:
        lines.append("Đã nhận: " + ", ".join(added) + ".")
    if pending:
        lines.append("Chưa chắc chắn: " + ", ".join(f"“{item['query']}”" for item in pending) + " — chọn thuốc đúng ở bên dưới.")
    if missing:
        lines.append("Không có trong dữ liệu Decagon: " + ", ".join(f"“{name}”" for name in missing) + ".")
    say(" ".join(lines) or "Các thuốc này đã có trong phác đồ.")
    if added:
        analyse_if_ready(system)


def choose_candidate(system: PolypharmacySystem, index: int, candidate: dict | None) -> None:
    item = st.session_state["pending"].pop(index)
    if candidate is None:
        say(f"Đã bỏ qua “{item['query']}”.")
        return
    st.session_state["regimen"][candidate["stitch_id"]] = candidate["name"]
    say(f"Đã nhận **{candidate['name']}** cho “{item['query']}”.")
    analyse_if_ready(system)


EVIDENCE_REF = re.compile(r"\s*\(CID\d+\)|\s*\((?:E\d+(?:\s*(?:,|;|và|and|&)\s*)?)+\)|\bE\d+\b")
TIER_ICON = {1: "🔵 Tier 1", 2: "🟠 Tier 2", 3: "🟣 Tier 3"}
CSS = """
<style>
.block-container {max-width: 1250px; padding-top: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
[data-testid="stMetricLabel"] {font-size: 0.85rem; color: #6b7280;}
.legend span {display:inline-block; margin-right:14px; font-size:0.85rem; color:#4b5563;}
.dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px;}
.effect-title {font-size:1.02rem; font-weight:600;}
.effect-sub {color:#6b7280; font-size:0.85rem; margin-bottom:0.4rem;}
</style>
"""


def clean_explanation(text: str) -> str:
    """The LLM cites evidence ids (E1, E2); users never see those ids."""
    text = EVIDENCE_REF.sub("", text)
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+([,.;])", r"", text)).strip()


def pair_text(pairs: list) -> str:
    return "; ".join(" + ".join(pair) for pair in pairs)


def legend() -> None:
    st.markdown(
        "<div class='legend'>"
        + "".join(
            f"<span><i class='dot' style='background:{TIER_COLORS[t]}'></i>{TIER_ICON[t].split(' ', 1)[1]}: "
            f"{label}</span>"
            for t, label in ((1, "đã ghi nhận"), (2, "khuếch đại qua protein chung"), (3, "giả thuyết, cần kiểm chứng"))
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def render_effect_detail(rank: int, item: dict) -> None:
    vi, en = item["side_effect_vi"], item["side_effect_en"]
    label = f"{rank}. {'⚠️ ' if item['serious'] else ''}{vi}" + (f"  ·  {en}" if vi != en else "")
    with st.expander(label, expanded=rank == 1):
        st.markdown(
            f"<div class='effect-sub'>{TIER_ICON[item['tier']]} · {item['tier_label']} &nbsp;|&nbsp; "
            f"{pair_text(item['pairs'])}</div>",
            unsafe_allow_html=True,
        )
        st.write(clean_explanation(item["explanation_vi"]))
        if item["amplified_pairs"]:
            st.info("Khuếch đại do protein đích chung với thuốc thứ ba ở: " + pair_text(item["amplified_pairs"]))
        st.caption(TIER_HELP[item["tier"]])
        for evidence in item["evidence"]:
            st.code(evidence["path_text"], language=None)


def effects_table(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Mức": [TIER_ICON[i["tier"]] for i in items],
            "Tác dụng phụ": [("⚠️ " if i["serious"] else "") + i["side_effect_vi"] for i in items],
            "Tên tiếng Anh": [i["side_effect_en"] for i in items],
            "Cặp thuốc": [pair_text(i["pairs"]) for i in items],
        }
    )


def render_result(result: dict, key: str) -> None:
    cov = result["coverage"]
    interactions = result["interactions"]
    serious = [i for i in interactions if i["serious"]]
    other = [i for i in interactions if not i["serious"]]

    if result["llm_status"] != "ok":
        st.warning(f"Giải thích đang dùng mẫu có sẵn, LLM không dùng được: {result['llm_error']}")

    cols = st.columns(4)
    cols[0].metric("Số thuốc", len(result["drugs"]))
    cols[1].metric("Cặp có dữ liệu", f"{cov['pairs_with_data']}/{cov['pairs_total']}")
    cols[2].metric("Nguy hiểm nhất", len(serious))
    cols[3].metric("Tổng đã ghi nhận", cov["documented_effects_total"])

    tab_main, tab_drugs, tab_mech = st.tabs(["⚠️ Tác dụng phụ khi phối hợp", "💊 Từng thuốc", "🧬 Cơ chế protein"])

    with tab_main:
        legend()
        if not serious:
            st.info("Không có tác dụng phụ nghiêm trọng nào được ghi nhận cho phác đồ này.")
        else:
            st.dataframe(effects_table(serious), hide_index=True, width="stretch",
                         height=min(38 * len(serious) + 40, 420))
            st.markdown("**Giải thích chi tiết**")
            for index, item in enumerate(serious, start=1):
                render_effect_detail(index, item)
        if other:
            with st.expander(f"Tác dụng phụ khác thường gặp ({len(other)})"):
                st.dataframe(effects_table(other), hide_index=True, width="stretch")
                for index, item in enumerate(other, start=len(serious) + 1):
                    render_effect_detail(index, item)

    with tab_drugs:
        rows = []
        for drug in result["drugs"]:
            top = [("⚠️ " if e["serious"] else "") + e["name_vi"] for e in drug["individual_side_effects"][:4]]
            rows.append({
                "Thuốc": drug["name"],
                "Biệt dược": ", ".join(drug["aliases_vi"][:4]) or "–",
                "Protein đích": f"{drug['target_count']}: " + ", ".join(drug["target_genes"][:5])
                if drug["has_targets"] else "Không có dữ liệu",
                "Tác dụng phụ khi dùng riêng": "; ".join(top) or "–",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if cov["drugs_without_targets"]:
            st.caption("Không có dữ liệu protein đích: " + ", ".join(cov["drugs_without_targets"])
                       + ". Với các thuốc này chỉ có bằng chứng Tier 1.")

    with tab_mech:
        if result["pair_summary"]:
            st.markdown("**Dữ liệu theo từng cặp thuốc**")
            st.dataframe(
                pd.DataFrame([{"Cặp thuốc": " + ".join(p["pair"]), "Đã ghi nhận": p["documented_effects"],
                               "Nghiêm trọng": p["serious_effects"]} for p in result["pair_summary"]]),
                hide_index=True, width="stretch",
            )
        if result["shared_targets"]:
            st.markdown("**Protein đích chung của từ 3 thuốc trở lên**")
            for share in result["shared_targets"]:
                st.write(f"`{share['protein']}`: " + ", ".join(share["drugs"]))
        if result["mechanism_bridges"]:
            with st.expander("Liên kết protein giữa các cặp chưa có dữ liệu (chưa phải kết luận)"):
                for bridge in result["mechanism_bridges"]:
                    st.code(f"{' + '.join(bridge['drug_pair_names'])}: {bridge['path_text']}", language=None)
        elif not result["shared_targets"]:
            st.caption("Không có protein chung hay cầu nối protein nào cho phác đồ này.")

    st.caption(result["disclaimer"])
    st.download_button("Tải kết quả JSON", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="ket_qua_tuong_tac.json", mime="application/json", key=f"dl_{key}")


# ---------------------------------------------------------------- layout
st.markdown(CSS, unsafe_allow_html=True)
st.title("💊 Hỏi đáp tác dụng phụ khi phối hợp thuốc")
try:
    system = load_system()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Không nạp được dữ liệu: {exc}. Kiểm tra đường dẫn CSV trong .env")
    st.stop()
init_state()

# Handle the new message first so the sidebar and graph show the updated state.
prompt = st.chat_input("Nhập tên thuốc, ví dụ: Panadol, ibuprofen, warfarin")
if prompt:
    with st.spinner("Đang tra đồ thị và tạo giải thích…"):
        handle_message(system, prompt)

with st.sidebar:
    st.subheader("Phác đồ hiện tại")
    regimen: dict = st.session_state["regimen"]
    if not regimen:
        st.caption("Chưa có thuốc nào.")
    for stitch_id, name in list(regimen.items()):
        cols = st.columns([4, 1])
        cols[0].write(f"• {name}")
        if cols[1].button("✕", key=f"del_{stitch_id}", help=f"Bỏ {name}"):
            regimen.pop(stitch_id)
            say(f"Đã bỏ {name}.")
            analyse_if_ready(system)
            st.rerun()
    if st.button("🗑 Phác đồ mới", width="stretch"):
        st.session_state.update(messages=[{"role": "assistant", "content": WELCOME}], regimen={}, pending=[], result=None)
        st.rerun()

    st.divider()
    show_graph = st.toggle("Hiện đồ thị quan hệ", value=True)
    show_proteins = st.checkbox("Protein (đích chung, cầu nối)", value=True, disabled=not show_graph)
    show_effects = st.checkbox("Tác dụng phụ nghiêm trọng", value=True, disabled=not show_graph)

    st.divider()
    status = system.status()
    st.caption(f"Dữ liệu: Decagon, {status['drugs']} thuốc ({status['drugs_with_targets']} có protein đích).")
    st.caption("LLM: " + (f"`{status['llm_model']}`" if status["llm_configured"] else "chưa cấu hình, dùng giải thích mẫu"))

result = st.session_state["result"]
if show_graph and result:
    chat_col, graph_col = st.columns([3, 2], gap="large")
else:
    chat_col, graph_col = st.container(), None

with chat_col:
    messages = st.session_state["messages"]
    last_result_index = max((i for i, m in enumerate(messages) if m.get("result")), default=-1)
    for index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("result"):
                if index == last_result_index:
                    render_result(message["result"], f"m{index}")
                else:
                    with st.expander("Xem lại kết quả này"):
                        render_result(message["result"], f"m{index}")

    for index, item in enumerate(list(st.session_state["pending"])):
        with st.chat_message("assistant"):
            st.markdown(f"“{item['query']}” là thuốc nào?")
            cols = st.columns(len(item["candidates"][:3]) + 1)
            for col, candidate in zip(cols, item["candidates"][:3]):
                if col.button(candidate["name"], key=f"pick_{index}_{candidate['stitch_id']}"):
                    choose_candidate(system, index, candidate)
                    st.rerun()
            if cols[-1].button("Không phải", key=f"skip_{index}"):
                choose_candidate(system, index, None)
                st.rerun()

if graph_col is not None:
    with graph_col:
        st.subheader("Đồ thị quan hệ")
        st.caption("🔵 thuốc · 🔶 protein đích chung · 🟣 protein cầu nối · 🟥 tác dụng phụ nghiêm trọng. "
                   "Kéo, phóng to, rê chuột để xem chi tiết.")
        try:
            html = graph_html(json.dumps(result, ensure_ascii=False), show_proteins, show_effects)
        except ImportError:
            st.warning("Thiếu thư viện `pyvis` trong môi trường Python đang chạy app. "
                       "Cài bằng `python -m pip install -r requirements.txt` (đúng venv của dự án) rồi tải lại trang.")
        else:
            if hasattr(st, "iframe"):  # Streamlit >= 1.52 replaces components.html
                st.iframe(html, height=640)
            else:
                components.html(html, height=640, scrolling=False)
