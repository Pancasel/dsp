"""Chat UI: type the drugs you take, get the dangerous combined side effects.

    streamlit run app.py

The Graph-RAG pipeline runs in this process (no separate API server). Each
analysis sends at most one LLM request.
"""

import json
import unicodedata

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


def badge(tier: int, label: str) -> str:
    return (
        f"<span style='background:{TIER_COLORS[tier]};color:white;padding:2px 8px;"
        f"border-radius:10px;font-size:0.8em'>Tier {tier} · {label}</span>"
    )


def render_interaction(item: dict, key: str) -> None:
    with st.container(border=True):
        mark = "⚠️ " if item["serious"] else ""
        english = (
            f" <span style='color:gray'>({item['side_effect_en']})</span>"
            if item["side_effect_vi"] != item["side_effect_en"] else ""
        )
        st.markdown(f"{badge(item['tier'], item['tier_label'])} &nbsp; **{mark}{item['side_effect_vi']}**{english}",
                    unsafe_allow_html=True)
        st.caption("Cặp thuốc: " + "; ".join(" + ".join(pair) for pair in item["pairs"]))
        if item["amplified_pairs"]:
            st.caption("Khuếch đại (protein đích chung với thuốc thứ ba) ở: "
                       + "; ".join(" + ".join(pair) for pair in item["amplified_pairs"]))
        st.write(item["explanation_vi"])
        with st.expander("Bằng chứng"):
            st.caption(TIER_HELP[item["tier"]])
            for evidence in item["evidence"]:
                st.code(evidence["path_text"], language=None)


def render_result(result: dict, key: str) -> None:
    st.caption(result["disclaimer"])
    if result["llm_status"] != "ok":
        st.caption(f"Giải thích đang dùng mẫu có sẵn (LLM không dùng được: {result['llm_error']}).")

    with st.expander("💊 Thông tin từng thuốc", expanded=False):
        for drug in result["drugs"]:
            aliases = f" — biệt dược: {', '.join(drug['aliases_vi'][:4])}" if drug["aliases_vi"] else ""
            st.markdown(f"**{drug['name']}**{aliases}")
            targets = (f"Protein đích ({drug['target_count']}): {', '.join(drug['target_genes'][:6])}"
                       if drug["has_targets"] else "Không có dữ liệu protein đích")
            st.caption(f"{drug['stitch_id']} · {targets}")
            effects = [("⚠️ " if e["serious"] else "") + e["name_vi"] for e in drug["individual_side_effects"][:6]]
            if effects:
                st.caption("Tác dụng phụ khi dùng riêng: " + "; ".join(effects))

    serious = [item for item in result["interactions"] if item["serious"]]
    other = [item for item in result["interactions"] if not item["serious"]]
    st.markdown("**⚠️ Tác dụng phụ nguy hiểm nhất khi phối hợp**")
    if not serious:
        st.info("Không có tác dụng phụ nghiêm trọng nào được ghi nhận cho phác đồ này.")
    for index, item in enumerate(serious):
        render_interaction(item, f"{key}_s{index}")
    if other:
        with st.expander(f"Tác dụng phụ khác thường gặp ({len(other)})"):
            for index, item in enumerate(other):
                render_interaction(item, f"{key}_o{index}")

    cov = result["coverage"]
    notes = [f"{cov['pairs_with_data']}/{cov['pairs_total']} cặp có dữ liệu",
             f"{cov['documented_effects_total']} tác dụng phụ đã ghi nhận (đã lọc ra những cái nguy hiểm nhất)"]
    if cov["drugs_without_targets"]:
        notes.append("không có dữ liệu protein: " + ", ".join(cov["drugs_without_targets"]))
    st.caption(" · ".join(notes))
    if result["mechanism_bridges"]:
        with st.expander("Liên kết protein giữa các cặp chưa có dữ liệu (chưa phải kết luận)"):
            for bridge in result["mechanism_bridges"]:
                st.code(f"{' + '.join(bridge['drug_pair_names'])}: {bridge['path_text']}", language=None)
    st.download_button("Tải kết quả JSON", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="ket_qua_tuong_tac.json", mime="application/json", key=f"dl_{key}")


# ---------------------------------------------------------------- layout
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
    if st.button("🗑 Phác đồ mới", use_container_width=True):
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
