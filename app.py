"""Chat UI: type the drugs you take, get the dangerous combined side effects.

    streamlit run app.py

The Graph-RAG pipeline runs in this process (no separate API server). Each
analysis sends at most one LLM request.
"""

import html
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
    "Chào bạn! Hãy cho mình biết bạn đang dùng những thuốc nào, ví dụ "
    "*“tôi đang uống Panadol, ibuprofen với warfarin”*. Gõ tên hoạt chất hoặc biệt dược đều được. "
    "Khi có từ 3 thuốc khác nhau, mình sẽ nêu các tác dụng phụ đáng chú ý nhất khi dùng chung.\n\n"
    "Lệnh: **bỏ <tên thuốc>** để bỏ một thuốc, **làm lại** để bắt đầu phác đồ mới."
)

st.set_page_config(page_title="Tác dụng phụ khi dùng nhiều thuốc", page_icon="💊", layout="wide")


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
TOP_VISIBLE = 5  # effect cards shown before "see more"
EXAMPLES = (
    "Panadol, Ibuprofen, Warfarin",
    "Nurofen, Plavix, Lipitor",
    "Zoloft, Xanax, Seroquel",
)
CSS = """
<style>
.block-container {max-width: 960px; padding-top: 2.2rem;}
h1 {font-size: 1.7rem !important; margin-bottom: 0.2rem;}
.fx-head {display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 4px;}
.fx-rank {background: #eef2ff; color: #3730a3; font-weight: 700; border-radius: 50%;
          width: 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center;
          font-size: 0.85rem; flex: none;}
.fx-name {font-size: 1.05rem; font-weight: 650;}
.fx-en {color: #6b7280; font-size: 0.85rem;}
.pill {color: #fff; border-radius: 999px; padding: 1px 10px; font-size: 0.75rem; font-weight: 600;}
.fx-meta {color: #6b7280; font-size: 0.85rem; margin: 2px 0 4px 0;}
.chip {display: inline-block; background: #f1f5f9; border-radius: 999px; padding: 2px 12px;
       margin: 2px 4px 2px 0; font-size: 0.9rem;}
.summary {color: #475569; font-size: 0.92rem; margin-bottom: 0.6rem;}
</style>
"""


def clean_explanation(text: str) -> str:
    """The LLM may cite evidence ids (E1, E2) or CID codes; users never see those."""
    text = EVIDENCE_REF.sub("", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def pair_text(pairs: list) -> str:
    return "; ".join(" + ".join(pair) for pair in pairs)


def effect_card(rank: int, item: dict) -> None:
    vi, en = html.escape(item["side_effect_vi"]), html.escape(item["side_effect_en"])
    color = TIER_COLORS[item["tier"]]
    warn = "⚠️ " if item["serious"] else ""
    english = f"<span class='fx-en'>{en}</span>" if vi != en else ""
    with st.container(border=True):
        st.markdown(
            f"<div class='fx-head'><span class='fx-rank'>{rank}</span>"
            f"<span class='fx-name'>{warn}{vi}</span>{english}"
            f"<span class='pill' style='background:{color}'>Tier {item['tier']} · "
            f"{html.escape(item['tier_label'])}</span></div>",
            unsafe_allow_html=True,
        )
        st.write(clean_explanation(item["explanation_vi"]))
        st.markdown(
            f"<div class='fx-meta'>Được báo cáo ở: {html.escape(pair_text(item['pairs']))}</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Bằng chứng"):
            st.caption(TIER_HELP[item["tier"]])
            if item["amplified_pairs"]:
                st.caption("Có protein đích chung với thuốc thứ ba ở: " + pair_text(item["amplified_pairs"]))
            for evidence in item["evidence"]:
                st.code(evidence["path_text"], language=None)


def render_details(result: dict, key: str) -> None:
    """Secondary information: per-drug cards, protein mechanisms, relation graph, raw JSON."""
    cov = result["coverage"]
    tab_drugs, tab_mech, tab_graph = st.tabs(["Từng thuốc", "Cơ chế protein", "Đồ thị quan hệ"])

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
            st.dataframe(
                pd.DataFrame([{"Cặp thuốc": " + ".join(p["pair"]), "Đã ghi nhận": p["documented_effects"],
                               "Nghiêm trọng": p["serious_effects"]} for p in result["pair_summary"]]),
                hide_index=True, width="stretch",
            )
        for share in result["shared_targets"]:
            st.write(f"Protein chung `{share['protein']}`: " + ", ".join(share["drugs"]))
        for bridge in result["mechanism_bridges"]:
            st.code(f"{' + '.join(bridge['drug_pair_names'])}: {bridge['path_text']}", language=None)
        if result["mechanism_bridges"]:
            st.caption("Các liên kết protein trên là giả thuyết cho cặp chưa có dữ liệu, chưa phải kết luận.")
        elif not result["shared_targets"]:
            st.caption("Không có protein chung hay cầu nối protein nào cho phác đồ này.")

    with tab_graph:
        st.caption("🔵 thuốc · 🔶 protein đích chung · 🟣 protein cầu nối · 🟥 tác dụng phụ nghiêm trọng. "
                   "Kéo, phóng to, rê chuột để xem chi tiết.")
        try:
            page = graph_html(json.dumps(result, ensure_ascii=False), True, True)
        except ImportError:
            st.warning("Thiếu thư viện `pyvis` trong môi trường Python đang chạy app. "
                       "Cài bằng `python -m pip install -r requirements.txt` (đúng venv của dự án) rồi tải lại trang.")
        else:
            if hasattr(st, "iframe"):  # Streamlit >= 1.52 replaces components.html
                st.iframe(page, height=560)
            else:
                components.html(page, height=560, scrolling=False)

    st.download_button("Tải kết quả JSON", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="ket_qua_tuong_tac.json", mime="application/json", key=f"dl_{key}")


def render_answer(result: dict, key: str) -> None:
    """The bot's answer: the most relevant effects as cards, details folded away."""
    cov = result["coverage"]
    serious = [i for i in result["interactions"] if i["serious"]]
    other = [i for i in result["interactions"] if not i["serious"]]

    st.markdown(
        f"<div class='summary'>{cov['pairs_with_data']}/{cov['pairs_total']} cặp thuốc có dữ liệu · "
        f"{cov['documented_effects_total']} tác dụng phụ đã được ghi nhận · "
        f"hiển thị {len(serious)} mục nghiêm trọng nhất</div>",
        unsafe_allow_html=True,
    )
    if result["llm_status"] != "ok":
        st.warning(f"Phần giải thích đang dùng câu mẫu vì LLM không dùng được: {result['llm_error']}")
    if not serious:
        st.info("Không có tác dụng phụ nghiêm trọng nào được ghi nhận cho phác đồ này.")
    for rank, item in enumerate(serious[:TOP_VISIBLE], start=1):
        effect_card(rank, item)
    if len(serious) > TOP_VISIBLE:
        with st.expander(f"Xem thêm {len(serious) - TOP_VISIBLE} tác dụng phụ nghiêm trọng"):
            for rank, item in enumerate(serious[TOP_VISIBLE:], start=TOP_VISIBLE + 1):
                effect_card(rank, item)
    if other:
        with st.expander(f"Tác dụng phụ khác thường gặp ({len(other)})"):
            for rank, item in enumerate(other, start=len(serious) + 1):
                effect_card(rank, item)

    with st.expander("Chi tiết thêm: từng thuốc, protein, đồ thị"):
        render_details(result, key)
    st.caption(result["disclaimer"] + " Phần mô tả tác dụng phụ do AI viết, mang tính chung và không lấy từ dữ liệu.")


# ---------------------------------------------------------------- layout
st.markdown(CSS, unsafe_allow_html=True)
st.title("💊 Hỏi đáp tác dụng phụ khi dùng nhiều thuốc")
st.caption("Nhập các thuốc bạn đang dùng (từ 3 thuốc). Dữ liệu: Decagon, tác dụng phụ đã được báo cáo cho từng cặp thuốc.")
try:
    system = load_system()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Không nạp được dữ liệu: {exc}. Kiểm tra đường dẫn CSV trong .env")
    st.stop()
init_state()

# Handle the new message first so the sidebar shows the updated state.
prompt = st.chat_input("Nhập tên thuốc, ví dụ: Panadol, ibuprofen, warfarin") or st.session_state.pop("queued", None)
if prompt:
    with st.spinner("Đang tra dữ liệu và viết giải thích…"):
        handle_message(system, prompt)

with st.sidebar:
    st.subheader("Phác đồ hiện tại")
    regimen: dict = st.session_state["regimen"]
    if not regimen:
        st.caption("Chưa có thuốc nào.")
    for stitch_id, name in list(regimen.items()):
        cols = st.columns([5, 1], vertical_alignment="center")
        cols[0].markdown(f"<span class='chip'>💊 {html.escape(name)}</span>", unsafe_allow_html=True)
        if cols[1].button("✕", key=f"del_{stitch_id}", help=f"Bỏ {name}"):
            regimen.pop(stitch_id)
            say(f"Đã bỏ {name}.")
            analyse_if_ready(system)
            st.rerun()
    if st.button("🗑 Phác đồ mới", width="stretch"):
        st.session_state.update(messages=[{"role": "assistant", "content": WELCOME}], regimen={}, pending=[], result=None)
        st.rerun()

    st.divider()
    st.caption("Thử nhanh")
    for example in EXAMPLES:
        if st.button(example, key=f"ex_{example}", width="stretch"):
            st.session_state["queued"] = example
            st.rerun()

    st.divider()
    status = system.status()
    st.caption(f"Dữ liệu: {status['drugs']} thuốc ({status['drugs_with_targets']} có protein đích).")
    st.caption("LLM: " + (f"`{status['llm_model']}`" if status["llm_configured"] else "chưa cấu hình, dùng giải thích mẫu"))

messages = st.session_state["messages"]
last_result_index = max((i for i, m in enumerate(messages) if m.get("result")), default=-1)
for index, message in enumerate(messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("result"):
            if index == last_result_index:
                render_answer(message["result"], f"m{index}")
            else:
                with st.expander("Xem lại kết quả này"):
                    render_answer(message["result"], f"m{index}")

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
