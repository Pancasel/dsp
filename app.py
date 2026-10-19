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
    "Chào bạn! Bạn đang dùng những thuốc nào? Cứ nhập tên hoạt chất hoặc biệt dược như "
    "*“Panadol, ibuprofen với warfarin”* nhé. Khi có ít nhất 3 thuốc khác nhau, mình "
    "có thể tra các tác dụng phụ được ghi nhận trong dữ liệu.\n\n"
    "Muốn sửa danh sách, gõ **bỏ <tên thuốc>**; muốn bắt đầu lại, gõ **làm lại**."
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
        current = f"Mình đã ghi nhận {names}" if names else "Mình chưa ghi nhận thuốc nào"
        say(f"{current}. Thêm ít nhất {need} thuốc nữa thì mình mới tra được phác đồ từ 3 thuốc.")
        return
    try:
        result = system.predict(list(regimen))
    except RegimenError as exc:
        say(f"Chưa phân tích được: {exc}")
        return
    st.session_state["result"] = result
    serious = sum(item["serious"] for item in result["interactions"])
    if serious:
        response = (
            f"Mình đã tra xong **{names}**. Trong dữ liệu có {serious} mục được "
            "xếp nhóm nghiêm trọng để bạn chú ý."
        )
    else:
        response = (
            f"Mình đã tra xong **{names}**. Hiện chưa thấy mục nào được xếp nhóm "
            "nghiêm trọng trong dữ liệu; điều đó không đảm bảo là không có rủi ro."
        )
    say(
        response,
        result,
    )


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
        say("Mình đã xóa danh sách thuốc rồi. Bạn muốn tra những thuốc nào?")
        return
    lowered = unicodedata.normalize("NFC", text).strip().casefold()
    if lowered.startswith(REMOVE_PREFIXES):
        targets = system.resolver.extract(lowered.split(" ", 1)[1])
        removed = [st.session_state["regimen"].pop(item.stitch_id) for item in targets
                   if item.stitch_id in st.session_state["regimen"]]
        if not removed:
            say("Mình chưa thấy thuốc đó trong danh sách hiện tại.")
            return
        say(f"Đã bỏ {', '.join(removed)} khỏi danh sách nhé.")
        analyse_if_ready(system)
        return

    found = system.resolver.extract(text)
    if not found:
        say(
            "Mình chưa nhận ra tên thuốc trong tin nhắn này. Bạn thử nhập tên hoạt chất "
            "hoặc biệt dược, chẳng hạn “Panadol, ibuprofen, warfarin” nhé."
        )
        return
    added = add_drugs(found)
    pending = [item.to_dict() for item in found if item.stitch_id is None and item.candidates and item.method == "ambiguous"]
    missing = [item.query for item in found
               if item.stitch_id is None and not (item.method == "ambiguous" and item.candidates)]
    st.session_state["pending"].extend(pending)
    lines = []
    if added:
        lines.append("Mình nhận ra " + ", ".join(added) + ".")
    if pending:
        lines.append(
            "Mình chưa chắc “"
            + "”, “".join(item["query"] for item in pending)
            + "” là thuốc nào; bạn chọn giúp mình ở bên dưới nhé."
        )
    if missing:
        lines.append(
            "Mình chưa tìm thấy "
            + ", ".join(f"“{name}”" for name in missing)
            + " trong dữ liệu thuốc."
        )
    say(" ".join(lines) or "Mấy thuốc này đã có trong danh sách rồi.")
    if added:
        analyse_if_ready(system)


def choose_candidate(system: PolypharmacySystem, index: int, candidate: dict | None) -> None:
    item = st.session_state["pending"].pop(index)
    if candidate is None:
        say(f"Được rồi, mình sẽ bỏ qua “{item['query']}”.")
        return
    st.session_state["regimen"][candidate["stitch_id"]] = candidate["name"]
    say(f"Mình hiểu “{item['query']}” là **{candidate['name']}** nhé.")
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
:root {
  --ink: #17352f;
  --muted: #668078;
  --line: rgba(103, 151, 135, .17);
  --mint: #14977d;
}
.stApp {
  background:
    radial-gradient(ellipse at 4% 8%, rgba(92, 205, 174, .14), transparent 31rem),
    radial-gradient(ellipse at 96% 43%, rgba(122, 181, 232, .12), transparent 34rem),
    linear-gradient(145deg, #f5fbf8 0%, #f7fafc 50%, #f2f8f7 100%);
  color: var(--ink);
}
.stApp::before, .stApp::after {
  content: "";
  position: fixed;
  z-index: 0;
  pointer-events: none;
  width: 25rem;
  height: 25rem;
  border-radius: 50%;
  filter: blur(72px);
  opacity: .18;
  animation: drift 22s ease-in-out infinite alternate;
}
.stApp::before {background: #72d7b4; top: 8%; left: -17rem;}
.stApp::after {background: #9ac9ee; right: -17rem; bottom: 3%; animation-delay: -9s;}
@keyframes drift {
  from {transform: translate3d(0, -12px, 0) scale(.94);}
  to {transform: translate3d(32px, 30px, 0) scale(1.08);}
}
.block-container {
  position: relative;
  z-index: 1;
  max-width: 1040px;
  padding-top: 2rem;
  padding-bottom: 3rem;
}
h1, h2, h3, [data-testid="stMetricValue"] {
  color: var(--ink) !important;
  font-family: 'Aptos Display', 'Segoe UI', sans-serif;
  letter-spacing: -.035em;
}
h1 {font-size: 2rem !important; margin-bottom: .35rem;}
p, label, li {font-family: 'Aptos', 'Segoe UI', sans-serif;}
[data-testid="stCaptionContainer"] {color: var(--muted);}
[data-testid="stSidebar"] {
  background: rgba(246, 251, 248, .88);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child {background: transparent;}
[data-testid="stChatMessage"] {
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1rem 1.15rem;
  background: rgba(255, 255, 255, .76);
  box-shadow: 0 8px 28px rgba(34, 83, 68, .045);
  animation: rise-in .35s ease-out both;
  backdrop-filter: blur(10px);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: linear-gradient(135deg, rgba(229, 247, 240, .96), rgba(238, 247, 251, .94));
  border-color: rgba(72, 155, 128, .2);
}
@keyframes rise-in {
  from {opacity: 0; transform: translateY(8px);}
  to {opacity: 1; transform: translateY(0);}
}
[data-testid="stChatInput"] textarea {
  border: 1px solid rgba(67, 137, 114, .25);
  border-radius: 18px;
  background: rgba(255, 255, 255, .9);
  box-shadow: 0 8px 28px rgba(34, 83, 68, .07);
}
[data-testid="stChatInput"] textarea:focus {
  border-color: rgba(20, 151, 125, .72);
  box-shadow: 0 0 0 3px rgba(20, 151, 125, .12);
}
div.stButton > button, [data-testid="stDownloadButton"] button {
  border-radius: 12px;
  border: 1px solid rgba(61, 138, 115, .18);
  color: #24574a;
  background: rgba(255, 255, 255, .78);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
div.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
  transform: translateY(-2px);
  border-color: rgba(20, 151, 125, .48);
  box-shadow: 0 8px 18px rgba(34, 83, 68, .1);
  color: #116b59;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--line) !important;
  border-radius: 18px !important;
  background: rgba(255, 255, 255, .76);
  box-shadow: 0 8px 28px rgba(34, 83, 68, .045);
  transition: transform .2s ease, box-shadow .2s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 13px 32px rgba(34, 83, 68, .09);
}
[data-testid="stExpander"] {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255, 255, 255, .58);
}
[data-testid="stTabs"] [role="tab"] {border-radius: 10px 10px 0 0;}
[data-testid="stAlert"] {border-radius: 14px;}
.fx-head {display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 4px;}
.fx-rank {
  background: linear-gradient(145deg, #e7f4ff, #e5f7ef);
  color: #256552;
  font-weight: 700;
  border: 1px solid rgba(86, 157, 132, .16);
  border-radius: 50%;
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85rem;
  flex: none;
}
.fx-name {font-size: 1.05rem; font-weight: 700; color: var(--ink);}
.fx-en {color: var(--muted); font-size: 0.85rem;}
.pill {color: #fff; border-radius: 999px; padding: 3px 11px; font-size: 0.75rem; font-weight: 650;}
.fx-meta {color: var(--muted); font-size: 0.85rem; margin: 5px 0 4px;}
.chip {
  display: inline-block;
  color: #24574a;
  background: linear-gradient(135deg, #e8f6ef, #eef6fb);
  border: 1px solid rgba(86, 157, 132, .14);
  border-radius: 999px;
  padding: 4px 12px;
  margin: 2px 4px 2px 0;
  font-size: 0.9rem;
}
.summary {
  color: #45675d;
  font-size: 0.92rem;
  margin-bottom: .8rem;
  padding: .75rem 1rem;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: linear-gradient(110deg, rgba(230, 247, 239, .78), rgba(239, 247, 251, .76));
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: .01ms !important;
  }
}
.st-key-chat-view, .st-key-regimen-view {
  animation: page-enter .3s cubic-bezier(.2, .75, .25, 1) both;
}
@keyframes page-enter {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
}
[data-testid="stRadio"] [role="radiogroup"] {gap: .5rem;}
[data-testid="stRadio"] label {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: .45rem .8rem;
  background: rgba(255, 255, 255, .58);
}
[data-testid="stRadio"] label:has(input:checked) {
  border-color: rgba(20, 151, 125, .35);
  background: linear-gradient(110deg, rgba(226, 246, 237, .95), rgba(234, 245, 251, .92));
}
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

page = st.radio(
    "Chọn cách tạo phác đồ",
    ["💬 Trò chuyện", "💊 Tạo phác đồ"],
    horizontal=True,
    label_visibility="collapsed",
    key="active_view",
)

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
        st.session_state["drug-search-text"] = ""
        st.session_state["drug_catalog_version"] = st.session_state.get("drug_catalog_version", 0) + 1
        st.rerun()

    st.divider()
    status = system.status()
    st.caption(f"Dữ liệu: {status['drugs']} thuốc ({status['drugs_with_targets']} có protein đích).")
    st.caption("LLM: " + (f"`{status['llm_model']}`" if status["llm_configured"] else "chưa cấu hình, dùng giải thích mẫu"))

if page == "💬 Trò chuyện":
    with st.container(key="chat-view"):
        prompt = st.chat_input(
            "Nhập tên thuốc, ví dụ: Panadol, ibuprofen, warfarin"
        ) or st.session_state.pop("queued", None)
        if prompt:
            with st.spinner("Đang tra dữ liệu và viết giải thích…"):
                handle_message(system, prompt)

        if not any(message["role"] == "user" for message in st.session_state["messages"]):
            st.caption("Thử nhanh với một phác đồ mẫu:")
            example_cols = st.columns(len(EXAMPLES))
            for col, example in zip(example_cols, EXAMPLES):
                if col.button(example, key=f"ex_{example}", width="stretch"):
                    st.session_state["queued"] = example
                    st.rerun()

        messages = st.session_state["messages"]
        last_result_index = max((i for i, message in enumerate(messages) if message.get("result")), default=-1)
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
else:
    with st.container(key="regimen-view"):
        st.subheader("Tạo phác đồ")
        st.caption("Tìm thuốc theo tên hoạt chất hoặc biệt dược, rồi thêm vào danh sách.")

        drug_ids = sorted(system.graph.drug_ids, key=lambda drug: system.names.drug_name(drug).casefold())
        base_labels: dict[str, str] = {}
        search_terms: dict[str, list[str]] = {}
        for drug_id in drug_ids:
            drug = system.names.drugs.get(drug_id)
            name = system.names.drug_name(drug_id)
            aliases = list(dict.fromkeys(drug.aliases_vi if drug else []))
            aliases = [alias for alias in aliases if normalize(alias) != normalize(name)][:2]
            base_labels[drug_id] = name + (f" · {', '.join(aliases)}" if aliases else "")
            search_terms[drug_id] = list(dict.fromkeys(
                normalize(term)
                for term in [name, *(drug.aliases_vi + drug.synonyms if drug else [])]
                if normalize(term)
            ))
        label_counts = {
            label: sum(candidate == label for candidate in base_labels.values())
            for label in set(base_labels.values())
        }
        drug_labels = {
            drug_id: f"{label} ({drug_id})" if label_counts[label] > 1 else label
            for drug_id, label in base_labels.items()
        }

        search_col, select_col, add_col = st.columns([3, 3, 1], vertical_alignment="bottom")
        search_text = search_col.text_input(
            "Tìm thuốc",
            placeholder="Tên hoạt chất hoặc biệt dược…",
            key="drug-search-text",
        )
        query = normalize(search_text)
        matches: list[str] = []
        if len(query) >= 2:
            ranked_matches = []
            for drug_id in drug_ids:
                matching_terms = [term for term in search_terms[drug_id] if query in term]
                if not matching_terms:
                    continue
                best_match = min(
                    (0 if term == query else 1 if term.startswith(query) else 2)
                    for term in matching_terms
                )
                ranked_matches.append((best_match, system.names.drug_name(drug_id).casefold(), drug_id))
            matches = [entry[2] for entry in sorted(ranked_matches)[:30]]

        selected_drug = None
        if matches:
            selected_drug = select_col.selectbox(
                "Chọn thuốc phù hợp",
                options=matches,
                index=None,
                placeholder=f"{len(matches)} kết quả — chọn một thuốc",
                format_func=lambda drug_id: drug_labels[drug_id],
                key=(
                    f"drug-catalog-{st.session_state.get('drug_catalog_version', 0)}-"
                    f"{query}"
                ),
            )
        elif query:
            select_col.caption("Chưa thấy kết quả khớp. Thử tên hoạt chất hoặc biệt dược khác nhé.")
        else:
            select_col.caption("Nhập ít nhất 2 ký tự để tìm thuốc.")

        already_added = selected_drug in st.session_state["regimen"] if selected_drug else False
        if add_col.button(
            "Đã thêm" if already_added else "＋ Thêm",
            type="primary",
            disabled=selected_drug is None or already_added,
            width="stretch",
        ):
            st.session_state["regimen"][selected_drug] = system.names.drug_name(selected_drug)
            st.session_state["drug_catalog_version"] = st.session_state.get("drug_catalog_version", 0) + 1
            if len(st.session_state["regimen"]) >= 3:
                with st.spinner("Đang tra dữ liệu cho phác đồ…"):
                    analyse_if_ready(system)
            else:
                need = 3 - len(st.session_state["regimen"])
                st.toast(f"Đã thêm {system.names.drug_name(selected_drug)}. Thêm {need} thuốc nữa để tra phác đồ.")
            st.rerun()

        regimen = st.session_state["regimen"]
        st.markdown(f"**Thuốc trong phác đồ** · {len(regimen)}")
        if regimen:
            chip_cols = st.columns(min(len(regimen), 4))
            for index, (drug_id, name) in enumerate(list(regimen.items())):
                with chip_cols[index % len(chip_cols)]:
                    st.markdown(f"<span class='chip'>💊 {html.escape(name)}</span>", unsafe_allow_html=True)
                    if st.button("Bỏ thuốc", key=f"builder-remove-{drug_id}"):
                        regimen.pop(drug_id)
                        analyse_if_ready(system)
                        st.rerun()
        else:
            st.info("Danh sách đang trống. Tìm rồi thêm thuốc để bắt đầu.")

        remaining = max(0, 3 - len(regimen))
        if remaining:
            st.caption(f"Thêm {remaining} thuốc nữa thì mình có thể tra dữ liệu cho phác đồ.")
        elif st.session_state.get("result"):
            st.success("Đã có kết quả cho danh sách thuốc hiện tại. Chuyển sang **Trò chuyện** để xem chi tiết.")
        else:
            st.caption("Danh sách đã đủ điều kiện để tra.")
