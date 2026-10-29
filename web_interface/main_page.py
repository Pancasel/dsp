"""Page frame: title, data loading, the page switch, the sidebar with the current regimen."""

import html

import streamlit as st

from graphrag.pipeline.polypharmacy_system import PolypharmacySystem
from web_interface.cached_resources import load_system
from web_interface.chat_message_handling import analyse_regimen_if_ready
from web_interface.chat_page import render_chat_page
from web_interface.interface_texts import CHAT_PAGE_LABEL, REGIMEN_BUILDER_PAGE_LABEL
from web_interface.page_styles import PAGE_CSS
from web_interface.regimen_builder_page import render_regimen_builder_page
from web_interface.session_state import (
    add_assistant_message,
    initialize_session_state,
    start_new_regimen,
)


def run_main_page() -> None:
    # Must be the first Streamlit call of the run.
    st.set_page_config(page_title="Tác dụng phụ khi dùng nhiều thuốc", page_icon="💊", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.title("💊 Hỏi đáp tác dụng phụ khi dùng nhiều thuốc")
    st.caption(
        "Nhập các thuốc bạn đang dùng (từ 3 thuốc). "
        "Dữ liệu: Decagon, tác dụng phụ đã được báo cáo cho từng cặp thuốc."
    )
    try:
        system = load_system()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Không nạp được dữ liệu: {error}. Kiểm tra đường dẫn CSV trong .env")
        st.stop()
    initialize_session_state()

    chosen_page = st.radio(
        "Chọn cách tạo phác đồ",
        [CHAT_PAGE_LABEL, REGIMEN_BUILDER_PAGE_LABEL],
        horizontal=True,
        label_visibility="collapsed",
        key="active_view",
    )
    with st.sidebar:
        render_sidebar(system)
    if chosen_page == CHAT_PAGE_LABEL:
        render_chat_page(system)
    else:
        render_regimen_builder_page(system)


def render_sidebar(system: PolypharmacySystem) -> None:
    st.subheader("Phác đồ hiện tại")
    regimen: dict = st.session_state["regimen"]
    if not regimen:
        st.caption("Chưa có thuốc nào.")
    for stitch_id, name in list(regimen.items()):
        name_column, remove_column = st.columns([5, 1], vertical_alignment="center")
        name_column.markdown(f"<span class='drug-chip'>💊 {html.escape(name)}</span>", unsafe_allow_html=True)
        if remove_column.button("✕", key=f"del_{stitch_id}", help=f"Bỏ {name}"):
            regimen.pop(stitch_id)
            add_assistant_message(f"Đã bỏ {name}.")
            analyse_regimen_if_ready(system)
            st.rerun()
    if st.button("🗑 Phác đồ mới", width="stretch"):
        start_new_regimen()
        st.rerun()

    st.divider()
    status = system.status()
    st.caption(f"Dữ liệu: {status['drugs']} thuốc ({status['drugs_with_targets']} có protein đích).")
    if status["llm_configured"]:
        st.caption(f"LLM: `{status['llm_model']}`")
    else:
        st.caption("LLM: chưa cấu hình, dùng giải thích mẫu")
