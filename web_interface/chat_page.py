"""The "Trò chuyện" page: chat input, example buttons, history and candidate choices."""

import streamlit as st

from graphrag.pipeline.polypharmacy_system import PolypharmacySystem
from web_interface.answer_rendering import render_answer
from web_interface.chat_message_handling import apply_candidate_choice, handle_user_message
from web_interface.interface_texts import CANDIDATE_BUTTONS_SHOWN, EXAMPLE_REGIMENS


def render_chat_page(system: PolypharmacySystem) -> None:
    with st.container(key="chat-view"):
        # An example button stores its text in "queued" and reruns the page.
        typed_message = st.chat_input("Nhập tên thuốc, ví dụ: Panadol, ibuprofen, warfarin")
        message = typed_message or st.session_state.pop("queued", None)
        if message:
            with st.spinner("Đang tra dữ liệu và viết giải thích…"):
                handle_user_message(system, message)

        if not any(entry["role"] == "user" for entry in st.session_state["messages"]):
            _render_example_buttons()
        _render_history()
        _render_pending_choices(system)


def _render_example_buttons() -> None:
    st.caption("Thử nhanh với một phác đồ mẫu:")
    example_columns = st.columns(len(EXAMPLE_REGIMENS))
    for column, example in zip(example_columns, EXAMPLE_REGIMENS):
        if column.button(example, key=f"ex_{example}", width="stretch"):
            st.session_state["queued"] = example
            st.rerun()


def _render_history() -> None:
    """Only the latest result is open; older results are folded under "Xem lại"."""
    messages = st.session_state["messages"]
    last_result_index = -1
    for index, entry in enumerate(messages):
        if entry.get("result"):
            last_result_index = index
    for index, entry in enumerate(messages):
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if not entry.get("result"):
                continue
            if index == last_result_index:
                render_answer(entry["result"], f"m{index}")
            else:
                with st.expander("Xem lại kết quả này"):
                    render_answer(entry["result"], f"m{index}")


def _render_pending_choices(system: PolypharmacySystem) -> None:
    for pending_index, pending in enumerate(list(st.session_state["pending"])):
        with st.chat_message("assistant"):
            st.markdown(f"“{pending['query']}” là thuốc nào?")
            candidates = pending["candidates"][:CANDIDATE_BUTTONS_SHOWN]
            columns = st.columns(len(candidates) + 1)
            for column, candidate in zip(columns, candidates):
                if column.button(candidate["name"], key=f"pick_{pending_index}_{candidate['stitch_id']}"):
                    apply_candidate_choice(system, pending_index, candidate)
                    st.rerun()
            if columns[-1].button("Không phải", key=f"skip_{pending_index}"):
                apply_candidate_choice(system, pending_index, None)
                st.rerun()
