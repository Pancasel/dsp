"""What the page remembers between reruns (st.session_state).

Keys:
- "messages": chat history, each {"role", "content", "result" (analysis or None)}
- "regimen": drug STITCH ID -> display name, in the order added
- "pending": ambiguous inputs waiting for the user to pick a candidate (DrugNameResolution.to_dict())
- "result": latest analysis of the current regimen, None when it no longer matches
- "drug_catalog_version": increased to reset the drug picker of the regimen builder
"""

import streamlit as st

from web_interface.interface_texts import WELCOME_MESSAGE


def initialize_session_state() -> None:
    st.session_state.setdefault("messages", [{"role": "assistant", "content": WELCOME_MESSAGE}])
    st.session_state.setdefault("regimen", {})
    st.session_state.setdefault("pending", [])
    st.session_state.setdefault("result", None)


def add_assistant_message(content: str, result: dict | None = None) -> None:
    st.session_state["messages"].append({"role": "assistant", "content": content, "result": result})


def reset_drug_catalog_picker() -> None:
    st.session_state["drug_catalog_version"] = st.session_state.get("drug_catalog_version", 0) + 1


def start_new_regimen() -> None:
    """Forget the conversation, the drugs and the result."""
    st.session_state.update(
        messages=[{"role": "assistant", "content": WELCOME_MESSAGE}],
        regimen={},
        pending=[],
        result=None,
    )
    st.session_state["drug-search-text"] = ""
    reset_drug_catalog_picker()
