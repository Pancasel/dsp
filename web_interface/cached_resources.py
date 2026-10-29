"""Objects built once and reused across reruns: the loaded system and the graph HTML."""

import json

import streamlit as st

from graphrag.pipeline.polypharmacy_system import PolypharmacySystem
from graphrag.visualization.interaction_graph_html import build_graph_html  # imports pyvis lazily


@st.cache_resource(show_spinner="Đang nạp đồ thị Decagon (lần đầu mất khoảng 2 phút để tạo cache)…")
def load_system() -> PolypharmacySystem:
    return PolypharmacySystem.load()


@st.cache_data(show_spinner=False, max_entries=8)
def interaction_graph_html(result_json: str, show_proteins: bool, show_effects: bool) -> str:
    """The result is passed as JSON text so that Streamlit can use it as a cache key."""
    return build_graph_html(json.loads(result_json), show_proteins, show_effects)
