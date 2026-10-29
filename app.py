"""Chat UI: type the drugs you take, get the dangerous combined side effects.

    python -m streamlit run app.py

The Graph-RAG pipeline runs in this process (no separate API server). Each
analysis sends at most one LLM request. The interface code is in web_interface/.
"""

from web_interface.main_page import run_main_page

run_main_page()
