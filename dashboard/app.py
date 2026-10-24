"""Streamlit entry point: `streamlit run dashboard/app.py`.

Presentation only. All numbers come from stored reports; runs go through the existing
orchestration API. Public mode (default) allows the offline mock provider only.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Community Cloud runs this file directly; make the repository importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard.components import apply_css, mode_badge  # noqa: E402

st.set_page_config(page_title="AI Reliability", page_icon=":material/verified:", layout="wide")
apply_css()

PAGES = Path(__file__).resolve().parent / "pages"
navigation = st.navigation(
    [
        st.Page(PAGES / "overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page(PAGES / "history.py", title="History", icon=":material/timeline:"),
        st.Page(PAGES / "compare.py", title="Compare", icon=":material/compare_arrows:"),
        st.Page(PAGES / "rag_pipeline.py", title="RAG pipeline", icon=":material/account_tree:"),
        st.Page(PAGES / "run.py", title="Run", icon=":material/play_arrow:"),
    ]
)
mode_badge()
st.sidebar.caption("All data is synthetic. Evidence is redacted.")
navigation.run()
