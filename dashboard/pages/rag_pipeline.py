"""RAG pipeline: stages, the checks covering each stage, and the defect detection matrix."""

from __future__ import annotations

import streamlit as st

from dashboard import components as ui
from dashboard import data

ui.page_header(
    "RAG pipeline",
    "Reference RAG app: search -> access filter -> prompt assembly -> generation.",
)
mutation = data.latest_mutation(data.report_dir())
st.subheader("Stages, covering checks and caught defects")
st.dataframe(data.pipeline_rows(mutation), hide_index=True)

st.subheader("Defect detection matrix")
if mutation is None:
    st.info(
        "No mutation-test result found. Produce one with "
        "`python -m reliability.cli mutation-test` (it writes reports/mutation-<run>.json), "
        "or start a reference-app run on the Run page."
    )
else:
    st.dataframe(data.matrix_rows(mutation), hide_index=True)
    st.caption(
        f"Baseline without defects: {data.safe_decision(mutation.get('baseline_decision'))}. "
        "'(expected)' marks the detection the specification requires for that defect."
    )
    gaps = mutation.get("gaps") or []
    if gaps:
        st.warning("Evaluator gaps:")
        st.code("\n".join(str(g) for g in gaps), language=None)
    else:
        st.success("Every injected defect was detected by an expected check.")
    st.caption(
        "With the offline mock generator, detection of model-behaviour defects depends on how "
        "the canned answers were written (README: reference RAG app)."
    )
