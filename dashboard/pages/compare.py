"""Compare two runs: changed statuses, changed decision, metric deltas (reuses `compare`)."""

from __future__ import annotations

import streamlit as st

from dashboard import components as ui
from dashboard import data

ui.page_header(
    "Compare", "Two stored runs side by side, using the same logic as the CLI `compare`."
)
reports = ui.load_or_explain(data.load_reports(data.report_dir()))
if len(reports) == 1:
    st.info("Only one run is stored; compare needs two.")
elif len(reports) >= 2:
    ids = [r.run_id for r in reversed(reports)]
    by_id = {r.run_id: r for r in reports}
    left, right = st.columns(2)
    baseline_id = left.selectbox("Baseline", ids, index=1)
    current_id = right.selectbox("Current", ids, index=0)
    result = data.compare(by_id[baseline_id], by_id[current_id])
    st.subheader("Decision")
    st.text(
        f"{result.decision_before.value} -> {result.decision_after.value}"
        + ("   REGRESSION" if result.decision_regressed else "")
    )
    st.subheader("Changed check statuses")
    if result.changes:
        st.dataframe(
            [
                {
                    "scenario": c.scenario_id,
                    "check": c.check,
                    "before": c.before,
                    "after": c.after,
                    "regression": "yes" if c.regression else "",
                }
                for c in result.changes
            ],
            hide_index=True,
        )
    else:
        st.caption("No check changed status.")
    if result.added or result.removed:
        st.caption("Scenario sets differ:")
        added = ", ".join(result.added) or "-"
        removed = ", ".join(result.removed) or "-"
        st.code(f"added:   {added}\nremoved: {removed}", language=None)
    st.subheader("Metric deltas")
    rows = [
        {
            "metric": k,
            "baseline": before,
            "current": after,
            "delta": None if before is None or after is None else round(after - before, 4),
        }
        for k, (before, after) in result.deltas.items()
    ]
    st.dataframe(rows, hide_index=True)
    st.subheader("Result")
    st.text("REGRESSION" if result.has_regression else "no regression")
