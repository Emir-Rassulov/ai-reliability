"""History: trends across runs from history/runs.jsonl (or, if absent, the stored reports)."""

from __future__ import annotations

import streamlit as st

from dashboard import components as ui
from dashboard import data
from reliability.reporting.history import history_entry

ui.page_header(
    "History", "Trends across runs. Regressions are runs that got worse than the one before."
)
entries, problems = data.read_history(data.history_path())
for problem in problems:
    st.warning("A history line was skipped:")
    st.code(problem, language=None)
source = "history file"
if not entries:
    loaded = data.load_reports(data.report_dir())
    entries = [history_entry(r) for r in loaded.reports]
    source = "stored reports (no history file found)"

if not entries:
    st.info(
        "No history yet. CI appends one line per main-branch run to history/runs.jsonl on the "
        "`history` branch; locally use `python -m reliability.cli run --history-file "
        "history/runs.jsonl`."
    )
else:
    st.caption(f"{len(entries)} run(s) from {source}.")
    metric = st.selectbox(
        "Metric", list(data.TREND_METRICS), format_func=lambda k: data.TREND_METRICS[k]
    )
    points = data.trend(entries, metric)
    ui.chart(ui.trend_figure(points, data.TREND_METRICS[metric]))
    st.caption("Marker colour = release decision of that run.")
    st.subheader("Regressions")
    alerts = data.history_regressions(entries)
    if alerts:
        st.code("\n".join(alerts), language=None)
    else:
        st.caption("No run was worse than the run before it.")
