"""Run: start a bounded run through the orchestration API and open its report."""

from __future__ import annotations

import streamlit as st

from dashboard import components as ui
from dashboard import data
from dashboard.runner import RunRejected, RunRequest, available_scenarios, max_scenarios, start_run
from rag_app.defects import DEFECTS
from reliability.reporting.summary import render_summary

PUBLIC = data.public_mode()
ui.page_header(
    "Run",
    "Start one evaluation run. Offline and free with the mock provider; "
    f"at most {max_scenarios()} scenarios, one run at a time.",
)

TARGETS = {
    "scenarios": "Scenario suite (precomputed retrieval)",
    "rag-app": "Reference RAG app",
    "mutation-test": "Mutation test of the reference RAG app (all defects)",
}
scenarios = available_scenarios()

with st.form("run"):
    target = st.selectbox("Target", list(TARGETS), format_func=lambda k: TARGETS[k])
    category = st.selectbox("Category", ["all", "llm", "rag", "security", "operational"])
    scenario_ids = ["all"] + [sid for sid, _, _ in scenarios]
    scenario = st.selectbox("Single scenario", scenario_ids, help="Overrides nothing else.")
    defect = st.selectbox(
        "Injected defect (reference RAG app only)", ["none", *sorted(DEFECTS)], index=0
    )
    provider = "mock"
    model = None
    if not PUBLIC:
        provider = st.selectbox(
            "Provider",
            ["mock", "litellm"],
            help="litellm = a local model such as ollama/llama3.2. Paid models are never "
            "allowed from the dashboard.",
        )
        model = st.text_input("Local model (litellm)", value="ollama/llama3.2") or None
    submitted = st.form_submit_button("Start run")

if submitted:
    request = RunRequest(
        target=target,
        category=None if category == "all" else category,
        scenario_id=None if scenario == "all" else scenario,
        defect=None if defect == "none" else defect,
        provider=provider,
        model=model if provider != "mock" else None,
    )
    if request.defect and target != "rag-app":
        st.error("An injected defect applies only to the reference RAG app target.")
    else:
        try:
            with st.spinner("Running..."):
                result = start_run(
                    request,
                    report_dir=data.report_dir(),
                    history_file=data.history_path(),
                    public=PUBLIC,
                )
        except RunRejected as exc:
            st.error("Run refused:")
            st.code(str(exc), language=None)
        else:
            st.session_state["selected_run"] = result.report.run_id
            st.success("Run finished. Open Overview in the sidebar to explore this run.")
            st.text(f"Decision: {result.report.decision.decision.value}")
            st.code(render_summary(result.report), language=None)
