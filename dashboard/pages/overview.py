"""Overview: the selected (default latest) run - decision, metric cards, findings, RAG, security."""

from __future__ import annotations

import streamlit as st

from dashboard import components as ui
from dashboard import data

ui.page_header(
    "Overview", "Latest evaluation run, compared with the previous run against the same target."
)
reports = ui.load_or_explain(data.load_reports(data.report_dir()))
if reports:
    by_id = {r.run_id: r for r in reports}
    ids = [r.run_id for r in reversed(reports)]
    wanted = st.session_state.get("selected_run")
    index = ids.index(wanted) if wanted in ids else 0
    run_id = st.selectbox("Run (newest first)", ids, index=index)
    current = by_id[run_id]
    position = reports.index(current)
    previous = data.previous_comparable(reports, current)

    st.subheader("Release decision")
    st.text(
        f"{current.decision.decision.value}  (target {current.target}, provider {current.provider})"
    )
    st.code("\n".join(f"{r.rule}: {r.detail}" for r in current.decision.fired_rules), language=None)
    st.caption(
        f"Evaluator self-tests: {current.selftest.matched}/{current.selftest.total} reproduced "
        "their declared verdicts."
        + (
            f" Compared with run {previous.run_id} (same target)."
            if previous
            else " No earlier run against the same target to compare with."
        )
    )

    same_target = [r for r in reports[: position + 1] if r.target == current.target]
    window = same_target[-10:]
    sparks = {
        "Pass rate": [r.decision.pass_rate or 0.0 for r in window],
        "Failures": [float(r.counts.get("FAIL", 0)) for r in window],
        "Inconclusive": [float(r.counts.get("INCONCLUSIVE", 0)) for r in window],
        "Errors": [float(r.counts.get("ERROR", 0)) for r in window],
        "Median latency (ms)": [r.latency.median_ms or 0.0 for r in window],
    }
    ui.metric_cards(data.metric_cards(current, previous), sparks)

    st.subheader("Findings")
    c1, c2, c3, c4 = st.columns(4)
    statuses = c1.multiselect("Status", ["FAIL", "INCONCLUSIVE", "ERROR"], default=[])
    categories = c2.multiselect("Category", ["llm", "rag", "security", "operational"], default=[])
    severities = c3.multiselect("Severity", ["critical", "high", "medium", "low"], default=[])
    roles = c4.multiselect(
        "Role",
        ["gate", "selftest"],
        default=[],
    )
    st.caption(
        "Role: gate = the release candidate; selftest = known-bad fixtures the evaluator "
        "must catch (their failures are expected)."
    )
    rows = data.findings(
        current,
        statuses=set(statuses) or None,
        categories=set(categories) or None,
        severities=set(severities) or None,
        roles=set(roles) or None,
    )
    st.caption(f"{len(rows)} finding(s)")
    if rows:
        st.dataframe(
            [
                {
                    "status": f.status,
                    "role": f.role,
                    "scenario": f.scenario_id,
                    "check": f.check,
                    "severity": f.severity,
                    "explanation": f.explanation,
                }
                for f in rows
            ],
            hide_index=True,
        )
        choices = list(range(len(rows)))
        picked = st.selectbox(
            "Evidence for finding",
            choices,
            format_func=lambda i: f"#{i + 1}  {rows[i].scenario_id} / {rows[i].check}",
        )
        with st.expander("Evidence and response excerpt", expanded=False):
            st.caption("Scenario name")
            st.code(rows[picked].name, language=None)
            st.caption("Evidence (redacted)")
            st.code(rows[picked].evidence, language="json")
            st.caption("Response excerpt (redacted, shown as plain text)")
            st.code(rows[picked].excerpt or "(empty)", language=None)

    st.subheader("RAG quality (gate means)")
    quality = data.rag_quality(current)
    left, right = st.columns(2)
    with left:
        ui.chart(ui.bar_figure(quality["retrieval"], "Retrieval"))
    with right:
        outcomes = data.grounding_outcomes(current)
        if outcomes:
            ui.chart(ui.count_figure(outcomes, "Grounding outcomes"))
        else:
            st.caption("No grounding evidence in this run.")
        gen = quality["generation"]
        st.caption(
            "Citation accuracy: "
            + ("n/a" if gen["Citation accuracy"] is None else f"{gen['Citation accuracy']:.0%}")
            + " | unverified content ratio: "
            + (
                "n/a"
                if gen["Unverified content ratio"] is None
                else f"{gen['Unverified content ratio']:.0%}"
            )
        )

    st.subheader("Security findings by type")
    breakdown = data.security_breakdown(current)
    left, right = st.columns(2)
    for column, role, title in (
        (left, "gate", "Gate (release candidate)"),
        (right, "selftest", "Known-bad fixtures detected"),
    ):
        with column:
            counts = breakdown.get(role, {})
            if counts:
                ui.chart(ui.donut_figure(counts, title))
            else:
                st.caption(f"{title}: none")
