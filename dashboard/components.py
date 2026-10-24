"""Reusable dashboard pieces. Figure builders are pure (return a Plotly figure) and testable.

Safety rule: report content (model output, evidence, explanations, names) is shown only
through `st.code`, `st.text` or `st.dataframe`, which display it as inert text. It is never
passed to `st.markdown`, `st.html`, widget labels, or anything with `unsafe_allow_html`.
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go
import streamlit as st

from dashboard import theme
from dashboard.data import Card, Loaded, public_mode, report_dir
from reliability.models import RunReport


def page_header(title: str, caption: str) -> None:
    st.title(title)
    st.caption(caption)


def apply_css() -> None:
    st.html(theme.CSS)  # constant styles only; see theme.py


def mode_badge() -> None:
    if public_mode():
        st.sidebar.caption("Public mode: offline mock provider only.")
    else:
        st.sidebar.caption("Private mode: local models allowed; paid models never from the UI.")


def load_or_explain(loaded: Loaded) -> list[RunReport]:
    for problem in loaded.problems:
        st.warning("A report file was skipped:")
        st.code(problem, language=None)
    if not loaded.reports:
        st.info(
            "No reports found yet. Start a run on the Run page, or locally with "
            "`python -m reliability.cli run`."
        )
        st.caption("Report folder:")
        st.code(str(report_dir()), language=None)
    return loaded.reports


def metric_cards(cards: Sequence[Card], sparklines: dict[str, list[float]] | None = None) -> None:
    sparklines = sparklines or {}
    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, card in zip(columns, cards[start : start + 3], strict=False):
            spark = sparklines.get(card.label)
            with column:
                st.metric(
                    card.label,
                    card.value,
                    delta=card.delta,
                    delta_color=card.delta_color,  # type: ignore[arg-type]
                    help=card.help,
                    border=True,
                    chart_data=spark if spark and len(spark) > 1 else None,
                    chart_type="line",
                    delta_arrow="off" if card.delta_color == "off" else "auto",
                )


def bar_figure(values: dict[str, float | None], title: str, *, percent: bool = True) -> go.Figure:
    names = [k for k, v in values.items() if v is not None]
    numbers = [float(v) for v in values.values() if v is not None]
    fig = go.Figure(
        go.Bar(
            x=numbers,
            y=names,
            orientation="h",
            marker_color=theme.COLORS["accent"],
            text=[f"{n:.0%}" if percent else f"{n:g}" for n in numbers],
            textposition="auto",
        )
    )
    xaxis = {"range": [0, 1], "tickformat": ".0%"} if percent else {}
    fig.update_layout(
        **theme.plotly_layout(title=title, xaxis=xaxis, yaxis={"autorange": "reversed"})
    )
    return fig


def count_figure(counts: dict[str, int], title: str) -> go.Figure:
    colors = [theme.COLORS["pass"] if k == "supported" else theme.COLORS["fail"] for k in counts]
    fig = go.Figure(go.Bar(x=list(counts), y=list(counts.values()), marker_color=colors))
    fig.update_layout(**theme.plotly_layout(title=title))
    return fig


def donut_figure(counts: dict[str, int], title: str) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=list(counts),
            values=list(counts.values()),
            hole=0.6,
            marker={"colors": theme.SERIES[: len(counts)]},
            sort=False,
        )
    )
    fig.update_layout(**theme.plotly_layout(title=title, showlegend=True))
    return fig


def trend_figure(points: Sequence[tuple[str, float | None, str]], title: str) -> go.Figure:
    # One category per run, in order: "#3 07:34:50" (time of day when the label is a timestamp).
    xs = [f"#{i} {p[0][11:19] if len(p[0]) >= 19 else p[0]}" for i, p in enumerate(points, 1)]
    ys = [p[1] for p in points]
    colors = [theme.STATUS_COLORS.get(p[2], theme.COLORS["na"]) for p in points]
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line={"color": theme.COLORS["accent"]},
            marker={"color": colors, "size": 10},
            text=[p[2] for p in points],
            hovertemplate="%{x}<br>%{y}<br>decision %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        **theme.plotly_layout(
            title=title,
            xaxis={"type": "category", "gridcolor": theme.COLORS["border"], "tickangle": -30},
            margin={"l": 56, "r": 16, "t": 40, "b": 70},
        )
    )
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
