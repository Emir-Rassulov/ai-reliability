"""Colours, a small CSS block and a Plotly layout matching the dark Streamlit theme.

The CSS is a constant written here; it never contains report content.
"""

from __future__ import annotations

from typing import Any

COLORS: dict[str, str] = {
    "bg": "#0f1117",
    "panel": "#171a23",
    "border": "#2a2f3d",
    "text": "#e6e8ee",
    "muted": "#9aa3b5",
    "accent": "#6c8cff",
    "pass": "#3fbf7f",
    "fail": "#ff6b61",
    "review": "#e0b341",
    "error": "#b392f0",
    "na": "#7d8596",
}

STATUS_COLORS: dict[str, str] = {
    "PASS": COLORS["pass"],
    "FAIL": COLORS["fail"],
    "REVIEW_REQUIRED": COLORS["review"],
    "INCONCLUSIVE": COLORS["review"],
    "ERROR": COLORS["error"],
    "NOT_APPLICABLE": COLORS["na"],
}

SERIES = [COLORS["accent"], "#46c2d0", "#e58ac9", COLORS["review"], COLORS["pass"]]

CSS = f"""
<style>
[data-testid="stMetric"] {{
  background: {COLORS["panel"]};
  border: 1px solid {COLORS["border"]};
  border-radius: 10px;
  padding: 12px 14px;
}}
[data-testid="stMetricLabel"] p {{ color: {COLORS["muted"]}; font-size: 0.85rem; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }}
[data-testid="stExpander"] {{ border-color: {COLORS["border"]}; }}
</style>
"""


def plotly_layout(**overrides: Any) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": COLORS["text"], "size": 13},
        "margin": {"l": 10, "r": 10, "t": 36, "b": 10},
        "xaxis": {"gridcolor": COLORS["border"], "zeroline": False},
        "yaxis": {"gridcolor": COLORS["border"], "zeroline": False},
        "legend": {"orientation": "h", "y": -0.2},
        "colorway": SERIES,
        "height": 300,
    }
    layout.update(overrides)
    return layout
