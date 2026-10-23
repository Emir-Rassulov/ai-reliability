"""Self-contained HTML report built with the standard library only.

* No external scripts, styles, fonts or images; no network requests.
* Every value is HTML-escaped (model output is untrusted) after redaction.
* A Content-Security-Policy allows only the single inline script, pinned by its SHA-256
  hash, so even an escaping bug could not execute injected script.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

from reliability.models import RunReport, Status
from reliability.reporting.redaction import Redactor


@dataclass(frozen=True)
class HtmlTable:
    title: str
    columns: list[str]
    rows: list[list[str]]
    note: str = ""


CSS = """
:root{--bg:#f7f7f5;--fg:#1c1c1c;--muted:#666;--card:#fff;--line:#e2e2de;
--pass:#1f7a4d;--fail:#b3261e;--warn:#9a6700;--err:#6f42c1;--na:#888}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e8e8e6;--muted:#a0a0a0;
--card:#1e1e1e;--line:#333;--pass:#4cc38a;--fail:#ff6b61;--warn:#e0b341;--err:#b392f0;--na:#999}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:14px 16px;margin:10px 0}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;
border:1px solid currentColor}
.PASS{color:var(--pass)}.FAIL{color:var(--fail)}.REVIEW_REQUIRED,.INCONCLUSIVE{color:var(--warn)}
.ERROR{color:var(--err)}.NOT_APPLICABLE{color:var(--na)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.metric b{display:block;font-size:20px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}.scroll{overflow-x:auto}
pre{white-space:pre-wrap;word-break:break-word;margin:6px 0 0;font-size:12px}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
select,input{font:inherit;padding:4px 6px;background:var(--card);color:var(--fg);
border:1px solid var(--line);border-radius:6px}
"""

SCRIPT = (
    "(function(){var s=document.getElementById('f-status'),r=document.getElementById('f-role'),"
    "q=document.getElementById('f-text');function u(){var rows=document.querySelectorAll("
    "'#findings tbody tr');for(var i=0;i<rows.length;i++){var t=rows[i];var ok=(!s.value||"
    "t.dataset.status===s.value)&&(!r.value||t.dataset.role===r.value)&&(!q.value||"
    "t.textContent.toLowerCase().indexOf(q.value.toLowerCase())>=0);t.style.display=ok?'':"
    "'none';}}s.onchange=u;r.onchange=u;q.oninput=u;})();"
)


def _csp() -> str:
    digest = base64.b64encode(hashlib.sha256(SCRIPT.encode("utf-8")).digest()).decode("ascii")
    return (
        "default-src 'none'; style-src 'unsafe-inline'; "
        f"script-src 'sha256-{digest}'; img-src 'none'; connect-src 'none'; form-action 'none'"
    )


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:.3g}"


def _table(columns: Sequence[str], rows: Sequence[Sequence[str]], table_id: str = "") -> str:
    head = "".join(f"<th>{_e(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{_e(v)}</td>" for v in row) + "</tr>" for row in rows)
    ident = f' id="{_e(table_id)}"' if table_id else ""
    return (
        f'<div class="scroll"><table{ident}><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def render_html(report: RunReport, redactor: Redactor, tables: Sequence[HtmlTable] = ()) -> str:
    data = RunReport.model_validate(redactor.redact_obj(report.model_dump(mode="json")))
    d = data.decision
    parts: list[str] = []
    parts.append(
        f"<h1>AI reliability report</h1><div class='muted'>run {_e(data.run_id)} &middot; "
        f"{_e(data.started_at)} &middot; provider {_e(data.provider)} "
        f"({_e(data.model or 'unknown')})"
        f" &middot; target {_e(data.target)}"
        + (f" &middot; defects {_e(', '.join(data.defects))}" if data.defects else "")
        + "</div>"
    )
    rules = "".join(f"<li><b>{_e(r.rule)}</b>: {_e(r.detail)}</li>" for r in d.fired_rules)
    parts.append(
        f"<div class='card'><span class='badge {_e(d.decision.value)}'>"
        f"{_e(d.decision.value)}</span>"
        f"<ul>{rules}</ul></div>"
    )
    metrics = [
        ("pass rate", _fmt(d.pass_rate, True)),
        ("error rate", _fmt(d.error_rate, True)),
        ("inconclusive rate", _fmt(d.inconclusive_rate, True)),
        ("score (info)", _fmt(data.score)),
        ("self-tests", f"{data.selftest.matched}/{data.selftest.total}"),
        ("p95 latency ms", _fmt(data.latency.p95_ms)),
    ]
    cards = "".join(
        f"<div class='card metric'><span class='muted'>{_e(k)}</span><b>{_e(v)}</b></div>"
        for k, v in metrics
    )
    parts.append(f"<div class='grid'>{cards}</div>")
    parts.append("<h2>Gate checks by status</h2>")
    parts.append(_table(["status", "count"], [[k, str(v)] for k, v in data.counts.items()]))

    parts.append("<h2>RAG metrics (gate means)</h2>")
    parts.append(_table(["metric", "value"], [[k, _fmt(v)] for k, v in data.rag_metrics.items()]))

    parts.append(
        "<h2>Findings</h2><p class='muted'>Failed, inconclusive and errored checks. "
        "Selftest rows are known-bad fixtures the evaluator is expected to catch.</p>"
        "<div class='filters'><select id='f-status'><option value=''>all statuses</option>"
        "<option>FAIL</option><option>INCONCLUSIVE</option><option>ERROR</option></select>"
        "<select id='f-role'><option value=''>all roles</option><option>gate</option>"
        "<option>selftest</option></select><input id='f-text' placeholder='filter text'></div>"
    )
    rows: list[str] = []
    for s in data.scenarios:
        for c in s.checks:
            if c.status in (Status.PASS, Status.NOT_APPLICABLE):
                continue
            evidence = json.dumps(c.evidence, indent=1, sort_keys=True)
            rows.append(
                f"<tr data-status='{_e(c.status.value)}' data-role='{_e(s.role.value)}'>"
                f"<td class='{_e(c.status.value)}'>{_e(c.status.value)}</td>"
                f"<td>{_e(s.role.value)}</td>"
                f"<td>{_e(s.scenario_id)}<div class='muted'>{_e(s.name)}</div></td>"
                f"<td>{_e(c.check)}<div class='muted'>{_e(c.severity.value)}</div></td>"
                f"<td>{_e(c.explanation)}<details><summary>evidence</summary><pre>{_e(evidence)}</pre>"
                f"<div class='muted'>response excerpt</div><pre>{_e(s.response_excerpt)}</pre>"
                "</details></td></tr>"
            )
    head = "".join(f"<th>{h}</th>" for h in ("status", "role", "scenario", "check", "explanation"))
    parts.append(
        f"<div class='scroll'><table id='findings'><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    if not rows:
        parts.append("<p class='muted'>No findings.</p>")
    for table in tables:
        parts.append(f"<h2>{_e(table.title)}</h2>")
        if table.note:
            parts.append(f"<p class='muted'>{_e(table.note)}</p>")
        parts.append(_table(table.columns, table.rows))
    parts.append(
        f"<p class='muted'>Generated by ai-reliability {_e(data.tool_version)}. "
        "All data is synthetic. Evidence is redacted.</p>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<meta http-equiv='Content-Security-Policy' content=\"{_csp()}\">"
        f"<title>AI Reliability Report</title><style>{CSS}</style></head><body><main>"
        + "".join(parts)
        + f"</main><script>{SCRIPT}</script></body></html>\n"
    )
