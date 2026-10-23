"""Structured output: parse JSON and validate it against the scenario's JSON Schema.

Markdown code fences are stripped only when `checks.strip_code_fences` is enabled
(off by default: a model asked for raw JSON that wraps it in fences is not following the
format contract).
"""

from __future__ import annotations

import json
import re

from jsonschema import Draft202012Validator

from reliability.checks.base import BaseCheck
from reliability.models import CheckResult, Scenario, Status
from reliability.providers.base import ProviderResponse

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(?P<body>.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _looks_truncated(text: str) -> bool:
    stripped = text.strip()
    opens = stripped.count("{") + stripped.count("[")
    closes = stripped.count("}") + stripped.count("]")
    return opens > closes or stripped.count('"') % 2 == 1


class StructuredOutputCheck(BaseCheck):
    name = "structured_output"
    method = "json.loads + jsonschema Draft 2020-12 validation"

    def run(self, scenario: Scenario, response: ProviderResponse) -> CheckResult:
        if scenario.output_schema is None:
            return self.not_applicable(scenario, "scenario has no output_schema")
        text = response.text
        if self.ctx.config.checks.strip_code_fences and (fence := _FENCE.match(text)):
            text = fence.group("body")
        if not text.strip():
            return self.result(scenario, Status.FAIL, "empty output", expected="JSON document")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            kind = "truncated JSON" if _looks_truncated(text) else "invalid JSON"
            return self.result(
                scenario,
                Status.FAIL,
                f"{kind}: {exc.msg} at line {exc.lineno} column {exc.colno}",
                expected="JSON document",
                evidence=[{"error": kind, "position": exc.pos}],
            )
        validator = Draft202012Validator(scenario.output_schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
        if errors:
            evidence = [
                {
                    "path": "/" + "/".join(str(p) for p in err.absolute_path),
                    "validator": str(err.validator),
                    "message": err.message[:200],
                }
                for err in errors
            ]
            return self.result(
                scenario,
                Status.FAIL,
                f"{len(errors)} schema violation(s)",
                expected="valid against output_schema",
                evidence=evidence,
                metrics={"violations": len(errors)},
            )
        return self.result(scenario, Status.PASS, "valid JSON matching the schema")
