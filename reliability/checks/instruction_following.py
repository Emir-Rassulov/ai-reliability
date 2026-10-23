"""Completeness / instruction following: only what is annotated is assessed.

* `required_sections`: a line that starts with the heading text (optionally after `#`
  markers or `**`), case-insensitive.
* `max_words` / `max_chars`: measured on whitespace-collapsed text.
* `required_format`: at least two bullet (`-`, `*`, `•`) or numbered (`1.`/`1)`) lines.
"""

from __future__ import annotations

import re
from typing import Any

from reliability.checks.base import BaseCheck
from reliability.checks.text import collapse_ws
from reliability.models import CheckResult, Scenario, Status
from reliability.providers.base import ProviderResponse

_BULLET = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)


def _has_section(text: str, heading: str) -> bool:
    pattern = rf"^\s*(?:#{{1,6}}\s*)?(?:\*\*)?{re.escape(heading)}(?![\w])"
    return re.search(pattern, text, re.IGNORECASE | re.MULTILINE) is not None


class InstructionFollowingCheck(BaseCheck):
    name = "instruction_following"
    method = "annotated sections, length limits and list format"

    def run(self, scenario: Scenario, response: ProviderResponse) -> CheckResult:
        spec = scenario.instructions
        if spec is None:
            return self.not_applicable(scenario, "scenario has no instructions annotations")
        text = response.text
        flat = collapse_ws(text)
        problems: list[dict[str, Any]] = []
        for heading in spec.required_sections:
            if not _has_section(text, heading):
                problems.append({"rule": "required_section", "section": heading})
        words = len(flat.split())
        if spec.max_words is not None and words > spec.max_words:
            problems.append({"rule": "max_words", "limit": spec.max_words, "observed": words})
        if spec.max_chars is not None and len(flat) > spec.max_chars:
            problems.append({"rule": "max_chars", "limit": spec.max_chars, "observed": len(flat)})
        if spec.required_format is not None:
            pattern = _BULLET if spec.required_format == "bullet_list" else _NUMBERED
            if len(pattern.findall(text)) < 2:
                problems.append({"rule": "required_format", "format": spec.required_format})
        metrics: dict[str, float | int | bool | None] = {"words": words, "chars": len(flat)}
        if problems:
            rules = ", ".join(sorted({p["rule"] for p in problems}))
            return self.result(
                scenario,
                Status.FAIL,
                f"instruction violations: {rules}",
                expected=spec.model_dump(),
                evidence=problems,
                metrics=metrics,
            )
        return self.result(
            scenario,
            Status.PASS,
            "all annotated instructions followed",
            expected=spec.model_dump(),
            metrics=metrics,
        )
