"""Required facts: each annotated fact must be present, not negated, not contradicted.

Lexical verification (see `checks/text.py`). Per fact: present / missing / contradicted /
hedged. Any contradicted or missing fact fails the check; hedged-only facts make it
INCONCLUSIVE. A missing or contradicted `mandatory` fact also sets `mandatory_violation`,
which release rule 1 turns into FAIL regardless of severity.
"""

from __future__ import annotations

from typing import Any

from reliability.checks.base import BaseCheck
from reliability.checks.text import match_phrases
from reliability.models import CheckResult, Fact, Scenario, Status
from reliability.providers.base import ProviderResponse


def evaluate_facts(
    text: str, facts: list[Fact], window: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    evidence: list[dict[str, Any]] = []
    counts = {"present": 0, "missing": 0, "contradicted": 0, "hedged": 0}
    for fact in facts:
        found = match_phrases(
            text,
            fact.phrases,
            number=fact.number,
            unit=fact.unit,
            tolerance=fact.tolerance,
            window=window,
        )
        counts[found.status] += 1
        evidence.append(
            {
                "fact": fact.id,
                "result": found.status,
                "mandatory": fact.mandatory,
                "matched_phrase": found.phrase,
                "offset": found.offset,
                "reason": found.reason,
            }
        )
    return evidence, counts


class RequiredFactsCheck(BaseCheck):
    name = "required_facts"
    method = "lexical phrase match with word boundaries, negation/hedge window, number conflict"

    def run(self, scenario: Scenario, response: ProviderResponse) -> CheckResult:
        if not scenario.required_facts:
            return self.not_applicable(scenario, "scenario has no required_facts annotations")
        evidence, counts = evaluate_facts(
            response.text, scenario.required_facts, self.ctx.config.checks.negation_window
        )
        mandatory_violation = any(
            e["mandatory"] and e["result"] in ("missing", "contradicted") for e in evidence
        )
        expected = [f.id for f in scenario.required_facts]
        if counts["contradicted"] or counts["missing"]:
            bad = [e["fact"] for e in evidence if e["result"] in ("missing", "contradicted")]
            status, why = Status.FAIL, f"facts missing or contradicted: {', '.join(bad)}"
        elif counts["hedged"]:
            status, why = Status.INCONCLUSIVE, "some facts are stated only with hedged wording"
        else:
            status, why = Status.PASS, "all required facts present and not contradicted"
        return self.result(
            scenario,
            status,
            why,
            expected=expected,
            evidence=evidence,
            metrics=dict(counts),
            mandatory_violation=mandatory_violation,
        )
