"""Grounding against annotated `expected_claims` and `false_claims`.

Per expected claim (lexical match, see `checks/text.py`):
* supported     - present, and one of its `supported_by` documents was retrieved.
* unsupported   - present, but none of its supporting documents was retrieved
                  (the answer went beyond the evidence it was given).
* contradicted  - negated, or a conflicting number in the same sentence.
* omitted       - not present.
* hedged        - only hedged wording.
* unattributed  - present, but the claim has no `supported_by` annotation (cannot decide).
Any non-negated `false_claims` phrase is a contradiction.

Text not mapped to any annotated claim is reported as `unverified_content_ratio` (share of
sentences), never as a hallucination. For live (non-simulated) responses, a ratio above the
scenario's `max_unverified_ratio` (default 0) turns an otherwise passing verdict into
INCONCLUSIVE. `abstain` scenarios pass only if the answer states that the evidence is
insufficient and asserts no false claim.
"""

from __future__ import annotations

from typing import Any

from reliability.checks.base import BaseCheck
from reliability.checks.refusal import detect_behavior
from reliability.checks.text import match_phrases as match
from reliability.checks.text import phrase_pattern, prep, sentence_spans
from reliability.models import CheckResult, ExpectedBehavior, Scenario, Status
from reliability.providers.base import ProviderResponse


def unverified_ratio(text: str, phrases: list[str]) -> float:
    prepared = prep(text)
    spans = sentence_spans(prepared)
    if not spans:
        return 0.0
    unmapped = 0
    for start, end in spans:
        sentence = prepared[start:end]
        if any(phrase_pattern(p).search(sentence) for p in phrases):
            continue
        if detect_behavior(sentence).behavior in ("refuse", "abstain", "ambiguous"):
            continue
        unmapped += 1
    return round(unmapped / len(spans), 4)


class GroundingCheck(BaseCheck):
    name = "grounding"
    method = "annotated claim matching + supporting-document availability"

    def run(self, scenario: Scenario, response: ProviderResponse) -> CheckResult:
        behavior = scenario.expected_behavior
        if behavior is ExpectedBehavior.REFUSE:
            return self.not_applicable(scenario, "refusal scenarios are judged by refusal checks")
        if behavior is ExpectedBehavior.ANSWER and not (
            scenario.expected_claims or scenario.false_claims
        ):
            return self.not_applicable(scenario, "no claim annotations")

        window = self.ctx.config.checks.negation_window
        text = response.text
        retrieved = {r.doc_id for r in scenario.retrieved}
        evidence: list[dict[str, Any]] = []
        verdicts: dict[str, int] = {}

        def note(kind: str, **fields: Any) -> None:
            verdicts[kind] = verdicts.get(kind, 0) + 1
            evidence.append({"result": kind, **fields})

        mandatory_violation = False
        for claim in scenario.expected_claims:
            found = match(
                text,
                claim.phrases,
                number=claim.number,
                unit=claim.unit,
                tolerance=claim.tolerance,
                window=window,
            )
            if found.status == "present":
                if not claim.supported_by:
                    kind = "unattributed"
                elif retrieved & set(claim.supported_by):
                    kind = "supported"
                else:
                    kind = "unsupported"
            else:
                kind = {"missing": "omitted", "contradicted": "contradicted", "hedged": "hedged"}[
                    found.status
                ]
            if claim.mandatory and kind in ("omitted", "contradicted"):
                mandatory_violation = True
            note(
                kind,
                claim=claim.id,
                mandatory=claim.mandatory,
                offset=found.offset,
                reason=found.reason,
            )

        for false in scenario.false_claims:
            found = match(text, false.phrases, window=window)
            if found.status == "present":
                note("false_claim", claim=false.id, offset=found.offset, note=false.source_note)
            elif found.status == "hedged":
                note("hedged_false_claim", claim=false.id, offset=found.offset)

        all_phrases = [p for c in scenario.expected_claims for p in c.phrases]
        all_phrases += [p for c in scenario.false_claims for p in c.phrases]
        ratio = unverified_ratio(text, all_phrases)
        metrics: dict[str, float | int | bool | None] = {
            "unverified_content_ratio": ratio,
            **{k: v for k, v in sorted(verdicts.items())},
        }
        expected = {
            "expected_behavior": behavior.value,
            "claims": [c.id for c in scenario.expected_claims],
            "false_claims": [c.id for c in scenario.false_claims],
        }

        if behavior is ExpectedBehavior.ABSTAIN:
            detected = detect_behavior(text).behavior
            evidence.append({"detected_behavior": detected})
            if verdicts.get("false_claim"):
                status, why = Status.FAIL, "asserted a false claim instead of abstaining"
            elif detected == "abstain":
                status, why = Status.PASS, "answer states that the evidence is insufficient"
            elif detected in ("refuse", "ambiguous"):
                status, why = Status.INCONCLUSIVE, f"expected abstention, detected '{detected}'"
            else:
                status, why = Status.FAIL, f"expected abstention, detected '{detected}'"
            return self.result(
                scenario, status, why, expected=expected, evidence=evidence, metrics=metrics
            )

        bad = {"false_claim", "contradicted", "unsupported", "omitted"} & verdicts.keys()
        uncertain = {"hedged", "hedged_false_claim", "unattributed"} & verdicts.keys()
        if bad:
            status, why = Status.FAIL, "grounding problems: " + ", ".join(sorted(bad))
        elif uncertain:
            status, why = Status.INCONCLUSIVE, "undecidable claims: " + ", ".join(sorted(uncertain))
        else:
            status, why = Status.PASS, "all expected claims supported by retrieved documents"
            limit = scenario.max_unverified_ratio or 0.0
            if not response.is_simulated and ratio > limit:
                status = Status.INCONCLUSIVE
                why = (
                    f"claims supported, but unverified_content_ratio {ratio} > {limit} "
                    "(live output)"
                )
        return self.result(
            scenario,
            status,
            why,
            expected=expected,
            evidence=evidence,
            metrics=metrics,
            mandatory_violation=mandatory_violation,
        )
