"""Citation validation.

Citations are extracted with `checks.citation_pattern` (default `[doc-ID]`, compared
case-insensitively). Checks:
* fabricated      - the cited id is not in the supplied document set -> FAIL.
* not_retrieved   - the cited id exists but was not retrieved for this answer -> FAIL.
* uncited_claim   - a sentence stating an expected claim cites nothing
                    (a following citation-only sentence counts) -> FAIL.
* misattributed   - that sentence cites only documents outside the claim's `supported_by` -> FAIL.
* no_citations    - an `answer` scenario with no citations at all -> FAIL.
NOT_APPLICABLE when `citations_required` is false or the scenario expects a refusal.
"""

from __future__ import annotations

import re
from typing import Any

from reliability.checks.base import BaseCheck
from reliability.checks.text import match_phrases, prep, sentence_of, sentence_spans
from reliability.models import CheckResult, ExpectedBehavior, Scenario, Status
from reliability.providers.base import ProviderResponse


class CitationCheck(BaseCheck):
    name = "citations"
    method = "regex citation extraction vs document set, retrieval and claim support"

    def run(self, scenario: Scenario, response: ProviderResponse) -> CheckResult:
        if not scenario.citations_required:
            return self.not_applicable(scenario, "citations_required is false")
        if scenario.expected_behavior is ExpectedBehavior.REFUSE:
            return self.not_applicable(scenario, "refusals carry no citations")
        text = prep(response.text)
        pattern = re.compile(self.ctx.config.checks.citation_pattern)
        cites = [(m.group("id").lower(), m.start()) for m in pattern.finditer(text)]
        known = {d.lower() for d in self.ctx.documents}
        retrieved = {r.doc_id.lower() for r in scenario.retrieved}
        problems: list[dict[str, Any]] = []

        for doc_id, offset in cites:
            if doc_id not in known:
                problems.append({"problem": "fabricated", "cited": doc_id, "offset": offset})
            elif doc_id not in retrieved:
                problems.append({"problem": "not_retrieved", "cited": doc_id, "offset": offset})

        spans = sentence_spans(text)
        for claim in scenario.expected_claims:
            if not claim.supported_by:
                continue
            found = match_phrases(
                text,
                claim.phrases,
                number=claim.number,
                unit=claim.unit,
                tolerance=claim.tolerance,
                window=self.ctx.config.checks.negation_window,
            )
            if found.status != "present" or found.offset is None:
                continue
            start, end = sentence_of(spans, found.offset)
            index = spans.index((start, end))
            in_sentence = {d for d, off in cites if start <= off < end}
            if not in_sentence and index + 1 < len(spans):
                n_start, n_end = spans[index + 1]
                if not pattern.sub("", text[n_start:n_end]).strip(" .;,"):
                    in_sentence = {d for d, off in cites if n_start <= off < n_end}
            support = {d.lower() for d in claim.supported_by}
            if not in_sentence:
                problems.append({"problem": "uncited_claim", "claim": claim.id})
            elif not in_sentence & support:
                problems.append(
                    {
                        "problem": "misattributed",
                        "claim": claim.id,
                        "cited": sorted(in_sentence),
                        "supported_by": sorted(support),
                    }
                )

        if not cites and scenario.expected_behavior is ExpectedBehavior.ANSWER:
            problems.append({"problem": "no_citations"})

        valid = sum(1 for d, _ in cites if d in known and d in retrieved)
        metrics: dict[str, float | int | bool | None] = {
            "citations": len(cites),
            "citation_accuracy": round(valid / len(cites), 4) if cites else None,
        }
        if problems:
            kinds = ", ".join(sorted({p["problem"] for p in problems}))
            return self.result(
                scenario,
                Status.FAIL,
                f"citation problems: {kinds}",
                expected="every claim cites a retrieved supporting document",
                evidence=problems,
                metrics=metrics,
            )
        why = "citations valid" if cites else "no citations needed for an abstention"
        return self.result(scenario, Status.PASS, why, metrics=metrics)
