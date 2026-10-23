"""Operational check of the provider call itself (it judges a CallOutcome, not text).

* PASS  - a response arrived (possibly after retries; attempts are reported).
* ERROR - the call failed: non-retryable error, retries exhausted, or Retry-After too long.
          This is an infrastructure verdict, not an application one.
* FAIL  - a response arrived but is unusable: empty, cut off by the model
          (`finish_reason == "length"`), or longer than `limits.max_response_chars`.
"""

from __future__ import annotations

from typing import Any

from reliability.config import AppConfig
from reliability.evaluation.outcome import CallOutcome
from reliability.models import CheckResult, Scenario, Status


class ProviderCallCheck:
    name = "provider_call"
    method = "call outcome: error class, attempts, emptiness, truncation"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run_call(self, scenario: Scenario, outcome: CallOutcome) -> CheckResult:
        evidence: dict[str, Any] = {
            "attempts": outcome.attempts,
            "waits_s": list(outcome.waits_s),
        }
        if outcome.error is not None or outcome.response is None:
            kind = getattr(outcome.error, "kind", "unknown")
            evidence["error_kind"] = kind
            evidence["stop_reason"] = outcome.stop_reason
            status, why = Status.ERROR, f"provider call failed ({kind}): {outcome.stop_reason}"
        else:
            response = outcome.response
            evidence["finish_reason"] = response.finish_reason
            if not response.text.strip():
                status, why = Status.FAIL, "empty response"
            elif response.finish_reason == "length":
                status, why = Status.FAIL, "response truncated by the model (finish_reason=length)"
            elif outcome.truncated_by_limit:
                limit = self.config.limits.max_response_chars
                status, why = Status.FAIL, f"response exceeded max_response_chars={limit}"
            else:
                retried = " after retries" if outcome.attempts > 1 else ""
                status, why = Status.PASS, f"response received{retried}"
        return CheckResult(
            check=self.name,
            status=status,
            severity=scenario.severity_of(self.name),
            method=self.method,
            explanation=why,
            expected="a complete, non-empty response",
            evidence=[evidence],
            metrics={"attempts": outcome.attempts},
        )
