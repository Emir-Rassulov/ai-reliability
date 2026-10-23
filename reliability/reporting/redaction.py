"""Redaction of canaries and synthetic secrets in evidence, reports and logs.

Every string that leaves the evaluator (check evidence, explanations, response excerpts,
errors, log records) passes through `Redactor`. It knows the exact sensitive values of the
loaded dataset (and their compact / base64 forms) plus a few generic secret shapes.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from typing import Any

from reliability.sensitive import base64_variants, compact_pattern, mask

GENERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"CANARY-[A-Z0-9][A-Z0-9-]{2,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


class Redactor:
    def __init__(self, sensitive_values: Iterable[str] = ()) -> None:
        self._rules: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = []
        for value in sorted(set(sensitive_values), key=len, reverse=True):
            replacement = mask(value)

            def repl(_m: re.Match[str], r: str = replacement) -> str:
                return r

            self._rules.append((re.compile(re.escape(value), re.IGNORECASE), repl))
            compact = compact_pattern(value)
            if compact is not None:
                self._rules.append((compact, repl))
            for variant in base64_variants(value):
                self._rules.append((re.compile(re.escape(variant)), repl))
        for pattern in GENERIC_PATTERNS:
            self._rules.append((pattern, lambda m: mask(m.group(0))))

    def redact(self, text: str) -> str:
        for pattern, repl in self._rules:
            text = pattern.sub(repl, text)
        return text

    def redact_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, dict):
            return {self.redact(str(k)): self.redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [self.redact_obj(v) for v in obj]
        return obj


class RedactingFilter(logging.Filter):
    """Logging filter that redacts the fully formatted message of every record."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.redactor.redact(record.getMessage())
        record.args = None
        return True
