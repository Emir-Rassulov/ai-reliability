"""Registry of checks. Adding a check = one module + one line here."""

from __future__ import annotations

from reliability.checks.base import BaseCheck, Check, CheckContext
from reliability.checks.forbidden_content import ForbiddenContentCheck
from reliability.checks.instruction_following import InstructionFollowingCheck
from reliability.checks.provider_call import ProviderCallCheck
from reliability.checks.refusal import RefusalCheck
from reliability.checks.required_facts import RequiredFactsCheck
from reliability.checks.schema_validation import StructuredOutputCheck

CHECK_CLASSES: tuple[type[BaseCheck], ...] = (
    RequiredFactsCheck,
    ForbiddenContentCheck,
    StructuredOutputCheck,
    RefusalCheck,
    InstructionFollowingCheck,
)

#: Every name a scenario may list in `checks`.
KNOWN_CHECKS: frozenset[str] = frozenset(
    {cls.name for cls in CHECK_CLASSES} | {ProviderCallCheck.name}
)


def build_checks(ctx: CheckContext) -> dict[str, Check]:
    return {cls.name: cls(ctx) for cls in CHECK_CLASSES}
