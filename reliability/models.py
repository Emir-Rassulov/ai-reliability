"""Pydantic v2 data models: scenarios, documents, results, and run reports.

All input models forbid unknown fields, so a typo in a scenario file is a validation
error instead of a silently ignored annotation.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,99}$"


class Status(StrEnum):
    PASS = "PASS"  # noqa: S105 - a verdict name, not a password
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(StrEnum):
    LLM = "llm"
    RAG = "rag"
    SECURITY = "security"
    OPERATIONAL = "operational"


class ExpectedBehavior(StrEnum):
    ANSWER = "answer"
    REFUSE = "refuse"
    ABSTAIN = "abstain"


class Role(StrEnum):
    """`gate`: the release candidate's behavior, counted in the release decision.

    `selftest`: a known-bad (or known-tricky) fixture whose declared verdicts prove that
    the evaluator still detects what it should. Not counted in the release decision.
    """

    GATE = "gate"
    SELFTEST = "selftest"


class Decision(StrEnum):
    PASS = "PASS"  # noqa: S105 - a verdict name, not a password
    FAIL = "FAIL"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    ERROR = "ERROR"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- scenario input


class Message(_Strict):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ScenarioInput(_Strict):
    system: str = ""
    messages: list[Message] = Field(min_length=1)
    #: Reference context supplied outside retrieval (e.g. a pasted ticket thread).
    context: str | None = None
    #: Structured input fields (e.g. form fields of a support ticket).
    fields: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _last_message_is_user(self) -> ScenarioInput:
        if self.messages[-1].role != "user":
            raise ValueError("the last message must have role 'user'")
        return self


class RetrievedRef(_Strict):
    doc_id: str = Field(pattern=ID_PATTERN)
    rank: int = Field(ge=1)


class Fact(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    phrases: list[str] = Field(min_length=1)
    number: float | None = None
    #: Unit word for number-conflict detection; inferred from the phrases when omitted.
    unit: str | None = None
    tolerance: float = Field(default=0.0, ge=0)
    mandatory: bool = False

    @field_validator("phrases")
    @classmethod
    def _non_blank(cls, v: list[str]) -> list[str]:
        if any(not p.strip() for p in v):
            raise ValueError("phrases must not be blank")
        return v


class ExpectedClaim(Fact):
    supported_by: list[str] = Field(default_factory=list)


class FalseClaim(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    phrases: list[str] = Field(min_length=1)
    source_note: str = ""


class ForbiddenMarker(_Strict):
    value: str = Field(min_length=1)
    kind: Literal["literal", "regex"] = "literal"
    category: str = "forbidden"
    policy: Literal["any_occurrence", "outside_refusal"] = "any_occurrence"

    @model_validator(mode="after")
    def _regex_compiles(self) -> ForbiddenMarker:
        if self.kind == "regex":
            try:
                re.compile(self.value)
            except re.error as exc:
                raise ValueError(f"invalid regex: {exc}") from exc
        return self


class SecretSpec(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    value: str = Field(min_length=6)
    kind: str = "secret"


class ConflictPair(_Strict):
    doc_ids: tuple[str, str]


class Instructions(_Strict):
    required_sections: list[str] = Field(default_factory=list)
    max_words: int | None = Field(default=None, ge=1)
    max_chars: int | None = Field(default=None, ge=1)
    required_format: Literal["bullet_list", "numbered_list"] | None = None


class RetrievalThresholds(_Strict):
    k: int | None = Field(default=None, ge=1)
    min_precision_at_k: float | None = Field(default=None, ge=0, le=1)
    min_recall_at_k: float | None = Field(default=None, ge=0, le=1)
    min_mrr: float | None = Field(default=None, ge=0, le=1)
    min_context_recall: float | None = Field(default=None, ge=0, le=1)


class Scenario(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1)
    category: Category
    severity: Severity
    role: Role = Role.GATE
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    input: ScenarioInput
    #: Conversation session for StatefulMockProvider (cross-scenario leakage tests).
    session_id: str | None = None
    retrieved: list[RetrievedRef] = Field(default_factory=list)
    #: None means "no relevance labels": retrieval metrics are NOT_APPLICABLE.
    relevant_doc_ids: list[str] | None = None
    required_evidence_ids: list[str] = Field(default_factory=list)
    user_access: list[str] = Field(default_factory=lambda: ["public"])
    expected_behavior: ExpectedBehavior = ExpectedBehavior.ANSWER
    required_facts: list[Fact] = Field(default_factory=list)
    expected_claims: list[ExpectedClaim] = Field(default_factory=list)
    false_claims: list[FalseClaim] = Field(default_factory=list)
    citations_required: bool = False
    forbidden_markers: list[ForbiddenMarker] = Field(default_factory=list)
    canary_tokens: list[str] = Field(default_factory=list)
    secrets: list[SecretSpec] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    instructions: Instructions | None = None
    conflicts: list[ConflictPair] = Field(default_factory=list)
    retrieval_thresholds: RetrievalThresholds | None = None
    #: Live providers only: tolerated share of answer sentences not mapped to annotations.
    max_unverified_ratio: float | None = Field(default=None, ge=0, le=1)
    check_severity: dict[str, Severity] = Field(default_factory=dict)
    checks: list[str] = Field(min_length=1)
    #: Required for selftest scenarios: the verdict each check must produce.
    expected_statuses: dict[str, Status] = Field(default_factory=dict)
    #: Eligible for `--target rag-app` (phase 5).
    rag_app: bool = False

    @model_validator(mode="after")
    def _consistency(self) -> Scenario:
        if len(set(self.checks)) != len(self.checks):
            raise ValueError("checks contain duplicates")
        if self.role is Role.SELFTEST and not self.expected_statuses:
            raise ValueError("selftest scenarios must declare expected_statuses")
        if self.role is Role.GATE and self.expected_statuses:
            raise ValueError("gate scenarios must not declare expected_statuses")
        unknown = set(self.expected_statuses) - set(self.checks)
        if unknown:
            raise ValueError(
                f"expected_statuses names checks not listed in checks: {sorted(unknown)}"
            )
        unknown_sev = set(self.check_severity) - set(self.checks)
        if unknown_sev:
            raise ValueError(f"check_severity names checks not listed: {sorted(unknown_sev)}")
        for label, items in (
            ("required_facts", [f.id for f in self.required_facts]),
            ("expected_claims", [c.id for c in self.expected_claims]),
            ("false_claims", [c.id for c in self.false_claims]),
            ("secrets", [s.id for s in self.secrets]),
        ):
            if len(set(items)) != len(items):
                raise ValueError(f"{label} contain duplicate ids")
        if self.output_schema is not None:
            try:
                Draft202012Validator.check_schema(self.output_schema)
            except SchemaError as exc:
                raise ValueError(
                    f"output_schema is not a valid JSON Schema: {exc.message}"
                ) from exc
        return self

    @property
    def query(self) -> str:
        return self.input.messages[-1].content

    def severity_of(self, check_name: str) -> Severity:
        return self.check_severity.get(check_name, self.severity)

    def all_sensitive_values(self) -> list[str]:
        return [*self.canary_tokens, *(s.value for s in self.secrets)]


class ScenarioFile(_Strict):
    schema_version: Literal["1.0"]
    synthetic: Literal[True]
    scenarios: list[Scenario] = Field(min_length=1)


class Document(_Strict):
    doc_id: str = Field(pattern=ID_PATTERN)
    title: str
    text: str = Field(min_length=1)
    topic: str
    version: int = Field(ge=1)
    valid_until: date | None = None
    access: str = "public"


class DocumentSet(_Strict):
    schema_version: Literal["1.0"]
    synthetic: Literal[True]
    documents: list[Document] = Field(min_length=1)


# --------------------------------------------------------------------------- results


class TokenUsage(_Strict):
    input_tokens: int | None = None
    output_tokens: int | None = None


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    status: Status
    severity: Severity
    method: str
    explanation: str
    expected: Any = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, float | int | bool | None] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    mandatory_violation: bool = False
    duration_ms: float = 0.0


STATUS_PRECEDENCE: tuple[Status, ...] = (
    Status.FAIL,
    Status.ERROR,
    Status.INCONCLUSIVE,
    Status.PASS,
    Status.NOT_APPLICABLE,
)


def aggregate_status(statuses: list[Status]) -> Status:
    """Worst status wins: FAIL > ERROR > INCONCLUSIVE > PASS > NOT_APPLICABLE."""
    for candidate in STATUS_PRECEDENCE:
        if candidate in statuses:
            return candidate
    return Status.NOT_APPLICABLE


class ScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    name: str
    category: Category
    severity: Severity
    role: Role
    tags: list[str]
    status: Status
    checks: list[CheckResult]
    latency_ms: float | None = None
    latency_source: Literal["simulated", "measured"] | None = None
    attempts: int = 0
    error: dict[str, str] | None = None
    usage: TokenUsage | None = None
    response_excerpt: str = ""
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    selftest_match: bool | None = None
    selftest_mismatches: list[str] = Field(default_factory=list)


class FiredRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: str
    outcome: Decision
    detail: str


class ReleaseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    fired_rules: list[FiredRule]
    applicable_checks: int
    pass_rate: float | None
    error_rate: float | None
    inconclusive_rate: float | None
    latency_p95_ms: float | None


class SelftestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    matched: int
    mismatched: list[str]


class LatencyStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    median_ms: float | None
    p95_ms: float | None
    sources: list[str]


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_schema: Literal["1.0"] = "1.0"
    run_id: str
    started_at: str
    finished_at: str
    tool_version: str
    target: Literal["scenarios", "rag-app"]
    defects: list[str]
    provider: str
    model: str | None
    scenario_files: dict[str, str]
    dataset_hash: str | None
    config: dict[str, Any]
    counts: dict[str, int]
    selftest: SelftestSummary
    score: float | None
    decision: ReleaseDecision
    rag_metrics: dict[str, float | None]
    latency: LatencyStats
    token_usage: TokenUsage | None
    errors: list[str]
    warnings: list[str]
    scenarios: list[ScenarioResult]
