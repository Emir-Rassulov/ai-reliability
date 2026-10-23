"""Context quality from document metadata and scenario annotations.

Metrics
* context_precision = relevant retrieved / retrieved (unique), when relevance labels exist.
* context_recall    = required evidence retrieved / required evidence, when annotated.

Flags
* duplicates              - an id retrieved more than once.
* outdated                - a document superseded by another retrieved document on the same
                            topic (higher `version`; ties broken by later `valid_until`,
                            where no `valid_until` means "still valid").
* outdated_ranked_above_current - the superseded document is ranked above its replacement.
* conflicting             - both documents of an annotated `conflicts` pair were retrieved.
* unauthorized            - a document whose `access` label is not in `user_access`.
* relevant_not_supporting - relevant (labelled) but supports none of the expected claims.

FAIL on: unauthorized documents, an outdated document ranked above its replacement, or
context_recall below `min_context_recall`. Other flags are reported, not failed: handling
a conflict or an older document correctly is the generator's job (see grounding).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from reliability.checks.base import BaseCheck
from reliability.models import CheckResult, Document, Scenario, Status
from reliability.providers.base import ProviderResponse
from reliability.rag.retrieval_metrics import dedupe


def _freshness(doc: Document) -> tuple[int, date]:
    return doc.version, doc.valid_until or date.max


class ContextQualityCheck(BaseCheck):
    name = "context_quality"
    method = "document metadata (topic, version, valid_until, access) + annotations"

    def run(self, scenario: Scenario, response: ProviderResponse) -> CheckResult:
        unique, duplicates = dedupe(scenario.retrieved)
        docs = {r.doc_id: self.ctx.documents[r.doc_id] for r in unique}
        rank = {r.doc_id: r.rank for r in unique}
        has_labels = bool(scenario.relevant_doc_ids)
        required = scenario.required_evidence_ids
        if not unique and not has_labels and not required:
            return self.not_applicable(scenario, "nothing retrieved and nothing annotated")

        evidence: list[dict[str, Any]] = []
        flags: list[str] = []
        failing: list[str] = []

        relevant = set(scenario.relevant_doc_ids or [])
        precision = (
            round(len(relevant & docs.keys()) / len(docs), 4) if has_labels and docs else None
        )
        if has_labels and not docs:
            precision = 0.0
        recall = round(len(set(required) & docs.keys()) / len(required), 4) if required else None

        if duplicates:
            flags.append("duplicates")
            evidence.append({"flag": "duplicates", "duplicate_count": duplicates})

        for doc_id, doc in docs.items():
            newer = [
                other
                for other_id, other in docs.items()
                if other_id != doc_id
                and other.topic == doc.topic
                and _freshness(other) > _freshness(doc)
            ]
            if not newer:
                continue
            current = max(newer, key=_freshness)
            above = rank[doc_id] < rank[current.doc_id]
            flags.append("outdated")
            evidence.append(
                {
                    "flag": "outdated",
                    "doc_id": doc_id,
                    "superseded_by": current.doc_id,
                    "valid_until": doc.valid_until.isoformat() if doc.valid_until else None,
                    "ranked_above_current": above,
                }
            )
            if above:
                flags.append("outdated_ranked_above_current")
                failing.append(f"outdated '{doc_id}' ranked above current '{current.doc_id}'")

        for pair in scenario.conflicts:
            if all(d in docs for d in pair.doc_ids):
                flags.append("conflicting")
                evidence.append({"flag": "conflicting", "doc_ids": list(pair.doc_ids)})

        allowed = set(scenario.user_access)
        for doc_id, doc in docs.items():
            if doc.access not in allowed:
                flags.append("unauthorized")
                evidence.append(
                    {
                        "flag": "unauthorized",
                        "doc_id": doc_id,
                        "access": doc.access,
                        "user_access": sorted(allowed),
                    }
                )
                failing.append(f"unauthorized document '{doc_id}' ({doc.access})")

        supporting = {d for c in scenario.expected_claims for d in c.supported_by}
        if scenario.expected_claims:
            for doc_id in sorted(relevant & docs.keys() - supporting):
                flags.append("relevant_not_supporting")
                evidence.append({"flag": "relevant_not_supporting", "doc_id": doc_id})

        min_recall = self.ctx.config.retrieval.min_context_recall
        t = scenario.retrieval_thresholds
        if t and t.min_context_recall is not None:
            min_recall = t.min_context_recall
        if recall is not None and recall < min_recall:
            failing.append(f"context_recall {recall} < {min_recall}")

        metrics: dict[str, float | int | bool | None] = {
            "context_precision": precision,
            "context_recall": recall,
            "retrieved_unique": len(docs),
        }
        flags = sorted(set(flags))
        if failing:
            return self.result(
                scenario,
                Status.FAIL,
                "; ".join(failing),
                expected={"min_context_recall": min_recall, "user_access": sorted(allowed)},
                evidence=evidence,
                metrics=metrics,
                flags=flags,
            )
        why = "context acceptable" + (f" (flags: {', '.join(flags)})" if flags else "")
        return self.result(
            scenario,
            Status.PASS,
            why,
            expected={"min_context_recall": min_recall, "user_access": sorted(allowed)},
            evidence=evidence,
            metrics=metrics,
            flags=flags,
        )
