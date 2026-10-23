"""Offline generator with canned answers keyed by trace id (scenario) and delivered context.

An answer entry matches when its `docs` list equals the documents delivered to the
generator, in rank order, and its optional `grounded` / `untrusted_marked` flags equal the
prompt's flags. Entries with flags are preferred over entries without. If nothing matches,
the generator abstains - a realistic behaviour for a grounded model, and the honest default
for a canned one.

Limitation (stated in the README): with this generator, detection of model-behaviour
defects (`no_grounding_instruction`, `no_untrusted_marking`) depends on how the canned
answers were written. A real local model tests them genuinely, but nondeterministically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_app.pipeline import AssembledPrompt

FALLBACK = "The provided documents do not contain enough information to answer that."


class MockGenerator:
    name = "rag-app-mock-generator"
    simulated = True

    def __init__(self, answers: dict[str, list[dict[str, Any]]], fallback: str = FALLBACK) -> None:
        self.answers = answers
        self.fallback = fallback
        self.prompts: list[AssembledPrompt] = []

    @classmethod
    def from_file(cls, path: Path) -> MockGenerator:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(raw["answers"], raw.get("fallback", FALLBACK))

    def generate(self, prompt: AssembledPrompt, trace_id: str | None) -> str:
        self.prompts.append(prompt)
        candidates = [
            entry
            for entry in self.answers.get(trace_id or "", [])
            if tuple(entry["docs"]) == prompt.delivered
            and entry.get("grounded", prompt.grounded) == prompt.grounded
            and entry.get("untrusted_marked", prompt.untrusted_marked) == prompt.untrusted_marked
        ]
        if not candidates:
            return self.fallback
        candidates.sort(key=lambda e: -(("grounded" in e) + ("untrusted_marked" in e)))
        return str(candidates[0]["text"])
