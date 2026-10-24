"""Generator backed by any `complete(messages) -> str` function, e.g. a live model.

Keeps `rag_app` independent of the evaluator: the CLI (composition root) passes in a
function that calls the live provider. Documents are kept in an explicit untrusted block
unless the `no_untrusted_marking` defect moved them into the instructions.
"""

from __future__ import annotations

from collections.abc import Callable

from rag_app.pipeline import AssembledPrompt

Complete = Callable[[list[dict[str, str]]], str]


def prompt_messages(prompt: AssembledPrompt) -> list[dict[str, str]]:
    system = prompt.system
    if prompt.documents_block:
        system = f"{system}\n\n{prompt.documents_block}"
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt.query}]


class CallableGenerator:
    simulated = False

    def __init__(self, name: str, complete: Complete) -> None:
        self.name = name
        self.complete = complete

    def generate(self, prompt: AssembledPrompt, trace_id: str | None) -> str:
        return self.complete(prompt_messages(prompt))
