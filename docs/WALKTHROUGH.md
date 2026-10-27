# ai-reliability: a guided walkthrough

This document explains how the project is built and why it is built that way. It says
little about Python syntax. The focus is on the engineering decisions, the failure modes
and the QA principles behind each module.

Each major module is covered with five questions:
1. **What it is.**
2. **Why it exists.**
3. **What it gives us.**
4. **How it works.**
5. **How it is tested.**

---

## 0. The main idea in one minute

A typical "LLM tester" sends a prompt and looks at the answer. The goal here is wider: make
an evidence-based release decision, and prove that the tool making that decision is itself
reliable.

That gives four layers:

1. **Response checks** (`checks/`, `rag/`). Deterministic, explainable, annotation-based.
2. **Release decision** (`evaluation/release.py`). Rules that cannot be "averaged away".
3. **Testing the tester.** Self-test fixtures, a labeled set with precision/recall,
   metamorphic relations, and mutation testing on a reference RAG app.
4. **Safety and cost.** Offline by default, redaction of secrets, paid models blocked
   without an explicit unlock.

One principle runs through everything: **honesty over appearance**. If a verdict cannot be
derived from the annotations, it is `INCONCLUSIVE`, not `PASS`. If a check has a blind spot,
it is recorded in the tests as a `known_miss` and shown in the README.

---

## 1. Models and scenarios - `reliability/models.py`, `reliability/scenarios.py`

**What it is.** Pydantic models for scenarios, documents, check results and the report,
plus a loader that validates every file before a run starts.

**Why.** A scenario is a test case. A typo in a test case is especially dangerous: for
example, `expected_claim` instead of `expected_claims`. The check would silently get an empty
list, return `NOT_APPLICABLE`, and the test would be "green" without checking anything. That
is a classic false positive inside the test suite itself.

**What it gives us.** Unknown fields are forbidden (`extra="forbid"`). Every error names the
file, the scenario id, the field and the problem, for example:
`cases.json: scenario 'bad-1' field 'severity': Input should be ...`.
Semantics are checked too:
- check names exist;
- document ids exist in the dataset;
- every scenario has a canned mock response;
- a selftest scenario must declare its expected verdicts, and a gate scenario must not.

**How it works.** Validation in two stages:
1. **Structure** - Pydantic.
2. **Cross-file links** - `scenarios.py`: documents, checks, mock responses.

Errors are collected and reported as a list, not one at a time. That saves iterations.

**The gate and selftest roles - a key design decision.** The specification requires both:
- negative cases in the dataset;
- CI failing on any exit code other than 0.

The only way to satisfy both:
- bad fixtures (`selftest`) run **every time**, but do not count in the application's
  release decision;
- each of them must reproduce its pre-recorded verdicts;
- if even one does not, rule R0 turns the run into ERROR. A broken measuring instrument has
  no right to issue a verdict.

**How it is tested.** `tests/unit/test_models_config_providers.py` covers:
- unknown fields;
- invalid regexes and JSON Schemas;
- duplicate ids;
- a last message that is not from the user;
- the format of error messages;
- file size limits.

---

## 2. Configuration - `reliability/config.py`, `config/default.toml`

**What it is.** Weights, thresholds, limits, cost mode.

**Why.** Release thresholds are a team policy, not code. They must be changeable without
editing the source, and visible in the report.

**How it works.** Order of application:
1. `default.toml`.
2. The user's file (`--config`) on top of it, deep-merged.
3. `AIREL_*` environment variables.

Everything is validated: a threshold of `min_pass_rate = 7` is rejected. A snapshot of the
final configuration goes into every JSON report, so the decision is reproducible.

**How it is tested.** The tests cover:
- default values;
- overrides;
- broken TOML;
- a citation regex without an `id` group.

---

## 3. Providers and cost protection - `reliability/providers/`

**What it is.**
- The `Provider` protocol.
- The error hierarchy: timeout, 429, 5xx, authentication, malformed response.
- `MockProvider` and `StatefulMockProvider`.
- The cost guard and the budget guard.

**Why.** The project must be buildable and demonstrable for free. An accidental call to a
paid API from a test means real money and leaked data.

**How it works.**
- **Mock.** Returns canned text or simulates an error. With `fail_times` it fails only the
  first N times and then answers - that is how retries are tested.
- **Stateful mock.** Remembers messages within a `session_id`. The `echo_history` mode models
  a system that "hands over" the previous user's conversation. The cross-session leakage
  test is built on it.
- **Cost guard.** Classifies the model against an allowlist:
  - `mock`;
  - `local` - `ollama/...` on a loopback address;
  - `paid` - everything else, including unknown names.

  A paid model is refused **before the provider object is created**, so before any socket.
- **BudgetGuard.** Estimates the worst case: no more input tokens than UTF-8 bytes, no more
  output tokens than `max_tokens`. A request that could exceed the budget is not sent.
  Prices come only from the user's file and are never guessed.

**How it is tested.**
- A test with a "spy" on `socket.socket.__init__` proves that no socket is created when a
  model is refused.
- Separate tests cover classification, the budget and session isolation.

**The live provider** is covered separately, in section 14.

---

## 4. Lexical matching - `reliability/checks/text.py`

**What it is.** Shared primitives for facts and claims:
- a phrase on word boundaries;
- a sentence;
- a negation and hedging window;
- a number conflict.

**Why.** Without a semantic model, the method must be transparent and explainable. Every
verdict can be pointed at: "here are the characters, here is the rule".

**How it works.**
- "30 days" does not match "130 days" or "2.30 days": word boundaries and the decimal point
  are checked.
- Negation is looked for in the last 3 tokens **of the same clause**. A comma, semicolon or
  colon ends the scope of a negation (an idea from NegEx). Without this rule the sentence
  "if it does not stop, use kill -9" wrongly treated "kill -9" as negated. We caught this on
  one of our own scenarios.
- Number conflict: the same sentence has a different number with the same unit, with no
  negation before it. "Not 14 days as before - now 30 days" is not a conflict.

**The limits are written down explicitly.** Tense is not analysed ("used to be 30 days").
A negation after the phrase is not seen. There are no synonyms.

**How it is tested.** `tests/unit/test_text_matching.py`: a table of traps for word
boundaries, negation, hedging, sentences and commas.

---

## 5. LLM checks - `reliability/checks/`

| Check | Idea | Main trap it avoids |
|---|---|---|
| `required_facts` | the fact is present, not negated, not contradicted by a number | "130 days" ≠ "30 days" |
| `forbidden_content` | markers and regexes, `outside_refusal` policy | a codename inside a refusal is not a leak |
| `structured_output` | `json.loads` + JSON Schema | ```json``` fences break the contract unless explicitly allowed |
| `refusal` | refusal and abstention patterns | **over-refusal is a FAIL** just like an unsafe answer |
| `instruction_following` | only what is annotated: sections, length, list format | "In summary..." is not a section heading |
| `provider_call` | the call outcome: attempts, error class, empty or truncated response | an empty response is an application FAIL, an auth error is an infrastructure ERROR |

Why is an empty response a FAIL, but an authentication error an ERROR?

- **FAIL** - the application answered badly. It is a verdict about quality.
- **ERROR** - we could not get an answer. It is a verdict about infrastructure.

They must not be mixed: otherwise a network outage would look like a model regression.

**How it is tested.** `tests/unit/test_llm_checks.py`. Each check has good, bad and
borderline cases. Known limitations are pinned by tests too. For example, a paraphrase of a
forbidden phrase is **not** caught, and a test asserts exactly that, so the limitation stays
visible.

---

## 6. RAG checks - `reliability/rag/`

**Why separate checks.** In RAG an error can happen at three stages:
- search returned the wrong documents;
- the context contains outdated or someone else's data;
- the generator wrote something that is not in the context.

Each stage is checked separately, so a finding points at the culprit.

- **`retrieval_metrics`.** Precision@K, Recall@K, MRR.
  - Duplicates count once, at their best rank.
  - Empty retrieval gives 0 for every metric.
  - No relevance labels gives `NOT_APPLICABLE`, not an invented number.
- **`context_quality`.** Context precision and recall. Flags:
  - duplicates;
  - an outdated document (by `version`, then by `valid_until`);
  - an outdated document ranked **above** the current one - that is a FAIL;
  - an annotated conflict;
  - a disallowed access level - FAIL;
  - "relevant but supports no claim" (relevance ≠ support).
- **`grounding`.** Each expected claim gets one status: supported, unsupported (stated
  without a retrieved supporting document), contradicted, omitted or hedged. Claims from
  `false_claims` give FAIL. Unannotated text is measured as `unverified_content_ratio` and is
  **never** called a hallucination: we simply do not know.
- **`citations`.** Catches:
  - invented ids;
  - ids that were not retrieved;
  - a claim without a citation;
  - a citation to the wrong document.

**The checks are orthogonal on purpose.** In scenario `rag-015` the model cites an outdated
document that really was retrieved. So `citations` gives PASS: the citation is formally
correct. And `grounding` gives FAIL: the claim is false. The citation check does not judge
truth.

**How it is tested.** `tests/rag/test_rag_checks.py`. Metrics are compared with hand
calculations, including duplicates, empty retrieval, K larger than the result list and gaps
in the ranks.

---

## 7. Security - `prompt_injection`, `data_leakage`, `sensitive.py`, `redaction.py`

**Canary tokens.** Each malicious instruction carries a unique marker. If the marker shows up
in the answer, the injection worked. The marker is searched for in three forms:
- directly;
- spread out (`C A N A R Y - ...`);
- in base64.

Leaks of the system prompt are caught with 8-word "shingles" from the prompt text.

**Data leakage.** A secret is searched for in four forms:
- exact;
- without separators;
- in base64;
- by its last 4 characters as a standalone token ("card ending 1234").

A secret quoted **inside a refusal** is still a leak. Scenario `sec-009` shows partial
compliance: the model says "I can't", but names the last digits.

**Cross-session leakage.** The scenarios run in one `StatefulMockProvider` session:
- `sec-012` - user A shares a card number;
- `sec-013` - user B in the same session gets it back; the leak is caught;
- `sec-014` - a different session; isolation is confirmed.

If you run only `sec-013`, the CLI pulls in the earlier steps of the same session by itself.

**Redaction.** Everything that leaves the evaluator passes through `Redactor`:
- evidence;
- the response excerpt;
- errors;
- JSON;
- HTML;
- log lines.

The mask keeps only an alphabetic label: `CANARY-****Q7`, `sk-****Y0`. A card number keeps
only its last 2 digits: `****34`.

**How it is tested.** `tests/security/test_security.py`:
- a full run with logging, after which no form of any secret may appear in the reports,
  stdout, stderr or logs;
- a control test that the mocks really contain secrets - otherwise the previous check would
  be vacuous;
- an AST test for forbidden constructs (`eval`, `pickle`, `subprocess`...);
- a test that a full run opens no socket.

---

## 8. Orchestration and retries - `reliability/evaluation/orchestration.py`

**What it is.** The run loop: build the request, call the provider with bounded retries,
evaluate, assemble the report. A separate function, `run_rag_app`, does the same for any
system that implements `RAGSystem`.

**How it works.**
- Only timeouts, 429 and 503 are retried. At most `1 + max_retries` attempts in total.
- Backoff is exponential (0.5, 1, 2 ..., capped). If the server sent `Retry-After`, that
  value is used. A `Retry-After` above the limit stops the call instead of waiting 2 minutes.
- Sleep and the clock are injected. In tests the "sleep" records the delays without
  sleeping, so the tests are instant. With the mock provider the CLI does not sleep either
  (time is simulated), but the delays are recorded in the evidence.
- A response longer than `max_response_chars` is truncated and flagged: protection against
  huge responses.

**How it is tested.** `tests/integration/test_operational.py`:
- every failure path;
- the backoff cap;
- zero retries;
- latency measurement with an injected clock.

---

## 9. Evaluator, scoring, release decision - `evaluator.py`, `scoring.py`, `release.py`

**Evaluator.** Runs a scenario's checks. If the call failed, every check gets ERROR. A
crashing check also gets ERROR. In both cases - **never PASS** (the principle: an
incomplete critical check does not count as passed). The evaluator then compares the
verdicts of selftest scenarios with the declared ones.

**Score.** Informational only. The weighted share of PASS among PASS+FAIL.

**Decision.** Rules R0-R6 are applied in order. Every rule that fires is recorded; the first
one decides. Important details:
- R1 cannot be "diluted": one critical FAIL among a hundred PASS is a FAIL.
- The pass rate excludes ERROR and INCONCLUSIVE. Otherwise an infrastructure outage would
  trigger R2 (application FAIL) before R3 could say "this is infrastructure".
- A run with nothing to assess is REVIEW_REQUIRED, not PASS.

**How it is tested.** `tests/unit/test_release_and_scoring.py`. Every rule has an explicit
test, including the boundary exactly at the threshold and the firing order. A separate test
recomputes the decision from the stored JSON and compares it with the original: that is how
reproducibility is checked.

---

## 10. Reports - `reporting/`

- **JSON.** Keys are sorted, so two runs can be compared with a plain diff. Inside: hashes
  of the scenario files and the dataset, a configuration snapshot, the latency source
  (`simulated` or `measured`). Token counts are included only if the provider returned them.
- **Summary.** Text for a QA engineer:
  - the decision and its reasons;
  - self-tests;
  - critical findings;
  - RAG;
  - security;
  - operational issues;
  - concrete next steps.
- **Compare.** Changed statuses with regressions marked, the decision change, metric deltas.
  Exit code 1 on a regression.
- **HTML.** One self-contained file. Everything is escaped. The CSP allows only our own
  filter script, pinned by its SHA-256 hash. Even if escaping ever broke, an injected script
  would not run.
- **History.** One compact JSONL line per run, for trends. In CI it is written to the
  `history` branch.

---

## 11. Testing the tester

This is the most important part for an AI QA portfolio. The project has four independent
mechanisms:

1. **Self-test fixtures (every run).** 31 known-bad or tricky cases. Each must produce
   exactly its declared verdicts, otherwise R0 → ERROR.
2. **Labeled set (`check-quality`).** 80 cases with a "truth". For each check, TP, FP, FN,
   precision, recall and agreement are computed. Known misses are marked `known_miss` with a
   reason. The test fails in two situations:
   - an unexpected miss appears;
   - a known miss suddenly disappears. Then the label and the README have to be updated
     honestly.
3. **Metamorphic relations.** They need no "correct answer". They need to know how a verdict
   **must not** change: case, whitespace, document order, a neutral sentence, a negation,
   base64.

   Relation M1 found a real defect. The refusal patterns were written with single spaces,
   and a double space broke refusal detection.
4. **Mutation testing.** Five defects are injected into the reference app, one at a time.
   Each must be caught by its expected check and turn the decision into FAIL or REVIEW.

   The very first run revealed a gap. A hallucination (an invented restocking fee) was
   caught by `grounding`, but the release stayed PASS: one high-severity failure among 22
   checks did not push the pass rate below 90 %. The lesson: severity is a policy, and that
   policy has to be tested too.

An honest note. The labeled set and the scenarios were written by the same author as the
checks. So the numbers are optimistic. They show that the checks do what they claim on the
cases we thought of, and they make the blind spots explicit. They are not an estimate of
quality on real traffic.

---

## 12. Reference RAG app - `rag_app/`

**What it is.** A small but real RAG pipeline:
1. BM25 search.
2. A freshness filter.
3. An access filter.
4. Prompt assembly with an "untrusted data" marking.
5. Generation.

**Why.** Before this phase, the evaluator was tested on fixtures: "does it label prepared
answers correctly". Now there is a system under test, and we can prove that the evaluator
catches real bugs in a real pipeline.

**Architectural boundary.** `rag_app` does not import the evaluator. The evaluator knows only
`rag_app/interface.py` (the `RAGSystem` protocol). `cli.py` - the composition root - wires
them together. Both rules are checked by an AST test. So any other RAG system (vector-based,
HTTP) can be plugged in by implementing one protocol.

**Defects.** Each one breaks exactly one stage, and a test proves it: retrieval and the
prompt are compared with and without the defect.

| Defect | What it breaks |
|---|---|
| `no_access_filter` | skips the access filter |
| `stale_first` | keeps old versions and ranks them higher |
| `no_grounding_instruction` | removes the "answer only from the documents" instruction |
| `truncated_context` | truncates the context |
| `no_untrusted_marking` | inserts the documents as instructions |

**Limitation of the mock generator.** The model-behaviour defects
(`no_grounding_instruction`, `no_untrusted_marking`) are caught only as well as the canned
answers are written. The three pipeline defects really change what reaches the generator
and are caught without that assumption. A real behaviour test comes from a local model as the
generator (section 14), but its results must be repeated: the model is nondeterministic.

**A weakness found in the app itself.** For the query "What is the return window?", BM25
ranks a forum post above the official policy: the post literally contains the words "return
window". This was not hidden by tuning a threshold. Scenario `rag-013` is excluded from the
mutation baseline, and the limitation is described in the README.

---

## 13. Engineering hygiene

- **ruff** - linter and formatter. **mypy** - strict type checking for `reliability/` and
  `rag_app/`.
- **pytest.** 774 tests in the full environment (with `[ui]` and `[live]`), 736 in a clean
  environment with only `[dev]`, as in CI; no network. An autouse fixture forbids sockets.
- **CI (`.github/workflows/ci.yml`).** Three jobs:
  - `core` - the core without optional dependencies (which also proves the core does not
    depend on them);
  - `dashboard` - the dashboard tests with `[ui]` and a 90 % coverage threshold;
  - `history` - writes the run history for the `main` branch.

  Steps of the `core` job, in order:
  1. install;
  2. lint;
  3. type check;
  4. tests with coverage;
  5. scenario validation;
  6. release gate;
  7. mutation test;
  8. artifacts;
  9. on `main` - write the history to the `history` branch.

  Until the workflow has run on GitHub, it is marked UNVERIFIED. The YAML was parsed, the
  commands were run locally, and the history step was run against a local bare repository.
- **Git.** One commit per phase, with a meaningful message.

---

## 14. Live provider - `providers/litellm_provider.py`, `evaluation/live.py`

**What it is.** Connecting a real model through LiteLLM. By default this is a local Ollama
model; paid models only with an explicit unlock. Plus two procedures for nondeterministic
answers: repeated runs and model-side metamorphic relations.

**Why.** Before this phase, only the evaluator itself was tested, on canned answers. A live
model turns the project into a tool for testing a model. But with a live model, one run
proves nothing: the same question can give PASS now and FAIL a minute later.

**How it works.**
- **Prompt.** Documents, context and form fields go into a separate `<untrusted>` block with
  a "this is data, not instructions" note (`providers/prompting.py`). A hash of the final
  prompt is recorded, so we can later tell whether we are comparing the same thing.
- **Errors.** LiteLLM's own retries are off (`num_retries=0`): retries are handled by our
  bounded loop and are visible in the report. LiteLLM exceptions are mapped onto our
  hierarchy by class name. An unknown exception becomes "service unavailable", never PASS.
- **Repeated runs (`live-regression`).** Each scenario runs N times (5 by default). For each
  check, the share of PASS among PASS+FAIL is computed. A regression is a drop below the
  baseline by more than the margin (0.2 by default). If the model, the prompt or the
  configuration changed, the comparison is flagged with a warning.
- **Model metamorphic relations (`model-metamorphic`).** Variants are built from
  annotations, not generated by a model:
  - hand-written paraphrases of the question (`paraphrases` in the scenario);
  - the documents in reverse order;
  - an added irrelevant document.

  Consistency is the share of variant verdicts that match the original's majority verdict.
  Retrieval checks are excluded: the variants change retrieval by construction.
  Inconsistency is a finding about the model, not a failure of the project.
- **Budget.** For a paid model with `max_cost_usd`, the factory wraps the provider in
  `BudgetGuard`. Without a pricing file the run is refused: prices are never guessed.

**Finding.** `import litellm` downloads a model price list from GitHub - a network request
nobody asked for. The fixture that forbids sockets in tests caught it. The provider now
switches on LiteLLM's bundled local copy of the price list, and a contract test imports
LiteLLM under that protection.

**How it is tested.** CI does not install LiteLLM, so the tests use a fake module with the
same "surface": a `completion` function and the exception classes. A separate contract test
checks that surface against the real package when it is installed. Checked by hand:
- parsing a real `ModelResponse` (via the built-in `mock_response`, no network);
- a refused connection → 4 attempts → run status ERROR (exit code 3), not FAIL.

**What is not verified.** There was no run on a real model: the build environment has no
Ollama. The README marks this as UNVERIFIED.

---

## 15. Dashboard - `dashboard/`

**What it is.** A Streamlit app with five pages:
- Overview;
- History;
- Compare;
- RAG pipeline;
- Run.

**Why.** A portfolio reviewer can open a link and see the results without installing
anything.

**The main rule - presentation only.** `dashboard/` contains no checks, no scoring and no
release rules. Every number comes from stored reports. Comparison uses the same function as
the CLI `compare`. Runs go through the same orchestration API. Loading and aggregation live
in `data.py` as plain functions that are tested without Streamlit.

**Data honesty in practice.** On the first screenshot, a card showed a latency regression of
"+515 ms". In fact it compared a scenario-suite run (simulated latency) with a RAG-app run
(measured, about 0 ms). Now:
- a run is compared only with the previous run **against the same target**;
- no delta is shown between simulated and measured latency.

It is a good example of how a visualisation can lie even when every number in it is real.

**Safety.**
- **Public mode** (the default): mock provider only, no fields for models or keys.
- **Limits:** one run at a time, at most 60 scenarios.
- **Report content** is shown only through `st.code` and `st.dataframe`, that is, as text.
  `unsafe_allow_html` is not used anywhere - an AST test checks this.
- **Chart labels.** Plotly can render some HTML in labels. So values from files that end up
  in chart labels or in Markdown are restricted to a known set.

**How it is tested.** With Streamlit's built-in `AppTest`:
- every page renders without exceptions on real reports and on empty data;
- the filters change the visible findings;
- in public mode a live model cannot be selected;
- a `<script>` and a Markdown link from a report stay inert text.

The dashboard tests run in a separate CI job with `[ui]`, so the core does not depend on
Streamlit.

---

## 16. What's next (roadmap)

- **Run phase 7 on a real model.** A baseline, a regression run, metamorphic consistency,
  the mutation test with a live generator - and record the results in the README.
- **Phase 8.** Ragas and DeepEval with a local judge: the agreement rate with the
  deterministic checks and an analysis of every disagreement. Deliberately deferred: without
  a working judge model, only unverified adapters could be written, and this project is
  built on not presenting unverified work as finished.
- **A native LLM judge.** Behind its own protocol, with verdicts shown next to the
  deterministic ones, never merged with them.
