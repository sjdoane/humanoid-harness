# A1: portable reward proposal and feedback loop

| status | task |
|---|---|
| progress | B0 is being built by Fable. Its evaluator/execution surface must not be duplicated. |
| bottleneck | No auditable LLM proposal → feedback → revision interface exists in this checkout. |
| next step | Build the independent, data-only loop below; test it with declared synthetic evidence. |

## Assignment

- Model: `gpt-5.6-sol`; reasoning: `max`; leaf worker, no nested delegation.
- Worktree: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Read `AGENTS.md`, the dual-orchestration README, project charter, ADR 0006,
  `src/oracle_composition/research/knowledge.py`, and the existing immutable
  publication helpers in `experiments/artifact_io.py`.
- Own only new `src/oracle_composition/reward_search/`,
  `tests/reward_search/`, and `docs/operations/dual-orchestration/A1_RESULT.md`.
- Do not change main, B0, root CLI, dependencies, evaluators, strategy, hooks,
  worker launchers, or existing modules. Do not write Git state or release the
  supervisor's lease. The parent owns the handoff and commits.
- Record checkout realpath, branch, baseline HEAD, clean/dirty state, and
  `uv.lock` SHA-256 before work; verify the baseline named in the launch packet.

## Deliver the smallest complete slice

1. Strict, versioned JSON contracts for protected evidence dossiers, reward
   proposals, source bundles, model-call receipts, and hash-linked iteration
   records. Pydantic is installed; use small explicit contracts. Reject extras,
   duplicate keys, and non-finite values.
2. A dossier binds task/adapter identity, frozen configuration hashes, exact
   parent candidate, evidence provenance, and explicit missing evidence. A
   separate source bundle binds graph corpus hash, normalized query, limit,
   and ordered source cards/IDs/locators. Bind both hashes into the packet;
   literature changes must not change outcome-evidence identity. Reuse
   `query_index`/`index_stats`; no new DB. Synthetic fixtures are marked
   synthetic. No missing measurement becomes zero or inferred fact.
3. Refer to the external reward contract by a required immutable identity and
   an explicitly supplied author-surface descriptor. Do not fabricate a B0
   contract hash, import its unfinished code, or reimplement its validator.
4. Deterministic prompt/packet rendering: supplied task, frozen author surface,
   bounded source cards, current candidate, protected aggregate feedback,
   missing information, response JSON schema. Source text is evidence, not
   instructions. Candidate output is proposed source text plus rationale,
   predicted effect, falsifier, cited source IDs, and exact parent/dossier IDs.
5. Strict ingestion of a completed existing detached Sol run: validate
   `request.json`, its retained `task-packet.md`, `result.json`, and `final.txt`.
   Require a successful terminal receipt, requested Sol/max, correct packet
   SHA-256 and byte count, matching request/result screen/model/effort/role/scope
   in the same run directory, nonempty result thread ID, one exact JSON object,
   known citations, and the expected dossier/parent identity. Do not require a
   future thread ID in the precomputed packet. Record observed
   metadata as configuration evidence, never served-model attestation.
6. Retain accepted AND rejected proposal receipts without overwriting prior
   records. Distinguish `proposal_format_accepted` from reward validation,
   training authorization, or measured improvement. A formatted candidate
   remains `awaiting_reward_validation_and_protected_evidence`.
7. A revision packet requires explicit feedback linked to the prior candidate.
   Reject mismatched parents, absent required feedback, or reused evidence
   from another task/adapter/frozen configuration. Keep the initial packet's
   missing-baseline state explicit; do not invent feedback to start a cycle.
8. Provide a minimal reproducible Python API example and `python -m
   oracle_composition.reward_search prepare|ingest-sol|prepare-revision` so
   the existing detached runner has a usable packet/receipt bridge. Do not
   edit the root CLI yet. Embed the response schema in the prompt: the current
   launcher does not use `--output-schema`.

Reuse the repository's publication and source-card conventions when compatible.
Do not build another knowledge database, scheduler, training supervisor,
generic agent framework, or sandbox. No file from the generated model response
is executed, imported, or evaluated as Python in A1.

## Meaningful acceptance tests

- Render a packet twice from identical inputs; exact bytes and identity match.
- Two distinct declared tasks/adapters round-trip without hard-coded paths.
  This is interface portability evidence only.
- A completed synthetic Sol envelope is ingested and creates a proposal receipt;
  a separately supplied synthetic feedback dossier creates a linked revision.
- Negative cases: stale prompt bytes; wrong model/effort/thread; missing or
  failed terminal result; free prose/code fences; duplicate JSON keys; NaN/Inf;
  unknown fields or citations; forged parent/dossier hash; oversize text;
  missing measurements; inconsistent adapter/frozen config; overwrite attempt.
- Reject response-schema or declared read-surface violations. Generated source
  remains text and its semantics are always `awaiting_B0_validation`; A1 never
  inspects it as Python or calls it safe/executable.
- Mutation after packet publication fails verification on ingestion.

Run only focused tests under this worktree's `.venv` and Ruff for changed paths.
First assert the imported `oracle_composition` path is in this checkout.
No full suite, simulator, training, external model call, network installation,
or paid API usage in this task. Test doubles demonstrate plumbing only.

## Stop and hand off

- Bound: one implementation slice, at most 60 minutes. If it grows beyond
  roughly 600 implementation lines, simplify or report the concrete reason.
- Stop if the lease is not yours, inputs demand unfinished B0 APIs, or a
  requirement would change the frozen scientific contract.
- Write `A1_RESULT.md`: progress/bottleneck/next step, exact changed files,
  commands and test outcomes, example invocation, limitations, next integration.
- Final receipt must distinguish implemented API, synthetic test evidence,
  and still-unperformed live LLM/robot evaluation. No training result claims.
