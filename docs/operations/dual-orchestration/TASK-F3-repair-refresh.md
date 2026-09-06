# F3 repair and exact-baseline refresh

- One Sol/max builder, no delegation; stop within 20 minutes of immutable
  request creation. No candidate-model call, network, simulator, training,
  install, full suite, commit, push, or peer checkout activity.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`;
  branch `astra/reward-loop`. Record HEAD, dirty paths, lock hash and import root.
- Use its existing writer lease and `apply_patch`. Only these paths are writable:
  1. `src/oracle_composition/reward_search/t2_model_contracts.py`
  2. `src/oracle_composition/reward_search/t2_model_protocol.py`
  3. `tests/reward_search/test_t2_model_protocol.py`
  4. new `docs/operations/dual-orchestration/F3_REPAIR_RESULT.md`
- The first three files are pre-existing untracked F3 work. Preserve unrelated
  work and the unchanged original `F3_RESULT.md`. Do not edit accepted helpers,
  exports, dependencies, shared strategy, hooks, launcher, or historical records.

## Read

- Root `AGENTS.md`, dual-orchestration `README.md`.
- In this docs directory: `TASK-F3-initial-protocol.md`,
  `F3_PARENT_CHECKPOINT.md`, `F3_REVIEW_FINDINGS.md`, `F2_ACCEPTANCE.md`.
- Complete current F3 modules/test, plus the independent report and probes in
  `.orchestration/f3-independent-20260906/`.
- Verify the pre-repair snapshot `.orchestration/f3-review-snapshot-20260906.json`
  before editing. Archived exact originals are in
  `.orchestration/f3-pre-repair-20260906/`; never edit them.

## Repair confirmed findings

### F3-01: timestamp arithmetic

- Replace overflow-prone `started + timedelta(...)` with an elapsed-duration
  comparison or equivalent total bounded arithmetic.
- Preserve the declared inclusive 0..1,200-second duration rule, negative and
  over-deadline rejection, and normalized malformed-timestamp errors.
- Add tests for minimum/maximum representable timestamps, year boundaries,
  negative duration, exact deadline and one second beyond it. No wall-clock
  freshness rule or authenticated-time claim is introduced by this repair.
- Valid abstract fixture metadata must not crash ingestion; if rejected, retain
  bounded raw bytes and emit the existing negative outcome receipt.

### F3-02: all-or-none identities

- Non-retained run-artifact bindings require filename, SHA-256 and byte count
  individually `None`; retained bindings require all three.
- Rejected receipts require both proposal and recipe hashes individually
  `None`; accepted receipts require both.
- Use real parser tests covering all partial subsets, not only the original
  reproducer. Preserve rejection reasons and fixed artifact ordering.

### F3-03: exact public types before work

- Exact-check every bytes and native-Path argument at each public entry before
  record reconstruction, any source/run reads, or publication.
- Preserve full semantic validation afterward and later path-state validation.
- Test both path-taking APIs and every argument position. Use callback traps
  only to prove early refusal, not to replace the real success-path helpers.
- Keep wrong types, subclasses, forged typed records and nested mutations
  refused; never accept preconstructed models at a bytes boundary.

### F3-04: expected configuration is not observation

- Rename the fixed receipt fields to `expected_model`,
  `expected_reasoning_effort`, `expected_runner_kind`, `expected_owner`,
  `expected_role`, and `expected_scope`.
- Rename the receipt metadata literal to
  `expected_configuration_not_served_model_attestation`.
- Do not rename the launcher's real `requested_*` fields or modify A1 models.
  A malformed request has no invented observed values. Retained raw request
  bytes remain the evidence for what it actually said.
- F3 has never been accepted or used for a real call; update only its unaccepted
  schema/tests. Keep the original review and archived receipts unchanged.
- Test a mismatched model, malformed request JSON, and ordinary accepted fixture
  to ensure the fixed expected values cannot be read as observed model service.

## Refresh the exact baseline

- Read this immutable Git object from Astra, not any peer dirty file:
  `87d2e39c47b6747b505bc2657d505aeda2265b5e:experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json`.
- Exact identity: 1,773 bytes, no newline, SHA-256
  `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f`.
- Update F3's single pinned baseline Git identity/hash/count and exact test
  bytes. Reject the old 1,144-byte baseline at the refreshed public boundary.
  Preserve old fixtures and receipts in the pre-repair archive; do not relabel.
- Continue to reconstruct exact bytes/internal identities. This artifact adds
  declared admission-contract bindings; it does not prove runtime admission,
  a successful load, measurements, or candidate improvement.
- Keep tracking-only `r_task=+0.0`, no parent alpha/beta recipe, target 3.0 m/s,
  cadence 0.015 s and all accepted F2 formula/bounds/input hashes unchanged.
- Keep the prompt and initial dossier explicit about missing observed adapter
  certificate, candidate admission, execution/evaluator results and measurements.
  Do not add fake feedback or a revision API. Preserve only-alpha/beta authoring.

## Verify and return

- Use only Astra's `.venv`; run updated F3 tests plus
  `tests/reward_search/test_formula_proposal_loop.py`. No broader suite.
- Run Ruff lint/format and compile on the three changed Python files.
- Verify every predecessor hash in `.orchestration/f2-review-20260905T2237.json`
  and `uv.lock` remains unchanged. F3 pre-repair hashes intentionally change;
  verify their archive remains exact instead.
- Retain red/green regression results; do not run historical defect-asserting
  probes against changed production code and call their expected failure a new
  defect. The parent already reproduced all seven on the archived version.
- Write `F3_REPAIR_RESULT.md`: three status rows, exact edits and hashes,
  each finding addressed, baseline identity, tests actually run, failures and
  remaining gates. No commit. A successful builder is not independent closure.
