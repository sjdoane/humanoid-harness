# F3 independent verification

- One Sol/max leaf, no delegation. Deadline: 20 minutes from immutable request
  creation. Use only the local subscription launcher; no model-candidate call.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`;
  branch `astra/reward-loop`. Record HEAD, dirty paths, import origin, lock hash.
- This is a **test-output-only writer**. You may create files only under the
  initially absent `.orchestration/f3-independent-20260906/` directory.
- Do not edit/add/delete source, tests, tracked docs, dependencies, snapshots,
  launcher, hooks, lease machinery, peer files or other existing artifacts.
  Do not commit, push, install, call the network, launch subprocess agents,
  import a simulator, run training or a full suite.
- Use `apply_patch` for any new probe/report source; normal pytest temporary
  fixture writes are permitted only inside the declared output directory.
- You are not the builder. Independently determine whether the narrow initial
  T2 packet/ingestion implementation meets its contract.

## Read and verify

1. Root `AGENTS.md`, dual-orchestration `README.md`.
2. `TASK-F3-initial-protocol.md`, `F3_RESULT.md`, `F3_PARENT_CHECKPOINT.md`,
   `F2_ACCEPTANCE.md` in this docs directory.
3. The complete two F3 modules and their test file:
   `reward_search/t2_model_contracts.py`, `reward_search/t2_model_protocol.py`
   under `src/oracle_composition`, and
   `tests/reward_search/test_t2_model_protocol.py`.
4. Relevant reused A1 bounded-reader/parser/publication/envelope helpers and
   trusted F1/F2/T2 formula/input implementations; do not alter them.
5. Verify every hash in `.orchestration/f3-review-snapshot-20260906.json`
   before and after review. A changed source is a stop, not permission to repair.

F3's original baseline is a historical static fixture: 1,144 bytes / SHA-256
`0986d4fc907e94185e14f58c9ef6eabf8ec26a8f336c68ee450bbacd5d8224d4`
at `a51ea9e68aa9e27a573e372f67e36d6fa4d494cd`. It matched the builder packet.
Fable has since changed it. No actual candidate call has occurred. The refresh
is a required subsequent gate, not proof that the builder disobeyed its packet.

## Execute bounded verification

- Confirm the output directory is absent before creating it; never delete an
  existing directory to make this condition true. Stop and report if it exists.
- Use the checkout's `.venv`. Set `PYTHONDONTWRITEBYTECODE=1`,
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and `TMPDIR` to a fresh subdirectory of
  the authorized output directory. Keep pytest capture disabled (`-s`),
  cache provider disabled (`-p no:cacheprovider`), and `--assert=plain`.
- Run only the two focused files initially, with `--basetemp` set to a fresh
  non-existing path inside that directory (pytest clears basetemp):
  `tests/reward_search/test_t2_model_protocol.py` and
  `tests/reward_search/test_formula_proposal_loop.py`.
- Parent reproduced 79 tests; verify independently. Add small independent
  adversarial probes under your output directory only. Use a different fresh
  basetemp for each invocation. Keep all negative results.
- If permissions deny any step, report the exact denial and continue with
  allowed static checks. Do not weaken settings or retry outside that boundary.
- Do not use mocks to bypass the reader, parser, publisher or source checks.

## Review questions

- Are exact baseline, fixed target/cadence/formula, immutable context and
  semantic-vs-rendered hashes revalidated at every public boundary?
- Are all successfully bounded-read run bytes retained before parsing, with
  accurate missing/empty/oversized/nonregular/unreadable states?
- Can rejected artifact/receipt models carry partial identity fields despite
  their validator messages? If yes, demonstrate actual parser behavior and
  distinguish schema weakness from a reachable ingestion acceptance defect.
- On malformed/mismatched envelopes, do receipt fields distinguish the fixed
  expected model configuration from observed request metadata clearly enough?
- Does the model-facing prompt convey the true tracking-only baseline and
  missing evidence needed for an honest initial hypothesis? No invented parent.
- Are malformed JSON, duplicate keys, deep nesting, wrong types, subclasses,
  mutated/forged typed records and nonfinite parameters normalized safely?
- Are runtime/training/admission always missing or unauthorized, A1/F1 schemas
  isolated, and source identities/negative outcomes preserved?
- Do not expand into general adversarial Python execution or make numerical
  alpha/beta tuning into a robotics result. That is outside this slice.

## Return

- Write a concise `REVIEW.md` inside your allowed directory and a final response.
- Verdict: `ACCEPT_F3_STATIC_ONLY` or `REJECT_F3`. Acceptance authorizes neither
  a live candidate call nor runtime integration.
- List actionable findings with severity, file/line, exact reproduction,
  consequence and minimal repair. No speculative exploit lists.
- Include tests actually executed/counts, failed attempts, before/after hash
  checks, source/working-tree scope, remaining baseline-refresh gate and limits.
- Finish before your watcher deadline. Do not release or alter another lease;
  the launcher handles release of your own verification lease.
