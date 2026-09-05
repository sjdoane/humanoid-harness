# R1 parent-execution boundary repair

| status | current truth |
|---|---|
| progress | Parent source validation, capture, and drift checks are AST/data-only. Public dynamic entry points refuse; candidate compilation and evaluation exist only in the private child worker after environment and resource-limit setup. |
| bottleneck | Dynamic determinism, axiom, scale, containment, and runtime admission remain unreviewed and unmeasured. Historical B0 dynamic receipts were not regenerated or readmitted. |
| next step | Parent inspects this diff, runs an independent R1 review, and then scopes R2 host-fixture/runtime-containment work before enabling any candidate execution. |

## Identity and scope

- Worktree: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Branch/start: `astra/reward-loop` at
  `67722402c58374cf7f22347f7055abb979c9436f`; initially clean.
- Run: `.orchestration/sol-runs/20260905T032725Z-9908742a-6959-4e8b-b014-ecdfa5c0f247`.
- Launcher requested `gpt-5.6-sol`, reasoning `max`, one write builder. The
  request record is not provider/model attestation. No delegation was used.

## Changed paths

| path | result |
|---|---|
| `src/oracle_composition/rewards/static_validation.py` | Replaced executable validation with `StaticallyAcceptedTaskTermSourceV1` and an explicit static-only receipt; removed callable/determinism state; unified `if`/`IfExp` depth accounting. |
| `src/oracle_composition/rewards/_sandbox_worker.py` | Moved candidate compile/install/output checks into the private worker implementation reached after environment sanitization and resource limits. |
| `src/oracle_composition/rewards/sandbox.py` | Source capture/drift uses static validation; public worker start refuses while runtime admission is unreviewed. |
| `src/oracle_composition/rewards/scale_calibration.py` | Removed local duck-typed evaluation and receipt generation; requires an exact source-bound worker route, then refuses pending R2 admission review. |
| `src/oracle_composition/rewards/contract.py` | Validates JSON array structure and numeric leaf types before conversion; rejects wrong-dtype direct arrays. |
| `src/oracle_composition/experiments/reward_target_speed_manifest.py` | Updated only the static-validator source consumer. |
| `tests/rewards/test_{contract,static_validation,scale_calibration,sandbox}.py` | Added strict types, metadata, exact 8/9 depth, callable-sentinel, stale-receipt, and explicit R2 deferral coverage. |
| `tests/rewards/test_parent_execution_boundary.py` | Added parent no-installer/no-callback/no-in-process-execution and public runtime-refusal regressions. |

## Verification

| command | result |
|---|---|
| `.venv/bin/python -c 'import pathlib, oracle_composition; print(pathlib.Path(oracle_composition.__file__).resolve())'` | Resolved inside this checkout: `.../humanoid-harness-astra/src/oracle_composition/__init__.py`. |
| `.venv/bin/python -m pytest -q tests/rewards/test_contract.py tests/rewards/test_static_validation.py tests/rewards/test_scale_calibration.py tests/rewards/test_parent_execution_boundary.py tests/rewards/test_sandbox.py` | `49 passed, 36 skipped in 0.09s`. Skips: 3 dynamic scale/axiom cases and 33 inherited worker/OS-canary cases, all labeled for R2 host fixtures. |
| `.venv/bin/python -m pytest -q tests/reward_search tests/integration/test_research_mailbox.py` | `59 passed in 2.08s`. |
| `.venv/bin/ruff check <11 changed Python files>` | Passed. |
| `.venv/bin/ruff format --check <11 changed Python files>` | Passed. |
| `.venv/bin/python -m py_compile <11 changed Python files>` | Passed with no output. |
| `git diff --check` | Passed. |

No candidate source, reward subprocess, OS canary, model, simulator, training,
or full suite ran. No dynamic receipt was created. No historical receipt,
lease, dependency, evaluator, study constant, fingerprint, commit, or remote was
changed.

## Finding disposition

- Addressed in code, pending independent review: SCI-02/06 and ROB-01/10 from
  the R1 packet: parent execution, arbitrary callable fallback, mixed
  conditional depth, and array coercion.
- Still open: every R2 containment/admission finding and every R3
  lineage/receipt/rights finding listed in `A3_RESULT.md`; dynamic determinism,
  axioms, scaling, and runtime behavior have no fresh evidence.
- Out of scope: `tests/experiments/test_reward_target_speed_manifest.py` still
  imports the removed ambiguous validator name and invokes candidates. A later
  authorized edit must convert that test to static receipt assertions before a
  full-suite run; it was not executed here.
