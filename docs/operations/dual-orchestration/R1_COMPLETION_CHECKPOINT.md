# R1 parent completion, before re-review

| status | current truth |
|---|---|
| progress | Parent completed the static-only repair: 139 focused checks passed, 16 explicitly deferred; no candidate reward executed. |
| bottleneck | The first independent review timed out without a verdict. R1 remains unaccepted; R2/R3 still gate dynamic reward use. |
| next step | Freeze the completed diff and request one independent Sol/max review with a 20-minute deadline. |

## Retained review failure

- Baseline: `67722402c58374cf7f22347f7055abb979c9436f`, plus uncommitted R1.
- First review: `20260905T043347Z-3e92f3ac-fb43-45fb-84a8-ae1cb14ee3de`.
- Terminal: `INTERRUPTED_TERM`, exit 143, 2026-09-05 04:43:51 UTC;
  deadline watcher sent TERM after 600 seconds; no escalation or final verdict.
- The preceding checkpoint verified its 12-file snapshot before parent edits.
  That snapshot is historical, not the completed diff's identity.
- Read-only pytest encountered unavailable temporary storage. Preserve that
  limitation; the next reviewer can inspect temp-backed tests and use
  `-s -p no:cacheprovider` for selected no-temp checks. Do not widen its sandbox.

## Completion changes

| issue | repair and evidence |
|---|---|
| All 33 sandbox tests skipped | Restored 20 data-only or fully mocked checks; 13 subprocess/host cases remain deferred. Mocks do not establish OS containment. |
| Receipt version coercion | Exact integer `1` required; boolean, float, string, null, and unsupported integer versions rejected. |
| Metadata serialization callbacks | Whitelist JSON literal trees before canonical serialization; reject nested callbacks and custom containers without invoking them. Nested metadata round-trips through plain JSON. |
| Stale manifest-test consumer | Uses static acceptance and no-callable assertions; both exact donor fixtures also checked in the simulator-free reward test module. |
| Vacuous deferred scale cases | Each skipped case now fails explicitly if enabled before its worker-backed assertion exists. |

The metadata callback issue was visible in the incomplete review's probes;
the parent reproduced it as a regression test. There was no independent
acceptance attached to that partial output.

## Parent verification, 2026-09-05

```text
.venv/bin/python -m pytest -q \
  tests/rewards/test_contract.py tests/rewards/test_static_validation.py \
  tests/rewards/test_scale_calibration.py tests/rewards/test_parent_execution_boundary.py \
  tests/rewards/test_sandbox.py tests/reward_search tests/integration/test_research_mailbox.py
139 passed, 16 skipped in 2.26s
```

- Includes the unchanged 59 A1/mailbox tests, 20 restored sandbox checks,
  and 11 new completion cases. Earlier intermediate run: 136 passed/16 skipped.
- Ruff check and format passed on 15 scoped Python files; compile checks and
  `git diff --check` passed. One formatting miss was corrected before this run.
- Import origin: this checkout's `src/oracle_composition/__init__.py`.
- Manifest test source compiled and its removed-API references were checked;
  the full manifest module was not collected because its package imports
  Gymnasium, absent from Astra's dev-only environment. Its simulator/runtime
  evidence remains deferred, not silently skipped or simulated by aliases.
- No candidate execution, reward subprocess, live OS canary, simulator,
  training, paid API call, push, or historical receipt regeneration.

## Remaining boundary

- R1 static receipt is not a dynamics, determinism, axiom, or scale certificate.
- Public reward execution still refuses while admission is unreviewed.
- R2 must supply reviewed worker-backed validation and real runtime tests.
- R3 must bind candidate, evaluator, calibration, and rollout provenance.
- Do not accept or integrate this diff until the independent review completes
  and its blocking findings are resolved. Do not launch real candidates merely
  because these focused tests pass.
