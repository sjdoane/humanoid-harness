# OT1E-01: bind evaluator admission to the manifest

## Boundary

- One write Sol/max leaf; no nested agents. Read AGENTS.md and the dual-lane
  README first. Do not investigate the whole repository.
- Work only in `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`,
  branch `astra/reward-loop`. Code base: `3406bb5f0af3c2479d47cdd2d245c3e738a168fe`;
  your launch HEAD adds documentation only. Record exact starting HEAD and
  clean/dirty state, checkout realpath, branch and final changed paths.
- Lock SHA-256: `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40`.
- Use the existing worker lease; do not release or renew it yourself. The
  launcher owns renewal/cleanup. No peer checkout edits or dirty-file reads.
- Aim to return within 12 minutes. At 15 minutes stop expanding scope, retain
  the best tested slice and report gaps. Exact watchdog deadline: 20 minutes.
- No commit, push, install, network, candidate call, simulator, real/fake
  training, full suite, calibration generation or source-receipt regeneration.
  Parent commits after inspection; independent review remains required.

## Allowed changed files

| path | allowed change |
|---|---|
| `src/oracle_composition/phase_b/evaluation_supervision.py` | Manifest validation, frozen request binding, parent/child admission |
| `src/oracle_composition/harness/cycle_cli.py` | Only evaluation request/manifest byte wiring |
| `tests/phase_b/test_evaluation_supervision.py` | Update its manifest/request fixture to the actual bound schema; do not execute this file |
| `tests/phase_b/test_evaluation_admission.py` | New pure admission and boundary regression tests |
| `docs/operations/dual-orchestration/OT1E01_RESULT.md` | Concise implementation, checks and remaining limits |

No other source/doc edits. In particular, leave `training.py`, `runtime.py`,
`contracts.py`, `reward_study/`, pairing tests, scoring and calibration alone.
If another file is essential, return the exact reason instead of expanding.
Local development is authorized; peer proposal
`20260906T065456.009901Z-ffc472497475469ab2d1291ab13d3226` does not yet authorize
shared CLI integration into main.

## Defect and minimal repair

- OT1E review confirmed `_worker` checks source identity but ignores
  `evaluation_manifest_sha256` in the parent's admission message. The manifest
  must identify the evaluation the child actually admits, not just its receipt.
- Preserve current evaluator-source realpath/hash verification. The separate
  claim that it ignores imported source location was refuted; do not fix it.
- Add a bounded immutable manifest-byte/digest binding to the frozen request.
  Parent verifies the actual `PublishedArtifact` bytes/count/digest against the
  request before `_spawn`; child recomputes canonical-byte identity and verifies
  the admission digest before `evaluate_policy_checkpoint` or any constructor.
- Prefer existing `read_verified_artifact_bytes` for any new file read: it
  checks real bounded bytes and refuses unsafe paths. Never use an unbounded
  `read_bytes` for the new boundary, or trust artifact metadata alone.
- The child must bind the request's existing runtime selectors to the manifest:
  checkpoint SHA, starting-actor SHA, evaluator-source identity, target speeds,
  and the existing fixed cells/seeds/160-episode schedule. Retain current schema
  semantics; do not create a new experiment, metric or calibration policy.
- Have the actual child start/admit exchange carry and check the computed
  manifest identity on both sides. Missing, wrong, noncanonical, oversized,
  altered or selector-inconsistent manifests fail before checkpoint/episode
  construction. Merely checking equality of two unchecked digest strings is
  not enough.
- Source of actual schema: `_evaluate_policy` in `harness/cycle_cli.py` around
  lines 639-680 and `EvaluationLineage.manifest_record` in
  `phase_b/evaluation_lineage.py`. Existing test `_run` currently publishes only
  `{schema_version: 1}`; update that fixture, not the production validator.
- Parent preflight rejection may remain an explicit `ExperimentContractError`;
  child admission failure must follow the existing failure-message/terminal
  accounting path and never produce success evidence. Do not redesign cleanup
  or status schemas in this slice.
- This repair does not establish calibration provenance, resource isolation,
  cleanup completeness, bounded result-artifact reads or robot competence.
  Existing source-bound receipts may become stale; report, do not regenerate.

## Required checks

- Use real canonical fixtures and the actual admission functions/worker path.
  Lightweight injected IPC and constructor sentinels are appropriate. Do not
  mock away validation or assert prewritten final statuses as evidence.
- Positive: matching manifest/request/source/admit crosses the admission gate.
- Negatives: missing and wrong admission digest; changed manifest bytes or
  recorded count; noncanonical JSON; explicit size bound; selector mismatch
  (checkpoint, starting actor, targets, source and fixed schedule).
- Count entry into the checkpoint/episode function on actual worker negatives:
  it stays zero. Call `_worker` synchronously with a safe injected channel and
  stub only process-identity operations as needed; do not spawn a child or call
  `os.setsid` on the test runner. Preserve actual manifest/source validation.
- Parent mismatch must refuse before `_spawn`; test that with a spawn sentinel.
- Only run the new pure test file. Existing supervision tests invoke a fake
  trainer and are not authorized here. Use:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider --assert=plain tests/phase_b/test_evaluation_admission.py
```

- Confirm `oracle_composition.__file__` resolves inside this checkout. Run Ruff
  check/format on changed Python files only and `git diff --check`. Use
  `apply_patch` for local edits. Preserve unrelated changes if present.
- Retain meaningful red/green output in the tool transcript; do not write fake
  receipts. Final report: exactly progress/bottleneck/next step rows, changed
  paths, test counts, missing checks and scoped readiness verdict. Keep it brief.
