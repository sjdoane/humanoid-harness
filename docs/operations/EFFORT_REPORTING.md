# Recorded effort, not inferred efficiency

```bash
humanoid-harness --json effort --run /absolute/path/course_run_manifest.json SHA256
```

- Repeat `--run MANIFEST SHA256` to include additional completed GMT attempts.
- Read-only: no simulation, policy loading, source modification or provider call.
  The command does not import simulator or training libraries.
- Uses the expected receipt SHA-256 and existing bounded JSON reader. Repeated
  receipt bytes or file identities refuse; missing/malformed inputs refuse.
- The hash identifies bytes, not proof that training happened. This report
  extracts receipt-reported values; it does not replace native run admission,
  output verification or an independent behavioral evaluator.

| Quantity | Meaning |
|---|---|
| Reported training transitions | Sum from selected completed manifests; `training: null` plus `training_performed: false` contributes zero reported training |
| Summed run wall seconds | Worker durations including training, evaluation and artifact work; not campaign elapsed time or supervisor wall time |
| Complete search cost | Unknown: selecting receipts does not prove that every attempt was included |
| Total simulator steps | Unknown: training transitions do not include all evaluation steps |
| Revision rounds / human effort / model cost | Unknown: not present in these manifests |
| Steps to target success | Unknown: no independent success curve is inferred from training return or completion |

- Completed runs with failed task outcomes are included, not filtered out.
- Failed/incomplete native jobs without a completed course manifest are not
  supported inputs yet. Their costs must not be treated as zero or hidden in a
  claimed complete manual-versus-harness comparison.
- Different seeds/tasks may be included for accounting; their outcomes are not
  pooled. This tool does not choose a winner or claim fair experimental matching.
- Future full comparisons need separately receipted human activity, model usage,
  failed-job costs, revision ancestry and held-out task outcomes.

## Verification — 2026-09-08

| Check | Result |
|---|---|
| New feature tests | 33 passed, including fresh-process refusal of training/simulator imports |
| Feature plus project CLI/status | 45 passed |
| Independent Sol review | Context and final implementation accepted; reviewer reran all 33 feature tests |
| Broad suite, excluding slow test and historical receipt replay | 2,723 passed; 89 failed; 40 skipped; 2 deselected; 0 setup errors; 287.61 seconds |
| Existing real Humanoid runtime checks | All 4 passed within the broad suite; interface/instrumentation evidence only |
| Repository Ruff / changed-file format / compilation / diff check | Passed |

- The broad suite is **not green**. Failure groups: migration check (1), absent
  native Farama actor (3), F3's historical Fable-checkout requirement (77), T2
  historical artifact binding (3), and T2 execution-lineage checks (5).
  No evaluator, source seal, import-origin guard or legacy receipt was changed.
- Command used: `pytest tests -q -m 'not slow' --deselect
  tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`.
- The final broad run held source unchanged. An earlier run overlapped the
  module relocation and produced source-snapshot setup errors; it is retained
  only as a diagnostic, not final-tree evidence. Local reports are ignored:
  `.orchestration/effort-validation-final-20260908.xml` and
  `.orchestration/effort-validation-20260908.xml` respectively.
- Real read-only check: retained [Study 018 baseline](../../experiments/018_g1_four_state_reward_loop/RESULTS.md),
  manifest `33b5e25ea8dfa7a8e3a4f520e7386edce067b920d75201e8ffb81581bd96f705`,
  reports **131,072 training transitions** and **270.24257166701136 run seconds**.
  Unmeasured fields remain null. This reads an existing run; it is not new training.
