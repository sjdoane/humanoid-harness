# T1 smoke route: source audit and live admission

| status | current truth |
|---|---|
| progress | Independent source audit permits proposing the training-only smoke; parent exercised the real preflight. |
| bottleneck | Preflight refuses missing historical designer audit bundles. Three bound trace indexes and their payloads are also absent locally. Atomic peer resource exclusion is not yet established. |
| next step | Obtain the exact retained data from Fable, verify its existing bindings, rerun preflight, then request the concrete 20-minute slot. No training now. |

## Independent review

- Source: `2c2a32ba26c88ef253a47ce0b2823d846500c474`.
- Verdict: `READY_TO_PROPOSE_T1_SMOKE`, **not permission to launch**.
- Run: `.orchestration/sol-runs/20260906T115529Z-63dd62e7-a207-4d8a-a989-d5ec5940f097`.
- Terminal: `SUCCEEDED`, exit 0, 2026-09-06 12:07:29 UTC.
- Final SHA-256: `4c15af41f0d8f4182b8c8163594b0645d5db643057ec90d1706b7b4de0798a7c`.
- Watcher retained `terminal_observed`; stderr/stdout log is empty. The worker
  finished before the 12:15:29 deadline. Timeout termination was not exercised.
- Reviewer: requested Sol/max, read-only. No imports, tests or simulator ran.

| Reachable in this exact T1 smoke | Separate, still-open gates |
|---|---|
| Frozen oracle/reward/input admission; numeric reference path; PPO accounting; source identity; resource controls; checkpoint persistence | Protected utility evaluation and calibration; T2 reward candidate/pairing admission; cohort assembly and authorization |
| Clean exact commit, complete canonical command, fresh output, accepted reservation and exclusive heavy-job slot | Causal reference ablations, state-aware transition success and behavioral claims |

- Training writes `episodes=[]`, `calibration=None`; it does not call the
  protected evaluator. Do not repair that evaluator merely to propose this smoke.
- The review does not grant blanket FT2R1..R3 acceptance. It identifies the
  remaining live training/admission questions as appropriate smoke questions.
- The source's process scan is not atomic and does not exclude every heavy
  job. A peer-agreed single-owner token remains mandatory before launch.
- Existing runtime cap is **1,200 seconds**, seed `121901`, `196608` transitions,
  24 rollouts, no promotion. Preserve it rather than using the strategy's looser
  45-minute upper cap to extend the run.

## Parent live preflight: 2026-09-06, 12:32 UTC

- Clean checkout at `fba2f09`; source unchanged from the reviewed `2c2a32b`.
- Own import origin verified inside `humanoid-harness-astra/src/`.
- Called the real `validate_training_preflight` for existing T1 oracle and
  tracking-only reward. No supervisor, model/environment construction, simulator
  step, training or output-directory creation was requested.
- Observed refusal after 0.84 seconds:
  `EvidenceChainError: designer provenance is unverifiable without its raw audit bundle`.
- This is a retained-data admission failure, not a trained-policy failure.
  It occurs before a complete software/source/input preflight result exists.
  Do not describe the remaining preflight checks as passed.

## Exact local data handoff needed

Existing committed receipts are the authority; do not regenerate or weaken them.

| Evidence | Required retained files | Existing binding |
|---|---|---|
| Cycle 1 designer | Six named audit files under `.orchestration/sol-runs/20260905T183122Z-b0788053-4fd7-4887-af1d-8c2ef411c93b/`, plus `.orchestration/oracles/cycle_1_candidate.json` | `cycles/cycle_1/designer_provenance.json` |
| Cycle 2 designer | Six named audit files under `.orchestration/sol-runs/20260905T185111Z-78f29dc3-9fba-4f26-a393-b86bc71f4349/`, plus `.orchestration/oracles/cycle_2_candidate.json` | `cycles/cycle_2/designer_provenance.json` |
| Historical traces | `artifacts/experiments_003/cycle_{0,1,2}/content_index.json` and only the exact payloads referenced by those indexes | Each cycle's `scientific_receipt_v2.json`: 80, 20, 20 entries respectively |

- Six audit filenames: `events.jsonl`, `final.txt`, `launch.json`, `request.json`,
  `result.json`, `task-packet.md`. No worker restart or other run files needed.
- Receipt paths above are under `experiments/003_composition_speed_profile/`.
- Both designer bundles are needed: the preflight recursively verifies prior
  scientific receipts (`harness/evidence.py:1814`). Trace checks also default on.
- Index SHA-256 values: cycle 0 `041a81cac8d77d717aacb31e6e633fcae2a47e09457fc92eade79dc103e7f983`;
  cycle 1 `4d2c419140b71d1cc13f04c6af802d9fb569aaf690621cc17cd99ab5f4057097`;
  cycle 2 `a210e04509a1ac2266981c1ea0af1407ce5059d72d633d6ba62daf9970bdf775`.
- These raw records remain private local ignored artifacts. No Git publication,
  raw-log mailbox paste, changing peer file, missing-byte fabrication or receipt
  regeneration. Request an exact immutable handoff before copying.
- Verify regular files, safe bounded paths/sizes, exact hashes and source/copy
  equality. Inspect only these historical records, not the peer's current run.
- A future integrated-source import needs a delta review. The paired reward
  repairs remain Fable's work and do not need to be imported for this pinned T1.

## Next boundary

After retained data verifies, repeat preflight without changing sources. If it
fails, report the exact next admission cause; do not patch around the check.
Once it passes, bind that clean commit and returned input ledger into the full
reservation proposal. Obtain explicit peer agreement and the atomic resource
token before any heavy command. Fable previously offered to provide that token;
request its exact mechanism rather than creating a second competing lock.

Even a successful smoke would establish training mechanics only. It retains no
composition video or protected evaluation episodes and cannot establish causal
reference use, successful transitions or humanoid competence.
