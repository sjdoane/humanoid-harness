# OT1 narrow review results and settled-state repair

| status | current truth |
|---|---|
| progress | OT1C and OT1E returned scoped verdicts. Parent reproduced the missing settling check and repaired both task-success consumers; 23 focused tests pass. |
| bottleneck | Calibration origins, evaluator manifest/resources/cleanup and five-seed report assembly remain open. The small scoring repair still needs independent review. |
| next step | Review OT1C-02 only, then select one remaining bounded repair. No protected evaluation, training, reward call or shared pairing edit authorized. |

## Exact review receipts

| review | run under `.orchestration/sol-runs/` | terminal |
|---|---|---|
| OT1C | `20260906T050231Z-b95a7a98-2b7e-48db-a8c0-7f4f7078448b` | SUCCEEDED, 05:08:22 UTC; scoped non-acceptance |
| OT1E | `20260906T050232Z-a86f1638-2681-4f63-a768-eda2ceb342dc` | SUCCEEDED, 05:08:33 UTC; protected evaluation not accepted |

- Both watchers observed terminal completion before their deadlines.
- Final reports are `final.txt` in the exact run directories. No reviewer tests
  or robot runtime executed. Requested Sol/max is not served-model attestation.
- The earlier two broad OT1 reviews remain incomplete. These scoped verdicts
  do not close unrelated original findings.

## Finding ledger

| ID | finding | current disposition |
|---|---|---|
| OT1C-01 | Self-consistent calibration rows lack verified checkpoint/rollout origin | Open; bind real protected traces and checkpoints before scoring admission |
| OT1C-02 | Calibrated settled-state band omitted by evaluator and report scoring | Parent repaired and regression-tested; independent closure pending |
| OT1C-03 | No artifact-driven assembler for five evaluated policy seeds | Open workflow gap; not a training-only smoke defect |
| OT1E-01 | Child ignores evaluation-manifest digest in admission | Open; must fail before episode construction on mismatch |
| OT1E-02 | Evaluation does not apply declared environment/resource controls | Open; training controls cannot be claimed as evaluation controls |
| OT1E-03 | Cleanup can overwrite primary failures and miss surviving descendants | Open; preserve both outcomes and suppress success on cleanup failure |
| OT1E-04 | Evaluation artifact loader bounds reported size, not actual read size | Open; bound actual file bytes before materialization |
| Refuted concern | Direct imported `.evaluation` cross-checkout identity is unchecked | Imported modules are resolved and checked; keep bounded-attestation limitation explicit |

The calibration data and evaluator are protected, not generated-candidate
authority. Do not replace these gaps with an agent-written success receipt.

## OT1C-02: minimal implementation

- Base: `7f2298f9d0b741ffecce3803015a34f77472cd96`.
- Source changes: `phase_b/evaluation.py::_score_task_success` and
  `phase_b/report_v2.py::_derived_task_success`.
- Both require a present measurement at or below the calibrated band.
- No new threshold; no changed tracking scale, safety gate, latency rule,
  reference, reward, seed, trainer or runtime reservation.
- Separate evaluator/report calculations retained; report recomputation does
  not trust the cached episode `task_success` flag.
- Missing calibration stays non-scoring; hold cells remain diagnostic.

| verification | observed result |
|---|---|
| Own import origin | `humanoid-harness-astra/src/oracle_composition/__init__.py` |
| Pre-fix regression, after correcting a test-only field-name typo | 8 failed, 8 passed, 11 deselected; 1.13 s |
| Post-fix selected scoring/utility tests | 23 passed, 4 deselected; 1.23 s |
| Ruff and formatting, three changed Python files | Pass |
| Diff whitespace check | Pass |

- Retained red/green output: `.orchestration/ot1-s2-20260906/`.
- Tests cover zero/exact/next-float-over/missing measurement, changing the band,
  no calibration, hold cells, preserved fixed utility checks, and serialized
  episode rows whose stale success flags must not control endpoint counts.
- These are pure metric/endpoint tests, not a full report-file reload or a
  simulator trace. The report reload calls the same endpoint by inspection.
- No full suite, fake-training CLI test, simulator, calibration generation,
  source-receipt regeneration, cohort, candidate call or publication ran.
