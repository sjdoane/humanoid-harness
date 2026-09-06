# OT1 narrow review results and settled-state repair

| status | current truth |
|---|---|
| progress | OT1C-02 independently accepted at `3406bb5`; both scorers enforce the calibrated settling band. Parent recheck: 23 passed. |
| bottleneck | Calibration origins, evaluator manifest/resources/cleanup and five-seed report assembly remain open. No tracker effectiveness result. |
| next step | Bind evaluator manifest admission in the isolated OT1E-01 slice. No protected evaluation, training, reward call or shared pairing edit authorized. |

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
| OT1C-02 | Calibrated settled-state band omitted by evaluator and report scoring | Closed for scoring only at `3406bb5`; independent verdict below |
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

## OT1C-02 acceptance: 2026-09-06

- Exact repair: `3406bb5f0af3c2479d47cdd2d245c3e738a168fe`.
- Independent run: `.orchestration/sol-runs/20260906T054257Z-4bf059e0-eac3-4194-97a2-9159443620d8/`.
- Terminal: `SUCCEEDED`, exit 0, 05:46:56 UTC; watcher `terminal_observed`.
- Verdict: `ACCEPT_OT1C02_SCORING_ONLY`; no new blocking finding in that scope.
- `final.txt` SHA-256: `dd32542c605d591ebbdce3b123c4a668cd8bbf44f34bfba00d518b1f3fba6b23`.
- Reviewer inspected retained red/green evidence and the real serialized-row
  endpoint test; did not execute tests. The reload-to-endpoint source chain
  supports this repair without a second full reload test.
- Later parent recheck: 23 passed, 4 deselected in 1.79 s. All 13 review
  bindings reverified before this documentation update; the old document hash
  remains a historical review input, not the hash of this updated ledger.
- Exact code handoff to Fable: `20260906T062154.426532Z-c4a7970b7dca478a9020d244aa5fb85a`.
- No acceptance of calibration origin, evaluator resources/manifest/cleanup,
  cohort assembly, robot competence, smoke or training authority.

## Next bounded repair

- Packet: `TASK-OT1E01-manifest-admission.md`; code base `3406bb5`.
- Peer scope proposal: `20260906T065456.009901Z-ffc472497475469ab2d1291ab13d3226`.
- Isolated development only; shared CLI integration needs peer agreement and
  independent review of the finished commit. Fable's pairing paths stay untouched.
