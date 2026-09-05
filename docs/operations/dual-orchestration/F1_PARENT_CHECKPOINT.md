# F1 and T2 input review checkpoint

| status | current truth |
|---|---|
| progress | F1 builder finished; parent reproduced 203 passed/16 deferred. A separate T2 input contract passes 17 pure tests. M1 integration is independently accepted. |
| bottleneck | F1/T2 source is unreviewed. Formula v1 still accepts only B0 inputs; no compositor, actual-model ingestion, or training integration exists. |
| next step | Independent source/test review; bind T2 only after the peer accepts its exact interface and the review passes. |

- HEAD: `402ff4f3ff2a4d3c0f93f8052e9e691ebde9ae3f`.
- F1 run: `20260905T190056Z-c31db1f8-b68f-4757-b676-daace9c7e240`;
  succeeded 19:20:13Z, lease released, watchdog observed terminal.
- Seven new builder files only; existing tracked source/test diff is empty.
- Parent read all three F1 modules and reproduced the combined focused suite:
  `203 passed, 16 skipped in 2.52s`. Skips are the old R2 host/runtime cases.
- Builder's own seven-file scope and counts remain in `F1_RESULT.md`; this
  checkpoint adds two separate parent-written V2 files, without editing F1.
- V2: `rewards/task_inputs_v2.py` and its test define COM speed, fixed 3.0 m/s,
  0.015-second cadence metadata, exact builtin numeric types, serialization,
  and consumption-time validation. `17 passed in 0.03s`.
- V2 is not wired to the formula or simulator. Source metadata does not prove
  that the stock measurement was actually captured. The adapter must do so.
- The first V2 lint check found an unescaped test regex; the test was corrected
  before the review snapshot. No production behavior change was needed.
- Peer T2 answer: `20260905T190920.097946Z-c27e34fd06fd4488a8ea87503581d300`.
  Exact-path proposal: `20260905T193614.811073Z-cc47aa7840274926956386766cdb5c05`.
- Proposed T2: 1,000 steps, COM speed target 3.0, separate MAE/falls, evaluation
  seeds 97001-97020. These are reused development seeds, not sealed test data.
  Baseline reward, training seeds/budget/checkpoint rule and resources remain
  to be frozen before any run.

## M1 accepted and handed over

- Exact merge: `8a48f322a4e3e14bb4467daddd709de1926d0e1d`.
- Reviewer: `20260905T190055Z-74defa80-2b0b-47a1-93d1-3b190ecf9fb7`;
  `ACCEPT_INTEGRATION`, no actionable merge-induced findings, 19:06:56Z.
- Fable handoff: `20260905T193614.771225Z-c507e2a15ab44cd2866dc113ee62369f`.
  Fable owns promotion. This acceptance excludes the uncommitted formula files.
