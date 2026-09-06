# OT1C: three scientific closure questions only

- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`; branch `astra/reward-loop`.
- Source: `851e16d3b274e21a3acb8b5446519bca4d0fd811`; immutable bindings in
  `.orchestration/ot1-review-20260906/snapshot.json`. Later commits are docs only.
- Read AGENTS.md and the dual-orchestration README. This is an explicitly bounded
  leaf review, not a mandate to audit the repository. No nested agents.
- Read-only Sol/max. No file writes, tests, imports, installs, network, simulator,
  training, resource reservation or model candidate. Inspect source/Git only.
- Two prior broad OT1 reviews timed out with no verdict. Do not repeat them.
  Their partial comments are provisional and can be wrong; one apparent
  early-fall denominator issue was explicitly withdrawn after checking the ABI.
- Target an 8-minute report, at most 900 words. At 10 minutes stop reading and
  return findings/unverified items. Exact watchdog terminates at 20 minutes.
  Deliver your final even if incomplete; do not wait for perfect coverage.
- Start with progress/bottleneck/next step rows, then only the questions below.
  For each: confirmed/refuted/unverified, producer/consumer path and line,
  one minimum negative test. State a scoped verdict, never blanket OT1 closure.
- Findings are static inspection, not reproduced defects or behavioral evidence.
  Smoke/cohort gates remain unchanged. No code or reviewer can grant authority.

## Questions

1. Calibration provenance: `phase_b/calibration.py` accepts sample dictionaries
   with split IDs and checkpoint hash strings. Can a caller fabricate all rows,
   recompute their hashes and publish a scoring-admitted receipt without any
   verified checkpoint/rollout trace? Is another admission boundary proving
   origin before `load_calibration_receipt` is used? Follow only this call chain.
2. Settled-state semantics: does the computed
   `settled_state_normalized_error_band` actually affect task-success scoring
   on the protected evaluator/report path? Trace the value through
   `calibration.py`, `protected_metrics.py`, `report_v2.py`, `evaluation.py`
   and their focused tests. Distinguish fixed tracking gates from the separately
   calibrated task-success endpoint. Missing a calibration receipt must remain
   non-scoring.
3. Cohort assembly: `harness/cycle_cli.py` evaluates one checkpoint and passes
   `seeds=[seed]` to `build_scientific_receipt`. Is there an existing consumer
   that creates the five-seed family report without manually synthesizing facts?
   Inspect relevant CLI/report/export functions and tests, not the entire repo.
   If absent, classify the operational gap separately from correctness of a
   single-seed diagnostic or readiness of a non-promotable smoke.

Read `experiments/003_composition_speed_profile/phase_b/DESIGN.md` only for
these declared endpoints. Nearby prior IDs: FT1-SCI-04 and FT2-SCI-03/05.
Do not declare the other original findings closed. Recommend the smallest repair
or retained gate needed, without designing a new generalized framework.
