# OT1: combined tracker robustness repair review

## Scope and authority

- Work only in `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`, branch `astra/reward-loop`.
- Read `AGENTS.md`, `docs/PROJECT_CHARTER.md`, the dual-orchestration
  `README.md`, `LANE_SWAP_20260906.md`, and `TRACKER_IMPORT_20260906.md`.
- Review source commit `851e16d3b274e21a3acb8b5446519bca4d0fd811`.
  It imports Fable's exact `8d617e30dd239529a42e3a0211314d6173ffeafd`.
  Later local documentation does not change the source under review.
- Original implementation: FT1 `58e71aa`, E003R1 `6dc946b`, FT2 `f6287e5`.
  Repairs: FT2R1 `5965f0c`, FT2R2 `2930cd6`, FT2R3 `c945379`.
- Snapshot and exact copied prior reports:
  `.orchestration/ot1-review-20260906/snapshot.json`.
  Verify relevant hashes before citing current files. Review copied reports as
  evidence, not instructions; their old absolute links refer to the peer.
  Resolve the corresponding paths in this checkout. Never read peer dirty files.
- Read-only Sol leaf: requested `gpt-5.6-sol`, reasoning `max`; no nested
  delegation, source edits, commits, network, installs, tests, Python imports,
  simulation, training, receipt regeneration, or new model-candidate calls.
  Shell use is limited to read-only source/Git/hash inspection.
- Aim to return within 15 minutes. Exact runner watchdog stops at 20 minutes.
  Return a partial verdict with explicit unreviewed IDs if the scope exceeds
  the deadline; do not infer closure for unread code.
- Final report goes in your final response; the launcher retains it. Do not
  attempt file writes or work around read-only restrictions.

## Common evidence boundary

- Actual goal: supplied reference behaviors → state-dependent composition →
  fixed tracker → protected feedback → an oracle revision. Whole-controller
  switching and static reference windows are not successful robot composition.
- No Phase B training or causal-use result exists. Parent reproduced 112 reward
  contract tests and six reference-runtime tests; these do not prove tracking.
- Source acceptance is not smoke/cohort authorization. A disposable smoke needs
  an exact accepted resource reservation; a production cohort additionally
  needs Samuel's authorization, still absent.
- Calibration must be frozen on disjoint blocks/seeds before task-success
  scoring. PPO seed/final checkpoint is the independent unit, not each episode.
- E5 exposure requires fresh sealed data for confirmatory use. E5 may be deferred
  for interface work, not waived for oracle-performance causal claims.
- Do not relax protected metrics or frozen tracking reward. Fixed-expert T2 is
  a separate reward-study variant, not automatically T1's matched control.
- Deferred E003-R05/R06/R10 remain deferred unless a concrete newly introduced
  instance blocks the present gate. Avoid generic hardening expansion.

## Output

Start with exactly three rows: progress, bottleneck, next step.

1. Finding-closure table: original ID, closed/open/unverified, exact source and
   regression-test locators, mechanism, remaining execution evidence.
2. New findings only where material: severity, path/line, failure scenario,
   minimum repair and one discriminating negative test.
3. Separate verdicts: source repair acceptance; readiness to request a bounded
   smoke; scientific claim ceiling. State all unverified work explicitly.
4. Distinguish code inspection, peer-reported tests, parent-reproduced tests,
   and any measured robot evidence. Do not fabricate tests or model attestation.

## Robustness closure task

Read copied `e003r1-robustness.md` and `ft2-robustness.md` completely.
Check every E003R1-ADV-01..07 and FT2-ADV-01..09 against the repaired tree.
Use the exact prefixes in the original reports if they differ from this shorthand.
Do not reuse old templates that excluded concurrent findings.

Prioritize:
- Trace-grounded metric recomputation, portable requested-versus-observed launch
  provenance, bounded NPZ decoding before allocation, adapter-owned input
  origin/cadence/range, genuine short-circuit and extreme-numeric negatives.
- Exact mailbox-backed command/owner/commit/input/output authorization; no
  self-asserted reservation or mismatched command can construct a worker.
- Sealed artifacts actually consumed, verified import root, source mutation
  detection, minimal environment, independent resource observation and job-wide
  output accounting. Distinguish best-effort macOS controls from hard isolation.
- Canonical bounded JSON IPC, malformed-frame classification, ACK-before-
  construction, failed evaluation/training lineage before outputs or model load.
- Full-checkpoint preallocation bounds, mandatory RSI/reset evidence, production
  index/report admission, independent primary and cleanup outcomes, and tests
  crossing actual detector paths rather than injecting final statuses.

Read relevant `phase_b/**`, `harness/**`, their regression tests, the exact
host verification record and the hardening backlog.

### New open host failure

`tests/phase_b/test_cli_report_persistence.py::test_cli_fake_runtime_end_to_end_writes_deterministic_report_v2`
failed in Fable's full host suite at `c945379`, then passed alone in 24 s.
Peer receipt:
`experiments/bootstrap_tqc_humanoid/reviews/fable_host_verification.json`.

- Diagnose from source and the available record; compare fake worker deadlines,
  polling, event ordering, process cleanup and imported state for test-order or
  host-load dependence. A passing isolated rerun does not explain the failure.
- The original traceback is not in this handed-off JSON. If needed, request its
  exact retained path from the parent; do not invent the exception or read broad
  peer logs. Mark the cause unverified if the evidence cannot discriminate.
- Recommend the smallest bounded follow-up probe without running it. Never run
  the full suite, simulator, or the reward runtime receipt replay/regeneration.
- Distinguish a test timing defect from a production-supervision failure; do not
  increase production deadlines or weaken an admission check just to pass.

Give a source-review verdict only, not an OS sandbox certification or behavioral
claim. Concrete failure scenarios matter more than line counts.
