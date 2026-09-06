# T1: shortest safe training-only smoke route

- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Branch: `astra/reward-loop`; source commit `2c2a32b` (resolve full ID first).
- Read `AGENTS.md` and `docs/operations/dual-orchestration/README.md`.
- Read-only Sol/max leaf; no nested delegation, edits, imports, tests, installs,
  network, simulator, training, candidate calls, or receipt regeneration.
- Inspect pinned Git objects and exact retained prior review files only. Never
  read Fable's changing tree. Do not infer acceptance from a worker exit.
- Target 8 minutes and at most 900 words. Stop reading at 10 minutes and return
  a partial answer if necessary. The exact runner has a 20-minute watchdog.

## One decision

Which existing defects actually block **requesting** the frozen T1 training-only
smoke, as distinct from protected evaluation or a scientific performance claim?

We are not asking for a fresh repository-wide review or a new implementation.
Two earlier broad reviews timed out. Avoid repeating that scope. Narrowing an
execution path never closes an unexamined finding or grants launch authority.

The user wants real reference composition evidence, not endless infrastructure.
Our agreed route first tests numeric-reference training mechanics with a single
non-promotable smoke; then considers a state-aware joint task if feasible.
Fable is repairing T2 paired reward-study admission in its own checkout. Do not
duplicate that review or make unrelated T2 candidate admission a T1 prerequisite.

## Trace this exact route

`harness.cycle_cli train --experiment experiments/003_composition_speed_profile
--cycle 3 --oracle experiments/003_composition_speed_profile/phase_b/oracle_cycle_1_reference_v1.json
--reward experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json
--seeds 121901 --transitions 196608 --smoke`

This is an **incomplete command sketch**, not executable authorization: fresh
output and exact accepted reservation are deliberately omitted.

- `harness/cycle_cli.py::_train_command` calls preflight and training supervision,
  then writes a report with no evaluation episodes and no calibration.
- `phase_b/supervision.py::validate_reservation` requires 1,200 seconds for
  smoke. This existing 20-minute hard limit is stricter than the later strategy's
  45-minute upper cap. Do not change either runtime budget or frozen artifacts.
- Source/data/software admission, peer resource agreement, atomic reservation,
  clean exact sources and no conflicting heavy job remain mandatory.
- No five-seed cohort, checkpoint promotion, task-success score, evaluator
  invocation, causal reference-use claim or behavioral success is authorized.

## Evidence to inspect, in priority order

1. The training entrypoint and direct real runtime/supervisor/report call chain:
   `src/oracle_composition/harness/cycle_cli.py`,
   `src/oracle_composition/phase_b/{supervision,training,runtime,reference_runtime,persistence,report_v2}.py`.
   Read only relevant portions. Follow helpers where a concrete gate depends on them.
2. `experiments/003_composition_speed_profile/phase_b/DESIGN.md` (commands,
   resource policy, FT2R1..R3 repairs). Treat prose/test counts as reported,
   not independent source closure.
3. Prior originals `.orchestration/ot1-review-20260906/ft2-scientific.md` and
   `ft2-robustness.md`: read their finding lists, then classify only IDs reachable
   on this training-only path. Do not audit protected evaluation today.
4. `OT1_NARROW_RESULTS.md`, `OT1E01_RESULT.md`, `T2PAIR_REVIEW_RESULT.md` in
   this packet's directory. Note the earlier OT1E-01 ledger is superseded by its
   explicit acceptance receipt. Calibration origin, evaluator resource/cleanup/
   read bounds remain open for evaluation; do not silently waive them.
5. Relevant focused tests only as static corroboration. No execution.

## Output

Start with progress, bottleneck, next step rows. Then:

- At most six rows: gate, actual caller/source locator, reachable in this smoke
  yes/no/uncertain, existing evidence, smallest missing evidence.
- Explicitly assess whether evaluator gaps and T2 pairing/candidate gates are
  on this exact T1 path; avoid claims about future merged bytes.
- Name concrete remaining training safety/admission defects, if supported. If
  the earlier training repair acceptance is incomplete, say precisely which
  reachable checks remain unverified, rather than claiming blanket readiness.
- Explain what retained training facts could demonstrate about actual reference
  consumption, and what still needs a later causal ablation or physical trace.
- End with `READY_TO_PROPOSE_T1_SMOKE`, `REPAIRS_BEFORE_T1_PROPOSAL`, or
  `INSUFFICIENT_SCOPE_FOR_T1_VERDICT`; none is execution authorization.

No generic hardening, new controller, new evaluator, changed thresholds, new
experimental family or automatic retries. Return the useful bounded answer.
