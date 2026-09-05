# R2: propose the smallest worker-runtime repair protocol

- One read-only Sol/max planner; no delegation, edits, model calls, or peer messages.
- Worktree: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Baseline: `754e869c618dbe4f1b43c4f1a513c023ec139306` (accepted static-only R1).
- At most 20 minutes; final plan at most 800 words. No implementation or execution approval.

## Read

- `AGENTS.md`, `docs/operations/dual-orchestration/README.md`.
- `docs/operations/dual-orchestration/R1_ACCEPTANCE.md`, `A3_RESULT.md`, and
  `REWARD_CAPTURE_PATH_CHECK.md` in that same docs directory.
- `src/oracle_composition/rewards/sandbox.py` and `_sandbox_worker.py`;
  `tests/rewards/test_sandbox.py` and `test_scale_calibration.py`.
- Original detailed B0 reviews, read-only if needed:
  `/Users/samueldoane/Documents/ChatGPT/humanoid-harness/.orchestration/sol-runs/20260904T221620Z-cd03d6f4-3479-4c26-9410-ecd4c2d57dcd/final.txt`
  and `/Users/samueldoane/Documents/ChatGPT/humanoid-harness/.orchestration/sol-runs/20260904T221621Z-d07a5b95-5955-462f-834e-61f108bda886/final.txt`.
  Report missing evidence, do not invent finding details.

## Deliver a decision-ready protocol

- Map R2 findings SCI-03 and ROB-03/04/05/06/07/09 to actual entry paths,
  the minimum changed files, and substantive negative tests.
- Separate fixture-only runtime testing from admitting generated reward code.
  Public execution remains refused until the relevant gates are independently
  reviewed; R3 candidate/calibration/episode identity remains a later gate.
- Specify trusted fixed fixtures, exact resource/deadline bounds, cleanup,
  non-vacuous allowed-versus-denied controls, load-time source/runtime identity,
  categorical failure receipts, and complete public-call deadlines.
- Use the real production worker lifecycle, not just decoder/self-canary tests.
  No legacy receipt may authorize a new source/runtime. Do not relax OS guards
  or substitute missing dependencies to obtain a pass.
- Propose at most three small implementation/test slices, ordered to enable
  the earliest meaningful numerical worker test without weakening admission.
  Identify which tests are pure or fully mocked and which need the separately
  reviewed host-fixture run protocol and peer resource agreement.
- State the smallest evidence that would unblock the next reward-feedback
  iteration, and what remains insufficient to claim robot improvement.

## Stop boundary

No tests, subprocess reward workers, candidate compile/exec, OS/network probes,
simulator, training, installs, or broad literature survey. This is a source-
grounded protocol proposal only. Do not edit shared scientific contracts or
Fable's checkout. After reading and reasoning, return progress/bottleneck/next
step and the compact protocol; do not use the full budget for unrelated audits.
