# F2: smallest T2 formula-to-runtime handoff

- Mode: read-only Sol/max leaf; no subagents, edits, model calls, simulator,
  training, dependency installation, full suite or shell writes.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra` only.
- Work from this packet's committed revision. Record its exact HEAD.
- Deadline: 20 minutes from launch; return a concise plan, not code.

## Concrete question

What is the smallest reviewed change that lets an LLM's data-only target-speed
recipe enter Fable's Phase B training reward, using the already accepted
`CandidateTaskInputsV2`, without weakening the old Python runtime or relabeling
F1's supplied-byte inputs as authenticated model outputs?

## Read

1. Root `AGENTS.md`, then `docs/operations/dual-orchestration/README.md`,
   `F1_ACCEPTANCE.md`, `T2_INPUT_ACCEPTANCE.md`, and `F1_REVIEW_FINDINGS.md`
   (the last three share that documentation directory).
2. Under `src/oracle_composition/`: accepted `rewards/task_inputs_v2.py`,
   `rewards/target_speed_formula.py`, `reward_search/formula_contracts.py`,
   and `reward_search/formula_loop.py`; their focused tests under `tests/`;
   `experiments/family_b_target_speed_formula_v1/config.json`.
3. **Committed Git objects only** from peer commit
   `8d91a811680599631779892a2c2bedf2127a3a74`:
   `src/oracle_composition/phase_b/{contracts,reward,receipts}.py` and relevant
   focused tests. Use `git show` in Astra, never read peer dirty files.
4. Phase B decisions in committed
   `f57acc3:docs/strategy/RESEARCH_STRATEGY.md`, lines around 289-305.

## Fixed decisions

- T2 input source is the accepted `rewards/task_inputs_v2.py` from `3df7e0b`:
  exact finite builtin numbers, COM x velocity, target exactly 3.0 m/s.
  Cadence is adapter metadata 0.015 s, not another candidate-visible input.
- Fable accepted this interface in mailbox message
  `20260905T202243.309823Z-181767ca33654e69aad5743d92537aeb`.
- The temporary duplicate class still visible in committed `phase_b/contracts.py`
  is superseded. Fable owns removing it; do not propose adopting that duplicate.
- Formula recipe stays bounded, data-only. Current math is triangular
  target-speed shaping with alpha [0.25,4], beta [-10,10], output [-10,15].
  A new target family must not silently widen or overwrite old B0/F1 contracts.
- Training reward: frozen tracking reward plus task term; stock reward is
  telemetry only. Baseline task term is positive zero. Task metrics remain
  independently measured COM speed error and falls, not generated reward.
- F1 receipt and actual model provenance are separate. Identify the minimal
  remaining live-model adapter, but do not run or implement a call.
- No new evaluator, tracking reward, simulator, trainer, seeds, training
  protocol or evidence claim. Shared registry/compositor edits need exact peer
  agreement. A plan or accepted input module is not integration approval.

## Return

- Progress / bottleneck / next step table.
- A small interface table: input, output, source identity, consumer, owner.
- Exact proposed files and signatures for the next bounded builder; separate
  Astra-only implementation from Fable-owned integration and peer decisions.
- Explicit reuse-versus-new-version choice. Avoid duplicating F1's large loop
  or introducing a general framework to support one formula.
- At most six acceptance checks: true positive, wrong target/class/identity,
  reward stream separation, preserved old behavior, provenance boundary.
- Order the slices by which most directly enables one honest closed loop.
  Name what still prevents a real trained cycle. No behavioral claims.
