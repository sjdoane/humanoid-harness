# F3: minimal T2-aware model-call protocol

- Read-only Sol/max planner; 20-minute launch deadline; no delegation, writes,
  model candidate calls, network research, simulator, training or full suite.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Record the exact committed HEAD. This is a code/protocol plan, not an
  experiment redesign or authority to dispatch the proposed candidate call.

## Outcome

Specify the smallest implementation that can prepare one honest T2-aware
reward-parameter request, retain an actual detached Sol response, and admit or
reject its format with exact provenance. It must lead directly toward one
real reward-loop iteration after Fable's runtime is ready, not another general
framework or a duplicate of F1's approximately 1,000-line loop.

## Read

- Root `AGENTS.md`, `docs/operations/dual-orchestration/README.md`.
- In that documentation directory: `F2_ACCEPTANCE.md`, `F2_PLAN_RESULT.md`,
  `A2R_RESULT.md` and `F1_REVIEW_FINDINGS.md`.
- `src/oracle_composition/reward_search/{contracts,loop,formula_contracts,formula_loop,publication}.py`;
  only source/test portions needed for packet, retained bytes and run envelopes.
- Accepted `src/oracle_composition/rewards/{target_speed_formula,target_speed_formula_t2,task_inputs_v2}.py`.
- Existing launcher `scripts/start-sol-worker` and the concrete request/result
  schema classes consumed by A1. No launcher/hook/auth changes are proposed.
- Committed peer objects only from
  `a51ea9e68aa9e27a573e372f67e36d6fa4d494cd`: relevant phase-B contract,
  report and reward artifact definitions. Use `git show`/`git ls-tree` in Astra;
  never read another checkout's dirty files.

## Non-negotiable semantics

- Accepted F2 evaluator commit: `9bbb6587c1f9d955e684ffe4e1772e72602e3556`.
- Task: COM target 3.0 m/s. Candidate author surface: alpha/beta only, exact
  existing recipe parser and F2 runtime IDs/bounds. This is parameterized
  shaping, not arbitrary reward-program generation or established improvement.
- F1's old-target packet and absent-compositor bindings are not T2-aware. Do
  not wrap a T2 call around that dossier and relabel its receipt afterward.
- **Baseline is tracking_only/v1, with task reward +0.0.** It is not the
  alpha=1,beta=0 recipe. Alpha=0 is outside the admitted recipe family. Bind
  the true baseline reward artifact for an initial request; a later revision
  can bind an actual admitted recipe. Do not manufacture a parent recipe.
- The requested packet must distinguish known immutable formula/parser/input
  bytes from missing or not-yet-reviewed compositor/admission/runtime/evaluator
  evidence. Missing evidence stays explicit. No invented rollout measurements.
- Fable owns the inclusive [-25,25] COM admission gate, measurement origin,
  cadence 0.015 s, registry and training. Accepted T2 core bytes stay unchanged.
- Existing A1/A2R Python-source receipts and F1 supplied-byte receipts remain
  historical and unchanged. A new call needs its own source-class-specific
  receipt, including requested configuration, never served-model attestation.
- Protect against the known F1 failures: raw rejected bytes must survive;
  the model must see every identity it must echo (no self-referential prompt
  digest); forged/partial/mutated typed records must fail before callbacks.
- Model artifacts are data only. No dynamically executing emitted Python,
  candidate changes to evaluator/MDP/controller, private transcript copying,
  publication, API spending, training authorization or claim inflation.

## Return a compact implementation packet proposal

1. Exact files/functions to add or reuse; separate the smallest initial-call
   slice from later feedback-linked revision. Avoid duplicating existing
   bounded I/O/publication or silently widening old schema contracts.
2. A small table of packet/response/receipt fields and exact byte/hash bindings,
   including the true baseline, T2 contract, independent evidence and any
   explicit missing-admission state. Identify the actual artifact dependency
   Fable must hand over before any training-ready proposal can be admitted.
3. How to use the existing detached launcher for one initial canary, terminal
   failure retention and a hard stop. No retry or revision is implied by this
   planning packet. State what the canary would and would not prove.
4. At most six focused acceptance checks, including altered packet/envelope,
   failed/missing response, incorrect baseline/target, malformed parameters,
   callback/forged-record cases and preserved old receipt classes.
5. Distinguish local metadata consistency from authentication/attestation.
   Highlight any authority or unresolved peer dependency instead of inventing
   a placeholder that could later be mistaken for an observed runtime fact.

No source changes or tests are required for this plan. End with progress,
bottleneck and the next concrete bounded builder step; no behavioral claims.
