# F1: data-only target-speed formula and proposal core

- One Sol/max writer, no delegation or model calls. Work only in the Astra
  checkout `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Source baseline: merge `8a48f322a4e3e14bb4467daddd709de1926d0e1d`; only parent
  packet documentation may follow it. Verify realpath, branch, HEAD, dirty
  paths, own import origin, and `uv.lock` hash before editing.
- Maximum 20 minutes. At most one writer. Do not edit main or touch its lease.
- Read `AGENTS.md`, dual README, `MAIN_INTEGRATION_20260905.md`, `R1_ACCEPTANCE.md`,
  `A1_RESULT.md`, existing reward `contract.py`, and reward-search contracts,
  loop, publication helper and tests. Existing APIs/receipts stay unchanged.

## Why this slice

Independent route review chose this bounded family; Fable accepted the JSON
route and exploratory claim ceiling through peer proposal
`20260905T183421.793603Z-8b619f3d56a84cbb81d31d39e11c36ef`, accepted by Astra
`20260905T185408.005663Z-ff80151886fc485782897509ffe98696`.
This is reward parameterization, not arbitrary program generation. No runtime,
training, or live candidate call is authorized. Python R2/R3 stay unresolved.

## Allowed new files

1. `experiments/family_b_target_speed_formula_v1/config.json`
2. `src/oracle_composition/rewards/target_speed_formula.py`
3. `src/oracle_composition/reward_search/formula_contracts.py`
4. `src/oracle_composition/reward_search/formula_loop.py`
5. `tests/rewards/test_target_speed_formula.py`
6. `tests/reward_search/test_formula_proposal_loop.py`
7. `docs/operations/dual-orchestration/F1_RESULT.md`

No existing source/test/schema, shared runner/report, CLI, dependency, charter,
experiment-v1 receipt, or evaluator changes. Stop and report if truly required.
Use `apply_patch`; comments explain invariants, not every line. Do not commit.

## Trusted numerical core

- Recipe is at most 1,024 UTF-8 bytes; exact JSON keys `formula_id`, `alpha`,
  `beta`. One fixed versioned formula identifier, no expressions or source text.
- Finite alpha in [0.25, 4.0], beta in [-10.0, 10.0]; reject bools, callbacks,
  unknown/missing/duplicate keys, NaN/Infinity, malformed/deep/oversize JSON.
- Trusted base term: `1.25 * (1 - min(1, abs(v - target) / target))`.
  Output: `alpha * base + beta`. Validate inputs and finite output/envelope;
  no generated Python, compile/exec/eval, dynamic imports or user callables.
- Use the exact existing CandidateTaskInputsV1 read surface: COM x velocity
  and targets 0.5/1.0/1.5 m/s. Validate exact safe types before arithmetic;
  reject malformed/subclass/forged inputs without calling their callbacks.
- Do not widen B0 targets or substitute root speed. Main's composition profile
  is incompatible (root speed, targets about 5.52/0.885); peer question is open.
  Record this limit and test refusal. No total-reward compositor claim yet.
- Numerical unit evidence at alpha=1,beta=0,target=1:
  v=-1,0,1,2 -> 0,0,1.25,0. Additional alpha/beta and boundary checks.

## Separate data lineage, reuse existing helpers where suitable

- New versioned formula packet/proposal/receipt/iteration kinds. Never put JSON
  in A1 `proposed_source` or relabel its v1 code receipts. Keep old artifacts
  parseable under their own schema and rejected under the new schema.
- Prepare bounded deterministic prompt/record from canonical recipe, immutable
  config, source bundle, protected evidence (measured or explicitly missing),
  parent/prior iteration and independently supplied frozen-input bindings.
- Bind and reverify exact recipe/config, trusted formula implementation,
  read-contract/compositor or explicit absent compositor, dossier/source bundle,
  parent, feedback, independent evaluator and frozen configuration identities.
  Hashes must be computed from retained bytes, not just echo unverified labels.
- Ingest a response with recipe, rationale, predicted effect, falsifier and
  source IDs against that exact packet; retain immutable acceptance/rejection
  receipts. Revision requires exact prior receipt/iteration/candidate and new
  protected feedback. Mutating any bound artifact must fail.
- This slice may expose local prepare/ingest/revise functions, not a second
  CLI or cycle runner. No launch functionality. If ingesting supplied response
  bytes rather than verified detached-Sol envelopes, label origin explicitly
  unverified/synthetic; never manufacture an actual model-call receipt.
- No dynamic calibration or training admission. Source/dossier provenance
  stays distinct from tested numerical behavior and model-origin evidence.

## Focused acceptance only

- Positive prepare/ingest/revise fixture flow and retained rejection flow.
- Negative JSON/type/boundary/unknown formula tests; hostile object callback
  canaries; mutation of each bound identity; wrong parent/prior receipt;
  feedback mismatch, missing evidence and v1/v2-family separation.
- Imported core and tests must not import Gymnasium, MuJoCo, torch or the shared
  simulator evaluator. No fake module aliases to conceal missing dependencies.
- Run only the two new test files plus existing reward-search tests and pure
  R1 reward tests as needed, bounded to 60 seconds per command; lint/format and
  compile only touched files. No full suite, simulation, training, OS probes,
  subprocess reward workers, paid API, network, install, push or new LLM calls.
- Result: progress/bottleneck/next step; exact diff, counts and skips, import/
  lock identity, API usage, retained boundaries and next integration dependency.
  Do not claim a completed research loop or humanoid improvement.
