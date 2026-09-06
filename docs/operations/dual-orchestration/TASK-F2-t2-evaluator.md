# F2: additive T2 evaluator

- Role: one Sol/max builder; no delegation. Deadline: 20 minutes from launch.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`;
  branch `astra/reward-loop`. Record HEAD, clean/dirty state and lock hash.
- Use the existing writer lease and `apply_patch`. Do not commit or push.
- Only three allowed new files:
  1. `src/oracle_composition/rewards/target_speed_formula_t2.py`
  2. `tests/rewards/test_target_speed_formula_t2.py`
  3. `docs/operations/dual-orchestration/F2_RESULT.md`
- No changes to existing source, tests, config, exports, hooks, dependencies,
  the peer checkout or any shared scientific contract.

## Read

- Root `AGENTS.md` and `docs/operations/dual-orchestration/README.md`.
- In that documentation directory: `F1_ACCEPTANCE.md`, `F2_PLAN_RESULT.md`,
  `T2_INPUT_ACCEPTANCE.md`.
- Existing `src/oracle_composition/rewards/{target_speed_formula,task_inputs_v2}.py`
  and their tests. These are accepted dependencies, not editing targets.

## Implement

```python
FORMULA_RUNTIME_ID = "target_speed_triangular_affine_t2_adapter/v1"
PARSER_ID = "target_speed_triangular_affine_recipe/v1"


def evaluate_target_speed_formula_t2(
    recipe: TargetSpeedFormulaRecipeV1,
    inputs: CandidateTaskInputsV2,
) -> float: ...


def target_speed_formula_t2_bounds() -> dict[str, object]: ...
```

- Import the exact accepted recipe and T2 classes. Do not create duplicates,
  convert inputs to V1, or call the V1 evaluator with fabricated targets.
- Exact-type revalidate the recipe and all slots on every consumption. Reject
  partial/forged objects, subclasses, bools, unsafe numeric/string subclasses,
  NaN/Inf and unknown recipe IDs before invoking user-controlled callbacks.
  Reuse the trusted recipe constructor where safe; normalize errors into a
  small `TargetSpeedFormulaT2Error` value error.
- Call `validate_task_inputs_v2`; target is exactly 3.0 m/s. No new velocity
  clamp or semantic change to accepted T2. Cadence stays adapter metadata.
- Reuse the recipe ID `target_speed_triangular_affine/v1`, alpha [0.25,4],
  beta [-10,10], base scale 1.25. Duplicate only the few arithmetic lines:
  base = 1.25 * (1 - min(1, abs(v - target) / target)); task = alpha*base+beta.
  Saturate before division to preserve finite output for extreme finite v.
- Return an exact finite builtin float in [-10,15]. No access to references,
  oracle state, actions, contacts, reward telemetry, filesystem, subprocesses,
  dynamic code/imports or candidate callbacks during evaluation. Only fixed
  trusted imports are allowed; no simulator or training dependency.
- Bounds helper returns a fresh data-only object on every call, exactly:
  `{"parameters":{"alpha":{"minimum":0.25,"maximum":4.0},"beta":{"minimum":-10.0,"maximum":10.0}},"r_task":{"minimum":-10.0,"maximum":15.0}}`.
- Do not implement a new parser. `PARSER_ID` names existing
  `parse_target_speed_formula_recipe` in `target_speed_formula.py`; the caller
  supplies its fixed recipe ID plus the two flat numeric parameters. Record
  this mapping clearly in the result. No runtime registry admission follows.

## Focused acceptance checks

1. Target 3.0, half-target, zero/backward motion, large finite speeds, all
   parameter endpoints: exact expected value and finite output envelope.
2. Wrong target, V1/duplicate/subclass inputs, missing slots, forged recipe
   fields, hostile numeric conversions and unknown recipe ID: refusal with
   callback counters untouched.
3. Existing bounded recipe parser feeds this evaluator; new runtime ID is
   never passed off as the old recipe ID. Bounds are fresh and deterministic.
4. No filesystem/dynamic execution/heavy import in the callable path. Check
   actual AST module imports/calls if adding a source guard; do not check only
   imported aliases while missing ImportFrom.module.
5. Old formula/T2 sources remain byte-identical and old focused tests pass.

Use Astra's existing `.venv`; verify package import resolves here. Run only the
new focused test plus `test_target_speed_formula.py` and `test_task_inputs_v2.py`,
Ruff/format on the two new Python files, compile and diff checks. Do not run
the full suite, gym-marked tests, simulator, candidate model calls or training.

## Result

- Progress / bottleneck / next step; exact changed files and tests actually run.
- New formula file SHA-256; existing parser/source SHA-256; canonical bounds
  bytes and SHA-256 (sorted JSON, compact separators, finite values, UTF-8).
- Input source SHA-256 and schema identity; output envelope; exact API mapping.
- Old formula source must remain
  `c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb`;
  input source remains
  `9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2`.
- Claim ceiling: pure T2 evaluator and parameterization only. No compositor,
  real-model provenance, training integration or behavioral evidence.
