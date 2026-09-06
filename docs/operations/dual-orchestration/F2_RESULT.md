# F2: additive T2 evaluator result

| status | current truth |
|---|---|
| progress | Added the pure T2 formula adapter and adversarial focused tests; the final authorized selection passed 73 tests. |
| bottleneck | No compositor or runtime-registry entry consumes this adapter. Real-model provenance, training integration, and behavioral evidence remain absent. |
| next step | Independently review these exact three files and hashes, then propose the runtime/parser/input bindings to Fable for a separately agreed registry integration. |

## Launch boundary

| item | recorded value |
|---|---|
| checkout realpath | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra` |
| branch | `astra/reward-loop` |
| starting HEAD | `2c94710af868c7853672143cf91126ece6131c57` |
| starting dirty state | clean; `git status --short` returned no paths |
| writer lease | existing owner `astra-f2-builder-20260905`; scope was exactly the three files below |
| lease acquired | `2026-09-05T22:06:15Z` |
| final verification | `2026-09-05T22:13:46Z`; 7m31s after lease acquisition and before the 20-minute deadline |
| requested worker | `gpt-5.6-sol`, reasoning `max`, builder; no delegation used |
| `uv.lock` SHA-256 | `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40` |
| package import | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/src/oracle_composition/__init__.py` |

No existing lease, hook, dependency, export, configuration, source, test, peer
checkout, or shared scientific contract was changed. No commit or push was
made.

## Exact new files

| path | purpose | SHA-256 |
|---|---|---|
| `src/oracle_composition/rewards/target_speed_formula_t2.py` | Pure T2 evaluator and fresh bounds helper | `064393887bb4a7157d614981cc2000940252e12c31ad0aabc13109737fd94301` |
| `tests/rewards/test_target_speed_formula_t2.py` | Exact values, refusal/callback canaries, parser/runtime separation, bounds, and AST guard | `39e5d07c3291bd103376948c093cc0698791af6220fa91457cc99c61405e8e1e` |
| `docs/operations/dual-orchestration/F2_RESULT.md` | This result receipt | self-hash intentionally omitted |

## Accepted dependency identities

| dependency | identity |
|---|---|
| existing recipe/parser source | `src/oracle_composition/rewards/target_speed_formula.py`; SHA-256 `c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb` |
| existing T2 input source | `src/oracle_composition/rewards/task_inputs_v2.py`; SHA-256 `9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2` |
| T2 schema | `humanoid-fixed-com-speed/task-inputs/v2`; `CandidateTaskInputsV2(com_x_velocity_m_s, target_speed_m_s)` |
| fixed input semantics | stock COM x velocity; target exactly `3.0` m/s; `0.015` s cadence remains adapter metadata |

Both accepted source hashes were rechecked before and after the focused test
run and remained byte-identical.

## Exact API and parser mapping

| role | identifier or callable | binding |
|---|---|---|
| recipe | `target_speed_triangular_affine/v1` | Existing exact `TargetSpeedFormulaRecipeV1`; alpha in `[0.25, 4.0]`, beta in `[-10.0, 10.0]` |
| parser label | `target_speed_triangular_affine_recipe/v1` | Existing `parse_target_speed_formula_recipe` in `target_speed_formula.py`; no new parser |
| runtime | `target_speed_triangular_affine_t2_adapter/v1` | New `evaluate_target_speed_formula_t2(recipe, inputs)` |
| input | `humanoid-fixed-com-speed/task-inputs/v2` | Existing exact `CandidateTaskInputsV2`, revalidated by `validate_task_inputs_v2` on every call |

The trusted caller takes the registry's flat numeric `parameters` object
containing only `alpha` and `beta`, supplies the fixed recipe ID
`target_speed_triangular_affine/v1`, and passes those three fields to the
existing bounded parser. The resulting exact recipe and an accepted exact T2
input are passed to the new runtime. The runtime ID is never used as the recipe
ID. `PARSER_ID` is only the binding label for the existing parser. This slice
does not admit either ID to a runtime registry.

## Formula, bounds, and output

For fixed target `3.0` m/s:

```text
error = abs(com_x_velocity_m_s - target_speed_m_s)
normalized_error = 1.0 if error >= target_speed_m_s else error / target_speed_m_s
base = 1.25 * (1.0 - min(1.0, normalized_error))
r_task = alpha * base + beta
```

The saturation comparison occurs before division, preserving finite behavior
for extreme finite velocities without adding a velocity clamp. The result is
an exact builtin `float`, finite, in `[-10.0, 15.0]`.

Canonical bounds bytes use sorted keys, compact separators, finite JSON values,
and UTF-8:

```json
{"parameters":{"alpha":{"maximum":4.0,"minimum":0.25},"beta":{"maximum":10.0,"minimum":-10.0}},"r_task":{"maximum":15.0,"minimum":-10.0}}
```

- Byte count: `137`.
- SHA-256: `2c6030264231a52320a44fe0f4d4ed519bc2e635b892f1b9c0d8929166106933`.
- `target_speed_formula_t2_bounds()` returns an equal, fully fresh nested object
  on each call.

## Verification actually run

| command | result |
|---|---|
| `.venv/bin/python -m pytest -q tests/rewards/test_target_speed_formula_t2.py` | Final: **29 passed in 0.03s**. Earlier iterations were **22 passed, 1 failed** and then **23 passed**. The one failure was test setup: the accepted T2 constructor correctly rejected a subclass before the evaluator was reached; the fixture was changed to forge the subclass without construction. |
| `.venv/bin/python -m pytest -q tests/rewards/test_target_speed_formula_t2.py tests/rewards/test_target_speed_formula.py tests/rewards/test_task_inputs_v2.py` | Final: **73 passed in 0.04s**. An earlier pre-hardening run was **67 passed in 0.04s**. |
| `.venv/bin/ruff check src/oracle_composition/rewards/target_speed_formula_t2.py tests/rewards/test_target_speed_formula_t2.py` | Final pass. The first run found one unused accepted constant import; explicit recipe-ID revalidation resolved it. |
| `.venv/bin/ruff format --check src/oracle_composition/rewards/target_speed_formula_t2.py tests/rewards/test_target_speed_formula_t2.py` | Final pass: two files already formatted. |
| `PYTHONPYCACHEPREFIX=<temporary-directory> .venv/bin/python -m py_compile src/oracle_composition/rewards/target_speed_formula_t2.py tests/rewards/test_target_speed_formula_t2.py` | Pass; temporary compile artifacts removed. |
| `git diff --check` plus `git diff --no-index --check /dev/null <new-file>` for each new Python file | Pass; no tracked diff and no whitespace errors in either new Python file. |

The new tests cover target, half-target, zero, backward and extreme finite
velocities; all alpha/beta endpoint combinations; exact finite output type and
envelope; wrong target; V1, duplicate, subclass and incomplete inputs; forged
recipe/input slots; bool, nonfinite, oversized and subclassed numeric values;
unsafe string subclasses; callback-order canaries; existing-parser handoff;
runtime/recipe ID separation; fresh deterministic bounds; and an AST check of
actual `Import` and `ImportFrom.module` values plus dynamic/filesystem calls.

No full suite, gym-marked test, simulator, candidate-model call, training job,
filesystem action from the evaluator, subprocess, dynamic import, or dynamic
code execution was run.

## Claim ceiling

F2 establishes only a pure T2 evaluator and its bounded parameterization under
focused software tests. It establishes no compositor, runtime-registry
admission, real-model provenance, training integration, humanoid competence, or
behavioral improvement.
