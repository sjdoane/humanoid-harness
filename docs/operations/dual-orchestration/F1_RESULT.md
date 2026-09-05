# F1 data-only formula core result

| status | current truth |
|---|---|
| progress | Added the fixed target-speed formula, schema-v2 formula packet/proposal lineage, retained-byte bindings, and focused refusal tests. The bounded acceptance command passes 174 tests. |
| bottleneck | The composition profile still uses root-x speed and targets outside B0's COM-x `0.5/1.0/1.5 m/s` contract. The compositor is explicitly absent; Python R2/R3 remain unresolved. |
| next step | Obtain peer agreement on the reward-study speed/target adapter and compositor, then independently review this uncommitted slice before any integration or runtime work. |

## Checkout receipt

| item | observed value |
|---|---|
| checkout realpath | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra` |
| branch | `astra/reward-loop` |
| starting HEAD | `402ff4f3ff2a4d3c0f93f8052e9e691ebde9ae3f` |
| required merge ancestor | `8a48f322a4e3e14bb4467daddd709de1926d0e1d` |
| commits after required merge | One parent packet commit, `402ff4f`; it adds only `TASK-F1-formula-core.md` and `TASK-M1-integration-review.md` |
| starting dirty paths | None |
| package import | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/src/oracle_composition/__init__.py` |
| `uv.lock` SHA-256 | `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40` |

The default `uv` cache was sandbox-inaccessible. Verification used the existing
Astra `.venv`; no install, shared environment, or fallback checkout was used.

## Implemented capability

- `target_speed_formula.py` accepts only a 1,024-byte UTF-8 JSON recipe with
  exact keys `formula_id`, `alpha`, and `beta` and the single identifier
  `target_speed_triangular_affine/v1`.
- The trusted evaluator applies
  `1.25 * (1 - min(1, abs(v - target) / target))`, then `alpha * base + beta`.
  It accepts only exact `CandidateTaskInputsV1`, revalidates finite exact-float
  slots, and keeps B0's COM-x signal and three targets unchanged.
- Schema-v2 kinds are separate from A1's schema-v1 source-code artifacts:
  `target_speed_formula_packet_manifest`, `target_speed_formula_packet_record`,
  `target_speed_formula_proposal`, `target_speed_formula_ingestion_receipt`, and
  `target_speed_formula_iteration_record`.
- Packet records retain exact base64-encoded bytes and recompute hashes for the
  config, trusted formula implementation, candidate read contract, independent
  evaluator, and caller-supplied frozen configuration. The current compositor
  binding is explicitly absent.
- Prepare, render, supplied-byte ingest, and revision functions verify packet,
  recipe, config, implementation, read contract, dossier, source bundle,
  evaluator, frozen configuration, prior receipt, prior iteration, and feedback
  links. Publication is no-overwrite and retains both acceptance and rejection
  receipts.
- Supplied response bytes are labeled
  `unverified_synthetic_supplied_bytes`; receipts state that no model-call
  receipt was created, dynamic calibration was not performed, training is not
  authorized, and improvement was not measured.

Public F1 functions are local library APIs only:

- Core: `parse_target_speed_formula_recipe`,
  `evaluate_target_speed_formula`.
- Lineage: `create_formula_bindings`, `prepare_initial_formula_packet` (and its
  `prepare_formula_packet` alias), `render_formula_packet`,
  `ingest_formula_response`, and `prepare_formula_revision`.
- No package re-export, CLI, cycle runner, launch path, or existing API changed.

## Numerical and refusal evidence

| check | observed result |
|---|---|
| `alpha=1`, `beta=0`, `target=1`, velocities `-1,0,1,2` | `0,0,1.25,0` |
| lower affine boundary at target | `alpha=0.25`, `beta=-10` gives `-9.6875` |
| upper affine boundary at target | `alpha=4`, `beta=10` gives `15.0` |
| output envelope | `[-10.0, 15.0]` |
| incompatible composition target | `5.520768616125457 m/s` refused by the unchanged B0 input contract |
| hostile values | bytes/numeric/input subclasses, callback-bearing objects, and forged slots rejected without callback execution |
| JSON negatives | missing/unknown/duplicate keys, bools, nonfinite values, malformed/deep/oversize JSON, unknown formula, and bound violations rejected |
| lineage negatives | each manifest/binding identity, parent, dossier, source bundle, prior receipt/candidate, feedback, evaluator, and frozen configuration mutation rejected |
| family separation | A1 code proposals remain parseable as A1 and are rejected as formula-v2; formula-v2 proposals are rejected as A1 |

These are unit and software-contract observations. They are not total-reward
composition, candidate execution, protected simulator evidence, training, or a
humanoid-performance result.

## Exact verification

| command scope | outcome |
|---|---|
| two new test files, run separately | `59 passed`: 27 numerical/core and 32 proposal-lineage tests |
| two new files plus existing reward-search and selected pure R1 reward tests | `174 passed, 3 skipped in 0.57s` |
| skip reason | Three inherited scale-calibration cases require the unresolved R2 host fixture |
| Ruff check on five touched Python files | Passed |
| Ruff format check on five touched Python files | `5 files already formatted` |
| `py_compile` on five touched Python files | Passed |
| fresh import audit | All three F1 modules resolve inside Astra; Gymnasium, MuJoCo, and torch remain unloaded |
| no-index whitespace check across all seven new files | Passed |

Every command was bounded to 60 seconds. No full suite, simulator, training,
live candidate call, reward subprocess worker, paid API, network, dependency
install, push, or model call ran. The existing A1 tests' isolated import-origin
subprocess checks did run; they do not execute reward candidates.

## Exact diff

Only these seven new paths are present (2,003 lines total); no existing source, test, schema,
runner, report, CLI, dependency, charter, receipt, evaluator, or lease changed:

1. `experiments/family_b_target_speed_formula_v1/config.json`
2. `src/oracle_composition/rewards/target_speed_formula.py`
3. `src/oracle_composition/reward_search/formula_contracts.py`
4. `src/oracle_composition/reward_search/formula_loop.py`
5. `tests/rewards/test_target_speed_formula.py`
6. `tests/reward_search/test_formula_proposal_loop.py`
7. `docs/operations/dual-orchestration/F1_RESULT.md`

The slice is deliberately uncommitted. Its next integration dependency is an
explicit peer resolution for COM-x versus root-x speed, the admitted target
set, and the total-reward compositor boundary, followed by independent review.
