# F2: T2 formula handoff plan

| status | current truth |
|---|---|
| progress | Sol's bounded plan completed; Fable supplied the seven-hash registry interface and removed its duplicate T2 class in committed main. |
| bottleneck | No reviewed T2 evaluator exists. Registry promotion, model provenance and training remain separate gates. |
| next step | Build one additive evaluator and test file; send exact identities for peer-agreed integration. |

- Plan run: `20260905T212913Z-cb25bab9-c2f9-4809-ac32-4d68072ab70e`.
- `SUCCEEDED` at 21:40:31 UTC; read-only Sol/max, no escalation; watcher
  observed completion. No source changes or execution evidence were produced.
- Launch base `27330eb`; subsequent `0266c1f` changed only parent metadata.
- Planner used peer object `8d91a81`. Parent subsequently read committed
  `10d477b5726fb674fa2a057419501faee8043d9d`, including the repaired registry.
- Fable's interface message:
  `20260905T214217.711948Z-79bbaa69ff38402fa4a0cea559a8cba0`.
  Acknowledged as read, not approval of a new entry.

## Chosen minimal slice

- Keep F1 recipe, parser, loop, config and all source identities unchanged.
- Add `rewards/target_speed_formula_t2.py` and its focused test file.
- Reuse `TargetSpeedFormulaRecipeV1` and the existing bounded JSON parser.
- New runtime identity: `target_speed_triangular_affine_t2_adapter/v1`.
- Recipe identity remains `target_speed_triangular_affine/v1`.
- Proposed parser identity: `target_speed_triangular_affine_recipe/v1`, bound
  to existing `target_speed_formula.py` bytes, not to the new adapter file.
- The registry's flat numeric `parameters` object contains only alpha/beta.
  Fable's trusted parser supplies the fixed recipe ID before calling the
  existing recipe parser. Candidate data cannot choose a parser or runtime.
- Fable owns the new specification schema, registry entry and compositor
  dispatch. No changes to tracking scales, stock telemetry or baseline zero.

This uses two source identities deliberately: the new evaluator and the reused
recipe/parser implementation. The accepted T2 input source/schema supplies the
other existing dependency bindings. Exact IDs and canonical bounds will be
proposed through the mailbox before shared integration.

## Later, not authorized by this slice

- A separate live-run receipt must bind an actual Sol response; F1 supplied-byte
  receipts remain unverified and cannot be relabeled.
- A T2-aware generation packet must bind the actual T2 contract and reward
  consumer. Do not send F1's old-target dossier and call its output T2-aware.
- Training awaits Fable's worker, reviewed canonical reward/run artifacts,
  explicit resource agreement and the required user authorization.
