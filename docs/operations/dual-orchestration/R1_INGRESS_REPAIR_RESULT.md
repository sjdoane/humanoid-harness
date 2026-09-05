# R1 static-ingress repair result

| status | current truth |
|---|---|
| progress | The three independent static-boundary findings were repaired in the allowed ingress module and regression-tested; the focused suite reports 143 passed and 16 intentionally deferred. |
| bottleneck | This builder result is not independent acceptance. R1 and B0 remain unaccepted pending the parent’s targeted re-review. |
| next step | Parent verifies this bounded diff and obtains an independent static-only re-review before any R2 work or dynamic admission. |

## Snapshot gate

- Pre-repair HEAD matched `67722402c58374cf7f22347f7055abb979c9436f`.
- All 15 paths in `.orchestration/b0-repair/r1review/review-snapshot.json`
  matched their recorded SHA-256 values before the first edit. No drift was
  found. That snapshot is now historical.

## Exact repair

| finding | change | regression evidence |
|---|---|---|
| Text-subclass callback | `_source_bytes` now accepts only exact built-in `str` or `bytes` before encoding or decoding. | Exact built-in text and bytes pass; a hostile `str.encode` sentinel is rejected untouched. |
| Metadata callbacks | Receipt ingress now accepts only exact plain JSON `dict`/`list` containers and exact JSON scalar types. Caller-supplied mapping proxies and arbitrary containers are rejected before traversal, equality, or serialization. Deep immutable proxies are created only from the validated copy. The redundant accepted-source `metadata` constructor argument was removed; its public `metadata` property exposes the receipt’s immutable view. | Direct receipt, `from_dict`, nested-container, proxy-wrapped hostile mapping, hostile equality/traversal, plain JSON round-trip, mutation isolation, and deep immutability cases pass without touching sentinels. |
| Forged AST evidence | Every `StaticallyAcceptedTaskTermSourceV1` construction reparses its exact bytes, derives the complete v1 receipt, and compares canonical receipt bytes. The factory uses the same derivation path. | Valid direct/factory/JSON constructions pass. Invalid source with fabricated matching hash/length, valid source with altered `ast_nodes`, and stale source bytes are rejected. |

Repair-created changes are limited to:

- `src/oracle_composition/rewards/static_validation.py`
- `tests/rewards/test_parent_execution_boundary.py`
- `tests/rewards/test_scale_calibration.py`
- `docs/operations/dual-orchestration/R1_INGRESS_REPAIR_RESULT.md`

No coordination handoff, active pointer, snapshot, worker, sandbox, evaluator,
environment, trainer, dependency, study configuration, historical receipt, or
shared strategy file was edited for this repair.

## Verification

| check | result |
|---|---|
| Import origin | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/src/oracle_composition/__init__.py` |
| Prescribed focused pytest command | `143 passed, 16 skipped in 2.52s` |
| Deferred boundary | 13 host/subprocess sandbox cases and 3 dynamic scale cases remain skipped. |
| Ruff check | Passed on the four allowed Python paths. |
| Ruff format check | Four files already formatted. |
| Python compile | Passed for the three repair-modified Python files. No candidate source was compiled. |
| `git diff --check` | Passed. |

No candidate source was compiled, executed, installed, or invoked. No reward
subprocess, OS/network canary, simulator, model, training, full suite, commit,
or push was performed.

## Unresolved findings

- No known implementation item remains from the three specified ingress
  findings in this builder pass.
- Independent targeted re-review remains mandatory; these local checks do not
  accept R1 or B0.
- R2 worker containment and R3 provenance remain separate, out-of-scope gates.
