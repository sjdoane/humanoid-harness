# T2C2 descendant execution seal and heavy-job supervision

| status | current truth |
|---|---|
| progress | T2 execution admission now binds a clean provenance commit plus an exact source snapshot and accepts only safe record-only descendants; production training and evaluation validate the complete shared heavy-job token immediately before spawn and release its exact identity after terminal cleanup. |
| bottleneck | Fable must review and commit this source slice before the v2 seal can be generated at a clean HEAD. The retained T2C1 v1 artifacts are intentionally unchanged and do not authorize cycle 0. |
| next step | Fable commits T2C2, runs the protocol's one clean-commit re-seal command, commits only the three regenerated T2 record files as its child, and obtains independent review before any runtime reservation. |

## Preflight and boundary

| item | observed value |
|---|---|
| launch base | clean `8773961167ab797c7f911f704a7555a39f506e82` |
| writer lease | `CLAIMED` by `sol-builder-20260906-t2c2`; model `gpt-5.6-sol`; role `builder`; exact declared scope |
| import origin | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness/src/oracle_composition/__init__.py` |
| Git writes | none; the builder's sandbox denies `.git` writes |
| execution boundary | tests only; no model call, simulator command outside tests, smoke, training, cohort, protected evaluation, or real slot acquisition |
| Astra-owned helper files | `resource_slot.py`, its tests, and `HEAVY_JOB_SLOT.md` remained byte-identical |

## Seal-paradox repair

| contract | implemented rule |
|---|---|
| admission identity | `admission_commit` is the exact clean HEAD observed while admission reads and seals the runtime bindings. A second clean-HEAD observation closes the binding-read window. |
| sealed source set | Existing dependency lock, Gymnasium environment source, MuJoCo model asset, and six runtime source bindings remain exact; `source_snapshot_sha256` hashes those binding records canonically. |
| accepted runtime HEAD | The admission commit itself, or a Git descendant whose complete endpoint diff contains only admitted documentation and record paths. |
| refused runtime HEAD | Non-descendant; dirty tracked or untracked tree; any non-allowlisted diff including `src/**`, `tests/**`, `uv.lock`, environment/model assets, or T2 `payloads/**`; or any changed binding/snapshot. |
| run-receipt identity | `t2_runtime_execution_identity` validates the manifest and returns adjacent `admission_commit` and `execution_commit_observed` fields for the runtime receipt. |
| retained compatibility | The exact T2C1 v1 execution/study/seal bytes remain current until Fable's clean-commit re-seal; the legacy validator preserves their semantics without treating the old commit as forever-equal HEAD. |

The allowlist is exact: `README.md`, `docs/**`, non-payload paths below
`experiments/004_t2_reward_study/**`,
`experiments/bootstrap_tqc_humanoid/reviews/**`, and
`experiments/family_b_target_speed_v1/receipts/**`. An empty record-only
endpoint diff is valid; a source change that is still present at the observed
descendant is not.

## Clean re-seal boundary

`admit_t2_study` is exposed through
`python -m oracle_composition.reward_study.final_admission`. It requires the
canonical experiment directory and exact expected clean commit, retains the
already admitted candidate, regenerates the v2 execution manifest plus both
study-arm bindings and seal, and replaces only the three canonical record
files. The exact post-commit command is in the
[T2 protocol](../../../experiments/004_t2_reward_study/PROTOCOL.md).

The builder did not run it: this checkout is necessarily dirty with the source
repair. T2C1 remains the visible sealed set until Fable creates a clean source
checkpoint and invokes admission once there.

## Heavy-job slot integration

| boundary | behavior |
|---|---|
| training | Existing mailbox reservation admission runs first. Every serial seed then revalidates the complete slot immediately before its worker spawn. Production callers must supply the same positive expected wall used for slot reservation. |
| evaluation | `evaluate-policy` now admits an exact mailbox reservation before output creation; the evaluation supervisor validates its token immediately before the evaluation worker spawn. |
| token binding | Astra's helper compares owner, canonical argv, clean commit, proposal and acceptance identities, canonical reservation digest, expected and hard wall, and unexpired acceptance. |
| lifecycle | The first successful validation captures owner/token ID. The same identity must persist between workers and is held through worker/process cleanup; the outer terminal path releases only that exact pair on success or failure. |
| negative seams | Tempfile slots and injected spawn/cleanup observers cover missing, malformed, expired, command-mismatched, and expected-wall-mismatched tokens without acquiring the real shared slot or running the simulator/training. |

## Verification

| check | observed result |
|---|---|
| Focused descendant-seal and training/evaluation supervision tests | final-byte run `122 passed in 85.87s` |
| Full suite, named reward-lane receipt test deselected | `1933 passed, 18 skipped, 1 deselected, 44 failed in 367.94s` |
| Sandbox baseline comparison | cleared pytest cache contained exactly `44` failed nodes; sorted diff against `sandbox_baseline_failures_03a3.txt` was empty |
| Repository-wide Ruff lint / format | passed / all `427` files formatted |
| `git diff --check` | passed |
| Builder wall time | `75m` from lease acquisition through final validation and handoff readback |

The excluded node is exactly
`tests/experiments/test_reward_target_speed_manifest.py::test_recorded_no_learning_runtime_receipt_replays_exactly`.
It is not executed or regenerated by this slice.

One intermediate focused repetition observed two fake-runtime checkpoint
publication crashes caused by `ENOENT` on unique temporary files. Both nodes
passed immediately in isolation (`2 passed in 38.31s`), and the subsequent
complete focused run passed all 122 nodes. The final full suite also passed
both. This transient is not treated as a T2C2 pass or failure signal.

## Claim ceiling

This slice establishes execution-identity, admission, spawn-exclusion, and
terminal-cleanup interface semantics only. Passing tests do not establish
reward effect, reference use, tracking utility, training success, simulator
behavior, or humanoid competence.

Fable resume: review the descendant/dirty/binding negatives, runtime receipt
identity, exact training and evaluation spawn boundaries, and token cleanup;
commit the complete T2C2 source/docs/tests slice; run the protocol's re-seal
command once at that exact clean commit; commit only the three regenerated T2
record files; then obtain independent review. Keep cycle 0, smoke, training,
cohorts, protected evaluation, and all behavior claims withheld.
