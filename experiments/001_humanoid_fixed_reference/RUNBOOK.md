# Experiment 001 execution runbook

| | Status |
|---|---|
| **progress** | Disposable calibration completed at `1,623.85` environment steps/s; fail-closed commands separate review, training, evaluation, and aggregation. |
| **bottleneck** | Peak memory is unmeasured, the design remains proposed, and broader static-stand/naturalness decisions lack independently justified thresholds. |
| **next step** | Review the calibrated budget and criteria, then author—not mutate—a human-reviewed locked design version. |

## Commands

Install the declared runtime and development dependencies:

```bash
uv sync --locked --extra gym --extra train --extra dev
```

Measure one complete proposed PPO rollout. The model is never serialized and is
discarded before the receipt is published:

```bash
uv run humanoid-fixed-reference calibrate \
  --design experiments/001_humanoid_fixed_reference/configs/static_stand_precalibration_v0.study.json \
  --seed 911
```

The completed seed-`911` calibration is preserved at
[`receipts/2026-09-02_resource_calibration_seed_911.json`](receipts/2026-09-02_resource_calibration_seed_911.json).
It is resource-planning evidence only; rerunning the command creates an ignored
run receipt and does not authorize training.

After calibration review, author a new design file with
`design_status: locked_pre_behavioral`. Do not mutate the proposed file in place.
The locked design must differ from the exact calibrated proposed design only in
`design_version` and `design_status`. Inspect it and emit a manifest candidate
using the exact calibration receipt printed by the command above:

```bash
uv run humanoid-fixed-reference emit-manifest-candidate \
  --design PATH_TO_LOCKED_STUDY_JSON \
  --calibrated-design experiments/001_humanoid_fixed_reference/configs/static_stand_precalibration_v0.study.json \
  --calibration-receipt PATH_TO_CALIBRATION_RUN/calibration_receipt.json \
  --output PATH_TO_CANDIDATE_MANIFEST_JSON
```

Candidate emission revalidates the calibration receipt's exact design, criteria,
current runtime, rollout count, action-likelihood audit, and discarded-model
status. The candidate cannot authorize training. After a human checks its
runtime and source identities, explicitly finalize it with the same calibration
inputs. Finalization re-inspects and rehashes them; changed bytes fail closed.
The reviewer and review note become part of the immutable manifest:

```bash
uv run humanoid-fixed-reference finalize-reviewed-manifest \
  --design PATH_TO_LOCKED_STUDY_JSON \
  --calibrated-design experiments/001_humanoid_fixed_reference/configs/static_stand_precalibration_v0.study.json \
  --calibration-receipt PATH_TO_CALIBRATION_RUN/calibration_receipt.json \
  --candidate PATH_TO_CANDIDATE_MANIFEST_JSON \
  --output PATH_TO_REVIEWED_MANIFEST_JSON \
  --reviewer-id REVIEWER_ID \
  --review-note "WHAT_WAS_CHECKED"
```

One invocation trains exactly one seed from the design's predetermined schedule.
The runtime is re-inspected before PPO starts. Only the final-timestep SB3
checkpoint is retained:

```bash
uv run humanoid-fixed-reference train-one-seed \
  --design PATH_TO_LOCKED_STUDY_JSON \
  --manifest PATH_TO_REVIEWED_MANIFEST_JSON \
  --seed 101
```

Evaluate one exact checkpoint on every declared evaluation seed:

```bash
uv run humanoid-fixed-reference evaluate-checkpoint \
  --design PATH_TO_LOCKED_STUDY_JSON \
  --manifest PATH_TO_REVIEWED_MANIFEST_JSON \
  --checkpoint-receipt experiments/001_humanoid_fixed_reference/runs/TRAIN_RUN/checkpoint_receipt.json
```

After all five predetermined seeds are trained and evaluated, aggregate exactly
their five checkpoint receipts and five evaluation receipts. Supply one flag
per artifact; order is irrelevant because train-seed identities are checked:

```bash
uv run humanoid-static-study-summary \
  --design PATH_TO_LOCKED_STUDY_JSON \
  --manifest PATH_TO_REVIEWED_MANIFEST_JSON \
  --criteria experiments/001_humanoid_fixed_reference/configs/static_positive_control_criteria_v1.json \
  --checkpoint-receipt RUN_101/checkpoint_receipt.json \
  --checkpoint-receipt RUN_202/checkpoint_receipt.json \
  --checkpoint-receipt RUN_303/checkpoint_receipt.json \
  --checkpoint-receipt RUN_404/checkpoint_receipt.json \
  --checkpoint-receipt RUN_505/checkpoint_receipt.json \
  --evaluation-receipt EVAL_101/evaluation_receipt.json \
  --evaluation-receipt EVAL_202/evaluation_receipt.json \
  --evaluation-receipt EVAL_303/evaluation_receipt.json \
  --evaluation-receipt EVAL_404/evaluation_receipt.json \
  --evaluation-receipt EVAL_505/evaluation_receipt.json \
  --output experiments/001_humanoid_fixed_reference/runs/static-study-summary.json
```

The aggregator reopens each checkpoint receipt and sibling checkpoint, then
revalidates all design, manifest, runtime, evaluator, reference, criteria, and
seed identities. It refuses partial schedules, duplicates, changed summary
source, and output overwrite.

Generated calibration, training, and evaluation directories are atomically
published under the ignored `runs/` directory. Existing runs and manifest files
are never overwritten.

## Bound surfaces

The frozen runtime fingerprint includes:

- exact design semantics and design-file bytes;
- exact calibrated proposed-design semantics and bytes, plus calibration-receipt
  semantics and bytes;
- exact decision-criteria semantics and file bytes;
- Gymnasium, MuJoCo, NumPy, SB3, Torch, Python, operating system, and machine;
- exact `uv.lock` bytes from the checkout or force-included wheel resource,
  MuJoCo XML, observation/action shapes, dtypes, and exact Box bounds;
- environment construction, reference ABI, reward formula, wrapper, experiment
  contract, protected evaluator, runtime inspector, and execution source bytes;
- the five-seed study-summary source bytes;
- a bounded digest of every regular `.py` file in the local
  `src/oracle_composition` tree, including the reference artifact contract;
- exact reference/schema/reward identities;
- the local tanh-squashed Gaussian policy source, normalized and physical
  action Boxes, and the sole normalized-to-physical affine wrapper;
- no observation or reward normalizer;
- the exact Gym `TimeLimit` horizon; a design/runtime mismatch fails before PPO
  construction; and
- the reviewed candidate bytes, reviewer identifier, and review note.

Any drift blocks training and evaluation. Evaluation also rehashes the design
file, manifest file, checkpoint receipt, and checkpoint bytes before loading the
policy. Study designs and execution JSON artifacts use bounded strict UTF-8
parsing; duplicate keys, non-finite numbers, oversized inputs, unknown fields,
and malformed encoding are rejected rather than repaired.

The lock digest binds the declared dependency resolution. It is not proof that
the current environment was installed from that lock; observed package versions
are separately recorded in the runtime fingerprint and must match the reviewed
manifest.

The source-tree digest closes the on-disk Python-source set. It does not prove
that already-imported bytecode came from those bytes, so every authorizing CLI
stage must start in a fresh, clean process. Injected test hooks are marked
`noncanonical_callables_test_only/v1`; their manifests and receipts cannot enter
the five-seed aggregator.

## Current limitations

- The current rule reports only full-horizon/collapse-rule,
  reward-scale-occupancy, bounded-control, and prohibited-contact conformance.
  Reward-derived scales are not independent task-success thresholds. It records reset-relative horizontal
  displacement and per-axis root speed descriptively, but cannot establish
  station keeping, naturalness, or causal reference use without independent
  pre-data thresholds.
- PPO stores the same bounded normalized action used to compute its corrected
  likelihood. An outer action wrapper maps that value exactly once to the
  physical Humanoid actuator Box; endpoint probes fail closed on drift.
- Protected metrics read direct MuJoCo state, every physics-substep contact,
  per-joint torque capacity, and executed policy action. The scalar tracking
  return remains diagnostic only.
- Aggregation may report the explicitly named nonclaim conformance result. Broader
  `static_stand_feasibility_pass` and generic `study_pass` remain `null`, and
  nothing is promoted automatically.
- The runtime currently binds only the operating-system and machine families,
  not OS release, CPU identity, or thread/determinism settings. Results are
  single-host evidence and must not be merged across hosts. Bitwise-identical
  training is not claimed.
