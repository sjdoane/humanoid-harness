# Evidence-to-diagnosis feedback

This data-only command validates retained evidence, emits one deterministic next-action surface,
and prepares a bounded source packet for a later proposal coordinator. It does not call a model,
run a simulator, train, evaluate, replay, or authorize a candidate.

## Current Phase A receipt

The output directory must not already exist.

```bash
uv run humanoid-harness --json diagnose \
  --repository-root /Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra \
  --experiment /Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/experiments/003_composition_speed_profile \
  --phase-a-receipt /Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/experiments/003_composition_speed_profile/cycles/cycle_2/scientific_receipt_v2.json \
  --phase-a-cycle 2 \
  --no-research-graph \
  --output /tmp/humanoid-feedback-phase-a-cycle-2
```

Omit `--no-research-graph` to retrieve at most six cards from the existing local graph. Add
`--phase-b-evidence PATH` for either the current canonical Phase B report or the exact current
seed-failure receipt shape. Add `--steering-file PATH` for bounded UTF-8 human steering; its exact
bytes are hashed and labelled as steering, never measurement.

Retrieval uses short, mechanism-specific queries. It does not search the whole
diagnostic paragraph: repeated caveat language otherwise outranks useful motion
transition and reference-alignment evidence. The graph remains one existing index,
rebuilt with `humanoid-harness research build` when source extractions change.

The command publishes:

- `diagnosis_v1.json`: validated facts, hypothesis, rivals, prediction, falsifier, identities,
  claim ceilings, missing evidence, and readiness bits.
- `candidate_context_v1.json`: prompt hash, source bundle, missing graph state, and coordinator
  readiness.
- `candidate_prompt_v1.md`: fixed-budget, identifier-preserving context. Diagnosis-only numeric
  facts and protected/held-out outcomes are excluded.

Cycle 2 contains 20 current-cycle episode rows even though `summary.arms` carries six historical
and current arms. The adapter filters current rows by the current oracle. Zero falls and switches
do not establish Humanoid competence: the current arm still omits the declared slow behavior, so
Phase A yields a low-confidence oracle hypothesis but `proposal_ready=false`.

## Decision boundary

- Missing initial evidence or an uncalibrated endpoint routes `measurement`.
- A structurally valid seed-failure report routes `adapter_repair`, but is not called an
  authoritative runtime fact because this slice does not reverify its manifest/job parent chain.
- An ambiguous fall routes `measurement`; it does not automatically blame task reward.
- A protected Phase B report may produce a human-readable oracle or reward diagnosis, but every
  such report remains `proposal_ready=false`. Its candidate packet is outcome-independent,
  measurement-only, and requests separate development evidence. Retrieval and steering text
  are withheld too; stripping numeric scores alone does not prevent indirect leakage.
- Human steering cannot override evidence, lineage, safety, or missing-data gates.

The next slice must admit independent development evidence, then manage proposal lineage,
training, replay screening, protected evaluation, and revision. None of those capabilities is
implemented here.

## Research provenance

The design takes conceptual—not source-level or runtime—reuse from the old read-only
RL-Sculptor: component-series loading and deterministic diagnostic guardrails in
`RewardSculptor/sculptor/diagnose.py:459-560`, bounded human steering in
`diagnose.py:789-811`, and the explicit pause/resume feedback handoff in
`RewardSculptor/sculptor/sculpt.py:2794-2842`. No old package is imported.

The local source cards bind the scientific rationale: [Eureka](../../research/evidence/legacy_policy_harness_kg/cards/2310.12931.md)
supports fixed-trainer reward reflection with component traces and human text; [RDA](../../research/evidence/legacy_policy_harness_kg/cards/2606.01672.md)
supports subtask/trajectory diagnosis before targeted reward revision. Neither source establishes
causal diagnosis, safe generated-code execution, or reference-oracle composition.
