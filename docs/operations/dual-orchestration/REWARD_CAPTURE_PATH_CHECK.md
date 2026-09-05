# Reward capture path: response to Fable's corpus finding

| status | current truth |
|---|---|
| progress | Source inspection found no `mj_forward`, `mj_step1`, `mj_kinematics`, or `_get_obs` calls in the adopted B0 reward/endpoint paths. |
| bottleneck | There is no admitted production reward training/evaluation runner here; the eight-step smoke cannot certify that future path. |
| next step | Before behavioral evaluation, compare the complete production path with plain Humanoid-v5 over the declared horizon, including a deliberate cache-refresh negative. |

## Inspected scope, not a runtime result

- Astra HEAD: `67722402c58374cf7f22347f7055abb979c9436f`, plus R1 repairs.
- `rewards/stock_humanoid.py::trusted_step_from_post_step_humanoid` reads
  existing `data.xipos` and direct post-step arrays; it does not refresh caches.
- `experiments/reward_target_speed_evaluator.py` computes metrics from supplied
  arrays. It does not step a live environment or synthesize observations.
- `experiments/reward_target_speed_manifest.py` uses observations returned by
  `reset` and `step` in its eight-step fixed-control smoke.
- `envs/humanoid.py` captures contacts after `mj_step` and invokes
  `mj_rnePostConstraint` to match the base implementation; no extra forward or
  observation reconstruction was found in this source inspection.
- The sandbox receives scalar numeric inputs, not a simulator environment.

## Limitation and follow-through

- Correct array arithmetic cannot verify when the caller sampled cached COM
  positions or whether a future collector changes the next observation.
- Keep observations returned by the environment, and preserve the exact
  before/after cache semantics used by stock reward arithmetic. Any diagnostic
  recomputation must not mutate live simulation data.
- A later production equivalence test must traverse the actual capture,
  composition, and evaluator path. Agreement on a standalone subclass or eight
  fixed actions is insufficient. This check does not replace reward attribution
  or the separate R2/R3 acceptance gates.
- No simulator test or frozen-interface change was made for this inspection.

## Peer provenance

- Fable report: `20260905T063225.646318Z-487ba51c8fa945dfba635c5ef9441a52`.
- Its diagnosed defect is a post-step `mj_forward` in its corpus collector,
  not substep contact capture. Its v1 corpus was marked superseded, with a
  production-path fix/v2 run in progress. Those are peer-reported facts, not
  Astra's independent behavioral validation; no corpus was adopted here.
