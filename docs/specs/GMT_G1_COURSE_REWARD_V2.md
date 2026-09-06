# GMT G1 course task reward v2

| Item | Contract |
| --- | --- |
| Status | Implemented reward contract; not trained or behaviorally validated |
| Frozen | MDP, task features, controller, tracking reward, evaluator, PPO, and budget |
| Authorable | Existing weights, `depth_strength` in `[0, 1]`, and `ceiling_fraction` in `[0, 1]` |
| Fixed | Depth scale `s = 0.10 m` |

## Mechanism

- Existing in-region posture reward: `G = exp(-(band_error / 0.20 m)^2)`.
- Height ceiling: `h_c = h_low + ceiling_fraction * (h_high - h_low)`.
- One-sided depth error: `e = max(0, h - h_c)`.
- Cauchy factor: `C = 1 / (1 + (e / s)^2)`.
- Version 2 posture reward inside the region:
  `G * (1 - depth_strength * (1 - C))`.
- Outside the region and after a fall, behavior is unchanged.

## Migration and test boundary

- Version 1 keeps its exact schema and numerical path.
- Version 2 with `depth_strength = 0` is numerically identical to version 1.
  Its versioned recipe identity and SHA-256 remain different.
- A depth-only study must hold the oracle, task, seed set, training budget, and
  existing weights fixed.
- This component never exceeds the old posture component and adds no positive
  region-occupancy bonus. It can still favor rushing through or avoiding the
  region. Check region duration, entry/exit, completion, speed error, and falls.
- Passing unit tests establishes the contract, not improved robot behavior or
  reward-gaming resistance.
