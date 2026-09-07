# Transition preview: a candidate, not an implemented capability

| progress | Current code samples the full future window from the active segment. |
|---|---|
| bottleneck | A finite clip can preview a terminal hold before the spatial exit guard fires. Causality is untested. |
| next step | Finish the unchanged Study017 probe, then use its phase/hold trace to choose a bounded intervention. |

## Evidence and scope

- **Source:** [OGMP v3, II-1 and III-B](https://arxiv.org/html/2403.04205v3).
  Its oracle generates finite-horizon state references from current state;
  its model-based examples use simplified dynamics and preview/LQR control.
  This motivates examining horizon construction. It does not validate a
  constant-speed guard predictor or our GMT joint-space transitions.
- **Local code:** `ComposedReference.command` evaluates current-state guards,
  then samples every future offset from the selected segment. No future
  state-machine rollout or cross-stage preview is implemented.
- **Hypothesis:** compatible upcoming-stage references may reduce anticipatory
  stopping or abrupt handovers. Neither the source nor Study016 proves this.

## Smallest competing designs

| Candidate | Changed artifact | Main risk | Discriminating evidence |
|---|---|---|---|
| Donor continuation | New immutable reference with a longer recorded tail | Different exit gait/speed; still not closed-loop preview | Same prefix, first changed future row, hold exposure and actual guard timing |
| State-predicted preview | Oracle window crosses an estimated spatial guard | Prediction error; incompatible joint/contact phases | Predicted versus actual crossing, command continuity, contact and tracking errors |

- Keep actor, MDP, task reward, evaluator and budget fixed within either test.
- Preserve actual-state authority for executed transitions. Predicted future
  states are diagnostics, never observations or evidence that a guard fired.
- Require explicit reset/phase/frame/handover rules; do not splice height or
  velocity channels alone and call the result a consistent whole-body motion.
- No implementation, run, training admission or source-family conclusion is
  authorized by this note. Study017's current bundle and gates stay unchanged.
- Existing graph source: `2403.04205`; this note is project interpretation,
  not a second literature record or a reported result from that paper.
