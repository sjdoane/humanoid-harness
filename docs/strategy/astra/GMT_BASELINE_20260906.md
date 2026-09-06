# G1 baseline: real supplied-motion tracking

Historical baseline record. Composition and learning results now live in
[`G1_LEARNING_RESULTS_20260906.md`](G1_LEARNING_RESULTS_20260906.md).

| status | evidence |
|---|---|
| progress | One reconstructed GMT controller tracked `walk_stand` for 10 simulated seconds. |
| bottleneck | No novel composition, task-reward learning, or feedback revision yet. |
| next step | Test numerical-reference influence, then state-triggered switching and a fixed residual-learning interface. |

## Measured result

| measure | value |
|---|---|
| simulation wall time | 0.922 s |
| minimum root height | 0.749 m |
| joint RMSE, mean | 0.0880 rad |
| root-height error, mean | 0.0108 m |
| roll/pitch RMSE, mean | 0.0281 rad |
| displacement, world x/y | +4.405 / -2.185 m |
| raw action saturation | 0% |

- Clean execution commit: `3762bde0e5a7167b190d6d785b37b2fb27467db3`.
- One deterministic start; survival is only the declared 0.5 m root-height proxy.
- Recorded-state GIF: `gmt_walk_stand_pathfix_20260906_recorded.gif` in the
  sibling `humanoid-harness-probe-runs` directory. No dynamics rerun or learning.
- The earlier attempt stopped at MuJoCo import: restricted PATH omitted
  `/usr/sbin/sysctl`. Its failed receipt is retained; no robot steps ran.
- This is local evidence for the reconstructed numeric actor, not a proof of
  equivalence to the original executable model or general tracking competence.

## Decision

- Prioritize GMT/G1 for the first full-loop demonstration. Its observed local
  tracking and runtime make it a useful starting point; native Humanoid-v5
  remains a separate family, not an abandoned negative behavioral result.
- Freeze base controller, residual interface, observations, dynamics, trainer,
  tracking reward and evaluator before comparing authored oracle/task rewards.
- First proposed task: ordinary gait → posture-constrained region → ordinary
  gait. No physical obstacle exists; do not call this collision avoidance.
- Score forward progress, lateral/heading error, posture, falls and tracking
  independently of the authored reward. Sideways drift cannot be hidden by x progress.
- Define the crouch failure metric before testing: the supplied crouch clip's
  10th-percentile reference height is 0.480 m, below the walking proxy's 0.5 m.
  A posture-aware floor/contact/uprightness definition is required, not a
  post-result relaxation. The clip's full median is 0.726 m because it includes
  standing; use the actual crouch segment, not its whole-clip median.
- Use state-triggered transitions and explicit phase transfer. Test the
  20-frame future-window discontinuity before adding a learned residual.

## Receipts

Artifacts are in sibling `humanoid-harness-probe-runs/gmt_walk_stand_pathfix_20260906`.

| artifact | SHA-256 |
|---|---|
| resource receipt | `6726fd2e566cfb7f5bbf75b6829dec670b3edb7e4a2304f9f324add3a94424c4` |
| replay manifest | `f5806c25f7f78d5ca9d05b1afb5914561a7a6ef20ea1b9036165d165296db730` |
| numeric trace | `216036fd098b426090c6205d999182c03e1e9f7af4374de8874af04bde241a02` |
| recorded-state GIF | `605a225a440a947f451e85eed1c32ab0f67e67691b664658415e4336b416616c` |
| GIF receipt | `742cd457875e57b791e828820b3402e8515095c84c805325c64c437f3cee7ae2` |

- Fable exact acceptance: `20260906T191214.181548Z-01e5a21f13ab405f8e2f048769b15d4a`.
- Task/switch critique: `20260906T190445.385573Z-fec002dbf8f34b83b2e2808e68cdc696`.
