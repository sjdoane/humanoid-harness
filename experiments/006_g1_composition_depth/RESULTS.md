# G1 composition x depth-reward results

| progress | All five locked jobs completed; every larger arm was independently rescored from retained traces. |
|---|---|
| bottleneck | No full-task pass. Three larger arms fall; the fourth stops before leaving the region. |
| next step | Separate lower-learning-rate testing from a repeatable crouch/rise oracle. |

## Fixed comparison

- Execution: `48ce56f6ba3a99c1404165534407d5e6f0862959`; seed `20260906`.
- Same G1 plant, base weights, tracking reward, task, evaluator and trainer.
  Reward scale `1/64`; learning rate `3e-4`; final checkpoint only.
- Protocol: [PROTOCOL.md](PROTOCOL.md). No arm dropped or parameter changed
  after the pilot. These are development results, not held-out evidence.
- Cumulative G1 training: 29 runs, 1,343,488 transitions. No full task pass.

| arm | transitions | final behavior | switches | inside compliance | min inside height | inside mean-speed deviation | overall speed MAE | lateral max |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| O4b/r1 pilot | 32,768 | 20 s; stops at 1.83 m | 1 | 7.2% | 0.502 m | 0.600 m/s | 0.569 m/s | 0.597 m |
| O2/r1 | 131,072 | falls at 4.42 s | 2 | 59.3% | 0.517 m | 0.287 m/s | 0.365 m/s | 1.988 m* |
| O4b/r1 | 131,072 | 20 s; stops at 1.11 m | 1 | 0.0% | 0.671 m | 0.642 m/s | 0.608 m/s | 1.732 m |
| O2/r4 | 131,072 | falls at 4.08 s | 2 | 47.9% | 0.504 m | 0.377 m/s | 0.439 m/s | 0.425 m* |
| O4b/r4 | 131,072 | falls at 6.36 s; exit guard never fires | 1 | 67.4% | 0.526 m | 0.096 m/s | 0.562 m/s | 4.275 m* |

- `*`: shorter exposure; not a full-horizon improvement. Missing 10/20 s
  observations remain unavailable. Physical region exit is not oracle exit.
- O4b/r4 traverses the region and reaches the finish distance, but stays in
  the crouch state and falls. Neither traversal nor a good inside-speed mean
  overrides the other gates.
- Explained variance, last 16 of 256 updates: O2/r1 `0.770`, O4b/r1 `0.601`,
  O2/r4 `0.981`, O4b/r4 `0.906`. All meet the diagnostic threshold; none
  demonstrates successful behavior.

## Predictions, including failures

- Oracle at r1: compliance and inside-speed predictions fail because O4b
  stops. At r4 both numeric predictions pass; both region windows finish
  before the falls. This does not establish a robust interaction.
- Reward at O2: height drops `0.01294 m`, below the declared `0.02 m`; compliance
  worsens. Both predictions fail.
- Reward at O4b: height drops `0.14458 m` and compliance rises `0.67416`.
  Both numeric predictions pass. The control stands at the region edge;
  this is a stall-to-traversal contrast, not isolated evidence of deeper
  crouching during a matched traversal. Neither reaches the `0.50 m` gate.
- O4b/r1 beats its zero-residual overall MAE/lateral predictions by stopping.
  Report the numbers; do not credit them as task improvement.
- Fable predicted O4b/r4 would stop: that prediction fails. It moves and falls.

## Mechanism evidence

- Same-oracle initial policy and zero-residual trajectory identities match.
  Within the same recipe, zero-residual frames/evaluations also match.
- Recomputed every r4 reward frame: O2 `1,204`, O4b `1,318`. The depth term
  acts on 58/48 and 71/89 zero/final inside frames respectively. Outside/fall
  components retain r1 semantics. The multiplier was verified numerically,
  not inferred from a reward being below its weighted maximum.
- The 131k O2/r1 and O4b/r1 paths reproduce their respective first 64 rollout
  boundaries from 32k. They are longer executions of the same seeded path,
  not independent replications or optimizer-resume artifacts.
- O2/r1 residual RMS rises from `0.106` to `0.297`; final survival regresses.
  O4b/r1 RMS rises from `0.133` to `0.350`; stopping worsens.
- A finite crouch crop can reach its standing terminal hold before spatial
  exit. The O4b/r4 trajectory reaches the distance guard but not the `0.72 m`
  height condition. Zero-residual admission had only about `0.001 m` margin.
- These observations motivate optimization and reference-continuation tests.
  They do not identify a specific PPO bug or prove one reward family impossible.

## Receipts

Local runs are in sibling `humanoid-harness-probe-runs/`; large assets remain
untracked. Every larger directory follows
`gmt_course_{arm}_scale64_131072_seed20260906_20260907/`.

| arm | run manifest SHA-256 | resource receipt SHA-256 |
|---|---|---|
| O2/r1 | `d6dff70fa63bdf13e5297bc868abf7afca83a8477f27261a100909728ae3499d` | `36332722d5e77b520b51251c278e29e300a82eb87315549f8489d741a1e8e99e` |
| O4b/r1 | `49f1c002df9bff4d475b1b378f74a3bf02de7aaf9a8eee7202fd20cfcbd5dbed` | `9ad25d46209c170e3123c941756c3d6bc777b10870891ad665e0993f946c9175` |
| O2/r4 | `e9d8007864d79c16727afbd92f5b62e5f20a2355a827cb370c3457bcaae9beac` | `6d7f9e53d58904e89b3e9296a0c09d37b94545ff220d255f9dac212799179623` |
| O4b/r4 | `ced939ad10815995d317e4fe96cf254e63a6546d0a4ba4e4c9b316f0a810549d` | `9751930c00a1ef993b503c3440e338fa9294530899ad6e6168e38568d9538320` |

- Paired scorer: reviewed `0b8d6cd` + `d90f562`, integrated as `a538464` +
  `7cc2779`. Pins resource receipts and reconciles episode/fall totals against
  validated telemetry. Failed policies remain present.
- Local paired score artifact:
  `artifacts/gmt/course_configs/study006_paired_scores_verified_20260907.json`,
  SHA-256 `48e48a97f799d6517a324d42069a9d172d1b6811c697cb34f32c5fa334ac5e55`.
- Independent Fable review: mailbox `20260907T003820.924905Z-565bb05e069b4a51b8f6adbfe6c9d729`.
  Its omitted O4b numeric contrast and mistaken two-fall count are corrected
  above; the retained raw review is unchanged.
