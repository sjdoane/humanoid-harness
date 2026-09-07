# G1 `basic_walk` native-window audit

| status | current evidence |
|---|---|
| progress | All 64,239 native-frame windows from 18 through 74 intervals were compared on reference-only yaw, speed, height, and seam metrics. Three fixed nondominated tradeoff points are retained below. |
| bottleneck | No retained window combines the current phase-zero entry, near-zero net yaw, and low pose plus velocity seams. The two useful cycles start near 37 s and are already moving. |
| next step | Do not change the locked learning-rate study or first O5 zero probe. If a later oracle-only revision is declared, test the balanced window with an explicit entry treatment before making any tracking claim. |

## Decision boundary

- This is exploratory **reference numeric fitness**, not policy evaluation.
- No simulator, actor, dynamics, contact, stability, or task-success result was
  used to select or score a window.
- The three rows are nondominated over the twelve recorded objectives. With
  twelve objectives, nondominance is a weak filter rather than a scalar rank.
- `1118:1151` is the best retained seam balance, not a demonstrated feasible
  controller target. `1119:1148` trades worse seams for lower net yaw.
  `0:18` is an entry-preserving control with a poor repeat seam.
- This note does not authorize an O5/007 configuration change.

## Source identity

GMT upstream is pinned at
[`2a590de25a1eb08e47491977a738549c22f16e1f`](https://github.com/zixuan417/humanoid-general-motion-tracking/tree/2a590de25a1eb08e47491977a738549c22f16e1f).
The local analysis base is `fd1c559`; the resulting Git commit pins this note
and the analysis script.

| admitted motion | original pinned SHA-256 | converted numeric NPZ SHA-256 | frames | supplied FPS (`float64`) | runtime FPS (`float32`) |
|---|---|---|---:|---:|---:|
| `basic_walk` | `9e60b415c56edf163ea657dcf2120238c202dd82fb6b993387239d1987266a5d` | `b6ee3143e61b308daebb2a1f07d8b420459a2d19cbde76ecc8f4d42affcdb7b1` | 1,173 | 29.980814325303772 | 29.98081398010254 |
| `walk_stand` | `13610cc7bca2fca0d0c5b2ad7bcbca61538556bdd82f2645b39980d1eeee7b75` | `908ba0e4f6acf1ecf829b0ddb73ed7e649ba6e7ca9fa1a96b43a95e0fdc2e3b0` | 222 | 29.932279909706544 | 29.932279586791992 |

Reviewed local numeric semantics at base `fd1c559`:

| source | SHA-256 | role |
|---|---|---|
| `contracts.py` | `f574b2a8266d7b4529525f720ce901faa9924538a80288627c558f750ad92f31` | motion pins, frames, supplied FPS |
| `motions.py` | `1482a73601162f91e4bba1c86561842c94de678871dad84c93b6c9076eb505be` | bounded numeric-only NPZ admission |
| `reference_runtime.py` | `19039aac4223a47c4131131304f266b2e87bcc90647947e933d5cc1ddb2b725f` | finite differences, smoothing, reference features |
| `reference_math.py` | `6f82ce49056fc9beacf1215c817f1f4f0e91733fc2eb056b16bb9fec04c49b0e` | `xyzw` quaternion and local-frame math |
| `composition.py` | `6b0a740828be78c7449a3ffa597996fb10f3221c5fefdb4825d37f5cdd4b2bc8` | segment endpoints and entry-phase matching |

## Method

- Search space: every `basic_walk` interval of 18--74 native frame
  intervals, or 0.600383959--2.468245165 s at supplied FPS. Exactly 64,239
  windows have an available end-boundary frame.
- Endpoint convention: `[i:j)` contributes frames `i..j-1` to window means.
  The diagnostic source seam compares frame `j` with frame `i`.
- Features: root height in metres; roll/pitch in radians; root velocity rotated
  into the quaternion-local frame in m/s; local yaw rate in rad/s; and 23 joint
  positions in radians. Absolute yaw and root `x/y` are not actor features.
- Velocity: float32 first differences multiplied by runtime float32 FPS. The
  final raw difference copies the previous value. Linear, angular, and joint
  velocities then use the pinned 19-frame zero-padded box smoother. Angular
  velocity is formed from `xyzw` quaternion delta/exponential-map first.
- Standing-height anchor: mean root height of the final 30 `walk_stand`
  frames, `0.775490546 m`. This is a reference anchor, not a robot gate.
- Pareto objectives, all minimized: absolute mean yaw rate, mean absolute yaw
  rate, per-frame forward-speed MAE to 0.7 m/s, mean absolute lateral speed,
  mean-height error, height SD, and six source-boundary seam measures: height,
  roll/pitch RMS, joint-pose RMS, local-velocity L2, yaw-rate absolute jump,
  and joint-velocity RMS.

## Three retained numeric candidates

`Yaw/loop` integrates mean local yaw rate over one source-window duration. It
is a local-reference proxy, not a world-heading prediction.

| role; source frames `[i:j)` | supplied-FPS source time; duration | yaw mean / mean absolute; yaw/loop | local `vx` mean / MAE to 0.7; mean abs `vy` | root `z` mean +/- SD |
|---|---|---|---|---|
| balanced cycle `1118:1151` | 37.29051478953356--38.391218714448236 s; 1.1007039249146757 s | `+0.05444 / 0.26848 rad/s`; `+3.433 deg` | `0.67469 / 0.02552; 0.11016 m/s` | `0.769270 +/- 0.006661 m` |
| minimum-net-yaw cycle `1119:1148` | 37.323869453924914--38.291154721274175 s; 0.9672852673492606 s | `-0.00408 / 0.23949 rad/s`; `-0.226 deg` | `0.67231 / 0.02769; 0.10430 m/s` | `0.768799 +/- 0.006889 m` |
| current-entry control `0:18` | 0--0.6003839590443686 s; 0.6003839590443686 s | `-0.13438 / 0.13438 rad/s`; `-4.622 deg` | `0.62880 / 0.09780; 0.09582 m/s` | `0.763873 +/- 0.008962 m` |

| source frames | seam `dz` m | roll/pitch RMS rad | joint-pose RMS rad | local-velocity L2 m/s | yaw-rate jump rad/s | joint-velocity RMS rad/s |
|---|---:|---:|---:|---:|---:|---:|
| `1118:1151` | 0.002185 | 0.004418 | 0.081373 | 0.173709 | 0.000049 | 0.132027 |
| `1119:1148` | 0.009268 | 0.028246 | 0.143075 | 0.229355 | 0.051580 | 0.294426 |
| `0:18` | 0.000322 | 0.018272 | 0.232473 | 0.375208 | 0.009385 | 0.595424 |

The full `basic_walk` reference has mean/mean-absolute local yaw rate
`0.460312 / 0.476757 rad/s`, forward-speed mean `0.784646 m/s` with
`0.160138 m/s` MAE to 0.7, mean absolute lateral speed `0.096336 m/s`, and
root height `0.766781 +/- 0.009466 m`.

## Entry comparison

The current full-clip segment begins at source phase zero. Later controller
transitions can search only the first 15 s, using height, roll/pitch, and joint
pose; the matching score ignores all velocities.

| candidate start | pose distance to `walk_stand` phase zero | local-velocity jump m/s | yaw-rate jump rad/s | joint-velocity RMS jump rad/s | inside current source 0--15 s pool |
|---|---:|---:|---:|---:|---|
| frame 1118 | 0.159880 | 0.620910 | 0.380596 | 0.347030 | no |
| frame 1119 | 0.155971 | 0.624061 | 0.364191 | 0.353686 | no |
| frame 0 | 0.105097 | 0.366509 | 0.152086 | 0.341255 | yes; exact current source phase |

When supplied the `walk_stand` phase-zero pose proxy, the current matcher
selects frame 133, supplied-FPS source time `4.436170364050057 s` and runtime
phase `4.436170578 s`, with score `0.075369709`; phase zero scores
`0.105096510`. This is not an observed robot transition or a warmed-reset
state. It establishes only reference-pose proximity. The two late-cycle starts
have materially larger reference-velocity entry jumps, so a silent crop
replacement would not preserve the current entry condition.

## Supplied time versus configured runtime time

Source times above use the exact supplied float64 FPS for reporting. The
runtime stores FPS as float32. Proposed configuration bounds therefore use
`frame / runtime_float32_fps`, so float32 sampling lands on the intended frame.
This distinction is material for frame 1119: its supplied-FPS start converts to
runtime frame coordinate `1118.999878`, while the runtime-FPS bound maps to
exact coordinate `1119.0`.

| frames | proposed `start_seconds` | proposed exclusive `end_seconds` | runtime endpoint coordinates | maximum runtime/native seam-metric difference |
|---|---:|---:|---|---:|
| `1118:1151` | 37.29051521889921 | 38.39121915648747 | `1118.0 / 1151.0` | `1.79e-7` |
| `1119:1148` | 37.32386988367461 | 38.29115516216127 | `1119.0 / 1148.0` | `2.38e-7` |
| `0:18` | 0.0 | 0.6003839659572324 | `0.0 / 18.0` | `1.12e-8` |

`wrap_within_segment` treats the end as exclusive. Exact
`phase == duration` wraps to the start; a phase immediately below the duration
approaches the end boundary. The 50 Hz controller need not sample that exact
endpoint, so these seam values are continuous source-boundary diagnostics, not
a claim about the exact pair of samples observed at every discrete wrap.

## Reproduction

[`scripts/analyze_gmt_walk_windows.py`](../../../scripts/analyze_gmt_walk_windows.py)
fails closed on the two converted NPZ hashes, replays the declared 18--74-frame
search only to verify that the fixed three rows have no dominator, emits no
other candidates, and performs no policy or simulator call.

```bash
PYTHONPATH=src python scripts/analyze_gmt_walk_windows.py \
  --numeric-root <GMT_NUMERIC_MOTIONS_DIR>
```

The output records all unrounded values, units, source identities, float32
endpoint mappings, and claim limits. Numeric fitness alone does not establish
trackability, contact feasibility, stability, path behavior, or learned task
success.
