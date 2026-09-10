# Fixed input normalization: retained, not adopted

| progress | Both 131,072-transition jobs completed. The normalized policy survives 20 s and makes both switches. |
|---|---|
| bottleneck | Compliance falls below the locked advancement floor. Three original task gates still fail. |
| next step | Retain the opt-in profile and failed screen; diagnose posture timing/depth and lateral drift before another declared comparison. |

## Matched result

- [Protocol](PROTOCOL.md) committed before training; seed `20260906` only.
- Source: `1e5d84b672f4fce34a9727ab0464be34d696acb6`, frozen through both jobs.
- Only intervention: fixed normalization of the residual learner's first 2,154
  inputs. Task/oracle/reward/base controller and all other PPO settings unchanged.

| Measure | Raw control | Fixed normalizer |
|---|---:|---:|
| Duration / falls | 5.76 s / 1 | 20 s / 0 |
| Oracle switches | 2 | 2 |
| Inside samples / compliant | 79 / 54 | 81 / 52 |
| All-visits compliance | 68.35% | **64.20%** |
| Minimum inside root height | 0.515173 m | 0.513356 m |
| Inside mean-speed deviation | 0.016270 m/s | 0.032732 m/s |
| Overall speed MAE | 0.381304 m/s* | 0.295033 m/s |
| Maximum lateral error | 1.309305 m* | 3.419531 m |
| Joint / roll-pitch p95 | 0.256849 / 0.404168 rad* | 0.229084 / 0.178847 rad |
| Final residual RMS | 0.206030 | 0.191946 |
| Training episodes / falls | 154 / 34 | 151 / 29 |
| Last-16-update explained variance | 0.78885 | 0.35574 |
| Wall time | 330.08 s | 334.47 s |

`*` Unequal exposure: the control falls. Do not call its shorter-horizon
aggregate errors a fair 20 s baseline.

- Candidate passes six of seven advancement requirements. Compliance fails
  the exact `0.6835443037974683` floor. **Retained, not adopted**; no automatic
  seed replication or default change is authorized by this result.
- Original full-task failures: compliance <75%, minimum height >0.50 m,
  lateral maximum >0.75 m. All other development gates pass; no held-out claim.
- Diagnostic predictions also fail: RMS >0.15; late-quarter falls increase
  from 5/36 to 9/39 episodes, and from 5 to 9 per 32,768 transitions.
- All 256 measured normalized-input ranges are finite; maximum 16.62421.
  This is an observed range, not a safety or distribution guarantee.
- Survival establishes that this trained O2 policy can cross the transition.
  It does not establish normalization generality or isolate remaining causes.

## Verification and equal-exposure diagnostics

- Parent and Fable independently confirm all **ten** control outputs exactly
  reproduce the retained pre-integration control, including telemetry/policies.
- Candidate's three zero-residual files are byte-identical to control.
- Initial common trainable tensors match; both saved candidate policies retain
  exact normalizer buffers. Full feedback/resource reconstruction passes.
- Over the common first 288 samples: lateral maximum 1.309305 -> 0.899983 m;
  speed MAE 0.381304 -> 0.258854 m/s. These are descriptive paired prefixes.
- The retained score's extra prefix p95 diagnostics use NumPy linear
  interpolation. Recomputing with the evaluator's nearest-rank convention gives
  joint 0.256849 -> 0.244082 rad; roll-pitch 0.404168 -> 0.242630 rad.
  The score's actual objective gates already use the frozen evaluator unchanged.

## Receipts

Run directories are in sibling `humanoid-harness-probe-runs/`:

- Control: `gmt_course_o2r1_normalizer_control_20260907/`.
  - manifest: `5ad2fa92b167ced6dbda98fd20310f197b26f2742a9a5e14d0fb47fcaf7c7faa`;
  - resource: `18e6c9f0815a76e33726ba73e80c477dae32e5a1b3ab3885ac2e8377a0d55b50`.
- Candidate: `gmt_course_o2r1_fixed_normalizer_20260907/`.
  - manifest: `2a628f53520222f84c458369207a6501ce2ad1629d3e5afa94367ab6f76a2e2d`;
  - resource: `1fa459aa6905acca90f1316f75cea5c54d108f0002002d5e05d17fb815664027`.
- Score: `artifacts/gmt/course_configs/study011_fixed_normalizer_score_20260907.json`;
  SHA-256 `ac8fbbac1b1d556ccd49f3ee79d9592a40ca4cc539ead859108a73146dac504b`.
- Independent review: `20260907T030248.864146Z-a3e093c4d97c4f478c16e4e7673ec710`.
- Recorded-state GIF: sibling `gmt_course_fixed_normalizer_recorded_20260907.gif`;
  SHA-256 `ed07c48409e17dfeb982e8e004f94cde33fe803153830d966004035379a82e21`.
  401 frames; no policy or dynamics rerun.
