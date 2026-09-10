# Lower learning rate: mixed changes, no adoption

| progress | Both locked 131k runs and the 32k compatibility control completed and independently re-scored. |
|---|---|
| bottleneck | Neither lower-rate arm meets the adoption rule or the full task gates. |
| next step | Test the repeatable crouch/rise oracle separately; do not extend this rate sweep. |

## Matched results

- Source `3b99f97bb98151d3b3afa633e83583bff1e981a3`; seed `20260906`.
- Fresh seeded policies, 131,072 transitions, final checkpoint, reward scale `1/64`.
- Each rate contrast holds oracle, reward, assets, MDP, budget and evaluator fixed.
- Controls are the corresponding [study-006](../006_g1_composition_depth/RESULTS.md) cells.
- Initial policy and all three zero-residual artifacts are byte-exact within each contrast.

| oracle / reward | rate | final rollout | switches | inside samples | compliance | minimum height, m | inside speed deviation, m/s | residual RMS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| O2/r1 control | `3e-4` | fall at 4.42 s | 2 | 54 | .593 | .5169 | .2866 | .2965 |
| O2/r1 candidate | `3e-5` | fall at 5.76 s | 2 | 79 | .684 | .5152 | .0163 | .2060 |
| O4b/r4 control | `3e-4` | fall at 6.36 s | 1 | 89 | .674 | .5265 | .0958 | .3371 |
| O4b/r4 candidate | `3e-5` | survives 20 s; backtracks | 2 | 194 | .289 | .5131 | .3948 | .1988 |

- O2/r1: region speed and posture improve on the observed complete crossing.
  Survival fails; residual RMS exceeds `.15`; first-to-last-quarter fall rate rises.
- O4b/r4: the primary prediction passes (survival, two switches, region exit).
  Compliance regresses, so adoption fails. The robot reaches `x=5.91 m`, returns
  through the posture region while in `after`, and ends at `x=2.25 m`.
- All region visits count. Do not substitute first-crossing compliance for the
  complete 194-sample record or treat reaching the finish once as task success.
- Neither arm reaches the required `.50 m` dip; all full-task gates remain fixed.
- Partial-horizon lateral and whole-episode speed errors are not equal-exposure
  comparisons. Missing 10/20 s measurements remain unavailable after a fall.

## Learning diagnostics, not success metrics

| arm / rate | episodes / falls | EV, last 16 updates | falls / episodes, first 64 → last 64 rollouts | raw episode return, first 64 → last 64 |
|---|---|---:|---|---|
| O2/r1 / `3e-4` | 232 / 134 | .770 | 2/34 → 77/92 | 3699.5 → 1353.2 |
| O2/r1 / `3e-5` | 154 / 34 | .789 | 6/36 → 10/39 | 3420.2 → 2904.4 |
| O4b/r4 / `3e-4` | 181 / 71 | .906 | 19/46 → 8/38 | 2412.6 → 2437.9 |
| O4b/r4 / `3e-5` | 215 / 118 | .927 | 39/61 → 24/49 | 1920.6 → 2417.3 |

- Every arm retains 256 updates and eight 32-rollout blocks in the score artifact.
- Episode means are weighted by completed episode count, not means of rollout
  means. Different lengths and censored partial episodes limit interpretation.
- `sb3_n_updates` counts attempted epochs, including KL-stopped partial epochs.
  It does not prove every optimizer minibatch ran. KL is not a hard bound.
- The smaller rate changes optimization and behavior, but does not identify a
  unique optimizer defect or prove reward/oracle design is the sole cause.

## Compatibility and provenance

- The 32k compatibility run reproduces all ten manifest outputs byte-for-byte:
  config, initial/final policies, zero/final trajectories, frame ledgers,
  evaluations and telemetry.
- Independent feedback reconstruction verifies objective signals, source/config,
  frozen trainer/base weights, output hashes and resource receipts for both cells.
- Frozen base state before/after: `3b81bab42253cfe540f76aff06267eca14427d2ed8c3da25136c4e99eabc9a1c`.
- Local run root: sibling `humanoid-harness-probe-runs/`. Large evidence stays local.

| run suffix | manifest SHA-256 | resource receipt SHA-256 |
|---|---|---|
| `o2r1_rate_source_control_20260907` | `ffa33c173b91f5a697861ecc0b88763f9af30e63ac2e522a643050e2e3174cfa` | `ae1d36b120ec2b51dc260be63578e38a6e0fcf39891099b6676e69450c6ffc1b` |
| `o2r1_lr3e5_131072_seed20260906_20260907` | `3f2ea5f917b5136dc1483fb428848830f999fe2db7d2f1f8ed01933889a8158e` | `92a748b0cdfc3b30c4deff989b26d52bd16a93c6fd66fe357daacca522181e20` |
| `o4br4_lr3e5_131072_seed20260906_20260907` | `4a2ccf38d2a7e99d3af92d68a17c9d1e310cc75c2158d54036ddac4ff85fa6b2` | `5ed8e998ca89ab1466b94a2a7360578c90dc5a6cdd2a3729cb3e47f27b3df5b5` |

Run directory names have prefix `gmt_course_`.

| local score in `artifacts/gmt/course_configs/` | SHA-256 |
|---|---|
| `rate_compat_verified_20260907.json` | `55eef6ef0c8c8c5daaad09c1705b2b7ecd58dd4dfe7506554b02293533e394aa` |
| `study007_o2r1_paired_score_20260907.json` | `033c23d13367c071a6fae5ce6a32686a109f4dba357fabf574e202eda1798237` |
| `study007_o4br4_paired_score_20260907.json` | `016955a4f410ba4f1f148a74695a6e14fcc6fa594612247a7c18a132d9d60cb9` |

## Review disposition

- Fable independently checked compatibility, initial/zero parity and outcomes.
- Parent corrected unweighted episode-return means and an overclaim that an
  attempted-epoch count proves complete epochs. Fable accepted both corrections.
- Final interpretation: two local, one-seed rate contrasts; no adoption, no
  full-task pass, no held-out or cross-dataset generalization claim.
