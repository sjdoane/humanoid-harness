# Trainer screen — 6 September 2026

| status | evidence |
|---|---|
| progress | The 1/64 trainer passes the preregistered single-seed screen. |
| bottleneck | Full task still fails: inside speed, posture compliance and depth. |
| next step | Run predetermined seeds 20260907/08 and re-establish the reward contrast under the scaled trainer. |

- Execution: `4ed558eeac27d2c0391d0514fa4d999938d70df5`.
- Both arms: O2/r1, seed 20260906, 32,768 transitions, final checkpoint only.
- Raw control reproduces all nine original artifacts and telemetry byte-for-byte.
- Scaled initial policy and all three zero-residual artifacts match exactly.
- Both effective trainer objects were compared with retained config; base weights unchanged.

| measurement | raw | scaled | screen criterion |
|---|---:|---:|---|
| last-16 explained variance, mean | -0.000049 | 0.923524 | ≥0.5: pass |
| final rollout duration / falls | 20 s / 0 | 20 s / 0 | pass |
| oracle switches | 2 | 2 | pass |
| posture compliance | 0.584615 | 0.548387 | ≥0.534: pass |
| overall speed MAE | 0.306657 m/s | 0.267944 m/s | ≤0.35: pass |
| maximum lateral error | 2.970430 m | 0.525993 m | ≤2.970430: pass |
| joint / roll-pitch p95 | 0.219 / 0.125 rad | 0.218 / 0.158 rad | existing gates: pass |
| full task gate | fail | fail | not earned |

- The lateral task gate changes to pass; inside mean-speed deviation is still
  0.165073 m/s, posture compliance 0.548387, minimum height 0.515190 m.
- Training episodes/falls: raw 45/18; scaled 34/2. These are descriptive learning
  paths, not equally sized episode cohorts or generalization evidence.
- All 64 distinct update indices were checked. Boundary records contain
  `previous_update`; the last record contains `update`. Read both, exactly once.
- Value-loss magnitudes have different units and are not compared.
- Fable independently reproduced parity and the screen verdict. This is one
  seed and one fixed start, not a robust trainer or task-success claim.

## Receipts

| artifact | SHA-256 |
|---|---|
| raw manifest | `c22c2f6da8b629e6b18910edbf163ed174a7d35246d0515d83d600150b1828ba` |
| raw resource receipt | `5bdb1c541641d76464be45f473aeac185a3480160d58742e474dd457a5262fb3` |
| scaled manifest | `5fd0e92e58a75fe77853bb06a455e6f722e9f968352e1c184edbcdc748a9cc17` |
| scaled resource receipt | `84861997a64e8fb0d8d5bddcf788c61e0e29f56749e47ac23db1dcb7cefdecdd` |
| scaled telemetry | `7e7730756799b3a7ff3f3ceb6858be7f8876471f6fbd69cddd2f70c576abacb7` |
| verified scaled feedback | `cda7bc397542fb1548e0d27ce37816eac1d0653dee5269dc021f23489f4fc82a` |

- Local directories: sibling `humanoid-harness-probe-runs/`,
  `gmt_course_o2r1_scale_source_control_20260906` and
  `gmt_course_o2r1_scale64_train_20260906`.
- One scaled dispatch was refused before spawn because an authority JSON
  roundtrip changed `ent_coef: 0.0` to `0`. No output or transition resulted.
  The corrected request and reservation preserve Python canonical numeric bytes.
- Initial exploratory scoring read only the final update; its count of one was
  rejected. The reported mean uses all 64 updates, with 16 in the final block.

## Independent oracle probe

- O4 substitutes `basic_walk` for the walking primitive, with entry phase ≤15 s.
  Crouch crop, state guards, reward, base controller and MDP remain fixed.
- Zero residual: 83.1% posture compliance and 0.702 m/s inside speed, but falls
  after 5.4 s. Roll/pitch p95 is 0.447 rad; full-horizon admission fails.
- Reject O4 as a ready-to-train reference. Diagnose its exit transition first.
- The shorter rollout's lower lateral maximum is not a full-horizon improvement.
- Manifest: `a014cd4545a1ec4e4955ba4890b869b0720987785cb13aad11c968c161b22ff1`.
