# O5 repeatable crouch: rejected feasibility candidate

| progress | Control parity passed. A real feedback-linked LLM oracle revision ran and was independently scored. |
|---|---|
| bottleneck | O5 turned back and fell before its rise guard; rise and after transitions remain untested. |
| next step | Close the reference-input causal-use gate; review an after-only walking crop while preserving the known finite crouch crossing. |

## Locked readout

| Measure | O4b four-state control | O5 candidate |
|---|---:|---:|
| Duration / falls | 20 s / 0 | 6.4 s / 1 |
| State switches | 2 | 1 |
| Maximum forward progress | 11.069 m | 1.897 m |
| Region samples / compliant samples, all visits | 71 / 59 | 147 / 100 |
| Region posture compliance | 83.10% | 68.03% |
| Minimum region root height | 0.5157 m | 0.4944 m |
| Mean region forward speed | 0.7021 m/s | 0.0092 m/s |
| Maximum lateral error | 7.9883 m | 2.3699 m |
| Full unchanged development task | FAIL | FAIL |

- O5 passes only the sample-count prediction. Survival, three switches and
  compliance predictions fail. Rise-entry location and walk re-entry distance
  are unobserved, not successful or zero-valued.
- Different observation durations prevent interpreting lower total lateral
  error as improvement. A depth dip in a failed traversal is not adoption.
- First 92 frame records are byte-identical. O5 enters crouch at decision
  step 92. Its reported phase wraps at recorded steps 201, 249 and 297.
- At those wraps, heading error is 0.732, 1.537 and 2.327 rad; forward speed
  is 0.392, -0.355 and -1.689 m/s. These are observations, not unique causal
  attribution to a seam, yaw rate or controller defect.
- Forward progress never reaches 2.05 m. The rise guard never holds; neither
  the rise segment nor the later walking segment executes. Terminal failure
  is non-foot ground contact at step 320.

## Verified mechanism and limits

- Both probes used clean source `96d46c76c422ed5693701286ed3f8548dedccc9c`;
  source tree `4ede2c2ab2bf1dad11f799ba699a1c14c3b1722f0595ec896c6c54fa2044f446`,
  188 bound files. Each supervisor completed and released its resource slot.
- Control trajectory and frame bytes equal legacy O4b. Its evaluation differs
  only in declared reset-profile metadata; original/new values were checked.
- Timing clarification was recorded at 01:32 UTC, before the 01:33 control:
  local `artifacts/gmt/course_configs/008_control_parity_clarification_20260907.md`;
  peer message `20260907T013209.206977Z-387d301a6d3e43e289d7c422114c30cd`;
  pre-launch acknowledgment `20260907T013303.204926Z-1055af804baf42159a9b5c8d9f645163`.
  The protocol wording correction is documented, not an outcome-driven gate change.
- Fable's raw JSON proposal was retained with one final newline added. The
  actual `g1 revise` command verified parent, feedback, source manifest and
  proposal identities, then changed only `oracle` and `segments`. Parent
  inspection checked every candidate number against the locked specification.
- All output hashes, frame/trajectory crosslinks and independent boundary
  metrics were checked through the full feedback builder. All 320 O5 reported
  phases and post-step reference targets were separately reconstructed exactly.
- This course trace does not retain the actual actor reference window. The
  separate five-arm intervention trace is required for closed-loop causal use.
- No training, held-out evaluation, external motion admission, recovery test,
  physical obstacle test or full-task success is established by this study.

## Reproduction receipts

Paths below are under sibling `humanoid-harness-probe-runs/`.

| Artifact | SHA-256 |
|---|---|
| `gmt_course_o4b_fourstate_probe_20260907/course_run_manifest.json` | `3c6764a695331d94edcb9aae44566cd4d50cb4156e9087f0abce70ee0e9bd461` |
| `gmt_course_o4b_fourstate_feedback_20260907/feedback_v1.json` | `1e314f66504b33d40f009ccca9f2c87bb4a55dfa2c6945ee7acafe82980708e3` |
| `gmt_course_o5_revision_20260907/proposal.input.json` | `2eeb0e05e4f67d49f2287f566bcc70440ebdef45ae9e452b23c6e3b6a2d901d2` |
| `gmt_course_o5_revision_20260907/candidate_config.json` | `31902376416a0f699c393611d1473708e078bcdb42a467af3c8c0a1f5ff83299` |
| `gmt_course_o5_revision_20260907/revision_receipt_v1.json` | `b5814c0129e6ce706b81031ac0d1ee75254bb426295a4eef743e36001f90b659` |
| `gmt_course_o5_probe_20260907/course_run_manifest.json` | `d1c2a202e5dc18dfccefaa4d378d4a4012c81d18bbdf00eb51f317b4464354fd` |
| `gmt_course_o5_probe_20260907/gmt_probe_resource_receipt_v1.json` | `432aedd520b68bbad5fcdb527d88fecc5afb9c555a1afdeeedb57453c7684e98` |
| `gmt_course_o5_feedback_20260907/feedback_v1.json` | `f3c5d9160a989e838696f61a728c628f9a9f14e5fd0d8a8c7f5a809b620d3067` |
| `gmt_course_o5_recorded_failure_20260907.gif` | `8c2e2e4678450c58aa4e41b83415aef5170f54e6145e6b81427b55743fe88cba` |

- GIF: 129 recorded-state frames, 480 x 360, 20 fps. Rendering does not rerun
  dynamics or the policy. It illustrates the failed probe, not learned behavior.
- Raw proposal mailbox body SHA: `f4a5fc81178a14e6df08319d3451c7b61026bbde85fcd1a73960d175ac6eb78b`.
  Its rounded compliance wording does not replace the locked exact 59/71 gate.
