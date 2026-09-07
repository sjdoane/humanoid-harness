# After feedback: containment improves, fixed law rejected

| progress | Matched control reproduced; enabled oracle completed 20 s and all four stages. |
|---|---|
| bottleneck | Heading screen passes, but 5.043 m lateral drift fails the 2.5 m screen. Full task still fails depth and lateral gates. |
| next step | Reject this fixed law for advancement. Test the execution-derived reference separately; no gain sweep or training promotion. |

## Matched result

- Seed `20260906`; residual zero; no training transitions.
- Same actor, task, reward, crops, guards, cadence, initial state and evaluator.
- Intervention: after-only state-derived yaw correction **plus total-rate clamp**.
- Source `08d7338299f83b65e00ffe769e9bcea3a62a333a`; 195-file executable
  tree `4730ba29718e6aea7c16399eee7249dc784d77693ca1571e13fde9c0d1caee07`.
- Fresh control reproduces all three Study015 candidate outputs byte-for-byte.
- Candidate preserves 263 complete frame/action records and 264 state rows;
  first after target changes. All 737 after corrections reconstruct from raw state.

| Measure | Legacy O7b control | After feedback |
|---|---:|---:|
| Duration / falls / switches | 20 s / 0 / 3 | 20 s / 0 / 3 |
| After maximum absolute unwrapped heading | 1.396641 rad | **0.386388 rad** |
| Whole-run maximum lateral error | 10.939883 m | **5.042953 m** |
| Final forward progress | 12.946624 m | 16.872768 m |
| Overall speed MAE | 0.241548 m/s | 0.263187 m/s |
| After speed MAE | 0.214348 m/s | 0.243709 m/s |
| Inside minimum height | 0.515773 m | identical |
| Inside compliant samples | 56 / 74 | identical |
| Joint / roll-pitch tracking p95 | 0.239778 / 0.183717 rad | identical |
| Original task gates | 9 / 11 | 9 / 11 |
| Predeclared feedback screen | comparison only | **FAIL: lateral** |

- P2 heading `<0.5 rad`: pass. P1 lateral `<2.5 m`: fail.
- All screen survival/traversal/tracking conditions pass; no later region revisit.
- Original depth `<=0.50 m` and lateral `<=0.75 m` gates remain unchanged.
- Earlier depth/lateral failures cannot be erased by an after-only intervention.

## Mechanism and limits

- Actual entry is **pre-action qpos row 263**: yaw `0.09244311649786376 rad`,
  lateral `1.4109614858362196 m`. Row 264 is the first resulting state, not entry.
- Entry-relative maximum heading excursion: `0.2939448878927521 rad`.
  Final heading: `0.07008305396366651 rad`; net change from entry `-0.022360 rad`.
- Mean applied correction `-0.444214 rad/s`; mean issued endpoint yaw rate
  `-0.240087 rad/s`; issued integral `-3.538879 rad` over 737 commands.
- Mean full-window saturation fraction: `0.611805`; native-limit exceedance
  fraction `0.213026`. These are window-row averages, not endpoint counts.
- Native endpoint differs from native window row zero on 66 after actions;
  issued endpoint also differs on 66. Both clocks remain unchanged and verified.
- First two seconds: maximum reset-relative heading `0.386388 rad`,
  entry-relative excursion `0.293945 rad`, lateral maximum `2.016057 m`.
- Post-hoc descriptive check: mean native local lateral velocity `-0.002244 m/s`,
  versus actual finite-difference displacement rotated by post-action heading
  `+0.106461 m/s`; actual world lateral velocity averages `+0.246404 m/s`.
  Residual heading alone does not explain the full observed lateral motion.
- This contrast measures the combined intervention, not correction-only
  causality. It does not establish general steering, dataset transfer, recovery,
  learned composition, or a full-task policy.

## Receipts

- Score: `artifacts/gmt/course_configs/study016_after_feedback_score_20260907.json`
  SHA `35be86a65f7507066379f1d219d20f4a4908660521f0cc156c8b1bea6c9ee56d`.
- Score inputs: sibling `study016_r2_score_inputs_20260907.json`.
- Control directory: `gmt_course_o7b_study016_control_r2_20260907`.
  Manifest `ad0c42bc32cdfe0bb9e660e6d1f6cb1b42717f2b7416037f6fecbe75e34993c3`;
  resource `90d68b428a46b3942eea964b24554fbc836475276d8234e640b00a5723452960`;
  reservation `fcee6cbecf69bcacde6390fb51d55458b21662e7cf76591318a8ed2b5265ed79`.
- Candidate directory: `gmt_course_o7b_study016_feedback_r2_20260907`.
  Manifest `0f1aaa53daa261dd0184118eac93dbd8b747ae6384905b669893e88bd8fca6c7`;
  resource `32aec014004b1f73c685db71a04edbf0396367e15d492f72b1a90f18c28568b0`;
  reservation `af3cd7f0f8fc5cebf06503a12185f19dc8e9788bc43da01388e0bd862709a79f`.
- Verified feedback SHA
  `6c981229d8b63455864ce5f304b49c2c57295d8456266fff79c845ccbbd40ea6`.
- Recorded GIF: `gmt_course_study016_after_feedback_recorded_20260907.gif`,
  401 retained-state frames; SHA
  `03a1b7a34dd958ee9661152b9ec1c3f75329dccbc737d818a71ed7560b45eb9b`.
  Rendering reruns no policy or dynamics. Blue region is not a physical obstacle.
- Run/media directories above are under sibling `humanoid-harness-probe-runs/`.
- Fable independently confirmed the screen decision; post-action-only entry
  diagnostics were corrected against the declared pre-action boundary.

## Integration failure retained

- Earlier source `e865a80`: control reproduced; candidate failed on the
  evaluator's strict frame-key allowlist. No candidate trajectory was published.
- Repair `1decb52` wires the explicit runtime into the same scoring arithmetic.
  Regression drives real Gym adapter → rollout → evaluator → JSONL/NPZ → validator.
- Fresh source/requests and both `r2` runs followed. Failed directory was neither
  overwritten nor scored as a negative behavioral trial.
