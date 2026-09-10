# Study019: earlier entry rejected

| progress | One feedback-linked oracle proposal was validated and tested against an exact control. |
|---|---|
| bottleneck | The candidate fell at4.34 s; the intended entry-phase shift did not occur. |
| next step | Keep O7b unchanged. Diagnose the saved initial/final018 policies before another training change. |

Internal development record; one seed, no uncertainty estimate, no training.
Independent agent readbacks agree; human scientific review remains pending.

## Decision and measured result

- **Reject.** The only behavior-bearing change was the before→inside progress
  guard:0.65→0.40 m. The candidate also has its own immutable oracle ID.
  The candidate passes3/11 original task gates and1/7 study-screen rows.
- The sole passing screen row is no after-mode revisit; after never executes.
  This is a vacuous guardrail pass, not demonstrated recovery.
- The intended entry phase[1.45,1.80] s and target height<0.50 m both fail.
  The result does not isolate a pure phase effect or establish that phase
  alignment cannot help. Earlier entry also changes the handover state/history.
- No candidate adoption, training admission, threshold revision or guard sweep.
  Study018 B remains stopped. Training totals remain38 runs/2,424,832 transitions;
  no full-task pass has been established.

| Measurement | Exact O7b control | Earlier-entry candidate |
|---|---:|---:|
| Duration / falls / switches |20 s /0 /3 |4.34 s /1 /1 |
| Physical-region compliant samples |56/74 (75.68%) |31/47 (65.96%) |
| First physical-entry action index |150 |125 |
| Entry post-step reference phase |1.18 s |1.10 s |
| Entry post-step target height |0.638942 m |0.633678 m |
| Minimum physical-region height |0.515773 m |0.506630 m |
| Inside mean-speed deviation |0.032602 m/s |0.411346 m/s |
| Overall speed MAE |0.241548 m/s |0.466279 m/s |
| Maximum lateral error |10.939883 m |2.089397 m, early termination |
| Joint / roll-pitch tracking p95 |0.239778 /0.183717 rad |0.292847 /0.383098 rad |
| Original task gates passed |9/11 |3/11 |

- Action indices are zero-based. Physical-region measurements use post-step
  progress in[1,2) m across all visits, not oracle mode membership.
- The smaller candidate lateral maximum is **not improvement evidence**:
  observation ends at4.34 s versus20 s. Whole-episode means also have unequal
  exposure. All correlated samples belong to one episode per arm.
- Candidate executes71 before commands and146 inside commands, no rise/after.
  It crosses the region but does not reach the3.5 m finish.
- Termination occurs at boundary217: torso-up0.459431, with
  `substep_uprightness_failure` and `torso_up_below_0.5`. Zero recorded non-foot
  ground contacts does not contradict this uprightness-defined fall.

## Post-hoc boundary readback

Reconstructed from retained qpos/qvel, reset task frame and command records;
no new rollout. These measurements describe coupled changes, not isolated causes.

| Pre-action boundary | Control | Candidate |
|---|---:|---:|
| Inside switch index |92 |71 |
| Incoming previous-interval forward speed |0.582878 m/s |0.774049 m/s |
| Instantaneous projected qvel |0.589267 m/s |0.778672 m/s |
| Heading at switch |0.386250 rad |0.193204 rad |
| First inside-exit guard satisfaction index |227 |176 |
| Progress / native phase at that boundary |2.050197 m /2.70 s |2.064268 m /2.10 s |
| Actual rise switch index |239 |Not observed |
| Exit deferral |0.24 s, completed |0.82 s observed before termination, censored |

The sealed scorer's exit-deferral block is null when rise never occurs. Null
does not mean zero delay or that the guard was never satisfied. This table
supplies the missing descriptive timing without changing the sealed scorer.
Neither speed excess nor the pending exit is established as the cause of falling.

## Integrity and sequence

- Clean source `9d7396add76122995da192210ebb771a1d5daca1`; executable tree196
  files, `6010bf0ce11bb56c30fe71fdb8b368235f7c5cde867dba8179571dff0ba11f15`.
- Actual Fable proposal passed through `g1 revise`; exact corrected proposal
  bytes, feedback, parent, candidate config and revision receipt are retained.
  Predata repair changed hypothesis labels/causal wording only, not the oracle.
- Fresh control reproduced all three retained015 output files byte-for-byte.
  Its verifier succeeded in a separate call **before** candidate approval/launch.
- The first changed command is index71 (command72):71 complete frame/action
  rows and72 state boundaries match. Other than the guard and oracle ID,
  all config fields remain fixed.
- Initial control-score input used a file digest where the native reservation
  requires a canonical-JSON digest. It failed closed; retained r2 input corrected
  that pin. No source, threshold, rollout or outcome-selected rerun changed.
- Initial revision publication lacked its output parent directory and failed
  before publication. Creating that parent allowed normal publication; no
  candidate or run was overwritten. Original proposal packaging and repair
  records remain retained rather than relabeled as new behavioral trials.
- Root, Fable and independent Sol agree on rejection. Sol reproduced the pair
  score byte-for-byte. Root/Fable each ran71 selected checks before dispatch;
  Sol ran22 focused and66 adjacent checks. Original015/017 scorers are unchanged.
- Both supervisors completed/released. Source freeze ended after independent
  scoring at13:40 UTC. Rescore at the pinned clean source, not a later HEAD.

## Local evidence

- Control: sibling `humanoid-harness-probe-runs/gmt_course_o7b_study019_control_20260907`.
- Candidate: sibling `humanoid-harness-probe-runs/gmt_course_study019_candidate_20260907`.
- Pair inputs/score: `artifacts/gmt/course_configs/study019_pair_score_inputs_20260907.json`
  and `study019_pair_score_20260907.json` in the Astra checkout.
- Revision chain: `artifacts/gmt/revisions/study019_entry_guard_20260907/`.
- Replay: sibling `humanoid-harness-probe-runs/gmt_course_study019_early_entry_recorded_20260907.gif`.
  It renders87 retained states at20 fps; no policy/dynamics rerun. The blue
  region marks a task constraint, not a physical obstacle. It shows a rejection.

| Receipt | SHA-256 |
|---|---|
| Protocol | `3265ca0804475d1144a351cac947fe255fcebb5fe36250cb3e1db7540cf03151` |
| Scorer | `fd7fa96618ce451b0eb42c13e010beeb2c51deb36de680c79e464d33e10dbfb7` |
| Full predata seal | `0ba9161fd35948376a67246abe301f1d0dfa0f50a90d5de90498811f4c902072` |
| Exact corrected proposal | `104b627cb3967a32f8a542de8cd395c94ddbf247f2b306d0500eb2e8f174483c` |
| Revision receipt | `22716278f8189af59a2d8bf152572900f7536089c5607a1196fee2d1e3728f98` |
| Control verification | `da1bb85dbcade607df42a2109f5a81ea03d4b1c99a8724e26d369caa101cb164` |
| Control / candidate manifests | `ec6d8f055ff4d3b9cf44278b491d084b176ee1d424b1bf2c4d044db261aff42e` / `4087ae2de4abfa112633fe216bcd6191e19294ea55f488e08e1ed48aaf8becb2` |
| Pair inputs | `4ce664b8680ad5cc72ff9c7a13979687b279abaceedda5a6940e8b5fce22c1b2` |
| Pair score | `fe8e9865d26add2d041a4af4f446c545a2fd85d34e3f76105d4c5947b4b65c80` |
| Replay | `fc5ff5bbd341f348b38a93997e537eb1e071e85bbe4585e6921fa757461f3566` |

Generated trajectories, checkpoints and private coordination remain local/uncommitted.
