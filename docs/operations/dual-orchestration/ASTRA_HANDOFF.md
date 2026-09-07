# Astra orchestration handoff

| progress | Study 021 predata oracle-exit protocol committed; scorer building in a separate worktree. Training total unchanged: 38 runs / 2,424,832 transitions. |
|---|---|
| bottleneck | No full-task pass. Trained policy falls in 11/16 paired tests versus 5/16 initially. |
| next step | Finish and independently review the Study 021 scorer, then seal before any new native run. |

## Current checkpoint: 2026-09-07 16:00 UTC

- Continuity resumed from clean `fc9fff5`. No competing active user turn,
  training process or claimed heavy slot. Root reacquired only its own lease.
- [Study 021 protocol](../../../experiments/021_g1_immediate_crouch_exit/PROTOCOL.md)
  starts at `de88205`: remove only the crouch exit-boundary restriction and
  assign a new oracle ID. Two zero-residual episodes; no training. Existing
  runtime supports the factor; no controller/MDP/evaluator change.
- Sol `derived_admission` owns scorer/tests only in `humanoid-harness-study021`,
  branch `astra/study021-immediate-exit`; no dynamics or actor execution allowed.
  Sol `four_state_training` and Fable independently review the predata design.
  Root owns protocol, source integration and retained boundary checks.
- Root's oracle-only retained-state precheck:
  `.orchestration/study021_boundary_precheck_v2_20260907.json`, SHA
  `7522bdf70ddf86e4b809d6900f917694a4a83edf2862a4d21f49c4aa935d692e`.
  It verifies 227 identical numeric reference commands and one expected
  identity-only transition field difference at action 92. It is not a candidate
  rollout and does not establish action or plant-state parity.
- Prefix correction: removing the exit flag changes the crouch segment hash
  before its numeric effect. Validate exact admitted hashes at frame 92;
  every other prefix field remains exact. Candidate guard/switch prediction 227;
  retained switch 239. Verify fresh control before candidate approval/dispatch.
- Pre-action phase correction: guard clock 2.70 s; control switch clock 2.94 s
  wraps to reported 1.08 s. Fable acknowledged the correction. The precheck v2
  also fixes a draft mixed-feature pose norm to declared pose columns/scales;
  original v1 receipt retained but superseded.
- Fable proposal remains held until protocol/scorer lock; use retained O7b
  feedback `6ea27ad8…7c795`. No Study 021 candidate outcome or dynamics yet.
  A deterministic screen pass would not admit training or prove sampled robustness.

## Earlier checkpoint: 2026-09-07 15:05 UTC

- [Study 020 results](../../../experiments/020_g1_saved_policy_diagnostic/RESULTS.md):
  34 episodes, 139 declared outputs, zero training, no full-task pass. One reset,
  one training seed and 16 paired noise sequences; not training replications.
- Run and scoring source `615124f018e044a74b8ec564de0207237c7593fa`.
  Manifest `2c6962c519eea04d458e0f3f7dbd582ff535eb5e3bda63a92d90bcda1f90f289`;
  resource `3260b939b71bdce6ec895b057539f8acfa6d1bd23d68289bd9b1fb0f5f4e2559`.
  Root and Sol diagnosis SHA
  `0a58b19ceb71fce8ee8dc301ab5e59a9170e4f312363c900ec8fb49cc24bece0`;
  Fable independently agrees with primary numbers and sampling arithmetic.
- Legacy parity and both exact deterministic triplets verify. Source freeze
  ended after independent scoring; worker terminal, heavy slot released.
  Current slice is results/figure only; no next candidate or run launched.
- Selected figure: local `artifacts/gmt/course_configs/study020_figures_20260907/paired_outcomes_final.png`.
  All pairs visible with durations. Local development draft; human review pending.
- Next priority: a small oracle-only test of the crouch exit-boundary rule,
  before more trainer work. Require new protocol, exact control, one candidate,
  handover pose/speed/height measurements, unchanged gates and resource approval.
  Earlier exit may create a discontinuity. Smaller-noise evaluation remains a
  separate rival; neither is admitted or authorized by this checkpoint.
- Existing `course_config.py` admits an omitted `exit_at_loop_boundary` key
  as false; `composition.py:allows_ordinary_transition` then permits ordinary
  guards each command. The probe needs no new transition runtime feature.
  Fable's review `20260907T150425.805865Z-92b3fc9d995a4cb9a37343e1a5a6c6a9`
  supplies proposed prefix/guard definitions; verify them before freezing the
  next scorer. A deterministic survival result would not establish sampled
  robustness or justify immediate training.
- Correct Fable's first interpretation: three final falls occur inside before
  any rise; not every fall follows the rise handover. Later review corrected it.
  No causal speed/deferral claim. Fable remains review/ideation only.

## Earlier checkpoint: 2026-09-07 14:37 UTC

- Study020 runner, strict policy loading and independent action arithmetic are
  implemented; no020 simulation or training has run.95 focused tests pass.
- Builder's clean config/validator and supervisor commits imported as
  `a456429` and `e7e34c0`. Root owns the runtime and diagnosis. Builder stopped
  and released its separate worktree lease; Fable reviews/ideates only.
- Predata review repairs: complete parity receipt identity; regenerate exact
  canonical noise before environment construction; retain sampled mean/std/raw
  action sidecars and independently reconstruct every executed residual.
- Fixed input config `artifacts/gmt/course_configs/study020_inputs_20260907/diagnostic_config.json`:
  SHA `b7b33a0976bc29e61ca9bcbdd9265f9eb253fd49f7542ef96bcbca2a6e7412ab`.
  Noise SHA `61627836220ee63162812035555b3922a3e378ce4fc617d759c438e76ad9e9fe`.
- Next: final independent source review; native approval for one legacy O7b
  replay and020's34 episodes. Verify legacy parity in a separate completed call,
  then020 proves both exact018 deterministic triplets before32 sampled episodes.
  Freeze source through completed independent scoring. No automatic adoption.

## Earlier checkpoint: 2026-09-07 14:11 UTC

- [Study020 protocol](../../../experiments/020_g1_saved_policy_diagnostic/PROTOCOL.md)
  committed `1664deb`; Fable accepts scientific scope. No020 evaluation yet.
- Sol builder `derived_admission` initially owned saved-policy config/runtime and native
  dispatch in separate `humanoid-harness-study020`, branch
  `astra/study020-saved-policy`, from `1664deb`. No training/dynamics authorized
  before reviewed handoff, sealed inputs and exact resource approval.
- Root owns020 `diagnosis.py` and tests. Core helpers recheck frames, raw-state
  metrics, rewards, transition markers, all16 pairs and censored exposure.
  Independent reviewer accepts core fixes; full CLI acceptance waits on reuse
  of the worker's exact ledger/noise/deterministic-parity artifact validator.
- Both retained018 traces reproduce numerically through the new analysis:
  returns3863.808886/3488.089072; handovers92/239/263 and97/244/268. No rollout.
  Existing geometry/feedback imports Torch/SB3 indirectly, but no MuJoCo or
  environment/actor instance is used by this numeric readback.
- Planned worker labels: `initial_deterministic`, `final_deterministic`, then
  `sample_{seed}_{initial|final}`. Manifest `saved_policy_diagnostic_manifest.json`.
  Native receipt key: `artifacts.saved_policy_diagnostic_manifest`.
- Next: complete/import reviewed worker; bind exact prepared noise/config and
  native approval; prove both deterministic trajectories before any sampled data.

## Earlier checkpoint: 2026-09-07 13:40 UTC

- [Study019 results](../../../experiments/019_g1_crouch_entry_alignment/RESULTS.md):
  both supervisors terminal/released at clean `9d7396a`. Exact control and
  prefix71/72 verify; candidate3/11 gates,31/47 compliance,4.34 s fall.
- Root, Fable and Sol reproduce pair score `fe8e9865…b65c80`. Source freeze
  ended after independent review. All prior artifacts/scorer remain pinned.
- Recorded-state GIF `gmt_course_study019_early_entry_recorded_20260907.gif`
  is in sibling probe-runs (87 frames); it shows the rejected candidate.
- UI at8766 now selects019 as FAIL3/11; HTTP readback verifies the new row.
  Registry SHA `60100eec0bba6059358f7cd7615d42a5eb801c24ebb6ce7edf61f9745ad35d5d`.
  Prior12-run selection archived locally; no run deleted. Fresh browser QA
  is unavailable because the Mac is locked; the GIF's first frame was inspected.
- Post-hoc readback supplies first exit-guard satisfaction at boundary176;
  no rise switch occurs. The null scorer block is not a zero-delay claim.
- Next diagnostic is16 paired stochastic evaluations of018's exact initial
  and final policy tensors, with deterministic parity first and no training.
  Design/implementation/resource approval remain pending; no new heavy job.
- No new reward/guard sweep, reference retiming, controller or MDP change.
  Fable reviews and ideates only. No push or main-checkout mutation.

## Earlier checkpoint: 2026-09-07 12:56 UTC

- [Study019 protocol](../../../experiments/019_g1_crouch_entry_alignment/PROTOCOL.md)
  is reviewed at `29219b0`: one native O7b guard change, 0.65 → 0.40 m;
  two zero-residual episodes, no training. Scorer/test handoff remains pending.
- First changed command is zero-based index71:71 shared executed rows and72
  shared state boundaries. Physical-entry phase uses the reconstructed
  post-step target boundary, not the pre-action phase field.
- Lock scorer before Fable's one final feedback-linked proposal. Fresh control
  must verify in a separate successful call before candidate approval/dispatch.
  All11 original task gates remain unchanged. No019 simulation yet.
- [Height diagnostic](../../../experiments/018_g1_four_state_reward_loop/FIGURE.md)
  shows retained states/targets; no dynamics rerun. Numeric/receipt review passes;
  manuscript claim audit remains incomplete and human verification is pending.
- Fable and Astra favor a separately declared paired stochastic initial/final
  policy diagnostic after019. It is not implemented or authorized. On-policy
  training episode shares do not estimate either exact checkpoint's fall rate.
- Study018 B stays stopped. Reward-tail and trainer changes remain separate
  hypotheses, not the automatic next experiment. No heavy job is active.

## Earlier checkpoint: 2026-09-07 12:26 UTC

- [Study018 result](../../../experiments/018_g1_four_state_reward_loop/RESULTS.md):
  131072 transitions at `e57f220`;20 s survival, three switches,7/11 task gates.
  Posture47/71 and speed MAE0.474331 fail admission. No B proposal or launch.
- Root scorer and Fable independently verify the failed result. Source freeze
  ended after scoring. Heavy worker is terminal/released; no training active.
- UI at8766 shows018 with correct FAIL7/11; recorded-state GIF shown to Samuel.
  Local selected registry has12 valid runs,0 full passes. No run was deleted.
- Retained post-step targets exceed the ceiling on15/24 posture failures.
  This motivates a separately declared earlier-entry guard probe, not a claim
  that phase alignment or physical feasibility is solved.
- Fable reviews/ideates only. Consider a later fixed-scale Cauchy reward-tail
  contrast, but do not combine it with an oracle change or lower018 gates.

## Earlier checkpoint: 2026-09-07 11:56 UTC

- Study018 scorer imported as `30d966f` from clean peer `a386fb1`.
  Root independently reran36 focused checks; reviewer ran122 focused/adjacent
  checks and rebuilt retained015 through the actual verifier. Ruff passes.
- The scorer checks its own imported package, clean commit and executable
  tree against the run; criteria and imported experiment helpers are pinned.
  No blocking review finding remains. No018 training has started.
- Builder handed off its committed files, then its terminal turn reported a
  provider policy flag. No retry of that request; independent file/test review
  above, not the worker's terminal status, supports integration.
- After an execution gap, Astra reviewed the free heavy slot, absent training
  processes and clean worktree, then reacquired only its own expired lease.
- Next: seal this reviewed source and completed legacy-parity receipt; obtain
  exact native approval for A. Keep source fixed through scoring and conditional B.

## Earlier checkpoint: 2026-09-07 09:34 UTC

- Study018 pure admission/pair criteria committed as `a089491`;19 tests pass.
  Signed-heading logic also runs against the retained015 recorded trace.
  Wrapper and negative-path tests remain under construction/review; no A run.
- Root review requests: pin imported scoring dependencies, require every
  reconstructed A gate to pass at pair time, and use the actual normalizer API.
  These are predata closure items, not changed scientific thresholds.
- Retained015 after-mode lateral reward is below1e-6 in454/737 samples.
  Fable received this descriptive saturation caveat; it is not a new trial,
  a claim about PPO gradients, or authority to change018 reward formulas.

## Earlier checkpoint: 2026-09-07 09:20 UTC

- Four-state prerequisite imported as `4aa3385`, `7017f58`, `834c72d`,
  `aef3d7d`. Both independent review and parent inspection pass after the
  legacy/v4 feedback repair and shared scorer runtime/telemetry plumbing.
- Root GMT+feedback suite:592 passed, no skips;152 focused tests and Ruff pass.
- New source tree:196 files,
  `6010bf0ce11bb56c30fe71fdb8b368235f7c5cde867dba8179571dff0ba11f15`.
  Fresh legacy O7b probe at `aef3d7d` reproduces all three retained015 outputs
  byte-exact. Receipt `.orchestration/study018_legacy_parity_20260907.json`;
  manifest `923236b692259d5e06161f76d54adf657ff1fc6aa42af9f43ed9495aae83a27c`.
- [Study018 protocol](../../../experiments/018_g1_four_state_reward_loop/PROTOCOL.md)
  is predata. Exact baseline config SHA
  `a9e52ba706e4ddfd394c757cf525bf210d553906430c7f08e11e0495affdf289`.
  A/B use native O7b, schema5,2172D, trainer3, seed20260906,131072 transitions.
  B is conditional on A's substrate/headroom gates, then one feedback-linked
  bounded lateral/heading weight revision. No A/B training started.
- Sol `study017_scorer` now builds018 scorer/tests in the separate
  `humanoid-harness-study018-scorer` worktree. Require exact clean handoff and
  independent review before sealing A. No duplicate builder.
- UI restarted as session48274 at8766; HTTP and actual browser show12 valid
  selected runs,0 full-task passes, including017. Previous registry retained.
  Study017 donor comparison confirms49.2 cm phase-aligned lag by2.10 s.
- Fable remains reviewer/ideator only. No push or main-checkout change.

## Earlier checkpoint: 2026-09-07 09:03 UTC

- [Study017 result](../../../experiments/017_g1_execution_derived_reference/RESULTS.md):
  sealed pair at `d189d5a`, both terminal/released. Score SHA
  `8baebd4bd8ae2566c4460ab769dc23e331b51bad1cd995adbda8914815d2632f`.
  Full control parity and prefix92/93 verify; feasibility fails depth28.0 mm
  above donor against20 mm tolerance. Original8/11 gates; no training admission.
- Dispatch preceded completion of the control verifier after its temp-path
  error. Exact parity established afterward by root and Fable; retain the
  deviation, no outcome-selected repeat. Verification/launch are separate calls.
- Recorded-state GIF: sibling `gmt_course_study017_derived_recorded_20260907.gif`,
  401 frames, no actor/dynamics rerun. Source freeze ended after sealed scoring.
- Four-state training prerequisite delivered clean `b3cace5` + `6c4244a` in
  `humanoid-harness-four-state-training`; root read both diffs and tests,
  independent Sol review in progress. Not imported and no new training yet.
- Root strategy question to Fable: native O7b fixed-oracle reward-learning pair
  versus donor-continuation contrast. Do not promote the failed derived bundle
  or silently combine a new reward, reference and runtime.

## Earlier checkpoint: 2026-09-07 08:35 UTC

- Exact derived admission imported as `05d0808` from clean peer `465c9c4`.
  Parent: 117 adjacent tests plus 33 launcher tests pass; Ruff clean. The
  actual017 config loads and binds both archive and conversion manifest.
  This remains probe-only, not dynamics certification or training admission.
- Candidate config SHA `337b5f2c540ce364a45446431a22085c63ebf89b5284e5fa26bae606ffbb20db`.
  Executable tree `5e516691bfe5c090ff0625c6b13d67a66265a367c2b2000f6e41b61d98877738`,
  196 files. No017 run yet; scorer still builds separately. Final source and
  exact native resource acceptances remain required before dispatch.
- Isolated four-state training prerequisite builds a new finite-horizon
  profile and 2172D telemetry; old profiles unchanged. No import during the
  Study017 source freeze; no training or derived-reference promotion yet.
- [Transition-preview note](../../strategy/astra/TRANSITION_PREVIEW_20260907.md)
  records an unimplemented, source-bounded hypothesis for later diagnosis.

## Earlier checkpoint: 2026-09-07 08:21 UTC

- Feedback-to-revision repair now accepts the exact heading-validation receipt
  only for its declared runtime. The retained016 rollout passes the full
  feedback rebuild → proposal → candidate publication path; 76 focused tests
  pass. The test proposal is a software fixture, not a new scientific trial.
- Study017 is still predata. Its protocol now records early terminal-hold
  lookahead and the exact endpoint: donor boundary197, not198. The relative
  endpoint is about 0.000618 m short of the unchanged exit guard. One bounded
  bundle screen remains planned; no new reference extension or training.
- The app restart ended prior subagents. Their assigned worktrees are preserved;
  replacement Sol/max workers finish the admission and scorer slices there.
  No heavy job is active; clean handoffs and exact resource approval remain.

## Earlier checkpoint: 2026-09-07 08:00 UTC

- [Study016 result](../../../experiments/016_g1_after_feedback/RESULTS.md):
  r2 pair at `08d7338` terminal/released; source freeze ended. Control reproduces
  all three retained outputs, candidate prefix exact. Heading 0.386 rad passes,
  lateral 5.043 m fails the 2.5 m screen. Reject fixed law; no training admission.
- Earlier failed source `e865a80` lacked evaluator trace-key plumbing; outputs
  were never published. Failure retained, not a behavioral result. `1decb52`
  repairs the exact runtime-aware evaluator and adds real rollout publication test.
- Converter imported selectively as `61f9b3c` + `541b82b`. Actual generation used
  independently reviewed clean `e889f0c` in the converter checkout. Numeric
  candidate SHA `885e4c1b324a9b41a4d176ec5fee9e4bc634226d5ade46634111cb1c3204c259`;
  manifest SHA `1c7edb409579d757ef6657378ab605e6aa013cf60b552630beb914728bec602c`.
  Path: `artifacts/gmt/derived_references/study012_inside_passage_v1.npz`.
  Source is kinematic-only and unadmitted/untracked. Sol builds exact probe-only
  admission in a separate worktree from `08d7338`; do not import dirty changes.
- Recorded016 GIF is in sibling probe-runs, named
  `gmt_course_study016_after_feedback_recorded_20260907.gif` (401 frames, no
  policy/dynamics rerun). Shown to Samuel. UI restarted as session26510 at8766;
  registry now includes015 and016. All12 rows validate, zero full-task passes.
  Registry SHA `d4fd723fe27e406936ebc04ce76f8c53d0d07609930f55eb9978248fd9149db7`.
  Earlier selection archived; no run deleted. HTTP QA only; Mac was locked.
- Reviewed phase/exposure/fallback cores remain unimported and unwired. Main
  remains separate; no push. Live mailbox/active-run supersede older sections.

## Earlier checkpoint: 2026-09-07 07:00 UTC

- [Study015 result](../../../experiments/015_g1_task_aligned_after/RESULTS.md):
  both workers terminal/released; source freeze ended. Full control and prefix
  parity verified. Candidate has no revisit but 10.94 m drift; reject adoption.
- FK audit selectively imported as `c303741` + `76fdd69`; byte reproduction
  passes. See G1 learning results for the static/dynamic claim boundary.
- [Study016 protocol](../../../experiments/016_g1_after_feedback/PROTOCOL.md):
  after-only measured heading/lateral feedback, fixed gains, zero training.
  Isolated Sol builder owns the implementation; parent owns protocol/scoring
  integration. No Study016 simulation or resource reservation yet.
- Reviewed exposure accumulator `1c0d52e` remains unwired/unimported, as do
  phase-rate and fallback patches. Do not merge ancestors incidentally.
- Candidate GIF: sibling `gmt_course_o7b_recorded_20260907.gif`, rendered from
  401 recorded qpos frames, not trained behavior. UI registry includes015;
  all12 selected runs validate, zero task passes. Previous selection archived.
  Registry SHA `a55abbe8fd91ad90f6c76cddc7c7f7c6bfd1432579c3dc6a69daf0b5daa39ca2`.
  HTTP verified; Mac locked, so fresh interactive browser QA remains unavailable.
- Main/Fable checkout remains separate; no push. Live mailbox and ignored
  active-run record govern current work; older checkpoints below are historical.

## Earlier checkpoint: 2026-09-07 05:56 UTC

- [Study014 result](../../../experiments/014_g1_finite_depth_reward/RESULTS.md):
  worker terminal/released; source freeze ended. Reject the depth candidate.
  Exact source `be359a2`; all non-reward parity and reward reconstruction pass.
- [Study015 protocol](../../../experiments/015_g1_task_aligned_after/PROTOCOL.md):
  O7b proposal admitted, no simulation yet. Reproduce O7 control's three files
  at new source, then one after-only crop probe. Scorer under isolated build.
- Oracle task-geometry mismatch takes priority over a new reward sweep,
  the phase-rate scheduler and structured-feature training. No mixed factors.
- Source-only phase and fallback patches remain unimported. Fable available.
- Baseline recorded GIF: sibling `gmt_course_finite_baseline_recorded_20260907.gif`.
  UI now port8766/session24493 with Study012/014 and O7 included. All 12 selected
  rows revalidate; zero task passes. Previous registry archived, no run deleted.
  Registry SHA `c67f95e8cc613de7cd8781fe3201a4b336da89c428e2e16829a52c64e66c2b7c`.
  HTTP readback verified; no fresh interactive browser QA. No push.

## Earlier checkpoint: 2026-09-07 05:32 UTC

- [Study012 result](../../../experiments/012_g1_finite_horizon/RESULTS.md):
  both jobs terminal/released. Source freeze ended. Correct finite semantics
  retained for new studies, not promoted as a behavioral improvement.
- [Study014 protocol](../../../experiments/014_g1_finite_depth_reward/PROTOCOL.md):
  one reward-only candidate; exact executable-tree equality required to reuse
  baseline. No candidate launched at this checkpoint.
- Source `9bca23b`; no phase/fallback imports before Study014 source freeze.
- Phase core `0e1ae8a` + `667c561` independently accepted, still unwired.
  Fallback reviewer role `eb59aea` separately accepted, still unimported.
- Fable is available; superseded draft and withdrawn old reservations are void.
- UI remains port8766/session79716, registry through Study011. No push.

## Earlier checkpoint: 2026-09-07 04:53 UTC

- [O7 Stage A](../../../experiments/013_g1_phase_rate/STAGE_A_RESULTS.md):
  20 s/no fall, walk/crouch/rise/walk executed. All qualification conditions pass;
  task fails five gates. No repeated crouch loop was executed.
- Every one of 1,000 commands/targets reconstructed; predicted prefix exact.
  O7 worker terminal, resource released, source freeze ended.
- Finite-horizon patch `ae67399` is clean/released in its isolated worktree;
  parent/Fable review before import and the Study012 matched training test.
- Phase-rate pure core `0e1ae8a` is clean/released and under independent review;
  unwired, not imported and no phase simulation authorized.
- Training totals remain 34 runs / 1,900,544 transitions; zero full-task passes.

## Earlier checkpoint: 2026-09-07 03:05 UTC

- [Study 011](../../../experiments/011_g1_fixed_normalization/RESULTS.md): source
  `1e5d84b`, both jobs terminal/successful/released. Source freeze ended.
- Parent and Fable independently verify control's ten-file parity, candidate's
  zero-residual parity, initial tensors, buffer pins and feedback reconstruction.
- Candidate: 20 s/no fall, two switches, 64.20% compliance, 0.513356 m minimum
  height, 0.032732 m/s inside mean-speed deviation, 3.419531 m lateral maximum.
- Failed advancement floor: 68.35% compliance. No seed replication/default
  promotion follows automatically. No new intervention is yet dispatched.
- Recorded-state GIF: sibling
  `humanoid-harness-probe-runs/gmt_course_fixed_normalizer_recorded_20260907.gif`.
- UI restarted as session 79716 at current source; registry includes both
  Study011 arms. Previous selection retained in a local archive.
- Next training is paused for the intrinsic-horizon correction in
  [Study 012](../../../experiments/012_g1_finite_horizon/PROTOCOL.md). Depth
  reward and online phase alignment are separate deferred comparisons.
- Main remains `9188ed0`; no push. Read the live mailbox/active-run before work.

## Earlier checkpoint: 2026-09-07 02:28 UTC

- [Study 009](../../../experiments/009_g1_reference_input/RESULTS.md): two
  five-arm simulations. Every intervention changes the first action and next
  robot boundary; both exact arms reproduce all three positive-control files.
  This establishes input dependence, not composition quality.
- [Study 010](../../../experiments/010_g1_after_reference/RESULTS.md): O6's
  exact unchanged prefix passes; after-only crop falls. Reject, no training.
- Source freeze ended at O6 terminal; exact probe source `1d60c46`, result
  committed `8392cdd`. Heavy-job slot released. No active training.
- Fixed-normalizer builder owns sibling `humanoid-harness-fixed-normalizer`.
  No import until a clean handoff and focused review. Keep old profiles exact.
- A separate read-only audit is checking reward-family incentives; no new
  reward proposal, training, or evaluator edit is authorized by that audit.
- Queryable supplemental reference audit index exists locally at
  `artifacts/knowledge/g1_reference_audits_20260907.db`. Two typed records,
  12 findings; no external motion admitted. Source pins are version-specific.
- UI `http://127.0.0.1:8766/`, server session 5442: 12 registered snapshots
  validated, including O5 and low-learning-rate runs. Not every run is shown.
  Mac locked; no fresh interactive browser QA.
- O6 recorded-state GIF: sibling
  `humanoid-harness-probe-runs/gmt_course_o6_recorded_failure_20260907.gif`.
- Main remains `9188ed0`; no push. Native Humanoid-v5 remains a separate,
  blocked family. Consult the live mailbox and active-run record before dispatch.

## Earlier 01:50 UTC checkpoint (historical)

- All study-007/008 workers and O5 renderer are terminal at 01:50 UTC.
- Study 008 result: `../../../experiments/008_g1_repeatable_crouch/RESULTS.md`.
  The source freeze ended after both probes. Their exact source was `96d46c7`.
- Current result: `../../../experiments/007_g1_learning_rate/RESULTS.md`.
- O5 and its revision-path/scorer-fixture repairs are integrated through
  `cc3922a`; 440 focused tests pass. O5 simulation failed; no O5 training.
- Next protocol: `../../../experiments/008_g1_repeatable_crouch/PROTOCOL.md`.
  The new profile is permanently probe-only; future training needs a versioned
  admission. A separate five-arm actor-reference ablation is being implemented;
  matched-state action sensitivity is not closed-loop evidence.
- The external-reference and native walking-window audits are integrated at
  `1857cef`. No external motion has been admitted. The lower-yaw native crops
  remain kinematic candidates for a later oracle revision.
- Main remains `9188ed0`; no push. Native family remains separate and blocked.
- UI `http://127.0.0.1:8766/`: selected registry now includes two low-rate
  rows instead of two r0 replication rows. Server still needs a restart for
  O5-profile evidence and current-registry API validation.
  The previous selection is archived locally; no run evidence was removed.
- UI busy-state fix has executable DOM regression coverage. A new browser
  screenshot could not be checked because the Mac was locked.
- Read the live mailbox and `.orchestration/astra-active-run.json` before dispatch.
- Entries below retain earlier checkpoints; they are not current launch authority.

- Current evidence: `../../strategy/astra/G1_LEARNING_RESULTS_20260906.md`.
- GIF: sibling `humanoid-harness-probe-runs/gmt_course_o2r1_task_overlay_v1_recorded.gif`.
- First 13 training runs used `004592f`; O3 used reporting-only descendant `1b501d6`.
- Integrated task-overlay renderer `b609bf6`, feedback builder `6d30af1`, raw
  boundary verification `a8f7431`. Parent: 200 focused + 9 CLI tests pass.
- O3 is rejected: trained rollout falls at 6.96 s; its five preregistered predictions fail.
- Heading-only r3 trained and was rejected: lateral drift 8.34 versus 2.97 m;
  speed MAE 0.424 versus 0.307 m/s. No full task pass.
- Main fast-forwarded to `9188ed0`; no push. Astra telemetry source `0923a5a`:
  all nine O2r1 output artifacts byte-identical, 266 focused tests pass.
- Initial raw telemetry: near-zero explained variance throughout 64 updates;
  149 attempted epochs, not completed epochs. The resulting scaling screen is
  now completed; see the current result above.
- Latest screen and exact receipts: `../../../experiments/005_g1_training_conditioning/RESULTS.md`.
- Current source includes reviewed UI commits through `9dd3e6b`; 330 focused
  adapter/feedback/CLI/UI tests pass. No full task pass; main remains `9188ed0`.
- Scale64 seed 06: explained variance 0.924, lateral 0.526 m, 20 s/no fall;
  posture/timing/inside-speed still fail. O4 zero-residual probe falls at 5.4 s.
- All five replication jobs and O4b probe completed at frozen `22b5ee4`.
  Every retained output/parity check and feedback recomputation passes.
- O4b: 20 s/no fall,83.1% compliance, joint/roll-pitch p95 0.249/0.247;
  depth, lateral and overall-speed gates still fail. Do not repeat O2b or
  start the unnecessary dwell/destination probes after this admission pass.
- Refresh `.orchestration/astra-active-run.json` and mailbox before launching.
  No heavy job at 23:58 UTC. Local G1 UI is on port 8766, separate from old 8765.
- Whole-suite failures remain; see `../TEST_MATRIX_20260906.md`. Focused G1 checks pass.
- Native likelihood chain through `44f65b6` remains unmerged; do not retry it automatically.
- Everything below is historical context, not current dispatch authority.

## Earlier attempts (retained)

- GMT attempt: `humanoid-harness-probe-runs/gmt_walk_stand_20260906`, outside
  this worktree; 0.58 s wall, 234 MB peak RSS, no robot steps or trace.
- Resource receipt SHA-256:
  `23dbb2f663f3d039d8424585fa4d878ddd0c9559db7fd4e4180675259e4ea849`.
- Launcher repair adds only `/usr/sbin`; it does not inherit arbitrary PATH.
  The regression imports real MuJoCo under the exact child environment/limits.

- Current attempt record: `../../strategy/astra/T1_ATTEMPT_20260906.md`.
- Native terminal: 27.44 s, `likelihood_failure`; no automatic training retry.
- Renderer: integrated from `2065c10`; 29 GMT tests pass. No replay/render yet.
- Fable independently verified the four terminal hashes and reproduced the
  numerical mechanism. Actual offending sample is missing, not reconstructed.

- 18:35 UTC: Fable reproduced the race on `0ab5cc5`; a listed `.pending`
  checkpoint was renamed before `stat`, incorrectly crashing a healthy worker.
- Repair: one no-follow `stat`; skip only vanished entries, retain symlink and
  other I/O failures. Output accounting remains an instantaneous sample.
- The prior `182250...5d86...` proposal is superseded, not launch authority.
- Separate evaluation cleanup `4369261` and GMT renderer `2065c10` remain
  unmerged. No protected evaluation or new simulation has run at this checkpoint.

## Earlier system-lead checkpoint: 2026-09-06 (historical)

- Authority: `../../strategy/astra/SYSTEM_LEAD_PLAN_20260906.md`; Samuel assigns
  Astra implementation, integration and bounded local training across both knobs.
  Fable reviews and ideates only after its already-running T2C2 checkpoint.
- Diagnosis: `humanoid-harness diagnose`; real Phase A cycle 2 validates 20
  current rows, 0 falls, no intended slow segment. This is not reference competence.
- Protected Phase B diagnoses are human-only. Candidate context excludes their
  routing, outcomes, retrieval and steering. Development evidence is the next input.
- Development ablation: `scripts/phase_b_development_reference_ablation.py`;
  four in-sample reset blocks, exact/zero/shuffled/shifted actor references.
  Production smoke and exact corpus lineage are required. No protected splits.
- GMT: numeric-only conversion and reconstructed actor in `adapters/gmt`;
  original TorchScript and motion pickles were not executed. Golden-output
  equivalence and tracking competence remain untested. The isolated GMT worker
  is implementing a headless playback adapter, with no simulator run yet.
- Fable critique: `../../strategy/astra/MDP_AND_FABLE_REVIEW_20260906.md`.
  Its read-only CLI call could not read files; no code-review claim.
- Live Fable follow-up: `20260906T175932.763617Z-9825de420bbd4fbbab8b551ffc087ec3`.
  Native T1 first; T2 is an incremental reward-loop result, not full composition.
- Role acknowledgment: `20260906T175818.868830Z-6266274cc43349c9a49af2daa9e7be95`.
  Parent accepted ownership of main promotion; no new Fable implementation lane.
- Heavy-job slot is free at this checkpoint. No new training or simulator result.
- The entries below are retained history, not current dispatch authority.

## Historical lane handoff

- Date: 2026-09-05.
- Last checkpoint: 2026-09-06, 15:56 UTC; helper accepted and committed for peer integration.
- Current lane: oracle/reference composition and tracker integration.
- Current transfer authority: `LANE_SWAP_20260906.md`. Historical reward
  sequences below are evidence records, not current Astra dispatch authority.
- Workspace: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Branch: `astra/reward-loop`.
- Orchestrator role: Astra, this Codex task
  `01a05e77-9737-7240-b79e-4e3902d1664c`.
- Model label identifies the requested role; use actual tool/request receipts
  for model settings, never sender names as attestation.
- Original checkout belongs to Fable. Its B0 worker was actively renewing the
  main writer lease at setup; its uncommitted files were not copied.

## Earlier oracle slice (historical)

- **Accepted source:** `b7b67b7925650db2e999953e6b85e81f865c7d34`.
  `RESOURCE_SLOT_01_ACCEPTANCE.md` records the exact closure and caller duties.
- Closure verdict `ACCEPT_RESOURCE_SLOT_HELPER_ONLY`, terminal 15:29:43 UTC,
  exit 0; watcher observed completion. Both defects closed, all five hashes
  verified; final parent recheck 37 passed in 2.69 s. No live Astra worker.
- Fable owns supervisor integration and main promotion. Do not duplicate the
  helper review, modify its peer checkout, or acquire a real compute slot.
  Refresh T1 preflight only after the reviewed final integration is available.
- Fable reports pairing closure and one initial F3 call. Its `2806729` ledger
  records alpha 1.0 / beta 0.0, hypothesis only, not admitted or measured.
  Astra read the committed record and acknowledged the message, not raw logs.

### Repair checkpoint (retained)

- Current result: `RESOURCE_SLOT_01_REVIEW_RESULT.md`; original independent
  verdict `REPAIRS_RESOURCE_SLOT_01`, terminal 14:54:15 UTC, watcher observed
  completion. Parent reproduced 8 failing new cases before repair; final
  two-file suite has 37 passes. Exact repaired hashes are retained for closure.
- Current packet: `TASK-RESOURCE-SLOT-01-CLOSURE.md`. Use active-run pointer;
  do not duplicate a live review or edit its five frozen implementation files.
- Fable reports its own combined readback of T2PAIRR2 and F3PINR2. Handoff
  acknowledged; no duplicate pairing review or reward dispatch from Astra.

### Original builder and first review (retained)

- Current checkpoint: `RESOURCE_SLOT_01_PARENT_RESULT.md`. The builder is
  terminal `INTERRUPTED_TERM`, exit 143 at 14:21:51 UTC, lease released; exact
  watchdog recorded `deadline_term_sent`. No final handoff; its diff is retained.
- Parent re-ran the two allowed test files: 28 passed in 2.35 s, Ruff/format and
  diff checks pass. Direct helper load works on Python 3.9.6 with stdlib only.
- Current independent packet: `TASK-RESOURCE-SLOT-01-REVIEW.md`. It freezes all
  five implementation hashes; no implementation commit or promotion before
  review. Consult `.orchestration/astra-active-run.json` for dispatch identity;
  never relaunch the stopped builder or duplicate a live reviewer.

### Accepted builder scope (retained)

- Token scope explicitly accepted in
  `20260906T133444.291029Z-5169f292a3bd4f48a65d05a997521904`, replying to exact
  proposal `20260906T132352.198444Z-b86226c1a7b544e997ae9ce2324a4145`.
  No more scope approval is needed for those five files; no compute is implied.
- Builder packet: `TASK-RESOURCE-SLOT-01.md`, committed at `b028542`;
  SHA-256 `9cdcf4430df96600df181de758cb8e7528521d928e51b276c5699ab39aa1ad17`,
  6,095 bytes. Base `6a39aaaf26455caa30e30cd527100ec7df375bc9`, docs-only
  descendants allowed. Requested Sol/max, write mode, no nested agents.
- Dispatch once as owner `astra-resource-slot-builder-20260906`. The launcher
  creates the exact durable run/request and writer lease. Read the active-run
  pointer and matching request before retrying; a prepared packet is not a launch.
- Target 12 minutes, stop work at 18, exact watchdog 20. Retain watcher errors
  and verify detached startup. Only five agreed files and two focused test files;
  no actual shared token, supervisor edit, simulator, full suite or training.
- After terminal/lease release: inspect exact diff and tests, then independent
  closure review before committing/promoting the implementation. Fable retains
  its combined review, supervisor integration and main promotion.

### Pre-dispatch evidence (retained)

- **Current T1 evidence:** `T1_SMOKE_ROUTE_RESULT.md`. Fable accepted the exact
  local-only data handoff in `20260906T124406.664008Z-62f7bdad6c224e699f9316dc1928c5bd`.
  All 137 files / 85,097,632 bytes copied with source/copy hash verification;
  existing private receipts unchanged. Real preflight passed in 9.45 s from
  clean `f18d698`, with 103 sealed inputs. No simulator construction or training.
- Copy ledger retained at `.orchestration/t1-history-handoff/verified-uar5nco2/copy_receipt.json`.
  Its full identity and live preflight digests are in the T1 result above.
- Fable confirms there is no implemented atomic heavy-job token. Returned a
  formal scope proposal because its offer was embedded in an acceptance:
  `20260906T132352.198444Z-b86226c1a7b544e997ae9ce2324a4145`.
  Astra owns the proposed pure `harness/resource_slot.py`, focused tests and
  mailbox adapters; Fable retains supervisor integration and main promotion.
  Await explicit scope agreement before dispatching this shared-interface slice.
  No token, resource slot or heavy-job authority has been claimed.
- **Current pairing verdict:** `REPAIRS_T2PAIRR1`, retained at
  `T2PAIRR1_REVIEW_RESULT.md`; PAIR-01/03 closed, PAIR-02 sink-level regression
  still open. Terminal at 12:49:59 UTC, exit 0; watcher `terminal_observed`.
  Final SHA-256 `ee7aec165c938c9f7c808a4dec1c987dd68cf1e8ce09e192f07785a107a8bafe`.
- Findings returned to Fable via review
  `20260906T131903.412132Z-030ff3ac4ab84fe09e332b3e8311ed20`.
  It retains all pairing repairs and the combined T2AR2/F3 review. No worker
  remains running in Astra; do not redispatch the completed closure packet.

### Earlier admission and dispatch checkpoints (retained)

- T1 verdict and actual preflight refusal: `T1_SMOKE_ROUTE_RESULT.md`.
  Read-only audit returned `READY_TO_PROPOSE_T1_SMOKE` at 12:07:29 UTC;
  watcher retained `terminal_observed`, with no timeout path exercised.
  Parent's real preflight refused missing raw designer audit data after 0.84 s;
  no simulator construction/step or training ran. Later checks remain unverified.
- Exact local-only data-handoff proposal:
  `20260906T124209.502736Z-57bb88fcef6d4b6e8bfe994f6b0ad51e`.
  It requests both historical six-file designer bundles, their original oracle
  files, and three content indexes with 80/20/20 bound traces. Nothing is copied
  yet; raw data remains private and ignored. Do not fabricate or regenerate it.
- The same proposal asks Fable for its promised atomic heavy-job token's exact
  mechanism/status. It is not a compute reservation. Complete live preflight
  before binding a future exact clean commit/input ledger into a smoke request.
- Pairing repair handoff received/acknowledged as read:
  `20260906T123327.297282Z-f8b27d09b938465baa5175c6277249cc`.
  Source `df3c41d5980ac31efb5024798e20e473f36fa17b`; inspect Git objects only.
  Parent verified contracts SHA-256 `c25d61de985f06ab69945842185fd004be3a623d3c984df11718ca1af53855f1`
  and pairing receipt SHA-256 `1a2b7ece139974117fd5c75e040d9cc52cd51a4e42c9b5afd794a60b02232348`.
  The committed full-suite summary reports **1 failure**, not a green run.
  `78f6c7b` subsequently changes the historical reward no-learning receipt and
  its regeneration record; do not infer a reproduced green suite from that.
- Active read-only Sol/max closure packet `TASK-T2PAIRR1-CLOSURE.md` at `3de76b4`;
  3,661 bytes, SHA-256 `6ae7798fcf7458a44d442acc3e9d8e52de1fff2f03a5a39d93c5a58c4132eb41`.
  Only PAIR-01..03; Fable keeps the combined T2AR2/F3 dispatch review.
- Run `.orchestration/sol-runs/20260906T124421Z-f2726334-bd50-4caa-a9f1-c512af01f0be`;
  owner `astra-t2pairr1-review-20260906`; launched 12:44:21 UTC;
  exact watchdog deadline **13:04:21 UTC**. No imports, tests or runtime.
- Separate-session watcher PID 34036 survived the tool return (PPID 1);
  start receipt binds exact Bash runner 34038. Error output is retained in the
  run's `watcher.stdout-stderr.log`. Check the final outcome before attesting it.

### Completed T1 audit dispatch (retained)

- Audit: `TASK-T1-SMOKE-ROUTE-AUDIT.md`, committed at `9e567a6`;
  4,941 bytes, SHA-256 `3a7cdb299b44c139c5ac8e28c5c2682fee66a8d594202d51eca473640d34905b`.
- Pinned source: `2c2a32ba26c88ef253a47ce0b2823d846500c474` in Astra.
  Scope: actual training-only prerequisites, not another broad tracker review.
  Determine which gates are reachable; do not waive incomplete source reviews.
- Read-only Sol/max run:
  `.orchestration/sol-runs/20260906T115529Z-63dd62e7-a207-4d8a-a989-d5ec5940f097`.
  Owner `astra-t1-route-review-20260906`; launched 11:55:29 UTC;
  exact watchdog deadline 12:15:29 UTC. No imports, tests, simulator or training.
- Watcher started in a separate session; PID 19521 remained alive with PPID 1
  after the launch tool returned. Start receipt names exact Bash runner 18882;
  stdout/stderr retained in the run's `watcher.stdout-stderr.log` (empty at check).
  This verifies durable startup, not eventual deadline enforcement.
- The launch wrapper first misparsed the launcher's text as JSON. The worker
  had already launched successfully; no duplicate was started. Watcher attached
  11 seconds later to the exact retained run, still using the original deadline.
- Parent source check: the existing smoke reservation requires 1,200 seconds
  (20 minutes), stricter than the strategy's 45-minute upper cap. Preserve it.
  The training command produces an empty-evaluation/non-scoring report; it does
  not invoke `evaluate-policy`. The independent audit must check remaining
  reachable training gates before a concrete smoke reservation can be proposed.
- Fable progress `20260906T114656.703183Z-aa616f999a0e4fc1a5179107ead9a225`
  reports T2AR2 at `d780bf9`, re-seal `487a796`, and T2PAIRR1 starting.
  Parent verified no `contracts.py` delta from `0b4a8a1` to `d780bf9`; SHA-256
  `c8b6fd3cb61f0a79a086156f1a3440e64a5dae7395086735c59a886b2ec8448a`.
  Peer test counts are not independently reproduced. No source imported.
- Fable keeps T2PAIRR1 and its planned combined review. Astra progress reply:
  `20260906T115626.178150Z-9e793b05e8864fdf88fe2b0f886f592f`.
  Any future shared training/runtime/contracts import needs a delta assessment;
  a verdict on this pinned Astra source cannot attest future integrated bytes.

### Completed pairing review

- Current verdict: `REPAIRS_T2PAIR`; exact findings and source locators in
  `T2PAIR_REVIEW_RESULT.md`. No pairing or study execution acceptance.
- Review returned to Fable in
  `20260906T112039.620591Z-a4aa4e27ac6d4bfaa8c88e57f8a71362`, replying to its
  exact T2PAIR handoff. Its existing T2AR2 writer retains repair ownership.
- Reviewer terminal at 10:50:41 UTC, exit 0, before its declared deadline;
  final SHA-256 `340ef1bf94c79a5522b2a8461b8b852e1922dcb42109eaaa327f7ea3b936bc33`.
  The watcher-start record exists but its final record is absent; deadline
  enforcement is not verified. No runner/watcher remained at collection.
- Preserve existing records. On the next launch verify durable watcher life
  and retain its error output. Do not fabricate a final-watch outcome.
- The original pairing review is finished. Fable owns the repair and contract
  path; the current Astra worker above is only its narrow read-only closure.

### Pairing dispatch checkpoint (retained)

- Current review packet: `TASK-T2PAIR-INDEPENDENT-REVIEW.md`, committed at
  `00dcf8e`; SHA-256 `d37d98bd6e890c02f28f77a2bc7f630f0f5b3a96f269a4b940237f5d9fca2e47`,
  4,715 bytes. Reviewer reads exact Git objects, never the peer's dirty files.
- Detached run: `.orchestration/sol-runs/20260906T103625Z-03f6ffe1-99f1-4b50-bc11-d817301ec04e`.
  Owner `astra-t2pair-review-20260906`; read-only Sol/max, no nested agents.
  Launched 10:36:25 UTC; exact watcher attached in the same call, deadline
  10:56:25 UTC. Review is static: no tests, fake runtime, simulator or training.
- Reviewed slice: `0b4a8a1fda322eac1a12071593a79747cff1b8b3`, base
  `7698256dfe721c020c2aa264432867523e1dc99c`. Parent verified pairing-receipt
  Git bytes: 9,770 bytes, SHA-256
  `e5351c3b49ba97cc362ccd68a0bcf797c5077fb0c2ace78535b24e5567e0a259`.
  It is a synthetic primitive-stream receipt, not a physical rollout.
- Fable handoff `20260906T102035.254055Z-d65272207ef446eeba0cc6d16c5ec3e8`
  is acknowledged as read. T2AR1's reported verdict is ACCEPT-WITH-REPAIRS;
  final-ready report admission, registry-resolved candidate and execution-seal
  verification remain with Fable's T2AR2. Do not duplicate that worker.
- Any later T2AR2 change to `phase_b/contracts.py` needs delta review before
  reusing a T2PAIR acceptance. This review cannot attest changing future bytes.
- Current main `03a482600656f1a91d2f588ea4f62388eb81367e` now includes
  `d47f528` by ancestry and contains all three advisory documents. This closes
  the earlier title-versus-tree discrepancy below; no peer source was imported.

### Earlier coordination checkpoints (retained)

- Fable explicitly accepted research proposal
  `20260906T075217.479909Z-0fe7bb48513649d5a37100b9bbc5c810` in
  `20260906T083628.043657Z-ead8dd69e6734d1693d2ce05fe7374d4`.
  Its independent assessment is committed at `49c0511` in main's
  `docs/strategy/RESEARCH_STRATEGY.md`, section "First-principles challenge".
- Accepted route: conditional benefit/cost, strong simpler baselines, optional
  vision, and one non-promotable T1 mechanism smoke capped at 196,608 transitions
  or 45 minutes. This accepts a route, not a run, resource allocation or cohort.
- Proposed order: T1 mechanism check; separate T2 expert-hold reward study;
  then the joint speed-zone comparison after the reference-use gate. T2 alone
  is a reward intervention, not a joint reference/reward demonstration.
- Source ancestry verified: `57add6e` is included in main
  `7698256dfe721c020c2aa264432867523e1dc99c`. That merge's title also names
  `d47f528`, but its tree contains no `docs/strategy/astra/` and the advisory
  commit is not an ancestor. Advisory-document integration is still pending,
  consistent with Fable's mailbox body; do not infer it from the title.
- Fable reports T2AR1 committed at `e50a307`, receipt regeneration at `cca4600`,
  and T2PAIR dispatched from `7698256`. Packet SHA-256:
  `515516fc81e925e8d056bb26fab128115733405332cf81d82233bf0789a93d60`.
  Peer review is ongoing. No new peer source imported into Astra; preserve
  the existing source until the exact finished handoff is reviewed.

- Latest user priority: investigate composition from scale and policy ICL, then
  make the smallest feasible joint reference/reward demonstration. Advisory
  note: `docs/strategy/astra/FIRST_PRINCIPLES_20260906.md`; companion source
  ledger and `REFERENCE_LIBRARY_SETUP_20260906.md` retain evidence boundaries.
- Drop universal claims that references are insufficient or ICL cannot solve
  humanoid tasks. Compare conditional adaptation value and cost instead.
- Native corpus staged locally: 477 files / 128 NPZ; 36 training clips decoded
  and all library artifact hashes verified. No simulator or training ran.
- OT1E01 builder terminal at 07:38:42 UTC, lease released. Parent 20 pure tests
  pass; independent `ACCEPT_OT1E01_MANIFEST_ONLY`; commit
  `57add6e8713fab4aa06bbfb1c70179202bb55787`. See `OT1E01_RESULT.md`.
- T2PAIR source scope accepted on main `ed9f1d3`, with T2AR1 prerequisite:
  reply `20260906T072706.830333Z-a50cb508191d4d629b4ba6c8f9e42103`.
  This supersedes the earlier pending-base notes below, but does not approve
  implementation, compute or scientific results. Preserve Fable's paths.
- At the 07:50 checkpoint no worker was running in Astra. The current bounded
  review above supersedes that state; no new builder has been dispatched.

### Prior dispatch context (retained)

- Accepted scoring repair and exact receipts: `OT1_NARROW_RESULTS.md`.
- Next code base: `3406bb5f0af3c2479d47cdd2d245c3e738a168fe`; dispatch adds docs only.
- OT1E-01 addresses the ignored admission-manifest digest before checkpoint or
  episode construction. It does not change any scientific metric or threshold.
- One bounded Sol/max builder; exact run/deadline live in ignored
  `.orchestration/astra-active-run.json`. Inspect receipts before resuming;
  a prepared packet alone does not mean a worker started.
- Preserve Fable's pairing files. Its converged proposal still lacks the
  promised exact T2A integration base and resolved test-file choice.
- Shared CLI integration proposal: `20260906T065456.009901Z-ffc472497475469ab2d1291ab13d3226`.
  Independent development is allowed; promotion is not yet accepted.

## Completed and historical reward work

| task | owner | state | acceptance |
|---|---|---|---|
| Coordination design review | Sol `dual_lane_review` | complete; applicable fixes folded | Explicit accept/reject, named promotion owner, evaluator separation, venv identity |
| Reward loop decomposition | Sol `reward_lane_plan` | complete; fixes folded into A1 | Separate sources/evidence, concrete module CLI, receipt binding |
| Local mailbox + tests | Sol `mailbox_builder` | complete; parent reproduced 9 passed | Real-worktree shared root, 36 concurrent sends, recipient checks, explicit acks |
| Protocol and lane authority | Astra + Fable | proposal explicitly accepted 22:14Z | Exact acceptance message recorded below; B0 exception remains |
| A1 implementation | detached Sol/max | accepted after repairs and review | `TASK-A1-reward-proposal-loop.md`; details in `A1_RESULT.md` |
| A1 independent review | detached Sol/max | completed 20:57:15Z; accept-with-repairs | Five blocking findings and one low-severity origin check |
| A1 repair | detached Sol/max | completed 21:42:41Z; parent checks pass | Six findings addressed; `A1_RESULT.md` |
| A1 final targeted review | detached Sol/max | ACCEPT at 22:06:14Z | All six original findings closed; parent corrected B0 sequencing wording |
| A2 live-model protocol | Astra | stopped; one initial call, no revision | `A2_RESULT.md`; transport timeout and retained rejection, no retry |
| A2R changed-condition canary | Astra | complete; both responses admitted | `A2R_RESULT.md`; text format/lineage only, no candidate execution |
| Independent goal interpretation | Fable | substantive reply received and acknowledged | `FABLE_GOAL_ALIGNMENT.md`; composition plan reframed, not demonstrated |
| B0 transfer | Fable → Astra | ownership transferred with P1 repair gates | Exact `89d4c34d1b04eb9efca2b75d1ffc828a13de55d7`; not execution-approved |
| A3 B0 adoption audit | read-only Sol/max, then parent | leaf stopped on ambiguous path; parent completed dependency check | `A3_RESULT.md`; exact 25-file donor snapshot, no execution approval |
| R1 parent-execution boundary | Sol/max builder | terminal succeeded; source uncommitted, not accepted | `R1_RESULT.md`; 49 passed/36 skipped and 59 A1 tests reported, known stale consumer |
| R1 first independent review | read-only Sol/max | timed out 04:43:51Z; no verdict | Exact run retained; no escalation; historical source snapshot verified before parent edits |
| R1 parent completion | Astra | 139 passed/16 deferred; not accepted | `R1_COMPLETION_CHECKPOINT.md`; restored safe coverage and repaired strict metadata/version/API tests |
| R1 completion review | read-only Sol/max | REJECT at 06:18:34Z; terminal receipt retained | `R1_REVIEW_FINDINGS.md`; three constructor/ingress defects; 37 no-temp tests reproduced |
| R1 ingress repair | Sol/max builder | finished 06:49:16Z, lease released; parent 143 passed/16 deferred | `R1_INGRESS_REPAIR_RESULT.md`; three findings repaired, not independently closed |
| R1 targeted final review | read-only Sol/max | timed out 07:35:25Z; partial closure, no verdict | `R1_FINAL_REVIEW_PARTIAL.md`; three findings reported closed, copy limitation open |
| R1 acceptance decision | read-only Sol/max | ACCEPT at 14:58:59Z | `R1_ACCEPTANCE.md`; three findings closed, copy-API limitation non-blocking and documented/tested |
| R2 runtime protocol | read-only Sol/max planner | finished 17:20:01Z, proposal only | `R2_PLAN_RESULT.md`; no execution or implementation approval |
| Cycle-first convergence | Astra + Fable | explicit proposal/acceptance exchanged | `MAIN_INTEGRATION_20260905.md`; one runner/report, exploratory boundary, JSON first |
| Minimum reward path review | read-only Sol/max | completed 18:28:40Z; chose JSON formula family | Existing Python execution gates remain unchanged |
| Pinned-main integration | Astra + Sol reviewer | ACCEPT_INTEGRATION at 19:06:56Z; handed to Fable | `8a48f32`; excludes later uncommitted F1 files |
| F1 formula core | Sol builder + reviewer | REJECT_F1 at 19:48:21Z; three lineage defects | `F1_REVIEW_FINDINGS.md`; numerical core/input gates accepted within static scope |
| T2 input module | Astra + reviewer | ACCEPT_T2_INPUT_ONLY at 19:48:21Z; 17 tests pass | `T2_INPUT_ACCEPTANCE.md`; not wired into formula or simulator |
| F1 lineage repair | Sol writer + Astra | finished 20:34:23Z; parent 232 passed/16 deferred | `F1_REPAIR_PARENT_CHECKPOINT.md`; includes one FIFO-open regression fix |
| F1 closure review | read-only Sol/max | ACCEPT_F1_STATIC_ONLY at 20:59:48Z | `F1_ACCEPTANCE.md`; three findings and FIFO follow-up closed |
| F2 T2/runtime interface plan | read-only Sol/max | completed 21:40:31Z | `F2_PLAN_RESULT.md`; additive adapter, no duplicate F1 loop |
| F2 T2 evaluator | Sol/max builder + Astra | finished 22:14:39Z; parent 73 focused passes | `F2_PARENT_CHECKPOINT.md`; three authorized new files, uncommitted |
| F2 independent review | read-only Sol/max | ACCEPT_T2_EVALUATOR_ONLY at 22:45:28Z | `F2_ACCEPTANCE.md`; static review, reviewer pytest 0 due temp-directory boundary |
| F3 T2-aware model protocol | read-only Sol/max planner | finished 23:29:33Z | `F3_PLAN_RESULT.md`; parent corrected deadline and read-failure semantics |
| F3 initial packet/ingestion | Sol/max builder + Astra | completed 00:13:01Z; parent 79 focused passes | `F3_PARENT_CHECKPOINT.md`; four new files, not accepted |
| F3 independent verification | Sol/max test-output-only writer | REJECT_F3 at 01:21:49Z; 79 focused tests and seven probes | `F3_REVIEW_FINDINGS.md`; parent reproduced defects, all 18 hashes unchanged |
| F3 repair and baseline refresh | Sol/max builder + Astra | succeeded 01:56:27Z; parent 112 focused passes | `F3_REPAIR_PARENT_CHECKPOINT.md`; not independently closed |
| F3 closure and conditional call review | Sol/max test-output-only reviewer | ACCEPT_F3_STATIC_ONLY and APPROVE_F3_ONE_CALL_PROTOCOL at 02:38:18Z | `F3_ACCEPTANCE.md`; zero candidate calls; new owner Fable after transfer |

- `F3_ONE_CALL_PROTOCOL.md` received conditional approval. Samuel's lane swap
  stops Astra dispatch; Fable must review/adapt ownership and re-pin its
  integration before any call. No revision, retry, simulator or training.
- Fable baseline confirmation: `20260906T021853.719294Z-e5ffd026d5ea41c1b039b7219513acc1`.
  Formatting-only commit for its reported historical F2 fence:
  `949f686e3ffa2a766fa3ac2b87c1733393fbc833`; original launch bytes retained.

- Review source and negative fixtures are retained; do not overwrite them.
  `.orchestration/f3-pre-repair-20260906/` holds the four original F3 files
  under their relative paths; parent verified all four original hashes.
- Current repair findings and precise review timing correction:
  `F3_REVIEW_FINDINGS.md`. This is not static acceptance or model-call authority.

- Exact F2 promotion accepted by
  `20260906T002718.313618Z-cbaf7f3f6326473ab7226e0c183ea53a`.
  Parent verified committed main `87d2e39c47b6747b505bc2657d505aeda2265b5e`
  integrates F1/F2. This supersedes pending-promotion notes below.
- New committed tracking-only baseline: 1,773 bytes, SHA-256
  `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f`.
  The old F3 baseline is a historical static fixture only; live calls remain
  gated on review and an explicit exact-baseline refresh. See parent checkpoint.

- Parent verified baseline `0986d4fc...224d4`, 1,144 bytes, from peer object
  `a51ea9e`. Its run manifest agrees; `phase_b/DESIGN.md` still names a stale
  prose hash. No peer file was edited. F3 binds the actual blob, not that table.

- Accepted F2 commit: `9bbb6587c1f9d955e684ffe4e1772e72602e3556`.
- Reviewed handoff sent:
  `20260905T231526.597779Z-051fd2dcdda14f949a570a5294b478df`.
- Exact-bounds/adapter-admission proposal:
  `20260905T231526.889802Z-539e60b2549f48459fda183b385842df`.
- F3 must bind the actual tracking-only baseline (+0.0 task reward), not an
  invented alpha=1,beta=0 parent or an out-of-bounds alpha=0 recipe.

- F2 builder: `20260905T220615Z-651f932a-0627-4e28-a7cb-0719bdf94475`;
  terminal success and lease release verified; watcher observed completion.
- Fable accepted interface proposal
  `20260905T220539.436910Z-95e3e2a4b5954edc81e2bfe32f171cd7` via
  `20260905T230846.027071Z-a969235f97034a998d56358ecec842a6`.
  The exact canonical bounds hash still needs promotion confirmation because
  the reply used interval-array shorthand. The proposed physical input range
  belongs in Fable's reviewed adapter certificate, not an unreviewed V2.1.
- Current F2 acceptance and handoff identities: `F2_ACCEPTANCE.md`.
  No raw response may be relabeled T2-aware while its packet still binds F1's
  old target set and absent compositor.

- Accepted F1 commit: `27330eb4f50feae27df3ab118e8c9c0d165112a8` (local only).
- Completed F2 plan: `20260905T212913Z-cb25bab9-c2f9-4809-ac32-4d68072ab70e`;
  exact next builder run and deadline are retained in the active pointer.
- Peer registry reply `20260905T214217.711948Z-79bbaa69ff38402fa4a0cea559a8cba0`
  is acknowledged as read. New formula IDs and promotion still need explicit
  agreement; Astra-only evaluator work does not modify the shared interface.
- Current peer object inspected: `10d477b5726fb674fa2a057419501faee8043d9d`.
  The duplicate phase-B T2 class is removed; input source/schema are registry-bound.

- Current acceptance: `F1_ACCEPTANCE.md`. Its retained negative records an
  accidentally selected simulator test failing on missing `gymnasium`; no
  simulator started. Use `-m 'not gym'` for this dev-only checkout's pure suite.
- Peer main observed at committed `8d91a811680599631779892a2c2bedf2127a3a74`:
  reviewed M1 merge and accepted T2 module integrated. No peer dirty files read
  or edited. Phase B registry/compositor remains Fable-owned.

- T2 exact interface accepted by Fable in
  `20260905T202243.309823Z-181767ca33654e69aad5743d92537aeb`.
  No duplicate phase-B input source was adopted; Fable owns removing its copy.

- Current checkpoint and exact receipts: `F1_PARENT_CHECKPOINT.md`.
- Fable selected new T2 COM speed target 3.0, not phase-A root speed; the old
  B0 contract stays unchanged. The explicit interface proposal is
  `20260905T193614.811073Z-cc47aa7840274926956386766cdb5c05`.

- Current authority supersedes the earlier pending proposal notes below:
  Fable proposal `20260905T183421.793603Z-8b619f3d56a84cbb81d31d39e11c36ef`
  accepted by `20260905T185408.005663Z-ff80151886fc485782897509ffe98696`.
  The narrower family is the first execution route, not permission to run it
  before its own review or to open the refused Python runtime.

- Fable proposal `20260905T175216.965624Z-a00f8a13b1a444978292d1eb1aa991ee`
  was read/acknowledged. Committed main base inspected: `a0a0a3b`.
  Main's dirty cycle builder is untouched. No lane merge/rebase performed.
- A non-checkout merge-tree check found 12 add/add conflicts with the pinned
  main base, including a historical receipt. Astra proposes a reviewed merge
  preserving accepted commit identities and both receipts, not a blind rebase.
- Public Python execution remains refused. The narrower JSON family is a
  proposal for the first cycle, not an implemented capability or a substitute
  for the broader reward-program research target.

- R1 history: `R1_PARENT_CHECKPOINT.md` and `R1_COMPLETION_CHECKPOINT.md`.
  Current repair checkpoint: `R1_REVIEW_FINDINGS.md` and
  `R1_INGRESS_REPAIR_RESULT.md`. No source/test edits during final review;
  both previous review snapshots are historical.

- Final review: `.orchestration/sol-runs/20260905T071423Z-bc589f15-291b-4341-94aa-730484edcd39`.
  Fresh 15-file snapshot: `.orchestration/b0-repair/r1final/review-snapshot.json`.
  Read-only Sol/max requested; deadline TERM, no escalation or final verdict.
  Independent partial findings and final hash gate are retained separately.

- Accepted decision: `.orchestration/sol-runs/20260905T135846Z-4adbb20a-a0ee-468a-a4ef-3dd4a807c825`.
  `R1_ACCEPTANCE.md` records scope, follow-up checks, and the late-watchdog
  caveat. Future launches and watcher attachment must occur in the same tool
  call. Local watchers now use the immutable launch time for their deadline.

- Accepted R1 commit: `754e869c618dbe4f1b43c4f1a513c023ec139306` (local only).
  Clean worktree verified after commit; final parent 144 passed/16 deferred.

- Completion review: `.orchestration/sol-runs/20260905T060349Z-d2435712-4e9a-4af1-a94b-fd183f40c96e`.
  Read-only `gpt-5.6-sol/max` requested; no nested agents; verdict REJECT.
  Snapshot and one-shot deadline watcher: `.orchestration/b0-repair/r1review/`.
  The expired parent lease was reviewed after both previous workers had
  terminal receipts; only that same parent's lease was released/reacquired.

- Fable's production collector cache-refresh finding was acknowledged and
  checked against adopted B0 source: `REWARD_CAPTURE_PATH_CHECK.md`. No matching
  mutation found in that narrow source scope; no runtime equivalence claim.

- Accepted A1 commit: `76a0fe6e86f5c4e5b52231d3b69df3d0c9b317d2` (not pushed or integrated into main).
- Final acceptance reproduction: `59 passed in 2.35s`, Ruff/format/diff checks pass.
- A2 inputs: `.orchestration/a2-canary/20260904T2250/`; `baseline.json` binds exact contract, adapter, feedback, corpus, dossier, packet, and source hashes. Every input is synthetic, not a real B0 binding.
- A2 initial run: `.orchestration/sol-runs/20260904T225403Z-f5fc865b-3d10-496b-8a5b-27e70aad70a4`; packet SHA-256 `b07d7a87a916e39aad7c3a6172bfa9fd8b1bf5944626c30e7665e5fec85f0f71`, 5,925 bytes. Calls launched: 1 of 2.
- A2 terminal: `INTERRUPTED_TERM`, exit 143, `23:25:32Z`, no escalation. CLI transport idle timeout observed; cause not established. A1 ingestion rejected missing `final.txt`; immutable rejection receipt and iteration are under `initial-records/`. The two-call protocol stopped after its first call; its dependent revision is not authorized.

- A1 builder run: `.orchestration/sol-runs/20260904T201331Z-58120f9b-e141-4e1b-be92-32f634829778`.
- Builder thread: `01a06e0e-1873-79b3-8b98-52704c2551fb`; terminal
  `SUCCEEDED`, lease `RELEASED`. This is execution status, not acceptance.
- Parent checkpoint tests: `35 passed in 2.69s`; module CLI lists `prepare`,
  `ingest-sol`, and `prepare-revision`.
- Review run: `.orchestration/sol-runs/20260904T204653Z-226c4610-e72e-4154-8c34-cf91843c0b2b`.
- Accepted repair scope: retained prior receipt/packet verification, immutable
  synthetic classification, coherent source-graph snapshot, bounded JSON reads,
  read-only proposal admission, and local publication-helper origin checks.
- Repair run: `.orchestration/sol-runs/20260904T212420Z-1ea01a7b-e7c5-4af6-866a-8e287d9145a7`; `SUCCEEDED`, lease `RELEASED`.
- Post-repair parent checks: `59 passed in 2.33s`; Ruff/format and diff checks passed. Import resolves in this worktree.
- Parent clarification: the repair result incorrectly adds B0 handoff as a
  prerequisite for the synthetic-input A2 canary. A2 does not execute or bind
  B0; actual B0 validation/training remains separately gated. The parent
  corrected the result wording after the final read-only review stopped.
- Fable acceptance: `20260904T221414.221774Z-a293af4af0eb4362a66d1a71554d3c15`, replying to the exact original lane proposal. Astra acknowledged receipt. Proposed reward-study contracts remain unlocked.
- B0 transfer: `20260905T010239.496518Z-74c32a4658554eb0960b1a3359f767c4`; acknowledged at 02:20 UTC, not accepted for execution. Both reviews require P1 repairs and re-review; transferred paths and exact review runs are in that message. Fable will not start another reward builder. Shared ADR/protocol/config changes still require peer agreement.
- Samuel's alignment request: `20260904T221845.220763Z-d00b214439d24b7e81ae0ff37e28e8af`; substantive Fable reply `20260905T010211.593536Z-f9e533bf69b64867b07778cbe87b137e` received. Fable reports a full private transcript read and proposes reframing Experiment 003 as composition. See the separate alignment checkpoint for interpretation versus chosen design.
- Central concern: a small timing adjustment on a single reference clip is a supporting diagnostic, not a demonstration of LLM-designed composition of reference behaviors into a task. Fable should distinguish source-grounded goals from its chosen oracle representation and propose the smallest meaningful composition/feedback demo.
- Reviewer accidentally started and then stopped a nested reviewer; no output
  was used. The repair packet explicitly prohibits nested work.
- Samuel asked whether continuity should remain. The existing dual-lane loop
  remains useful and ACTIVE: it resumes Astra only and defers during active
  user work. No duplicate or takeover loop was created.

## Verification and source reading

- `.venv/bin/python -m pytest -q tests/integration/test_research_mailbox.py`:
  `9 passed in 1.94s` in the final parent check, invoking the actual executable.
- Direct execution exposed macOS Python 3.9's missing `datetime.UTC`; fixed
  with the compatible timezone constant and switched tests to the real CLI.
- Changed-path Ruff passed; `git diff --check` passed; executable CLI verified.
- Local import resolves to `humanoid-harness-astra/src/oracle_composition`.
- Raw Lokesh transcript and two-page proposal re-read; both match the source
  manifest SHA-256 values. Lane split preserves their two-factor contract.
- Assessment of Fable's progress: `FABLE_PROGRESS_SNAPSHOT.md`.
- One existing heartbeat was updated to dual-lane continuity; interval remains
  30 minutes. No duplicate automation was created.

## Resume

1. Read this file, the dual-lane protocol, and the shared mailbox.
2. Check the lease in this worktree. Never use main's lease to authorize Astra
   writes. Never release another owner's lease.
3. Collect current worker receipts. Native subagents belong to the active
   Codex turn; use the existing detached Sol launcher for work that must
   survive the turn. Exact active run is in ignored
   `.orchestration/astra-active-run.json`; it records launch baseline, role,
   packet, run directory, and next action. Do not launch duplicate workers.
4. Take the next bounded packet only after prerequisites are met. Preserve
   tests, sources, and claim ceilings; update this handoff after each slice.
5. Send Fable a compact progress/dependency message. Keep routine heartbeat
   notifications quiet unless a result, failure, or Samuel's action matters.

## Historical sequence — not current dispatch authority

1. A1 builder completes its bounded packet and `A1_RESULT.md`.
2. Inspect diff and focused checks. Launch the independent read-only
   `TASK-A1-review.md` through the detached Sol launcher; record its run.
3. Resolve accepted findings in one bounded repair packet. Review/test again
   only where necessary, then commit the complete A1 slice.
4. A2: two subscription-authenticated read-only Sol canaries on initial and
   synthetic-feedback packets. Use A1 ingestion to verify exact parent and
   dossier identities. This proves real LLM plumbing only; no code execution,
   training, or robot-result claim. Write its concrete packet before dispatch.
   **This attempt stopped after call 1.** Do not resume this numbered step as
   permission to retry; follow the current next-step row and `A2_RESULT.md`.
5. A2R completed its separate two-call protocol; retain its initial/revision
   receipts and the failed original A2. No further candidate calls in this series.
6. B0 ownership transferred with execution gates. The A3 leaf stopped without
   audit conclusions; parent inspection found no missing predecessor source.
   The exact 25-file snapshot is now retained for repair, excluding shared
   operational docs. Do not rerun the ambiguous A3 packet.
7. R1 removes parent execution and fixes pure validation defects. Collect its
   exact run, inspect changes, and independently review the bounded repair.
   R2 then addresses runtime containment; R3 closes lineage and receipt gates.
   All P1s remain required before real generated-candidate execution/training.
   No unattended heavy job before an atomic shared reservation exists.

If A1 is incomplete, preserve its diff and write a continuation packet with
specific missing deliverables; do not silently relaunch the original task.

No training, behavioral evaluation, paid API calls, publication, or push has
been performed by the Astra lane at this checkpoint.
