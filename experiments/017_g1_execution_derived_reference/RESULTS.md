# Execution-derived reference: survives, fails depth fidelity

| progress | One control and one candidate probe completed; all four stages execute without a fall. |
|---|---|
| bottleneck | The candidate misses the predeclared depth-fidelity screen and three original task gates. |
| next step | Reject this exact reference bundle for training; use the failure to choose the next controlled comparison. |

## Decision

- **Reject automatic training admission.** Minimum physical-region height is
  0.532951 m: 0.028008 m above the donor, outside the fixed 0.020 m tolerance.
- Original task gates remain unchanged. The candidate passes 8/11, not the task.
- The result concerns this extraction, transformation, entry and terminal-hold
  bundle. It does not reject execution-derived references generally.
- Exploratory result with the dispatch deviation below. No outcome-selected repeat.

| Measurement | Retained O7b / fresh control | Derived inside reference |
|---|---:|---:|
| Survival / switches | 20 s / 3 | 20 s / 3 |
| Physical-region samples | 74 | 85 |
| Posture-compliant samples | 56/74 (75.7%) | 29/85 (34.1%) |
| Minimum height in physical region | 0.515773 m | 0.532951 m |
| Maximum lateral error | 10.939883 m | 13.452166 m |
| Overall speed MAE | 0.241548 m/s | 0.342003 m/s |
| Joint / roll-pitch tracking p95 | 0.239778 / 0.183717 rad | 0.144464 / 0.102899 rad |
| Original gates passed | 9/11 | 8/11 |

Better joint tracking did not imply better task behavior. The reference itself
changed, so tracking error is not an independent measure of task quality.

## Timing and terminal exposure

- Inside starts at action92, phase0; rise at236; after at260.
- Future windows first include the terminal hold at action102, local phase0.20 s.
- First held **post-step target**: action196, boundary197, phase2.10 s,
  progress1.557475 m. There are 40 held post-step targets, representing0.80 s.
- First held **pre-action reference** occurs one command later; its count is39.
  These are different sampling boundaries, not competing measurements.
- Inside exits at raw phase2.88 s and measured progress2.052327 m. The robot
  keeps advancing during the hold; it does not stall or fall.
- These observations are consistent with the predeclared timing/hold risk.
  They do not isolate anticipation, transformed velocities, entry mismatch,
  root-frame transformation, or handover as the cause of the missed depth.
- The scorer reconstructs windows from recorded boundary states. Windows were
  not directly retained by this probe; no policy or dynamics rerun is claimed.

## Integrity and dispatch deviation

- Both runs use clean source `d189d5aa0fbbcd0a46b3e64b4ce3694da3c30046`.
- Fresh control reproduces all three retained Study015 outputs byte for byte.
- Candidate prefix: 92 complete frame/action rows and 93 qpos/qvel boundaries
  are identical. First changed inside target occurs at the declared boundary.
- All 1,000 oracle commands and post-step targets reconstruct; native resource
  approvals, archive/manifest provenance, contacts and evaluator outputs verify.
- **Deviation:** the candidate dispatched after the control finished but before
  the control verification completed. Verification first rejected an unresolved
  macOS temporary-directory symlink; an unconditional shell sequence continued.
- Corrected path resolution and two independent audits established exact parity
  afterward. No source, config or criterion changed. Report the sequence honestly;
  do not label dispatch fully preregistration-compliant.
- Future verification and launch use separate successful calls. A failed check
  must never be followed by an unconditional launch.

## Local receipts

All generated files remain local; no checkpoint or private data is committed.

| Artifact | SHA-256 |
|---|---|
| Predata seal | `c535bee5dc4bd8e2b3eb767a6de5e8301ac2ebffd732eaedd53f4c16ca0cbb81` |
| Scorer | `b16fdd4c5229a571bfa971adf950704302cbe854b125b7b19156a58b12f1f1ec` |
| Score | `8baebd4bd8ae2566c4460ab769dc23e331b51bad1cd995adbda8914815d2632f` |
| Control manifest | `1a3cca160749111958231799cf37c193e935927ec159c70dbe78fcad484f0dba` |
| Candidate manifest | `52989c8414133630ef28301930db8cd2d02bc44f5c6c778bcc02bf92bc11e7b8` |
| Control resource receipt | `bba43d2d50c6c0c815d9d79789b9252be2788543464030fdda86db7c82fa9b57` |
| Candidate resource receipt | `951f5bfb6ebb58561b82c8292cf4b98746d6468a74dd8c3f6dce4fa7b75a8d10` |
| Recorded-state GIF | `dce81ae76a21cee805e50e2a970629d3ec66b869a1a53af2a6ce52b3b5cd0193` |

- Score: `artifacts/gmt/course_configs/study017_execution_derived_score_20260907.json`.
- Seal/deviation: `.orchestration/study017_{seal,dispatch_deviation}_20260907.json`.
- Runs: sibling `humanoid-harness-probe-runs/gmt_course_o7b_study017_control_20260907`
  and `gmt_course_study017_derived_candidate_20260907`.
- GIF: sibling `humanoid-harness-probe-runs/gmt_course_study017_derived_recorded_20260907.gif`.
  It displays recorded states with a visual task marker, not a physical obstacle.
- Root: 60 adjacent scorer/admission tests plus donor-profile regression pass.
  Fable independently reviewed the clean scorer and retained run artifacts.
