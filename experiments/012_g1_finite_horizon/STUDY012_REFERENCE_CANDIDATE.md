# Study 012 executed-passage reference candidate

| progress | A hash-pinned converter is ready for the exact 106 retained Study 012 states spanning the executed inside passage. |
|---|---|
| bottleneck | The candidate has not been published, admitted, or re-tracked; it is not a dynamics certificate and its 0.50494 m minimum does not establish the task depth. |
| next step | Independently review the converter, then publish once and run one zero-residual re-tracking comparison against the frozen O2 reference schedule. |

## Candidate boundary

- Donor: finite-horizon Study 012, source commit
  `9bca23bf62ea386cd7f0492aa63626b6c2a219dc`, exact `qpos[92:198]`.
- Timing: 106 states, 105 intervals, 50 Hz, 2.10 s. Frame rows
  `[92,197)` prove the corresponding 2,100 executed simulator substeps.
- Root: initial-episode task-frame progress, rebased to zero; lateral position and
  yaw set to zero; retained root height and 23 GMT-ordered joints cast to
  float32. Roll and pitch are reconstructed as a unit `xyzw` quaternion.
- Velocities are not copied. The existing `ReferenceMotion` loader would derive
  finite differences and apply its fixed 19-frame smoothing kernel. That makes
  this a transformed trajectory with a new identity, not a replay of donor
  dynamics.
- The receipt binds the donor resource receipt, manifest, config, JSONL, NPZ,
  executable source tree, model XML, all 35 model meshes, converter sources,
  output archive, and every output array's C-order bytes.

## Measured scope

- Donor height is 0.77857 m at entry, 0.50494 m minimum, and 0.59544 m at exit.
  It advances 1.39447 m while the discarded run-specific lateral correction is
  -0.19142 m.
- Recorded execution has only the two allowed ankle-roll ground-contact bodies
  across the selected 2,100 substeps.
- Static `mj_forward` checks of all 106 transformed output poses have no
  non-foot ground contact. Allowed-foot penetration reaches 0.01543103 m; the
  candidate is **not** penetration-free.
- These facts establish a contact-screened, execution-derived kinematic source.
  They do not establish dynamic feasibility after the horizontal transform,
  controller support, reference tracking, reward quality, or task success.

## Single falsifying re-tracking test

After separate admission, replace only the O2 inside reference segment with
this exact candidate. Keep controller, reset, task, evaluator, reward, cadence,
horizon, seed, and zero residual fixed. Compare the full 20 s trajectory with
the unchanged O2 zero-residual control.

The candidate is rejected for this use if it introduces a non-foot contact or
fall, fails to traverse the actual `[1,2)` m region, or does not reduce the
inside root-height tracking gap without worsening the predeclared lateral and
tracking gates. Passing would support only this one re-tracking use, not depth
or learnability.

## Reviewed generation command (not run by this slice)

```bash
PYTHONPATH="$PWD/src" .venv/bin/python \
  scripts/convert_study012_reference_candidate.py \
  --output /reviewed/server-owned/path/study012_inside_passage_v1.npz
```

The converter refuses overwrite and does not update any motion registry or
course configuration.
