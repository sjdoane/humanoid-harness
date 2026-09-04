# Disposable TQC resource calibration

| | |
|---|---|
| progress | A version-pinned, receipt-only calibration is specified for the exact local Humanoid adapter. |
| bottleneck | TQC throughput, peak memory, and finite-update behavior have not been measured on this host. |
| next step | Review the code and design, then run the 100,000-step excluded-seed calibration once. |

## Purpose

- Measure whether Farama's TQC bootstrap method is affordable after adapting
  it to the project's different, frozen `Humanoid-v5` runtime.
- Emit resource and integrity evidence only.
- Discard the in-memory model. Save no controller, optimizer, checkpoint,
  replay buffer, or normalization state.

## Frozen calibration

| Item | Value |
|---|---:|
| Seeds | Run seed `92001`; exact worker seeds `92001`–`92005`; all permanently excluded from controller training and evaluation |
| Environment steps | `100,000` |
| Vectorization | `5 x DummyVecEnv`, CPU |
| Environment | `Humanoid-v5`; `terminate_when_unhealthy=false`; no normalization |
| Replay capacity | `1,000,000` transitions; no memory optimization |
| Replay allocation | `5,648,000,000` bytes (`5.26 GiB`); every page address written before learning |
| Sampled peak-RSS failure threshold | `12 GiB`; retrospective check, not an OS memory cap |
| Sampled free-disk failure threshold | `50 GiB` |
| Throughput gate | `116 environment steps/s`; 10,000-step warmup, then every 10,000 steps |
| Time projection | `14m 22s` for 100,000 training steps at the floor; `47h 54m` for 20 million |
| Logging | In-memory SB3 logger with no directory and no output formats |
| Declared research output | One descriptor-reserved, atomically finalized JSON receipt; no SB3 temp log, pickle, Torch archive, checkpoint, or replay |

Before learning, the runner writes zero to one address on every OS page backing
each replay array, plus its final byte. It verifies that the empty buffer's
position does not change. This proves the write traversal, not simultaneous
physical residency. Process-lifetime peak RSS is reported separately.

RSS is sampled once per vector step and every 4,096 replay pages (`64 MiB` on
the reviewed 16-KiB-page host). A transient allocation can cross `12 GiB`
between samples, and the OS can terminate the process first. The threshold
makes a surviving overshoot fail; it does not cap memory.

The exact machine-readable design is
[`configs/tqc_resource_calibration_v0.study.json`](configs/tqc_resource_calibration_v0.study.json).
The throughput floor projects a 20-million-step bootstrap run below 48 hours;
it does not authorize that run.

SB3 seeds the five vector workers from the run seed by rank (`92001` through
`92005`). Treat all five derived streams as calibration-only.

## Run after review

```bash
uv run humanoid-harness tracker calibrate-tqc \
  --design experiments/bootstrap_tqc_humanoid/configs/tqc_resource_calibration_v0.study.json \
  --output artifacts/bootstrap_tqc_humanoid/resource_calibration_seed_92001.json \
  --confirm-disposable-resource-probe
```

The final flag acknowledges a one-time `5.26-GiB` replay-address traversal and
the sampled `12-GiB` failure threshold. It is not an OS memory-limit control.
Before runtime inspection, the runner reserves the exact output name with a
valid `in_progress` JSON receipt. Normal completion atomically exchanges the
final receipt; an interrupted process leaves the provisional status visible.

## Claim boundary

- A passing receipt means the exact disposable workload completed within the
  declared resource gates and its inspected signals remained finite.
- It does not show balance, locomotion, tracking, reference use, oracle quality,
  or controller readiness.
- `sb3-contrib` TQC is a bootstrap candidate, not the research contribution.

## Primary sources

- [Farama Humanoid expert description](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/descriptions/Humanoid-expert.md)
- [Farama MuJoCo training script](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/train.py)
- [SB3-Contrib TQC `v2.9.0`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.9.0/sb3_contrib/tqc/tqc.py)
- [TQC paper](https://arxiv.org/abs/2005.04269)

The Stable-Baselines3 Scientific Agent Skill informed the executable calibration
pattern. The project records its method citation in Experiment 001.

The successful resource receipt advances only to the bounded
[E1 initialization-identity fixture](E1_INITIALIZATION_IDENTITY.md). E1 itself
remains blocked until exact trained actor bytes exist.
