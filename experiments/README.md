# Experiment registry

| ID | Purpose | Admission state |
|---|---|---|
| `000_toy_positive_control` | Prove the evaluator can detect useful state-conditioned selection | planned |
| `bootstrap_tqc_humanoid` | Calibrate a disposable stable-locomotion bootstrap path | E0 resource pass; E1 synthetic transfer fixture implemented; no trained controller |
| `001_humanoid_fixed_reference` | Establish static tracker feasibility only | interface and evaluator implemented; no trained checkpoint |
| `002_humanoid_reference_use` | Install a time-varying reference and test exact/zero/shuffled/shifted causal use | offline numeric-sensitivity probe passed; behavioral gate blocked by tracker admission |
| `003_humanoid_manual_oracle` | Test phase- and state-aware transitions/recovery | blocked by `002` |
| `004_humanoid_generated_oracle` | Compare bounded generated variants | blocked by `002` and `003` |

Each experiment directory will contain its hypothesis, frozen manifest,
preregistered metrics, seed/budget plan, acceptance gates, and compact receipts.
Run outputs belong under `runs/` and remain untracked until explicitly admitted
as content-addressed evidence.
