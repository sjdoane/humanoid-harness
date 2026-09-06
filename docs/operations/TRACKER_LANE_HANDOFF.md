# Oracle and tracker lane handoff (Fable to Astra, 2026-09-06)

| status | current truth |
|---|---|
| progress | The loop runs end to end on stock `Humanoid-v5` (Experiment 003 phase A, three cycles, infeasibility finding). The phase B fine-tuning runtime is implemented and repaired through FT2R3: full-authority warm start with a bitwise E1 receipt, nearest-state reference composition, tracking-only baseline reward, PPO worker with supervision, persistence, report v2, isolation, and the F2 reward registry entry. No training has run. |
| bottleneck | The combined scientific and robustness re-review of FT2R1 to FT2R3 has not run; production-load controls are unmeasured; the phase B endpoint tolerances need a disjoint calibration split; the sealed E5 blocks are exposed. |
| next step | Astra dispatches the re-reviews, proposes the disposable smoke reservation, and writes the next composition packet; any cohort needs Samuel's authorization. |

## Commit chain in `main` (oldest first, tracker lane only)

| commit | content |
|---|---|
| `41b2597`, `a09c804` | Slice 03B and 03BFIX: plain-runtime reference corpus v2 (`128/128` replay and plain-comparison certificates), E3 fork corpus (`28/36` admitted blocks), expert development screen failed (`19/20`) |
| `50fd60c`, `5dfd701`, `74d7aa5` | Experiment 003 phase A: cycle CLI, library manifest, task T1, cycles 0 to 2 |
| `6dc946b` | E003R1: harness evidence chain (sealed execution manifest, prior-chain validation, separated identities, deterministic scientific receipts, designer provenance, bounded recovery) |
| `58e71aa` | FT1: phase B contracts, full-authority warm start, phase transfer, tracking-only reward |
| `f6287e5` | FT2: training worker, supervision, persistence, report v2, CLI `train` and `evaluate-policy` |
| `5965f0c` | FT2R1: training-admission repairs (E1 admission binding, task-input admission certificate, execution seal, calibration procedure, bounded actor reload) |
| `2930cd6` | FT2R2: evaluator-owned protected metrics, evaluation lineage, non-scoring endpoint, rollout likelihood audit, report completeness, mailbox-bound reservations, cohort authority, F2 registry entry |
| FT2R3 | isolation: sealed-input lineage in the worker, clean environment and resource gates, bounded IPC, fail-closed RSI evidence, mechanism-level negatives (commit hash in the handoff message) |

Integration commits (`Integrate Slice …`) carry the handoff, the host verification record `experiments/bootstrap_tqc_humanoid/reviews/fable_host_verification.json`, and the regenerated reward-lane receipt.

## Where things are

| item | path |
|---|---|
| Strategy, preregistration, phase B design, corrections | `docs/strategy/RESEARCH_STRATEGY.md` sections "Corpus result", "Preregistration", "Alignment audit", "Composition loop results", "Corrections and phase B endpoint", "Phase B design decisions" |
| Decisions | `docs/decisions/0008_alignment_pivot_and_tracker_family.md`, `0009_lane_swap.md` |
| Harness cycle CLI and evidence chain | `src/oracle_composition/harness/` (`cycle_cli.py`: `prepare`, `evaluate`, `train`, `evaluate-policy`; `evidence.py`) |
| Phase B runtime | `src/oracle_composition/phase_b/` (`contracts.py` registry and seals, `policy.py`, `reference_runtime.py`, `reward.py`, `training.py`, `supervision.py`, `persistence.py`, `evaluation.py`, `protected_metrics.py`, `evaluation_lineage.py`, `isolation.py`, `report_v2.py`) |
| Experiment 003 | `experiments/003_composition_speed_profile/` (`TASK.md`, `PROTOCOL.md`, `task_spec_v1.json`, `library_manifest_v1.json`, `execution_manifest_v1.json`, `cycles/cycle_0..2/`, `phase_b/DESIGN.md` and frozen artifacts and receipts) |
| Corpus | `experiments/reference_corpus_v1/` (protocol, E3 manifest, validation manifests v1 and v2); ignored payloads under `artifacts/reference_corpus_v2/` |
| Reviews | `.orchestration/sol-runs/<run>/final.txt` for E003 (`20260905T191153Z-*`), FT1 and E003R1 (`20260905T214317Z-*`, `20260905T214318Z-f43d13f9-*`), FT2 (`20260905T231138Z-*`) |
| Packets | `.orchestration/task-packets/TASK-2026090[56]-*.md`; generic review packets `TASK-REVIEW-SCI-GENERIC.md`, `TASK-REVIEW-ROBUSTNESS-GENERIC.md` driven by `.orchestration/review-target-{sci,adv}.txt` |
| Hardening backlog | `docs/operations/HARDENING_BACKLOG.md` (E003-R05, R06, R10 deferred) |

## Open items transferred

1. Combined re-review of FT2R1 to FT2R3 (scientific and robustness), then fold findings.
2. Disposable smoke: seed `121901`, `196,608` transitions, expected `3 min`, hard `20 min`, `--smoke` non-promotable, label `interface_check`; needs a mailbox reservation accepted by the peer; the reservation is bound by digest to the acceptance record (FT2R2).
3. Five-seed cohort (`121001`-`121401`, `1,048,576` transitions each, about `106 min`, hard `120 min`) needs Samuel's explicit authorization; asked, not yet given.
4. Phase B endpoint tolerances: calibration on a disjoint split of blocks and seeds, procedure frozen before candidate evaluation (calibration module exists; no receipt yet).
5. Sealed E5 blocks `120201`-`120204` were read by the authority-gap analysis; redraw before any confirmatory E5; E5 is optional science under ADR 0008.
6. The expert failed its frozen development screen (`19/20`); recorded, not rerun; every downstream claim carries its own gates.
7. Composition cycles for phase B: the cycle-1 oracle (`experiments/003_composition_speed_profile/cycles/cycle_1/oracle_1.json`) is the reference program the runtime binds; the designer-prompt path and the read-only designer pattern (E003D1, E003D2 packets) are reusable; steering must be attributed to its author.

## Rules that travel with the lane

- Plain `Humanoid-v5` is the frozen MDP; the collector, screen, and certifier never call `mj_forward` or synthesize observations.
- No seed replacement; failed blocks stay in their role's denominator; admission comes from the E3 certificate of the corpus run a packet binds.
- The PPO seed and final checkpoint is the independent unit; episodes are repeated measures.
- Interface checks carry `interface_check`; trained cycles `exploratory_fine_tuning_cycle`; `matched_study` is never used on this chain.
