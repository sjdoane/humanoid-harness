# Humanoid Harness

An auditable research harness for generating and testing state-aware reference
oracles and executable task rewards around a fixed policy-training system.

```text
task + references + r0
          |
          v
  LLM design harness <------ literature graph
          ^   |              human steering
          |   | O_k, r_k
          |   v
  diagnosis   fixed training adapter ----> policy
          ^                              |
          +------ protected evaluator <--+
```

| Evidence class | Current state |
|---|---|
| **research target** | Autonomous, steerable iteration over reference oracle `O_k` and task reward `r_k` |
| **implemented capability** | Immutable oracle/reference contracts, deterministic state-machine runtime, observation-preserving corpus capture with per-step plain-runtime receipts, full-integration replay bundles with predecessor-transition cache reconstruction, E3 fork checks, Gymnasium adapter, protected mechanical metrics, canonical trace contract, research index, unified CLI, and read-only status UI |
| **measured evidence** | Run v2: `108/108` named three-actor corpus clips and `20/20` expert screen clips passed separate-process full-clip replay and per-step plain-runtime comparison. The corpus qualified `28/36` E3 blocks; the expert failed the predeclared screen because only `19/20` resets stayed healthy and upright, although velocity and displacement gates passed. No stable-tracking, formal causal-use, E4, E5, oracle-improvement, reward-improvement, naturalness, robustness, or cross-MDP result |
| **Experiment 003 phase A** | Closed after cycles 0-2 on seeds `97001`-`97020`; every determinism replay matched `20/20`. All `60/60` episodes across the three switching designs fell after the running-speed handover. The cycle-2 designer therefore held `expert` throughout: `0/20` falls and median speed MAE `2.0742 m/s`, exactly matching `single_fast` while ignoring the slow third. See the [phase-A protocol and result](experiments/003_composition_speed_profile/PROTOCOL.md), [cycle-2 report](experiments/003_composition_speed_profile/cycles/cycle_2/report_2.md), and [cycle record](experiments/003_composition_speed_profile/cycles/cycle_2/cycle_record.md). This is `exploratory_oracle_cycle`; controller switching stood in for tracker following and supports no oracle-quality or tracker claim. Phase B keeps T1 and uses a tracker-following fine-tuning runtime warm-started at the expert. |

## Quick start

Requires Python 3.12 or 3.13 and `uv`.

```bash
uv sync --locked --extra sources --extra gym --extra train --extra dev
uv run humanoid-harness doctor
uv run humanoid-harness research build
uv run humanoid-harness research query "phase recovery"
uv run humanoid-harness trace inspect PATH/TO/trace.json
uv run humanoid-harness ui
```

The UI opens a local, read-only view. It does not grade a run or keep a second
copy of research state. On loopback only, it can display the falling local
exploration and the reviewed Experiment 002A report. Missing, tampered,
unlisted, or symlinked artifacts fail closed. Neither view establishes stable
tracking or oracle quality.

## Research orchestration

The optional orchestration path keeps research strategy with a verified Fable
5.1 session and delegates bounded implementation and independent reviews to
GPT-5.6 Sol workers.

```bash
claude auth login
./scripts/orchestration-doctor
./scripts/start-fable-orchestrator
```

The launcher asks before its single no-tool identity request because
noninteractive Fable calls can draw usage credits without an in-app consent
dialog. Do not run the doctor's optional `--live` mode first unless you intend
to authorize a separate probe.

Read the [current handoff](docs/operations/CURRENT_RESEARCH_HANDOFF.md),
[orchestration guide](docs/operations/MULTI_AGENT_ORCHESTRATION.md), and
[strategy ledger](docs/strategy/RESEARCH_STRATEGY.md). This path does not
authorize the real 1M-step attempt or change the evidence boundary.

## First development adapter

| Component | Current choice | Role |
|---|---|---|
| Robot/MDP | Gymnasium `Humanoid-v5` | Public integration substrate |
| Stable-motion bootstrap | TQC | Resource probe first; never an oracle result |
| Tracker-family screen | Frozen-TQC residual PPO, expanded TQC, direct PPO + RSI | Development selection before tracker admission |
| First study | Static tracker-admission prerequisite | Establish static feasibility; causal reference use is a distinct next study |
| Final adapter | Lab-supplied controller/training system | Separate future experiment family |

The immediately falling Humanoid seen in the current smoke evidence is an
interface check. It is not a trained-policy or oracle demonstration.

## Source data boundary

| Source | Local use | Current ceiling |
|---|---|---|
| Farama `Humanoid-v5` TQC expert, medium, and simple actors | Hash-pinned source bytes projected to strict NPZ; `108` corpus and `20` expert screen trajectories retained locally with per-step plain-runtime receipts | Qualifying E3 fork corpus (`28/36` blocks) only; exact expert failed the predeclared screen (`19/20` healthy and upright), so no tracker or broader behavior claim |
| Minari `mujoco/humanoid/expert-v0` | Exact commit, file sizes, and SHA-256 values are registered; episode 0 projects data-only into the 45D reference ABI | Tier K only; root x/y is absent and dataset redistribution rights are unresolved |
| DeepMimic `humanoid3d` motions | Exact MIT-licensed source files are audited in the research tree | Format/source evidence only; retargeting and Tier-D certification remain undone |

The registered Minari importer rejects any file, source-record, HDF5 storage,
shape, or environment drift before reading episode arrays. It never loads an
external checkpoint or grants training admission.

## Repository map

```text
docs/                 charter, architecture, boundaries, protocols, decisions
research/             literature corpus, source records, and provenance
src/oracle_composition/
  contracts/          immutable artifact and ABI validation
  runtime/            deterministic oracle execution
  envs/               policy-training adapters
  tracking/           reference state and tracking-reward candidate
  evaluation/         structural evidence admission
  experiments/        frozen runner and protected evaluator
  research/           deterministic research index and retrieval
  sources/            bounded data-only source audits and projections
  traces/             bounded synchronized trajectory contract
  ui/                 read-only local evidence surface
tests/                 contract, negative, integration, and UI tests
experiments/           declared protocols; run outputs remain untracked
praxist/               optional outer-loop orchestration notes
```

## Evidence boundary

- A passing test proves software behavior, not humanoid competence.
- The agent may change only the factor declared by a study.
- Generated artifacts never compute their own success metric.
- Every claim must retain exact artifact, runtime, seed, and evaluator lineage.
- PRAXIST may orchestrate experiments; it does not own scientific truth.

Read the [project charter](docs/PROJECT_CHARTER.md), [system architecture](docs/SYSTEM_ARCHITECTURE.md), [scientific boundary](docs/SCIENTIFIC_BOUNDARY.md), and [engineering tool review](docs/ENGINEERING_TOOL_REVIEW.md) before extending the system.

The repository is currently private and no project license has been selected.
Imported research records retain source-level attribution and provenance.
