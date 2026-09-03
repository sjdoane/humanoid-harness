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
| **implemented capability** | Immutable oracle/reference contracts, deterministic state-machine runtime, Gymnasium adapter, protected mechanical metrics, canonical trace contract, research index, unified CLI, and read-only status UI |
| **measured evidence** | Interface and regression checks only; no trained tracker, oracle improvement, reward improvement, or cross-MDP result |

## Quick start

Requires Python 3.12 or 3.13 and `uv`.

```bash
uv sync --locked --extra gym --extra train --extra dev
uv run humanoid-harness doctor
uv run humanoid-harness research build
uv run humanoid-harness research query "phase recovery"
uv run humanoid-harness trace inspect PATH/TO/trace.json
uv run humanoid-harness ui
```

The UI opens a local, read-only view. It does not grade a run or keep a second
copy of research state.

## First development adapter

| Component | Current choice | Role |
|---|---|---|
| Robot/MDP | Gymnasium `Humanoid-v5` | Public integration substrate |
| Trainer | PPO | Development adapter |
| First study | Static tracker-admission prerequisite | Establish static feasibility; causal reference use is a distinct next study |
| Final adapter | Lab-supplied controller/training system | Separate future experiment family |

The immediately falling Humanoid seen in the current smoke evidence is an
interface check. It is not a trained-policy or oracle demonstration.

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
