# Engineering tool review

**Checked:** 2026-09-02
**Evidence class:** current engineering survey, not scientific result

## Decision table

| Decision | Tool | Use here | Boundary |
|---|---|---|---|
| Adopt the schema pattern now | [HumanTracker](https://github.com/GalaxyGeneralRobotics/HumanTracker) | Define one canonical trajectory with robot state, controller targets, contacts, references, and motion-quality metrics | Do not import its G1-specific model, preference score, data, or weights |
| Pilot after the trace exists | [Rerun](https://github.com/rerun-io/rerun) | Synchronized robot/reference, video, events, and time-series episode inspection | Derived viewer only; protected metrics read the canonical trace |
| Pilot after real runs exist | [MLflow](https://github.com/mlflow/mlflow) | Compare run parameters, artifacts, checkpoints, and LLM traces | Store links and hashes; do not select checkpoints or own claims |
| Use for a later graph view | [Cytoscape.js](https://github.com/cytoscape/cytoscape.js) | Visualize paper → mechanism → failure → metric → decision → run relations | SQLite remains the graph source of truth |
| Reconsider when commands grow | [Typer](https://github.com/fastapi/typer) | Typed nested commands and generated completion | Current standard-library CLI is sufficient for the first surface |
| Study only | [mink](https://github.com/kevinzakka/mink) | Prototype exact-ABI motion retargeting | IK gives a kinematic candidate, not dynamics feasibility |
| Adapter reference only | [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) and [mjlab](https://github.com/mujocolab/mjlab) | Learn controller-neutral interfaces and diagnostics | Different robot, DoF, observations, dynamics, and compute stack |

## Do not add now

| Tool family | Reason |
|---|---|
| LangGraph or PydanticAI | Duplicates PRAXIST orchestration before the scientific loop is runnable |
| GraphRAG | LLM-extracted edges cannot replace source-located scientific provenance |
| Streamlit or Gradio | Would create a second UI stack without solving trajectory inspection |
| Eureka or Text2Reward code | Useful reward-design references, but direct integration changes the stack; Text2Reward also lacks a repository license |

## Immediate architecture implication

```text
canonical trajectory trace
        |
        +--> protected evaluator --> immutable metrics
        |
        +--> Rerun recording -----> episode inspector

source extractions --> SQLite graph --> CLI now / Cytoscape later
run receipts -----------------------> MLflow links later
PRAXIST ----------------------------> bounded outer-loop scheduling
```

No external runtime dependency was added by this review. The next tool adoption
decision occurs only after `TrajectoryTrace/v1` is specified and tested.
