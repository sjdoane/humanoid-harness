# Humanoid Harness

Research software for **automating post-training for dynamic humanoid tasks**.
The harness turns task intent and rollout failures into revisions to two inputs:
**reference/oracle guidance** and **task reward**. It continues the RL-Sculptor
research line; the target integration is the lab's SONIC-based VIBE system.

The question is whether this loop improves task success with fewer training
interactions and less manual tuning. Reference composition supports that goal;
it is not the goal by itself.

```text
INPUT: task + reference library + initial reward
                         |
             human or LLM proposes a revision <---- development feedback
                         |                                  ^
               validate + version oracle/reward              |
                         |                                  |
              supplied post-training adapter ------> independent evaluation
                         |
             OUTPUT: policy + rollouts + metrics + effort

Frozen per comparison: task/scene, initial controller, permitted trainable
parameter set, training method/budget, feedback access and evaluator.
Current adapter: GMT/G1 proxy. Target adapter: VIBE/SONIC (not integrated).
```

## Current state · September 10, 2026

| | Status |
|---|---|
| **progress** | GMT/G1 implements state-triggered references, residual PPO, bounded reward recipes, rollout feedback and LLM-proposed revisions. Retained development records cover **38 training runs / 2,424,832 training transitions**. |
| **bottleneck** | **No full-task pass or demonstrated tuning-efficiency gain.** VIBE integration and the four semester task environments are not admitted. The full test suite is not green. |
| **next step** | Confirm the lab's controller/trainer interface and task assets, starting with loaded-cart readiness; complete effort accounting before a fixed/manual/harness comparison. |

- **Planned tasks:** volleyball, floor hockey, loaded-cart interception and
  basketball (dribble, shoot, layup). These are not completed demonstrations.
- **Existing evidence:** flat-ground walk/crouch/rise/walk studies. Real LLM
  oracle and reward revisions were trained or probed and rejected when they
  failed their criteria. A blue posture region is not a physical obstacle.
- **Scope:** GMT/G1 is a development proxy, not VIBE. Mode changes use measured
  state; within-clip playback remains clock-driven.
- **Automation boundary:** agents coordinate proposals and run launches; this
  is not yet an unattended, one-command post-training system.

[Current results and limits](docs/CURRENT_STATUS.md) ·
[Research goal and tasks](docs/PROJECT_CHARTER.md) ·
[Architecture](docs/SYSTEM_ARCHITECTURE.md)

## Quick start: inspect the project

Requires Python 3.12–3.13 and [uv](https://docs.astral.sh/uv/).
Run these commands from the cloned repository root:

```bash
git clone https://github.com/sjdoane/humanoid-harness.git
cd humanoid-harness
uv sync --locked
uv run humanoid-harness doctor
uv run humanoid-harness status
uv run humanoid-harness research build
uv run humanoid-harness research query "phase recovery"
uv run humanoid-harness g1 --help
```

This installs the inspection tools and builds a local literature index. It does
**not** download a controller, train a policy or certify experiment readiness.
Optional simulator/training dependencies, test commands and reproduction limits
are in [Contributing](CONTRIBUTING.md).

With the optional dependencies installed, `uv run humanoid-harness ui` starts a
read-only local evidence server. G1 replays
require a local artifact registry; videos and large run artifacts are not bundled
in Git. Committed study reports are readable without those downloads.

## Where to look

- [Documentation guide](docs/README.md): current contracts, results and history.
- [Research evidence](research/README.md): source provenance and queryable index.
- [Experiments](experiments/README.md): study families and measured outcomes.
- `src/oracle_composition/`: adapters, oracle/reward contracts, feedback, CLI and UI.
- `tests/`: contract, negative-path, integration and host-reproduction checks.

Research prototype, not a validated robotics release. No project license has
been selected; do not assume redistribution rights. Third-party sources retain
their own attribution and restrictions.
