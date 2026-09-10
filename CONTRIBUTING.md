# Contributing

Start with [current status](docs/CURRENT_STATUS.md), the
[charter](docs/PROJECT_CHARTER.md) and [AGENTS.md](AGENTS.md). This is research
software with unresolved reproducibility gaps, not a validated robotics release.
No project license is selected yet; confirm contribution and reuse terms with
the maintainer before contributing third-party code or data.

## Development setup

Use Python 3.12–3.13 and `uv`, from the repository root:

```bash
uv sync --locked --extra sources --extra gym --extra train --extra dev
uv run humanoid-harness doctor
uv run ruff check .
uv run ruff format --check src tests scripts
uv build
```

- The root quick start needs only core dependencies. The evidence UI and some
  test imports also require the simulator/training extras above, even when they
  do not launch training.
- Motion/controller payloads are separate assets, not Python dependencies.
  Installing extras does not admit a reference library or create a runnable task.
- `doctor` checks dependencies and project files, not scientific readiness.
- Repository-wide formatting has pre-existing failures. The exact count and
  changed-file checks are recorded in [verification](docs/operations/SHARE_READINESS_20260910.md).

## Portable checks

This bounded selection uses repository fixtures and temporary outputs. It does
not require the private lab stack or downloaded controller assets:

```bash
uv run pytest -q tests/unit \
  tests/contracts/test_reference_contract.py \
  tests/contracts/test_program_contract.py \
  tests/contracts/test_source_motion_manifest.py \
  tests/research/test_knowledge_index.py \
  tests/harness/test_composition_contract.py \
  tests/harness/test_executor.py \
  tests/harness/test_cycle_cli.py \
  tests/ui/test_evidence_ui.py
uv run python scripts/verify_migration_manifest.py --repository-root .
```

The full suite is `uv run pytest`. It contains historical host-, asset- and
source-identity-dependent studies and is **not currently green**. The portable
selection does not replace it. See [verification](docs/operations/SHARE_READINESS_20260910.md)
for observed results and failures; do not bypass gates to obtain a passing badge.

## Make a reviewable change

- Use a separate branch/worktree. Preserve other researchers' dirty files and
  active workers; never edit code while a run depends on its exact bytes.
- Keep contracts/validation in core modules and CLI/UI code thin.
- Add a regression test for a corrected failure, including its negative path.
- Keep independent task metrics outside generated rewards. An oracle study
  freezes the reward; a reward study freezes the oracle.
- Record the exact source, inputs, configuration and outputs of each study.
  New source does not inherit a historical execution seal.
- Document changed behavior and limits. Keep old results and failed candidates;
  put corrections in a new, dated record.
- Use short, concrete comments for units, frames, invariants and failure modes.

Do not commit credentials, raw private transcripts, checkpoints, videos or
unlicensed source data. Propose asset packaging separately, with provenance and
redistribution rights established first.
