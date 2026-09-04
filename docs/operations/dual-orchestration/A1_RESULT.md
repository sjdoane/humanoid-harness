# A1 reviewed admission repair result

| status | current truth |
|---|---|
| progress | Closed all six independent-review findings in the portable A1 slice. The required focused command passes 59 synthetic/software tests. |
| bottleneck | No live LLM call, B0 reward validation, simulator run, training, or protected robot evidence has been performed. |
| next step | Run the authorized two-call read-only A2 proposal/revision canary with synthetic inputs; real B0 binding, validation, execution, and training remain separately gated. |

## Checkout and repair receipt

| item | value |
|---|---|
| checkout | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra` |
| branch | `astra/reward-loop` |
| starting HEAD | `e49b2dfa5bcadb3eff4f3eabb0477dc8d535ada1` |
| `uv.lock` SHA-256 | `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40` |
| state before this repair | Parent-modified `ASTRA_HANDOFF.md`; untracked A1 package, tests, and result. Preserved as expected. |
| review source | `.orchestration/sol-runs/20260904T204653Z-226c4610-e72e-4154-8c34-cf91843c0b2b/final.txt` |
| requested builder configuration | `gpt-5.6-sol`, reasoning `max`, leaf/no delegation; configuration evidence only, not served-model attestation |
| pre-repair reproduction | Required focused command: `35 passed in 1.97s` |

The repair edited only the assigned A1 package, A1 tests, and this result. It did
not edit the pre-existing handoff change, Git state, dependencies, B0, research
or experiment modules, scientific contracts, hooks, launchers, or leases.

## Review finding dispositions

| # | disposition | regression evidence |
|---|---|---|
| 1 | Closed: revision preparation now requires the retained prior `ModelCallReceipt` and `PacketRecord` in the Python API and CLI. It re-renders the packet and verifies packet bytes/hash, task-packet artifact hash, dossier, source bundle, parent, proposal, candidate, receipt, disposition, stage, and preceding-iteration links before rendering the new revision. | Eleven isolated link/disposition/stage mutations and a chained preceding-iteration mutation are rejected. The synthetic initial-to-revision success path remains accepted. |
| 2 | Closed: dossier `synthetic` is now part of the frozen revision boundary. A synthetic/live change starts a separate lineage. | Both `true -> false` and `false -> true` feedback relabels are rejected. |
| 3 | Closed with the scoped fail-closed option: `index_stats` and `query_index` are bracketed by canonical database identity checks, followed by a second metadata read. File identity or metadata drift rejects the bundle. | A stable generation succeeds; atomic replacement between metadata and query is rejected. |
| 4 | Closed: CLI JSON and retained run files share a descriptor-bound regular-file reader. It rejects nonregular/symlink inputs before a blocking open, checks size before allocation, reads at most the bound plus one byte, and checks descriptor/path identity and state before and after. Sol run files still require mode `0600`. | An oversized sparse JSON file is rejected without `Path.read_bytes`; a FIFO is rejected without opening; growth, truncation, and path replacement during a read are rejected. |
| 5 | Closed: ingestion requires `request.mode == "read-only"`. | Changing only the valid fixture's mode to `write` retains a rejected receipt and publishes no proposal. |
| 6 | Closed: the private alias package path plus the helper module's `__file__` and import-spec origin must resolve to this package's canonical `experiments/artifact_io.py`. | Fresh import resolves to this checkout without importing Gymnasium; preloaded wrong-origin alias and module cases fail closed. |

## Implemented API and CLI

- `prepare_initial_packet`, `ingest_sol_run`, deterministic rendering, and
  no-overwrite publication retain their A1 behavior.
- `prepare_revision_packet` now takes required keyword-only `prior_receipt` and
  `prior_packet` artifacts in addition to the prior dossier, proposal, iteration,
  feedback dossier, and current source bundle.
- `prepare-revision` now requires `--prior-receipt` and
  `--prior-packet-record`; every JSON input uses the bounded descriptor reader.
- Accepted and rejected ingestion receipts remain immutable. Generated reward
  source remains text in `awaiting_B0_validation`; A1 never imports or executes it.

## Paths

- Repair-touched paths:
  - `src/oracle_composition/reward_search/cli.py`
  - `src/oracle_composition/reward_search/loop.py`
  - `src/oracle_composition/reward_search/publication.py`
  - `tests/reward_search/test_reward_proposal_loop.py`
  - `docs/operations/dual-orchestration/A1_RESULT.md`
- Retained A1 paths present before this repair and left unchanged by it:
  - `src/oracle_composition/reward_search/__init__.py`
  - `src/oracle_composition/reward_search/__main__.py`
  - `src/oracle_composition/reward_search/contracts.py`

## Exact verification

| command | outcome |
|---|---|
| `.venv/bin/python -c "import oracle_composition; print(oracle_composition.__file__)"` | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/src/oracle_composition/__init__.py` |
| `.venv/bin/python -m pytest -q tests/reward_search tests/integration/test_research_mailbox.py` before repair | `35 passed in 1.97s` |
| `.venv/bin/python -m pytest -q tests/reward_search tests/integration/test_research_mailbox.py` after repair | `59 passed in 2.83s` |
| `.venv/bin/ruff check src/oracle_composition/reward_search tests/reward_search` | `All checks passed!` |
| `.venv/bin/ruff format --check src/oracle_composition/reward_search tests/reward_search` | `7 files already formatted` |
| `.venv/bin/python -m oracle_composition.reward_search --help` | Passed; exposes `prepare`, `prepare-revision`, and `ingest-sol` |
| `.venv/bin/python -m oracle_composition.reward_search prepare-revision --help` | Passed; exposes both required prior-artifact flags |
| `.venv/bin/python -c "...print package, helper, Gymnasium state..."` | Package and helper resolve inside this checkout; `gymnasium` loaded: `False` |
| `git diff --check` | Passed |

All tests are software-contract or declared synthetic plumbing evidence. They do
not demonstrate a provider-served model identity, reward quality, training
improvement, humanoid behavior, or robot competence.

## Minimal revision use

```python
from pathlib import Path
from oracle_composition.reward_search import (
    IterationRecord,
    ModelCallReceipt,
    PacketRecord,
    ProtectedEvidenceDossier,
    RewardProposal,
    SourceBundle,
    load_model,
    prepare_revision_packet,
)

revision = prepare_revision_packet(
    load_model(Path("prior-dossier.json"), ProtectedEvidenceDossier),
    load_model(Path("prior-proposal.json"), RewardProposal),
    load_model(Path("prior-iteration.json"), IterationRecord),
    load_model(Path("feedback-dossier.json"), ProtectedEvidenceDossier),
    load_model(Path("sources.json"), SourceBundle),
    prior_receipt=load_model(Path("prior-receipt.json"), ModelCallReceipt),
    prior_packet=load_model(Path("prior-packet-record.json"), PacketRecord),
)
```

```bash
.venv/bin/python -m oracle_composition.reward_search prepare-revision \
  --prior-dossier prior-dossier.json \
  --prior-proposal prior-proposal.json \
  --prior-receipt prior-receipt.json \
  --prior-packet-record prior-packet-record.json \
  --prior-iteration prior-iteration.json \
  --feedback-dossier feedback-dossier.json \
  --source-bundle sources.json \
  --packet revision.md \
  --packet-record revision-record.json
```

## Limits and next canary gate

- Retained-artifact checks establish consistency among supplied artifacts. They
  are not provider attestation or tamper-proof trust against an actor that
  rewrites every linked artifact consistently.
- The graph repair uses before/after identity and metadata checks around the
  existing query APIs. It adds no database framework and fails closed on the
  reviewed atomic-replacement race.
- The publication-origin check is a local-origin guard, not a Python sandbox.
- Real B0 binding requires an externally reviewed reward-contract hash and
  author surface. A synthetic-input plumbing canary does not require B0 and
  does not implement or infer its semantics.
- The package is 1,058 nonblank Python lines, above the original rough 600-line
  guide. The pre-review slice was already reported at 880; the added lines are
  bounded input/origin/snapshot checks and explicit retained-lineage validation.
  No scheduler, evaluator, sandbox, trainer, or database framework was added.

Independent targeted review returned `ACCEPT` at 2026-09-04 22:06:14 UTC; all
six findings were closed. Receipt: `.orchestration/sol-runs/20260904T215938Z-74e2803d-4a92-44ed-ab6b-2e45d8068c83/final.txt`.
Parent reproduced `59 passed in 2.33s` and accepted the software slice at the
22:49 UTC checkpoint. The next step is two bounded detached Sol requests with
retained mode `read-only` and explicitly synthetic inputs. Ingestion may admit
only proposal format; B0 validation remains required before execution/training.
