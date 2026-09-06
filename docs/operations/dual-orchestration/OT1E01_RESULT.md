# OT1E-01 evaluator-manifest admission result

| status | current truth |
|---|---|
| progress | Bound canonical manifest bytes, SHA-256, byte count, frozen selectors, and executed evaluator-source identity through parent preflight and the child start/admit gate. The authorized pure suite passes 20 tests. |
| bottleneck | Existing source-bound receipts may be stale. Calibration provenance, resource isolation, cleanup completeness, bounded result reads, simulator behavior, and robot competence remain unverified. |
| next step | Parent inspects this uncommitted slice, obtains independent review, and decides whether to commit; shared CLI integration still requires peer acceptance. |

## Checkout receipt

| item | observed value |
|---|---|
| checkout realpath | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra` |
| branch | `astra/reward-loop` |
| code base | `3406bb5f0af3c2479d47cdd2d245c3e738a168fe` |
| starting HEAD | `10add85b070d693c22b1a7aa8509e5301a4942e9` |
| starting dirty state | clean |
| launch-HEAD delta | documentation only |
| `uv.lock` SHA-256 | `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40` |
| requested worker | `gpt-5.6-sol`, max reasoning, no nested agents |
| commit/push | not performed |

## Implemented capability

- The frozen request carries manifest path, digest, and byte count.
- Parent admission reads the real artifact with `read_verified_artifact_bytes`,
  checks the published/request binding, canonical schema, and frozen selectors
  before `_spawn`.
- The child independently rereads and validates the manifest, compares the full
  executed evaluator-source identity, and exchanges its computed digest/count
  in both `started` and `admit` before checkpoint/episode construction.
- The schema remains the existing v1 evaluation manifest: checkpoint and
  starting actor SHA-256s, three target speeds, four cells, seeds
  120101--120120, and 160 episodes.

## Verification

| check | observed result |
|---|---|
| pre-repair red | 1 failed: missing admission digest reached the checkpoint sentinel |
| authorized pure file after repair | 20 passed in 1.11s |
| worker negatives | 16 negative cases kept checkpoint/episode entry at zero and emitted no completion |
| parent negatives | 3 byte/digest/count mismatches refused before the spawn sentinel |
| Ruff check | passed on four changed Python files |
| Ruff format check | 4 files already formatted |
| import origin | `src/oracle_composition/__init__.py` inside this checkout |
| whitespace | `git diff --check` passed |

Not run: `test_evaluation_supervision.py`, full suite, simulator, fake or real
training, calibration/source-receipt generation, network, or candidate calls.
These software-contract checks do not establish evaluator calibration or
humanoid performance.

## Changed paths

1. `src/oracle_composition/phase_b/evaluation_supervision.py`
2. `src/oracle_composition/harness/cycle_cli.py`
3. `tests/phase_b/test_evaluation_supervision.py`
4. `tests/phase_b/test_evaluation_admission.py`
5. `docs/operations/dual-orchestration/OT1E01_RESULT.md`

## Parent closure, 2026-09-06

- Detached builder finished at 07:38:42 UTC, exit 0; writer lease released.
- Parent reproduced the pure admission suite: **20 passed in 1.29s**.
- Parent inspected the production diff and reproduced Ruff; both passed.
- Independent read-only Sol/max reviewer: `ACCEPT_OT1E01_MANIFEST_ONLY`.
- Reviewer and parent hashes agree on all four Python files:

| file | SHA-256 |
|---|---|
| `evaluation_supervision.py` | `7993fa1abd1f63696a7f30e005552deb86d7664627548ccd1e672d26284dc2a4` |
| `cycle_cli.py` | `141d7ec4e7bc935c2d2487b804b88de3f3d4c6a5fae3ec922c6603c3436c06c5` |
| `test_evaluation_supervision.py` | `927756a77a69eefd15119debfbeac37e0bd03110f610b634685f74b74160fd22` |
| `test_evaluation_admission.py` | `d483ca02c29c061d41b95319da0932730303e92c2afb39cc04fb24997a4553e8` |

Review limit: the exported supervisor's parent path checks manifest integrity;
fresh pre-spawn source hashing is supplied by its production CLI caller. The
child always rehashes before evaluation. Static review did not reproduce tests;
the parent result above is separate evidence.

Scoped verdict: accepted for manifest admission only. Fable owns integration
into main. Calibration, resource isolation, cleanup, result-read limits and
scientific behavior remain separate gates; no receipt was regenerated.
