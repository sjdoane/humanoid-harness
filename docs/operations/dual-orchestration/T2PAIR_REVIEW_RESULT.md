# T2PAIR independent review

| status | current truth |
|---|---|
| progress | Static review completed; paired consumers use the common primitive streams by inspection. |
| bottleneck | Common-field comparison admits type aliases; production paired execution lacks a regression; one protocol hash is stale. |
| next step | Fable repairs the exact findings at its clean checkpoint; narrow re-review before pairing acceptance. |

## Verdict and retained evidence

- Verdict: **`REPAIRS_T2PAIR`**, not pairing or runtime acceptance.
- Reviewed commit: `0b4a8a1fda322eac1a12071593a79747cff1b8b3`.
- Direct parent: `7698256dfe721c020c2aa264432867523e1dc99c`.
- Run: `.orchestration/sol-runs/20260906T103625Z-03f6ffe1-99f1-4b50-bc11-d817301ec04e`.
- Requested reviewer: Sol/max, read-only, no nested agents.
- Terminal: `SUCCEEDED`, exit 0, 2026-09-06 10:50:41 UTC.
- Final report SHA-256: `340ef1bf94c79a5522b2a8461b8b852e1922dcb42109eaaa327f7ea3b936bc33`.
- Reviewer ran no tests, repository Python, fake runtime, simulator or training.
  Peer test counts were not reproduced or treated as independent evidence.

## Required repairs

| ID | committed location | issue | smallest closure evidence |
|---|---|---|---|
| PAIR-01 | `phase_b/contracts.py:1051` | Dictionary equality compares common projections, then only baseline is type-validated and hashed. `1048576` equals `1048576.0`; `true` equals `1`, despite different canonical bytes | Validate both projections; compare canonical bytes; hash those verified equal bytes. Actual validator regressions for integer/float and boolean/integer aliases |
| PAIR-02 | `tests/phase_b/test_pairing.py:67`; `tests/reward_study/test_reward_pairing.py:79`; `tests/phase_b/test_training.py:45` | Helpers/counters and their receipt generator are tested, but the PPO integration tests use legacy non-paired plans | A bounded paired production-consumer regression observing action primitives, minibatch permutations and per-slot reset/RSI routing; a broken consumer branch must fail it |
| PAIR-03 | `experiments/004_t2_reward_study/PROTOCOL.md:21` | Recorded current `pairing.py` digest is stale | Correct the source ledger from exact committed bytes and rebind impacted receipts under the reviewed repair workflow |

All line numbers refer to the pinned reviewed commit, not Fable's changing tree.
T2AR2 already touches contract admission. Keep repairs with its owner; no
concurrent Astra patch to that file and no approval of unseen future bytes.

## Parent verification

- Inspected the complete `T2RewardPairing.__post_init__` path in the committed
  blob: candidate common fields are not independently validated before the
  baseline-only hash calculation.
- A standalone stdlib check confirmed both type-alias examples compare equal
  in Python and serialize differently. This corroborates the language mechanism;
  it is **not** an executed project-validator regression.
- Actual `reward_study/pairing.py` blob: 27,753 bytes, SHA-256
  `6fe7aae67bd23557eb5e914e366bace5fcec7273c271b5c242d6f2546ce7261f`.
- Protocol instead names
  `242e094325b91d3c5f309c158d585b5335ead5f7302045ad4197a818f641099f`.
- Pairing receipt bytes verified: 9,770 bytes, SHA-256
  `e5351c3b49ba97cc362ccd68a0bcf797c5077fb0c2ace78535b24e5567e0a259`.
  That receipt binds the correct source hash; the protocol text does not.

## What the review supports

- Actual code statically selects paired action and minibatch primitives;
  reset/RSI indexing includes environment slot and episode/cycle identity.
- Full baseline/candidate execution hashes remain distinct.
- Legacy undeclared mode retains its original derivation.
- Common primitive draws do not imply equal actions, trajectories, episode
  lengths, returns or training outcomes.
- No measured cohort is known to be contaminated: no cohort ran. These are
  pre-execution contract and regression-coverage defects.
- T2AR2 report/candidate/execution-seal gaps, resource reservations and cohort
  authorization remain separate and unwaived.

## Watcher limitation

- The watcher-start receipt exists with launch 10:36:25 UTC and deadline
  10:56:25 UTC. The reviewer finished normally before that deadline.
- At collection, neither the exact runner nor watcher was active, but the
  watcher-final receipt was absent. **Do not claim `terminal_observed` or
  verified deadline enforcement for this run.** Cause is not established.
- No process was signaled or restarted at collection. Preserve the original
  records; verify durable watcher lifecycle and retain stderr on the next
  launch instead of relying only on its start receipt.
