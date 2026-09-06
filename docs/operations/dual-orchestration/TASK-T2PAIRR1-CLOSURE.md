# T2PAIRR1: close the three exact pairing findings

- Work in `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`,
  branch `astra/reward-loop`. Read AGENTS.md and the dual-orchestration README.
- Peer source is read-only **committed Git objects**, never its changing tree:
  `df3c41d5980ac31efb5024798e20e473f36fa17b` in the shared object database.
  Use `git show COMMIT:path` and `git diff`; do not import or check out its files.
- Prior reviewed source: `0b4a8a1fda322eac1a12071593a79747cff1b8b3`.
- Read-only Sol/max leaf. No nested agents, writes, imports, tests, simulator,
  fake runtime, training, installs, network, calls or receipt regeneration.
- Target 8 minutes, at most 900 words. At 10 minutes stop reading and return
  a scoped partial verdict. Exact watchdog stops the runner at 20 minutes.

## Scope

Read `T2PAIR_REVIEW_RESULT.md` in this directory. Close only PAIR-01..03;
Fable owns the concurrent combined T2AR2/F3-dispatch review. Do not duplicate it.

1. PAIR-01: both arm-common projections must be independently type-validated,
   compared as exact canonical bytes, and those verified bytes hashed. Check
   `phase_b/contracts.py` and its actual-constructor alias negatives in
   `tests/phase_b/test_contracts.py`.
2. PAIR-02: regression must traverse the real paired PPO/runtime consumers, not
   merely a synthetic receipt or direct seed helpers. Trace the new capture test
   and disabling seam through `phase_b/training.py`, `phase_b/runtime.py`,
   `tests/phase_b/test_pairing.py`. Are assertions non-vacuous and sensitive to
   missing/misrouted action, minibatch, per-slot reset and RSI consumers?
   Inspect the actual consumer, not only the helper return capture. Distinguish
   fake environment dynamics from real scheduling/PPO implementation.
3. PAIR-03: protocol source digests and pairing receipt must bind the exact
   committed source. Inspect `experiments/004_t2_reward_study/PROTOCOL.md`,
   `pairing_receipt_v1.json`, `src/oracle_composition/reward_study/pairing.py`,
   and relevant tests in `tests/reward_study/`.

Parent confirmed from exact committed blobs:

- `contracts.py`: `c25d61de985f06ab69945842185fd004be3a623d3c984df11718ca1af53855f1`.
- Pairing receipt: `1a2b7ece139974117fd5c75e040d9cc52cd51a4e42c9b5afd794a60b02232348`.
- Full-run summary at this commit reports **1 failed, 1922 passed, 16 skipped**;
  do not call it a green suite. Failure is the historical reward no-learning
  receipt replay at `tests/experiments/test_reward_target_speed_manifest.py`.
- Later committed `78f6c7b7e243a1aab4372256a0e4891a78c48138` changes only that
  reward receipt and its regeneration record. You may inspect those exact two
  Git blobs to classify the separate failure, but do not rerun or treat receipt
  regeneration alone as a reproduced full-suite pass.

## Output

Start with progress/bottleneck/next step rows. Then a three-row closure table:
ID, closed/open/unverified, actual source and regression locator, rationale.

- Material new defects only in this repair delta, with smallest closure test.
- Distinguish source inspection, peer test claims and actual reproduced tests
  (none authorized). Preserve legacy non-paired behavior and distinct full arm
  execution/reward hashes. Common draws do not imply equal trajectories.
- Verdict: `ACCEPT_T2PAIRR1_STATIC_ONLY`, `REPAIRS_T2PAIRR1`, or
  `INCOMPLETE_T2PAIRR1`. Do not grant F3 dispatch, source integration, runtime,
  resource reservation, cohort or scientific-behavior approval.
- No broad hardening backlog or future-source acceptance. Return useful bounded
  findings even if time expires before every requested edge can be checked.
