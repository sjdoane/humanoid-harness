# T2PAIR: independent pairing-only review

## Scope and stop rule

- One read-only `gpt-5.6-sol`, max reasoning, no nested agents.
- Aim for a decision in 10 minutes; stop expanding at 15 minutes. The existing
  detached-run watchdog ends the exact runner at 20 minutes.
- Read this checkout's `AGENTS.md` and the dual-orchestration `README.md`.
- Worktree: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Branch: `astra/reward-loop`. No checkout, merge, source edits or commits.
- Review **committed Git objects**, not either lane's changing working files.
  Use `git -C /Users/samueldoane/Documents/ChatGPT/humanoid-harness show COMMIT:PATH`
  and `git diff BASE COMMIT -- PATH` as read-only source access.
- Reviewed slice: `0b4a8a1fda322eac1a12071593a79747cff1b8b3`.
- Its base: `7698256dfe721c020c2aa264432867523e1dc99c`.
- Later context only: `03a482600656f1a91d2f588ea4f62388eb81367e` includes
  receipt regeneration and advisory docs. Do not infer those close review gaps.
- No network, installation, simulation, full suite, training (including fake
  runtime), candidate calls, receipt regeneration or process/lease mutation.
- Do not run repository Python or tests. This is static source/evidence review.
  Record that limit; do not report peer test counts as your reproduction.

## Authority and known exclusions

- Astra accepted the path scope in message
  `20260906T072706.830333Z-a50cb508191d4d629b4ba6c8f9e42103`.
- Fable's committed handoff:
  `20260906T102035.254055Z-d65272207ef446eeba0cc6d16c5ec3e8`.
- T2AR1 is ACCEPT-WITH-REPAIRS. Final-ready report admission, registry-resolved
  candidate, verified execution seal and deeper negatives remain under Fable's
  T2AR2 repair. Do not duplicate that repair or assess its dirty source.
- A pairing-only acceptance is **not** reward-study, dispatch, real-runtime,
  calibration, resource, training/cohort or behavioral approval.

## Source route

Inspect the T2PAIR diff, then follow only necessary committed dependencies:

1. `src/oracle_composition/phase_b/contracts.py`
2. `src/oracle_composition/phase_b/training.py`
3. `src/oracle_composition/phase_b/runtime.py`
4. `src/oracle_composition/reward_study/pairing.py`
5. `tests/phase_b/test_pairing.py`
6. `tests/reward_study/test_reward_pairing.py`
7. Changed fixtures in `tests/phase_b/test_contracts.py`, `test_training.py`,
   and `tests/reward_study/test_pairing.py`, `test_t2_artifacts.py`.
8. `experiments/004_t2_reward_study/PROTOCOL.md` and `pairing_receipt_v1.json`.

Expected committed pairing receipt: 9,770 bytes, SHA-256
`e5351c3b49ba97cc362ccd68a0bcf797c5077fb0c2ace78535b24e5567e0a259`.
Verify actual Git-blob bytes; this is a synthetic-runtime mechanism receipt,
not a trained policy or a simulator trace.

## Questions to decide

- Does declared paired mode require exact canonical study and both arm bytes,
  derive its common key, and reject forged/missing/mismatched authority before
  any policy or environment factory? A caller-supplied digest alone is not proof.
- Are only reward, arm label, output path and timestamps excluded? Retain each
  distinct full execution identity. A changed common task/config must change
  the pairing key; the reward-only change must preserve it.
- Do actual training consumers use the common key for action-noise primitives,
  minibatch permutations and composition/reset/RSI schedules? Check every
  event index and stream tag, not only a helper function or receipt generator.
- Are reset/RSI episode counters independent per environment slot so different
  termination times cannot shift another slot's indexed schedule?
- Does undeclared legacy mode preserve its old manifest-derived streams and
  explicitly remain non-paired? No silent partial opt-in.
- Do tests reach production paths and meaningful refusal boundaries? Distinguish
  a receipt recomputing expected hashes from independent observation of consumers.
- Is the claim limited to common primitive variates/indexed schedules? Different
  policies need not have equal actions, trajectories, episode lengths or returns.
- Identify issues introduced by this slice separately from the known T2AR2
  dispatch/admission gaps. Do not waive either category.

## Deliverable

- Verdict: `ACCEPT_T2PAIR_STATIC_ONLY`, `REPAIRS_T2PAIR`, or
  `INCOMPLETE_REVIEW`; no generic full-system acceptance.
- Give exact committed paths and line numbers for substantive findings.
- State strengths, blocking issues, residual limits and smallest next check.
- Include inspected commit/base and the verified receipt hash/byte count.
- Say explicitly that no tests, policy training or simulator ran in this review.
- Return final text to the launcher; do not create files yourself.
