# OT1E: evaluator execution boundary only

- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`; branch `astra/reward-loop`.
- Source: `851e16d3b274e21a3acb8b5446519bca4d0fd811`; immutable bindings in
  `.orchestration/ot1-review-20260906/snapshot.json`. Later commits are docs only.
- Read AGENTS.md and the dual-orchestration README. This is an explicitly bounded
  leaf review, not a mandate to audit the repository. No nested agents.
- Read-only Sol/max. No file writes, tests, imports, installs, network, simulator,
  training, resource reservation or model candidate. Inspect source/Git only.
- Two prior broad OT1 reviews timed out with no verdict. Do not repeat them.
  Their partial comments are provisional and can be wrong; one apparent
  early-fall denominator issue was explicitly withdrawn after checking the ABI.
- Target an 8-minute report, at most 900 words. At 10 minutes stop reading and
  return findings/unverified items. Exact watchdog terminates at 20 minutes.
  Deliver your final even if incomplete; do not wait for perfect coverage.
- Start with progress/bottleneck/next step rows, then only the questions below.
  For each: confirmed/refuted/unverified, producer/consumer path and line,
  one minimum negative test. State a scoped verdict, never blanket OT1 closure.
- Findings are static inspection, not reproduced defects or behavioral evidence.
  Smoke/cohort gates remain unchanged. No code or reviewer can grant authority.

## Questions

1. Executed-source identity: `phase_b/evaluation_supervision.py::_worker`
   computes a source digest at a supplied repository root, while its imported
   evaluation function may already come from elsewhere. Does it assert the
   actual imported module root/bytes? Does the child consume/verify the complete
   evaluation-manifest binding sent in the admit message? Trace only
   `evaluation_supervision.py`, `evaluation_lineage.py`,
   `harness/cycle_cli.py` and the reusable helpers they call.
2. Bounded failure behavior: which training-side resource/isolation guarantees
   are actually applied to protected evaluation? Inspect spawning, bounded IPC,
   input verification, wall/RSS/CPU/output controls and cleanup in the evaluator.
   Report only concrete gaps against the existing declared contract; do not
   demand a new universal sandbox. Does cleanup overwrite the primary outcome?
   Could both facts be preserved without relaxing the existing fail-closed gate?

The nearest prior IDs are FT2-SCI-02 and FT2-ADV-02/03/05/09. Inspect relevant
tests in `tests/phase_b/test_evaluation_supervision.py`. Do not repeat the
whole training/supervision audit or declare all previous findings closed.

Host-test note: original traceback was not retained; a later full suite passed.
`tests/phase_b/test_cli_report_persistence.py::_limits` sets 20 s per seed and
30 s total. Timeout is a hypothesis, not the observed cause. Do not run this
test or call it harmless. Recommend one bounded diagnostic if needed.

Separate blockers for running a training-only disposable smoke from blockers
for protected evaluation and published scientific claims.
