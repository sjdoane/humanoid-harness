# Independent review: A1 reward proposal loop

- Read `AGENTS.md`, dual-orchestration README, `TASK-A1-reward-proposal-loop.md`,
  `A1_RESULT.md`, and the exact A1 changed files.
- Role: independent read-only Sol/max reviewer, no nested delegation.
- Review only the Astra worktree's proposal/feedback implementation. Do not
  mutate files, start training, call a live model, or change the evaluator.
- Check prompt/dossier/parent binding, missing-data handling, strict response
  admission, byte limits, receipt completeness, immutable history, no accidental
  generated-code execution, and consistency with B0's future interface.
- Check that tests prove the negative paths claimed and do not merely mirror
  code. Confirm the imported module belongs to the Astra checkout.
- Return actionable findings with severity, concrete failure, exact path/line,
  regression test, and verdict: accept, accept with repairs, or reject.
- Separate software correctness from scientific validity. A1 has no live LLM,
  training, task-improvement, or cross-MDP behavioral evidence.
- Keep the report concise. The orchestrator decides and assigns repairs; this
  reviewer cannot edit or authorize its own findings.
