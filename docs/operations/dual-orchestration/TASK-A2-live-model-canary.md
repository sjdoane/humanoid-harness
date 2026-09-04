# A2: live model, synthetic inputs

| status | protocol |
|---|---|
| progress | A1 prepares and admits deterministic proposal/revision artifacts; focused software tests pass. |
| bottleneck | A1 final repair review accepted all six fixes. No live model round trip has been admitted yet. |
| next step | After A1 acceptance and commit, run exactly two read-only Sol calls: initial proposal, then synthetic-feedback revision. |

## Claim and limits

- Question: can the actual detached subscription-authenticated launcher return a strict proposal, then revise it through the retained A1 lineage?
- Claim ceiling: live model plumbing with synthetic evidence. No reward-semantic, robot-performance, generalization, or served-model-attestation claim.
- B0 handoff is **not** a prerequisite for this data-only canary. Its contract, author surface, feedback, and reference identities are explicitly synthetic.
- Reviewed B0 handoff remains mandatory before real contract binding, reward validation, execution, or training. Do not import unfinished B0 or copy its changing files.
- No generated source is executed, imported, evaluated, or called safe. Preserve A1's `awaiting_B0_validation` and `not_authorized` states.
- Two live calls total, sequential, `gpt-5.6-sol` / `max`, `read-only`, no nested work. No retry-to-pass selection. Retain failed calls as results.
- No network installs, paid API keys, simulator, full suite, heavy compute, push, or publication. Use this worktree's venv and existing launcher.

## Inputs frozen before call 1

Keep retained files in a fresh ignored `.orchestration/a2-canary/` child directory; never overwrite a previous run.

| input | required content |
|---|---|
| Synthetic author contract | A task-term function over numeric current/target velocity; explicit names, units, source-size bound; no real adapter claim |
| Synthetic baseline | Literal constant-zero source, SHA-256 of its exact bytes |
| Dossier | Synthetic IDs and provenance; fixed task/adapter/author/frozen configuration; explicit missing measurements |
| Source bundle | Explicitly synthetic fixture cards, with synthetic locators; no invented paper citation or actual-graph claim |
| Feedback fixture | Predeclared synthetic normalized target-error value `0.75`; not computed from a policy, reward, or simulator |

- Hash the retained canonical contract/config/feedback bytes; do not substitute arbitrary repeat-character hashes in the canary receipt.
- Task: propose a bounded numeric velocity-target reward term as text. State that the function ABI is synthetic and no environment ran.
- Initial missing evidence: real baseline trace, physical guardrails, B0 semantic validation, protected evaluator result.
- Feedback does not replace these missing measurements; it tests whether an explicit synthetic feedback field reaches a linked revision.

## Two-call sequence

1. Record checkout, branch, accepted A1 commit, lock hash, and all input hashes. Prepare/publish packet 1 with A1, then verify deterministic re-render.
2. Launch `scripts/start-sol-worker launch read-only` using that exact packet plus expected SHA-256/byte count. Record run directory before yielding. Do not wrap or edit the rendered packet after publication.
3. Wait for terminal receipt; ingest through A1 into a fresh records directory. Stop if rejected. Preserve request/result/final and rejection receipts; do not repair the response by hand.
4. If accepted, construct the feedback dossier with the exact returned candidate bytes as parent and the predeclared synthetic feedback fixture. All frozen identities and synthetic classification remain unchanged.
5. Prepare packet 2 through `prepare_revision_packet`, supplying packet 1, prior receipt, proposal, iteration, and dossier. Publish and launch the second exact packet read-only.
6. Ingest call 2. Report both outcomes, exact parent/packet/receipt/iteration hashes, requested model/effort and observed thread IDs, retained failures, source changes, and whether the revision addresses the declared feedback.

Do not equate changed source text with better reward quality. If the revision is unchanged, report it; do not retry. Assess feedback use qualitatively against its exact text, not fabricated numerical improvement.

## Completion and stop

- Success: both real envelopes are admitted; revision retains exact lineage and synthetic classification; outputs remain nonexecuted/nonvalidated text.
- Failure: malformed or mismatched response, missing terminal receipt, model/effort/mode mismatch, changed inputs, or quota/service error. Stop and retain evidence before designing any follow-up.
- Record source result in `A2_RESULT.md` with progress/bottleneck/next step and exact receipts. Keep verbose/private run artifacts out of Git.
- If the first call is still active at a checkpoint, poll only its recorded run. No duplicate canary.
- Next after this slice: reviewed B0 handoff and a thin contract/evaluator adapter; behavioral testing still needs peer-agreed compute and a frozen protocol.
