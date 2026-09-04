# Astra reward-lane handoff

| status | current truth |
|---|---|
| progress | Separate worktree and venv, reviewed dual-lane protocol, tested mailbox (9 passed), and independent A1 task/review packets are ready. |
| bottleneck | Fable acknowledgment and reviewed B0 handoff are pending. No live reward proposal/training loop is demonstrated. |
| next step | Commit this setup and launch A1. Inspect `.orchestration/astra-active-run.json` for the exact durable worker; consume B0 only after explicit handoff. |

- Date: 2026-09-04.
- Workspace: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Branch: `astra/reward-loop`.
- Orchestrator role: Astra, this Codex task
  `01a05e77-9737-7240-b79e-4e3902d1664c`.
- Model label identifies the requested role; use actual tool/request receipts
  for model settings, never sender names as attestation.
- Original checkout belongs to Fable. Its B0 worker was actively renewing the
  main writer lease at setup; its uncommitted files were not copied.

## Immediate work

| task | owner | state | acceptance |
|---|---|---|---|
| Coordination design review | Sol `dual_lane_review` | complete; applicable fixes folded | Explicit accept/reject, named promotion owner, evaluator separation, venv identity |
| Reward loop decomposition | Sol `reward_lane_plan` | complete; fixes folded into A1 | Separate sources/evidence, concrete module CLI, receipt binding |
| Local mailbox + tests | Sol `mailbox_builder` | complete; parent reproduced 9 passed | Real-worktree shared root, 36 concurrent sends, recipient checks, explicit acks |
| Protocol and lane authority | Astra | complete; Fable reply pending | Clear ownership, in-flight B0 exception, truthful continuity |
| A1 implementation | detached Sol/max | launch after setup commit | `TASK-A1-reward-proposal-loop.md` |

## Verification and source reading

- `.venv/bin/python -m pytest -q tests/integration/test_research_mailbox.py`:
  `9 passed in 1.94s` in the final parent check, invoking the actual executable.
- Direct execution exposed macOS Python 3.9's missing `datetime.UTC`; fixed
  with the compatible timezone constant and switched tests to the real CLI.
- Changed-path Ruff passed; `git diff --check` passed; executable CLI verified.
- Local import resolves to `humanoid-harness-astra/src/oracle_composition`.
- Raw Lokesh transcript and two-page proposal re-read; both match the source
  manifest SHA-256 values. Lane split preserves their two-factor contract.
- Assessment of Fable's progress: `FABLE_PROGRESS_SNAPSHOT.md`.
- One existing heartbeat was updated to dual-lane continuity; interval remains
  30 minutes. No duplicate automation was created.

## Resume

1. Read this file, the dual-lane protocol, and the shared mailbox.
2. Check the lease in this worktree. Never use main's lease to authorize Astra
   writes. Never release another owner's lease.
3. Collect current worker receipts. Native subagents belong to the active
   Codex turn; use the existing detached Sol launcher for work that must
   survive the turn. Exact active run is in ignored
   `.orchestration/astra-active-run.json`; it records launch baseline, role,
   packet, run directory, and next action. Do not launch duplicate workers.
4. Take the next bounded packet only after prerequisites are met. Preserve
   tests, sources, and claim ceilings; update this handoff after each slice.
5. Send Fable a compact progress/dependency message. Keep routine heartbeat
   notifications quiet unless a result, failure, or Samuel's action matters.

## Authorized next sequence

1. A1 builder completes its bounded packet and `A1_RESULT.md`.
2. Inspect diff and focused checks. Launch the independent read-only
   `TASK-A1-review.md` through the detached Sol launcher; record its run.
3. Resolve accepted findings in one bounded repair packet. Review/test again
   only where necessary, then commit the complete A1 slice.
4. A2: two subscription-authenticated read-only Sol canaries on initial and
   synthetic-feedback packets. Use A1 ingestion to verify exact parent and
   dossier identities. This proves real LLM plumbing only; no code execution,
   training, or robot-result claim. Write its concrete packet before dispatch.
5. Once Fable hands over reviewed B0, plan the thin validation/evaluator adapter
   and agree compute/run protocol. No unattended heavy job before an atomic
   shared reservation exists.

If A1 is incomplete, preserve its diff and write a continuation packet with
specific missing deliverables; do not silently relaunch the original task.

No training, behavioral evaluation, paid API calls, publication, or push has
been performed by the Astra lane at this checkpoint.
