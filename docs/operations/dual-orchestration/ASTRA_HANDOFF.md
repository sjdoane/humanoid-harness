# Astra reward-lane handoff

| status | current truth |
|---|---|
| progress | F1 independent closure accepts static/synthetic plumbing; all ten reviewed hashes match. Parent pure regression: 256 passed/16 deferred/1 simulator test deselected. |
| bottleneck | F1 still targets old B0 inputs; T2 formula/compositor mapping and authenticated model ingestion remain. |
| next step | Collect the bounded F2 interface plan and agree the minimal Phase B reward handoff with Fable. |

- Date: 2026-09-05.
- Last checkpoint: 2026-09-05, 21:25 UTC heartbeat; F1 closure collected and all ten hashes reverified before acceptance.
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
| Protocol and lane authority | Astra + Fable | proposal explicitly accepted 22:14Z | Exact acceptance message recorded below; B0 exception remains |
| A1 implementation | detached Sol/max | accepted after repairs and review | `TASK-A1-reward-proposal-loop.md`; details in `A1_RESULT.md` |
| A1 independent review | detached Sol/max | completed 20:57:15Z; accept-with-repairs | Five blocking findings and one low-severity origin check |
| A1 repair | detached Sol/max | completed 21:42:41Z; parent checks pass | Six findings addressed; `A1_RESULT.md` |
| A1 final targeted review | detached Sol/max | ACCEPT at 22:06:14Z | All six original findings closed; parent corrected B0 sequencing wording |
| A2 live-model protocol | Astra | stopped; one initial call, no revision | `A2_RESULT.md`; transport timeout and retained rejection, no retry |
| A2R changed-condition canary | Astra | complete; both responses admitted | `A2R_RESULT.md`; text format/lineage only, no candidate execution |
| Independent goal interpretation | Fable | substantive reply received and acknowledged | `FABLE_GOAL_ALIGNMENT.md`; composition plan reframed, not demonstrated |
| B0 transfer | Fable → Astra | ownership transferred with P1 repair gates | Exact `89d4c34d1b04eb9efca2b75d1ffc828a13de55d7`; not execution-approved |
| A3 B0 adoption audit | read-only Sol/max, then parent | leaf stopped on ambiguous path; parent completed dependency check | `A3_RESULT.md`; exact 25-file donor snapshot, no execution approval |
| R1 parent-execution boundary | Sol/max builder | terminal succeeded; source uncommitted, not accepted | `R1_RESULT.md`; 49 passed/36 skipped and 59 A1 tests reported, known stale consumer |
| R1 first independent review | read-only Sol/max | timed out 04:43:51Z; no verdict | Exact run retained; no escalation; historical source snapshot verified before parent edits |
| R1 parent completion | Astra | 139 passed/16 deferred; not accepted | `R1_COMPLETION_CHECKPOINT.md`; restored safe coverage and repaired strict metadata/version/API tests |
| R1 completion review | read-only Sol/max | REJECT at 06:18:34Z; terminal receipt retained | `R1_REVIEW_FINDINGS.md`; three constructor/ingress defects; 37 no-temp tests reproduced |
| R1 ingress repair | Sol/max builder | finished 06:49:16Z, lease released; parent 143 passed/16 deferred | `R1_INGRESS_REPAIR_RESULT.md`; three findings repaired, not independently closed |
| R1 targeted final review | read-only Sol/max | timed out 07:35:25Z; partial closure, no verdict | `R1_FINAL_REVIEW_PARTIAL.md`; three findings reported closed, copy limitation open |
| R1 acceptance decision | read-only Sol/max | ACCEPT at 14:58:59Z | `R1_ACCEPTANCE.md`; three findings closed, copy-API limitation non-blocking and documented/tested |
| R2 runtime protocol | read-only Sol/max planner | finished 17:20:01Z, proposal only | `R2_PLAN_RESULT.md`; no execution or implementation approval |
| Cycle-first convergence | Astra + Fable | explicit proposal/acceptance exchanged | `MAIN_INTEGRATION_20260905.md`; one runner/report, exploratory boundary, JSON first |
| Minimum reward path review | read-only Sol/max | completed 18:28:40Z; chose JSON formula family | Existing Python execution gates remain unchanged |
| Pinned-main integration | Astra + Sol reviewer | ACCEPT_INTEGRATION at 19:06:56Z; handed to Fable | `8a48f32`; excludes later uncommitted F1 files |
| F1 formula core | Sol builder + reviewer | REJECT_F1 at 19:48:21Z; three lineage defects | `F1_REVIEW_FINDINGS.md`; numerical core/input gates accepted within static scope |
| T2 input module | Astra + reviewer | ACCEPT_T2_INPUT_ONLY at 19:48:21Z; 17 tests pass | `T2_INPUT_ACCEPTANCE.md`; not wired into formula or simulator |
| F1 lineage repair | Sol writer + Astra | finished 20:34:23Z; parent 232 passed/16 deferred | `F1_REPAIR_PARENT_CHECKPOINT.md`; includes one FIFO-open regression fix |
| F1 closure review | read-only Sol/max | ACCEPT_F1_STATIC_ONLY at 20:59:48Z | `F1_ACCEPTANCE.md`; three findings and FIFO follow-up closed |
| F2 T2/runtime interface plan | read-only Sol/max | dispatch/collect exact active pointer | `TASK-F2-interface-plan.md`; proposal only, no integration authority |

- Accepted F1 commit: `27330eb4f50feae27df3ab118e8c9c0d165112a8` (local only).
- Active F2 plan: `20260905T212913Z-cb25bab9-c2f9-4809-ac32-4d68072ab70e`;
  read-only Sol/max, with a 20-minute immutable-launch deadline watcher attached
  in the launch tool call. No implementation or resource authority is delegated.

- Current acceptance: `F1_ACCEPTANCE.md`. Its retained negative records an
  accidentally selected simulator test failing on missing `gymnasium`; no
  simulator started. Use `-m 'not gym'` for this dev-only checkout's pure suite.
- Peer main observed at committed `8d91a811680599631779892a2c2bedf2127a3a74`:
  reviewed M1 merge and accepted T2 module integrated. No peer dirty files read
  or edited. Phase B registry/compositor remains Fable-owned.

- T2 exact interface accepted by Fable in
  `20260905T202243.309823Z-181767ca33654e69aad5743d92537aeb`.
  No duplicate phase-B input source was adopted; Fable owns removing its copy.

- Current checkpoint and exact receipts: `F1_PARENT_CHECKPOINT.md`.
- Fable selected new T2 COM speed target 3.0, not phase-A root speed; the old
  B0 contract stays unchanged. The explicit interface proposal is
  `20260905T193614.811073Z-cc47aa7840274926956386766cdb5c05`.

- Current authority supersedes the earlier pending proposal notes below:
  Fable proposal `20260905T183421.793603Z-8b619f3d56a84cbb81d31d39e11c36ef`
  accepted by `20260905T185408.005663Z-ff80151886fc485782897509ffe98696`.
  The narrower family is the first execution route, not permission to run it
  before its own review or to open the refused Python runtime.

- Fable proposal `20260905T175216.965624Z-a00f8a13b1a444978292d1eb1aa991ee`
  was read/acknowledged. Committed main base inspected: `a0a0a3b`.
  Main's dirty cycle builder is untouched. No lane merge/rebase performed.
- A non-checkout merge-tree check found 12 add/add conflicts with the pinned
  main base, including a historical receipt. Astra proposes a reviewed merge
  preserving accepted commit identities and both receipts, not a blind rebase.
- Public Python execution remains refused. The narrower JSON family is a
  proposal for the first cycle, not an implemented capability or a substitute
  for the broader reward-program research target.

- R1 history: `R1_PARENT_CHECKPOINT.md` and `R1_COMPLETION_CHECKPOINT.md`.
  Current repair checkpoint: `R1_REVIEW_FINDINGS.md` and
  `R1_INGRESS_REPAIR_RESULT.md`. No source/test edits during final review;
  both previous review snapshots are historical.

- Final review: `.orchestration/sol-runs/20260905T071423Z-bc589f15-291b-4341-94aa-730484edcd39`.
  Fresh 15-file snapshot: `.orchestration/b0-repair/r1final/review-snapshot.json`.
  Read-only Sol/max requested; deadline TERM, no escalation or final verdict.
  Independent partial findings and final hash gate are retained separately.

- Accepted decision: `.orchestration/sol-runs/20260905T135846Z-4adbb20a-a0ee-468a-a4ef-3dd4a807c825`.
  `R1_ACCEPTANCE.md` records scope, follow-up checks, and the late-watchdog
  caveat. Future launches and watcher attachment must occur in the same tool
  call. Local watchers now use the immutable launch time for their deadline.

- Accepted R1 commit: `754e869c618dbe4f1b43c4f1a513c023ec139306` (local only).
  Clean worktree verified after commit; final parent 144 passed/16 deferred.

- Completion review: `.orchestration/sol-runs/20260905T060349Z-d2435712-4e9a-4af1-a94b-fd183f40c96e`.
  Read-only `gpt-5.6-sol/max` requested; no nested agents; verdict REJECT.
  Snapshot and one-shot deadline watcher: `.orchestration/b0-repair/r1review/`.
  The expired parent lease was reviewed after both previous workers had
  terminal receipts; only that same parent's lease was released/reacquired.

- Fable's production collector cache-refresh finding was acknowledged and
  checked against adopted B0 source: `REWARD_CAPTURE_PATH_CHECK.md`. No matching
  mutation found in that narrow source scope; no runtime equivalence claim.

- Accepted A1 commit: `76a0fe6e86f5c4e5b52231d3b69df3d0c9b317d2` (not pushed or integrated into main).
- Final acceptance reproduction: `59 passed in 2.35s`, Ruff/format/diff checks pass.
- A2 inputs: `.orchestration/a2-canary/20260904T2250/`; `baseline.json` binds exact contract, adapter, feedback, corpus, dossier, packet, and source hashes. Every input is synthetic, not a real B0 binding.
- A2 initial run: `.orchestration/sol-runs/20260904T225403Z-f5fc865b-3d10-496b-8a5b-27e70aad70a4`; packet SHA-256 `b07d7a87a916e39aad7c3a6172bfa9fd8b1bf5944626c30e7665e5fec85f0f71`, 5,925 bytes. Calls launched: 1 of 2.
- A2 terminal: `INTERRUPTED_TERM`, exit 143, `23:25:32Z`, no escalation. CLI transport idle timeout observed; cause not established. A1 ingestion rejected missing `final.txt`; immutable rejection receipt and iteration are under `initial-records/`. The two-call protocol stopped after its first call; its dependent revision is not authorized.

- A1 builder run: `.orchestration/sol-runs/20260904T201331Z-58120f9b-e141-4e1b-be92-32f634829778`.
- Builder thread: `01a06e0e-1873-79b3-8b98-52704c2551fb`; terminal
  `SUCCEEDED`, lease `RELEASED`. This is execution status, not acceptance.
- Parent checkpoint tests: `35 passed in 2.69s`; module CLI lists `prepare`,
  `ingest-sol`, and `prepare-revision`.
- Review run: `.orchestration/sol-runs/20260904T204653Z-226c4610-e72e-4154-8c34-cf91843c0b2b`.
- Accepted repair scope: retained prior receipt/packet verification, immutable
  synthetic classification, coherent source-graph snapshot, bounded JSON reads,
  read-only proposal admission, and local publication-helper origin checks.
- Repair run: `.orchestration/sol-runs/20260904T212420Z-1ea01a7b-e7c5-4af6-866a-8e287d9145a7`; `SUCCEEDED`, lease `RELEASED`.
- Post-repair parent checks: `59 passed in 2.33s`; Ruff/format and diff checks passed. Import resolves in this worktree.
- Parent clarification: the repair result incorrectly adds B0 handoff as a
  prerequisite for the synthetic-input A2 canary. A2 does not execute or bind
  B0; actual B0 validation/training remains separately gated. The parent
  corrected the result wording after the final read-only review stopped.
- Fable acceptance: `20260904T221414.221774Z-a293af4af0eb4362a66d1a71554d3c15`, replying to the exact original lane proposal. Astra acknowledged receipt. Proposed reward-study contracts remain unlocked.
- B0 transfer: `20260905T010239.496518Z-74c32a4658554eb0960b1a3359f767c4`; acknowledged at 02:20 UTC, not accepted for execution. Both reviews require P1 repairs and re-review; transferred paths and exact review runs are in that message. Fable will not start another reward builder. Shared ADR/protocol/config changes still require peer agreement.
- Samuel's alignment request: `20260904T221845.220763Z-d00b214439d24b7e81ae0ff37e28e8af`; substantive Fable reply `20260905T010211.593536Z-f9e533bf69b64867b07778cbe87b137e` received. Fable reports a full private transcript read and proposes reframing Experiment 003 as composition. See the separate alignment checkpoint for interpretation versus chosen design.
- Central concern: a small timing adjustment on a single reference clip is a supporting diagnostic, not a demonstration of LLM-designed composition of reference behaviors into a task. Fable should distinguish source-grounded goals from its chosen oracle representation and propose the smallest meaningful composition/feedback demo.
- Reviewer accidentally started and then stopped a nested reviewer; no output
  was used. The repair packet explicitly prohibits nested work.
- Samuel asked whether continuity should remain. The existing dual-lane loop
  remains useful and ACTIVE: it resumes Astra only and defers during active
  user work. No duplicate or takeover loop was created.

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
   **This attempt stopped after call 1.** Do not resume this numbered step as
   permission to retry; follow the current next-step row and `A2_RESULT.md`.
5. A2R completed its separate two-call protocol; retain its initial/revision
   receipts and the failed original A2. No further candidate calls in this series.
6. B0 ownership transferred with execution gates. The A3 leaf stopped without
   audit conclusions; parent inspection found no missing predecessor source.
   The exact 25-file snapshot is now retained for repair, excluding shared
   operational docs. Do not rerun the ambiguous A3 packet.
7. R1 removes parent execution and fixes pure validation defects. Collect its
   exact run, inspect changes, and independently review the bounded repair.
   R2 then addresses runtime containment; R3 closes lineage and receipt gates.
   All P1s remain required before real generated-candidate execution/training.
   No unattended heavy job before an atomic shared reservation exists.

If A1 is incomplete, preserve its diff and write a continuation packet with
specific missing deliverables; do not silently relaunch the original task.

No training, behavioral evaluation, paid API calls, publication, or push has
been performed by the Astra lane at this checkpoint.
