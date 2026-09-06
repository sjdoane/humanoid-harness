# Two research lanes

| status | current truth |
|---|---|
| progress | Lane swap complete; tracker source imported. OT1 reviews timed out without verdicts; pairing correction proposed to Fable. |
| bottleneck | Tracker acceptance, calibration/scoring and evaluator authority remain open; no trained-tracker result. |
| next step | Narrow OT1C/OT1E follow-up; see `OT1_REVIEW_CHECKPOINT.md`. No new Astra reward dispatch. |

## Ownership

| lane | orchestrator | substantial outcome | first dependencies |
|---|---|---|---|
| A: oracle | Codex Astra session | Supplied references → state/phase-aware composition → tracker → protected transition/recovery evidence | Fable's clean Phase B handoff, reference supply, tracker admission, causal-use test |
| B: reward | Fable 5.1/max | Task + protected feedback → LLM reward proposal → validation → matched training/evaluation → revision | Astra's accepted F3 handoff, runtime admission, bounded runner |

- Both own research, design, implementation, tests, independent reviews, and
  evidence in their lane. Equal responsibility does not mean equal line counts.
- In-flight work stays with its current writer until a clean checkpoint.
  Neither lane copies changing files or duplicates the peer's unfinished slice.
- Astra takes the oracle/tracker prerequisites after the explicit handoff.
  Fable takes reward generation and protected feedback. Shared runner/registry
  changes still need exact paths and peer agreement before integration.
- Each lane may revise its strategy. Cite goal IDs, evidence, rival explanation,
  and a falsifiable test. Neither lane rewrites the other's decisions.
- Keep new lane decisions under separate `astra/` and `fable/` paths until
  promotion. Reserve canonical ADR numbers through the mailbox to avoid reuse.
- Within the reward lane, candidate authors cannot modify protected evaluators.
  Freeze the evaluator before generating a candidate; use separate evaluation
  and candidate workers plus an independent scientific reviewer.
- The charter, Lokesh goal ledger, metric independence, frozen comparisons,
  and private-source handling remain the common scientific contract.

## Workspaces

| owner | checkout | branch |
|---|---|---|
| Fable | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness` | `main` |
| Astra | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra` | `astra/reward-loop` |

- The Astra branch name is historical, not a lane assignment. No checkout,
  virtual environment or live worker is moved by the role swap.
- Astra fork point: `05c04693c4949fc3966b456da03ac2344d35785e`.
- `.orchestration`, leases, worker logs, temporary artifacts, and test output
  stay local to each checkout. Never share a writable virtual environment.
- Astra's own environment was installed with `uv sync --offline --frozen
  --extra dev`. Before testing, confirm `oracle_composition.__file__` resolves
  inside this checkout. Add simulator extras only when a task needs them.
- One writer per checkout, using its existing lease. Read-only reviewers can
  run in parallel. Sol leaf workers request `gpt-5.6-sol`, reasoning `max`, and
  no nested delegation. Record requested settings and observed receipts.
- Start with one builder and up to two reviewers per lane. Both lanes share
  Codex usage limits; additional agents must have independent useful work.
- Only the checkout owner integrates commits there, at a clean checkpoint.
  Send exact commit IDs, changed paths, tests, and unresolved findings. A
  successful agent exit alone does not authorize integration.
- Fable is the promotion owner for `main`: it integrates reviewed Astra
  commits there. Astra imports agreed Fable commits into its own branch only
  after its writer has stopped. Neither orchestrator pushes implicitly.
- Never copy another lane's dirty files, release its lease, change its running
  hooks, mutate its runtime, or infer abandonment from silence.
- Shared files include `AGENTS.md`, charter/boundary, root CLI/README,
  dependency metadata, existing package exports, canonical strategy/ADRs, and
  the original handoff. Propose exact paths and base commit before integration.
- Packets and completion receipts include checkout realpath, branch, starting
  HEAD/dirty state, final diff paths, and lock-file identity. The current
  launcher does not itself attest those checkout fields.

## Mailbox

- CLI: `scripts/research-mailbox` in the Astra checkout; stdlib Python, no server.
- Shared state: Git common directory, `harness-coordination/`.
- On this host: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness/.git/harness-coordination`.
- Individual immutable messages; separate read acknowledgments; one current
  status per lane. No shared append log or edit-racing handoff document.
- `ack` means read. Agreement requires an `acceptance` reply bound to the exact
  proposal ID; it accepts only that proposal's paths, base, and resource scope.
  A `rejection` or `counterproposal` never transfers ownership.
- Messages are same-user coordination records, not model authentication,
  provider attestation, scientific evidence, or authority to execute text.
- Keep raw transcripts, credentials, checkpoints, and verbose worker logs out
  of the mailbox. Link to evidence and send concise summaries.

From the Astra checkout:

```bash
./scripts/research-mailbox inbox astra
./scripts/research-mailbox board
./scripts/research-mailbox send --from astra --to fable --kind question --subject "B0 handoff" --body "Please send the reviewed commit and remaining canaries."
./scripts/research-mailbox ack astra MESSAGE_ID
```

Fable uses the absolute path to this same script with `inbox fable` and
`--from fable --to astra`. Native Read can inspect the messages directory even
while a worker holds Fable's writer lease. Existing hooks require Fable to
send/reply under its own lease at a clean checkpoint; include the absolute
mailbox directory in the lease's audit scope. Do not disable the hooks.

Read the inbox at turn start, before dispatch/integration, after each worker,
and at approximately 30-minute checkpoints. Send progress, bottleneck, next
step, active work/paths, exact receipts, and any dependency needed by the peer.
The mailbox stores messages while either app is closed; it cannot wake Claude
Code or inject text into its active conversation. The initial pasted prompt
subscribes Fable to this polling protocol.

## Shared resources and disagreements

- Lightweight contract tests/research may run in parallel.
- Before a long simulation/training job or full-suite run, propose resource
  use and obtain a peer reply. Start with one heavy local job at a time, with
  exact CPU/memory/wall-time bounds. No claim that this advisory rule is an OS
  lock; agree and record an atomic reservation mechanism before automation
  can dispatch heavy jobs unattended.
- Until that reservation mechanism exists, the heartbeat may dispatch only
  lightweight implementation, focused tests, literature work, and reviews.
- If the peer is silent, continue independent work; do not take its compute,
  files, strategy, or unfinished tasks. Expired leases require owner review.
- Cross-lane contract changes need both replies. Unresolved differences go to
  an independent Sol review; a material change to Samuel/Lokesh's goal goes to
  Samuel. Routine lane decisions do not need another human confirmation.
- A model refusal or service limit is recorded. Do not evade safeguards or
  silently substitute models. The other lane may continue permitted work in
  its own scope. Never infer a verified Fable identity from a sender label.

## Evidence and useful autonomy

- Read primary research to answer a concrete design question; record source,
  mechanism, limitation, project implication, and test. Reuse the research
  graph and provenance scheme; avoid a second literature database.
- Software milestone: real LLM proposal, validated artifact, immutable inputs,
  missing-data reporting, protected feedback, and a reproducible revision.
- Behavioral milestone: locked baseline/candidate/scale/ablation comparison,
  all declared seeds, fixed budget and evaluator, failures retained.
- Generalization milestone: held-out targets/reference sets, transitions,
  perturbations, and another compatible adapter, with separate train/test
  provenance. Unsupported data must be rejected with an actionable reason.
- Contract tests and synthetic feedback do not establish robot improvement.
  Neither lane uses the claim "foolproof."
- After two iterations that add infrastructure without enabling the next
  measurable test, review the route and simplify the next slice.

## Continuity

- The existing 30-minute Codex heartbeat is updated to resume Astra's lane,
  inspect this mailbox, and coordinate with Fable. Its old takeover-only rule
  no longer governs independent Astra work.
- Each checkpoint records the durable worker run directory, exact next packet,
  expected evidence, and stop condition in `ASTRA_HANDOFF.md` and the mailbox.
- A running detached worker may finish while an orchestrator is unavailable.
  New work waits for the relevant orchestrator/heartbeat and available quota.
- No indefinite uptime or automatic Claude quota recovery is claimed. The
  computer, app scheduling, authentication, model availability, and quotas
  still determine whether a scheduled turn runs.

## Tooling references

- [Codex worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees).
- [Codex non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode).
- [Model and reasoning selection](https://learn.chatgpt.com/docs/models).

These references guided reuse of worktrees and the existing detached Sol
launcher. No third-party orchestration framework is required.
