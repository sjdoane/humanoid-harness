# Current research handoff

| status | current truth |
|---|---|
| progress | The oracle/reward boundary, deterministic oracle contract, evidence UI, and research index exist. Fable-to-Sol orchestration is configured and its fail-closed local paths passed bounded tests. |
| bottleneck | No reference-aware Humanoid tracker is admitted. The only visible tracker falls immediately, causal reference use is unproved, and the uncommitted TQC-v2 launcher has unresolved lifecycle risks. |
| next step | Authenticate and identity-check Fable, let it audit the strategy against the Lokesh goal ledger, then choose the smallest prerequisite experiment. Do not launch the real 1M-step attempt. |

- Updated: `2026-09-04 04:43 PDT`
- Repository: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness`
- Branch: `main`
- Baseline HEAD and `origin/main` before orchestration setup: `dc6db640456c62b2d7ff2c57b20b33fe19060449`
- Control owner: human until a live Fable identity check passes
- Active Fable session: none observed
- Active write worker: none; lease status `UNCLAIMED`
- Active Codex task/thread IDs: none recorded
- Takeover authorization: disabled; no task may be launched by the heartbeat

## Claim boundary

| layer | current statement |
|---|---|
| research target | An LLM-guided harness revises a reference-composition oracle `O_k` and task reward `r_k`, using protected rollout evidence and optional human steering. |
| implemented capability | Typed oracle artifacts; a validated linear phase-window automaton; Gymnasium adapter; deterministic trace/metric contracts; CLI; read-only evidence UI; research-source ledger. Uncommitted TQC-v2 WIP is present but is not runnable or admitted. |
| measured evidence | Interface and regression checks; one non-admitted falling tracker exploration; one reviewed offline numeric-reference sensitivity probe; one 100k resource calibration. |
| not demonstrated | Stable Humanoid tracking; causal policy use of reference windows; better transitions; recovery; oracle improvement; reward improvement; cross-MDP generalization; an autonomous closed research loop. |

## Scientific dependency chain

```text
Experiment 001             Experiment 002                 Experiment 003+
admit stable tracker  -->  prove causal reference use --> compare oracle arms
       BLOCKED                    BLOCKED                   NOT AUTHORIZED
```

- A time-indexed reward is not proof that actor and critic consumed the
  immutable reference clock and window.
- Oracle quality cannot be inferred from the current falling video.
- Reference generation and reference composition are separate problems. This
  project currently owns composition.
- Task-reward generation remains a second scientific workstream. Hold it fixed
  during an oracle-only comparison.

## Working-tree state

- Pre-existing research WIP must be preserved:
  - `13` modified tracked files across `pyproject.toml`, TQC-v2 source, and tests.
  - `7` untracked TQC-v2 source/test files.
- The latest full-suite receipt was `1129 passed` before the newest WIP. It is
  stale and must not be reported as validation of the current tree.
- The E0 calibration observed `619.662` environment steps/s over `100,000`
  steps. It is a resource receipt, not behavioral evidence.
- The proposed real development run is TQC, `1,000,000` steps, `5` environments,
  training seed `95001`, with fixed evaluation seeds `96001-96020`.
- No real 1M-step attempt has started.

## Current no-go findings

These findings were observed against the uncommitted TQC-v2 working tree. They
must be retested after any repair.

| id | blocker |
|---|---|
| TQC-P1-01 | The one-attempt rule is directory-local and can be bypassed with another fresh directory. |
| TQC-P1-02 | A failure after process start can escape complete cleanup and can misreport whether a worker started. |
| TQC-P1-03 | Process-group signal authority can be inferred from the OS before worker self-report is authenticated. |
| TQC-P1-04 | Authenticated worker failures before manifest acknowledgement can be downgraded to generic integrity failures. |
| TQC-P1-05 | Signal and terminal-receipt ownership is not safe across every public launch path. |
| TQC-P1-06 | A transient macOS `EPERM` group probe can false-reject cleanup instead of remaining indeterminate within the deadline. |
| TQC-P1-07 | The training-to-persistence semantic bridge for policy, entropy, and optimizer state is unresolved. |
| TQC-P1-08 | Success does not yet revalidate every retained preflight, manifest, reservation, and worker receipt. |

Additional constraint:

- A production-ID canary would consume the sole attempt. Any canary must use a
  separate, explicitly non-authorizing identity.

## Orchestration state

| component | observed state |
|---|---|
| Claude Code | `2.1.260`; supports `claude-fable-5-1`; shell login still required |
| Fable policy | Exact model requested at launch; `switchModelsOnFlag=false`; live response identity gate available |
| session guard | Project hooks deny requested non-Fable switches and all Claude tool use when the recorded model is not Fable or active effort is not `max`; events remain under ignored `.orchestration/` state |
| Codex | App-bundled CLI authenticated with ChatGPT; live catalog includes `gpt-5.6-sol` at `max` reasoning |
| worker path | `scripts/start-sol-worker` is the strict default: detached GNU screen runner, direct Codex CLI, exact Sol/max request, compact status, and retained JSONL/final/PID/thread receipts; plugin `1.0.6` remains optional because it uses a thin Sonnet router |
| write control | Atomic 45-minute lease gates primary Fable and Sol write paths and is renewed by the Sol launcher; recorded path scope is audit-only; ambiguous or expired leases fail closed |
| continuity | Active 30-minute heartbeat is coordination-only; without a valid single-use takeover record it reports status and does not implement; no supported API pushes into a closed Claude Code session or reports its quota transition |

Setup verification note:

- A shell-shim mistake briefly started Codex thread
  `01a06bdc-8f87-7040-8264-23cd60052ee6` on a harmless launcher test packet.
- It produced only a startup message, no terminal receipt, no observed repository
  edit, and no scientific result. The process was absent when checked; its lease
  was manually released. Treat it as an aborted operational test, not evidence.
- Corrected fake-executable tests then proved detached launch across caller
  boundaries, compact live status, thread-ID capture, terminal receipt,
  exact Sol/max request recording, single-use takeover, packet SHA-256/size
  binding, and lease release without another model call.
- Packet-tamper tests failed before consuming authorization and at the launcher's
  owned-copy boundary. A malformed runner-receipt test emitted
  `PREFLIGHT_FAILED` and released its exact lease.
- The shared takeover-state lock blocked a concurrent clear. An expired lease
  could be released for review but could not be renewed or silently revived.
- Simulated Claude hook events accepted recorded Fable/max tool use, rejected
  non-max effort and a requested Opus switch, and removed tool authority after
  a recorded automatic substitution.
- Runtime test state was moved to Trash. Current takeover state is disabled,
  no screen worker is present, and the writer lease is `UNCLAIMED`.
- Independent goal-alignment and adversarial orchestration re-reviews reported
  no remaining P0/P1 after the accepted repairs. This clears the control-plane
  setup only; it does not clear the TQC-v2 or scientific no-go findings above.

## Resume order

1. Run `claude auth login` once.
2. Run `./scripts/orchestration-doctor` for the no-usage local checks.
3. Run `./scripts/start-fable-orchestrator`; authorize its one live identity
   probe when prompted.
4. Fable reads `CLAUDE.md`, the master prompt, this handoff, the strategy ledger,
   the private transcript, and the proposal.
5. Samuel runs `/status`; Fable also checks the latest session-start model event.
6. Fable records questions or a revised strategy before assigning implementation.
7. Fable uses its session owner for strategy edits, releases that lease, then
   launches one durable Sol writer through `scripts/start-sol-worker`. Separate
   read-only Sol workers review scientific validity and adversarial failures.
8. Update this file at least every 30 minutes during active work and at every
   control transfer.
9. The Codex heartbeat must back off while Fable or another write worker owns
   overlapping work.

## Handoff update contract

- Keep the three-row table first.
- Replace, do not append to, its current statements.
- Separate requests, implementation, and measurements.
- Name changed files, test receipts, remaining blockers, active worker/thread
  IDs, and the next bounded action.
- Never convert a green test, UI launch, or zero-action smoke test into a
  behavioral claim.
