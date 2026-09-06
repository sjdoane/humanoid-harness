# ADR 0009: lane swap between the Fable and Astra orchestrators

| | |
|---|---|
| progress | Samuel instructed both orchestrators on 2026-09-06 to switch lanes. Astra now leads oracle and reference composition and tracker integration; Fable leads task-reward generation and the protected feedback and revision loop. |
| bottleneck | The swap lands at a clean checkpoint: FT2R3 committed, the tracker-lane handoff written, Astra's accepted reward commits integrated into main. |
| next step | Astra dispatches the combined re-reviews of FT2R1 to FT2R3 and the next composition packet; Fable re-pins the F3 one-call protocol to its lane and plans reward cycle 0 for task T2. |

**Status:** accepted on Samuel's explicit instruction (Claude session and Astra session, 2026-09-06); acceptance sent to Astra's proposal `20260906T023515.308859Z-50b08e0c…`
**Date:** 2026-09-06
**Supersedes:** the lane assignment in ADR 0007 and `docs/operations/dual-orchestration/README.md`; nothing else. The Lokesh goal ledger, the frozen scientific contracts, the hard gates, and the writer guards are unchanged.

## Decision

| item | decision |
|---|---|
| Ownership | Astra: oracle programs, reference composition, tracker integration, the phase B fine-tuning runtime as a tracker, E4 utility and E5, the composition cycles of Experiment 003 phase B. Fable: reward specifications and their generation loop (A1, F1, F2, F3 lineage), the reward study T2 on the fine-tuning runtime, protected feedback and revision, the reward registry entries. |
| Mechanics | Physical checkouts, environments, and leases stay as they are: Fable in `main` (promotion owner), Astra in `humanoid-harness-astra` on `astra/reward-loop`. Reviewed commits are exchanged; no live process moves. Shared-runtime code stays in `main`; Astra merges main into its branch before building on it, and Fable integrates Astra's reviewed commits with merge commits. |
| Transfer content | `docs/operations/TRACKER_LANE_HANDOFF.md` (Fable to Astra) and Astra's `ASTRA_HANDOFF.md`, `F3_ACCEPTANCE.md`, and `LANE_SWAP_20260906.md` (Astra to Fable). |
| Reviews | Astra dispatches the combined scientific and robustness re-reviews of FT2R1 to FT2R3 and folds their findings in its lane. |
| Compute | Unchanged: any job over ten minutes needs a mailbox reservation accepted by the peer; any training cohort needs Samuel's explicit authorization; one heavy job at a time. |
| Strategy documents | `docs/strategy/RESEARCH_STRATEGY.md` stays the shared strategy record; each lane records its own decisions there under its section and cites goal IDs. |

## Consequences

- Fable's tracker-lane work ends at commit chain FT2R3; the smoke and cohort for the tracker utility are Astra's to propose.
- Fable's reward lane inherits accepted static plumbing (A1, R1, F1, F2, F3) and no executed reward cycle; its first milestone is one executed cycle on the fine-tuning runtime for task T2.
