# Independent review during provider unavailability

| progress | Astra owns execution; Fable's source review is retained. |
|---|---|
| bottleneck | Fable reported a usage limit before signing the exact Study012 launch requests. |
| next step | Permit an explicitly named Sol reviewer without weakening the launch checks. |

## Authority and scope

- Samuel requested Astra leadership, Sol sub-agents, and Codex continuation
  while Claude is unavailable. This implements that workflow locally.
- Fable remains the preferred scientific review/ideation partner when available.
- `sol-reviewer` means a separately tasked Codex reviewer, not Fable, not model
  attestation, and not permission for Astra to approve its own experiment.
- Initial use: Study012's already specified control reproduction and one finite-
  horizon candidate. Further experiments require their own bounded packets.
- No provider-limit circumvention, repeated Claude requests, billing change,
  paid compute, or manufactured acceptance. A monthly-limit message is not a
  reliable promise that a session reset restores access.

## Same launch boundary

```text
Astra: exact task + source + inputs + output + budget + protocol
                            ↓
separate Sol reviewer: source / experiment / resource review
                            ↓
named acceptance → Astra read acknowledgment → atomic heavy-job reservation
                            ↓
one supervised worker → terminal cleanup → independent artifact readback
                            ↓
verified result + compact feedback → next bounded proposal
```

- Inputs: immutable source, config, assets, command, output and expiry.
- Output: immutable named acceptance, acknowledgment and resource receipt.
- Frozen: evaluator, scientific factors, resource ceilings and exclusion rules.
- Feedback: reviewer can reject or request a specific repair; no auto-acceptance.
- The reviewer must not have authored the implementation or candidate it approves.
- Bind the exact proposal ID and authorizer. Self-acceptance, unknown identities,
  stale/malformed records and changed source/input bytes still fail closed.
- Request/sender labels are cooperative local identities, not cryptographic proof
  of which model served a response. Retain agent handoff and actual message IDs.
- Do not open a second heavy-job slot or change a live worker's source.

## Current migration

- Fable public message: `2026-09-07T04:58:45.161Z`; usage limit reported.
- Withdrawn **before launch**: Study012 proposals
  `20260907T045459.489708Z-a5b6d37dcd4a40d09ef787c935815b2c` and
  `20260907T045500.752667Z-84ab744bc2824a4b875999c9162b7f55`.
- No approvals existed for these requests. No worker or output was created.
- Regenerate exact requests after the reviewed mailbox-role change is committed.
- Fable can resume reviews later; never rewrite Sol records as Fable records.
