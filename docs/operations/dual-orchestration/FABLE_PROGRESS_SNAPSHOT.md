# Fable progress review — 2026-09-04

| status | assessment |
|---|---|
| progress | Clearer strategy and useful importer implementation; independent review identified substantive defects. |
| bottleneck | No stable tracker, causal reference-use result, or reward-improvement experiment has been demonstrated. |
| next step | Finish the smallest reviewed path to a real candidate → training → protected evaluation → revision result. |

Snapshot: main at `05c0469`, B0 unfinished, before dual-lane handoff. These are
observed project changes, not a controlled comparison of model quality.

| decision/change since Fable startup | evidence | assessment |
|---|---|---|
| Re-read Lokesh's sources and added LG-13–16: fine-tuning a supplied tracker, temporary Gym adapter, reward-only exercise, CLI-first | `4a0976d`; strategy and source manifest | Stronger alignment with the actual project |
| Parked the TQC-v2 supervisor route; preserved the work on `wip/tqc-v2-attempt-supervisor` | ADR 0005; `5bdae45` | Sensible reduction of a detour: 4,499 uncommitted lines and eight lifecycle findings had produced no controller |
| Chose a hash-pinned public Farama TQC actor as bootstrap material | ADR 0005; `1d6b461` | Practical prerequisite; imported actor is not reference-conditioned and is not an oracle result |
| Imported data-only actor bytes, strict NPZ, provenance, and fixed-batch equivalence | `1d6b461`; recorded 1099-test outside-sandbox suite, not rerun for this snapshot | Concrete implementation progress with appropriately limited claims |
| Opened reward-first Family B on stock Humanoid-v5 | ADR 0006; Samuel's recorded approval | Creates a path to test the reward loop independently of tracker readiness |
| Built reward contracts, static gate, compositor, sandbox, scale checks, protected endpoint | Current B0 working tree and handoff | Substantial draft; validation and OS canaries still pending |
| Scientific review found semantic fingerprint bypass, safe-global ordering gap, ineffective/missing negative tests | Review 06: accept with repairs, 0 P0 / 3 P1 | Review is useful and finds real problems; importer still needs repair before dependent use |
| Adversarial review 07 terminated with exit 1 | Its `result.json` at 18:24:17Z | A failed review is not a completed review; Fable must recover it |

Quality judgment:

- **Strategy: improved.** The route now addresses the two research outputs and
  treats controller development as a temporary prerequisite.
- **Evidence discipline: improved.** Source facts, software checks, and robot
  behavior are kept distinct; review findings are concrete.
- **Scientific results: still unproved.** No measured gain in oracle behavior,
  reward quality, transitions, recovery, or transfer is available here.
- **Efficiency: mixed.** Setup/sandbox failures still consume time and the
  importer alone adds substantial code. Two lanes should shorten the path to
  a measured loop, not double the infrastructure effort.

Source locations: original checkout's `docs/strategy/RESEARCH_STRATEGY.md`,
`docs/decisions/0005_public_expert_base_controller.md`, ADR 0006,
`docs/operations/CURRENT_RESEARCH_HANDOFF.md`, and review run directories
`20260904T181613Z-721ea785-8dc3-4d8f-9eb6-84b4672d1fb3` and
`20260904T181614Z-5bb41e36-fc24-49b0-8fb3-a73409210b5a`.
