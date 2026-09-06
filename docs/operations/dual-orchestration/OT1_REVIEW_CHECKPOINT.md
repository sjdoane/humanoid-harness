# OT1 review checkpoint: bounded follow-up

| status | current truth |
|---|---|
| progress | Both broad reviews reached their watchdog deadlines; all 350 frozen inputs still match. Useful partial comments retained, no final verdict. |
| bottleneck | Tracker not source-accepted; calibration/scoring and evaluator authority need narrow closure. Original host-test failure cause unknown. |
| next step | OT1C checks three scientific questions; OT1E checks evaluator execution. No broad review restart, training or reward dispatch. |

## Review receipts

| review | exact run suffix | terminal |
|---|---|---|
| Scientific | `20260906T042442Z-1fc46a05-b553-4022-8163-26393a0d400d` | `INTERRUPTED_TERM`, exit 143, 04:44:45 UTC |
| Robustness | `20260906T042442Z-160c5183-97d9-47d3-bd0d-af37dc924106` | `INTERRUPTED_TERM`, exit 143, 04:44:45 UTC |

- Runs are under Astra's `.orchestration/sol-runs/`.
- Both watchers sent TERM at the declared 20-minute deadline; no escalation.
- Neither produced `final.txt`. Progress comments are not acceptance.
- Partial comments retained under `.orchestration/ot1-review-20260906/`.
- Source commit remains `851e16d`; original 350-input snapshot remains valid.
- All previous findings remain open or unverified unless individually closed
  by a later recorded decision. No blanket closure from an exit or test count.

## Shared study pairing

- Fable proposal: `20260906T043826.945195Z-feeb1a9f0de44274b3da3cbb9708a944`.
- Astra counterproposal:
  `20260906T045824.333974Z-a0329add05c54f5b84d035b113cc6ff5`.
- Confirmed by source inspection: training action/minibatch RNG and RSI derive
  from the arm execution-manifest digest. Different reward manifests therefore
  change more than the reward in a supposedly common-random-number comparison.
- Proposed correction is T2 reward-only pairing, not a universal oracle-study
  fix. Share primitive random variates/schedules, not policy outputs; retain
  full, distinct arm identities. Require validated common inputs and complete
  consumer paths before agreeing to shared changes.
- Fable is preferred implementer of that bounded seam; Astra independently
  reviews the committed handoff. No shared-source edits accepted yet.

## Host test and formatting

- Fable reports a later full suite passed the formerly failing CLI test. The
  original traceback was not retained. Root cause remains unknown.
- Test-only limits are 20 s per seed and 30 s total; timing is a plausible
  follow-up question, not an established diagnosis. Production limits unchanged.
- Historical F3 packet: Python fence formatting only, focused Ruff check passes.
  Pre-format bytes retained in ignored review folder and historical Git commit:
  SHA-256 `c550cd2063a64e9db6c36483af810149fb8dcdfd0c291214041029002d4b1a43`.
  New document SHA-256:
  `5538169600226995a2fdd4728da71b06b5a2169bedfc5dabe98eb2f1596c04f9`.
- No launch packet or original receipt was rewritten; no F3 candidate call.
