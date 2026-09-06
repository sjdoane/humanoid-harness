# F3 one-call protocol

| status | current truth |
|---|---|
| progress | Initial-only protocol and exact tracking-only baseline are implemented; parent focused tests pass. |
| bottleneck | This protocol is a proposal until independent F3 closure and explicit review of this document are recorded. |
| next step | After both reviews pass, retain one exact initial packet, launch at most one read-only Sol candidate, and ingest its terminal artifacts. |

## Purpose and strict scope

- One local subscription-authenticated initial parameter hypothesis; zero
  revision calls, retries, Python reward execution, simulator or training.
- Research input: supplied tracking-only baseline plus accepted T2 contracts.
  Output: alpha/beta hypothesis, rationale, predicted effect and falsifier.
- No protected evaluation or measured feedback exists. Do not invent it.
- This is a software integration check, not reward efficacy or model attestation.

## Preconditions

1. Independent verdict `ACCEPT_F3_STATIC_ONLY` closes F3-01 through F3-04 and
   the exact-baseline refresh. Separate verdict `APPROVE_F3_ONE_CALL_PROTOCOL`
   approves this document. Any rejection or missing verdict stops dispatch.
2. Parent reproduces the bounded checks, verifies the reviewed snapshot, commits
   the accepted F3 slice and records exact HEAD/source hashes before publishing.
   No source or accepted contract may change between review and use.
3. Baseline is the exact 1,773-byte no-newline blob at
   `87d2e39c47b6747b505bc2657d505aeda2265b5e:experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json`,
   SHA-256 `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f`.
   A changed peer baseline must not silently replace it. Fable confirmed this
   identity at 02:18 UTC; it remains declared configuration, not observed load.
4. Own checkout only; parent owns its lease for preparation/ingestion. No other
   local writer. No heavy resources or new shared scientific interface.

## Exact preparation and dispatch

- Use a fresh, initially absent local directory:
  `.orchestration/f3-initial-call-20260906/`. Never erase/reuse an existing one;
  inspect its saved state on resume instead of starting another call.
- Read the immutable baseline Git blob as bytes and verify SHA-256/count.
  Use only reviewed `prepare_initial_t2_packet`, `render_initial_t2_prompt`
  and `publish_initial_t2_packet`; retain `record.json` and `task-packet.md`.
  No prefix, suffix or handwritten prompt substitution is permitted.
- Retain a preparation record with baseline, reviewed source, final prompt and
  record hashes/counts, exact HEAD, and missing-evidence classification.
- Before launch, write a durable dispatch-intent record: maximum calls 1,
  attempts remaining 0 after this intent, and expected launcher settings.
  An interrupted/uncertain launch must be reconciled from existing receipts;
  it is never permission to launch again.
- Existing `start-sol-worker` launcher only: `read-only`, owner
  `astra-f3-initial`, role `candidate`, scope
  `t2-initial-parameter-hypothesis-only`, requested `gpt-5.6-sol`, effort `max`.
  Use the exact published prompt SHA-256/count as launcher arguments.
- Attach the existing read-only exact-run watchdog in the same tool call as
  launch. Deadline: immutable request creation + 1,200 seconds. Preserve the
  watcher's start/terminal receipts and exact runner identity. Never infer a
  model/limit event from silence or substitute a model after a refusal.
- Store exact run directory/owner/packet and next action in the active pointer
  immediately. Stop rather than create a duplicate if any launch state is
  uncertain. The protocol authorizes one call total, not one per heartbeat.
- Release the preparation lease before waiting. Do not dispatch another Astra
  writer while this call is active; reacquire the parent lease after its terminal
  receipt and source checks, immediately before ingestion.

## Collect, retain, report

- On ordinary success or failure, wait for the exact terminal receipt. Inspect
  source hashes and confirm no other local writer before parent ingestion.
- Call only reviewed `ingest_initial_t2_sol_run` with retained record bytes,
  the exact run directory and a fresh `records` output beneath the call root.
  Retain request, packet, result, final response and an acceptance/rejection
  receipt as the implementation permits; missing bytes stay explicitly missing.
- Accept only the fixed read-only envelope, exact prompt, bounded successful
  terminal, and valid response echoes/parameters. A parse or transport failure
  is a retained negative outcome; no repaired response or retry is authorized.
- If publication/ingestion fails, preserve its exact error and partial files;
  do not delete, overwrite or invent a receipt. A fresh reviewed repair decision
  is required before further processing. No automatic retry loop.
- Format acceptance yields a hypothesis-only proposal and trusted recipe.
  Record expected configuration separately from raw request facts; neither is
  authenticated served-model evidence.
- Candidate admission, observed COM certificate, execution manifest, protected
  evaluator results, candidate measurements, runtime/training authorization and
  improvement remain missing or unauthorized. Do not send a runnable candidate
  to the peer registry or call the broader Python reward runner.
- Record outcome and evidence in a new result document; send Fable a concise
  retained-result handoff. The next scientific step needs its own matched,
  independently evaluated runtime protocol and resource agreement.
