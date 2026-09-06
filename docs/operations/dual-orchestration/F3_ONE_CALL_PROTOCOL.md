# F3 one-call protocol

| status | current truth |
|---|---|
| progress | The accepted initial-only protocol is re-pinned to Fable's reward lane in `main`; its one-call, zero-retry semantics are unchanged. |
| bottleneck | No F3 candidate call, candidate admission, protected evaluation, training, or reward result exists. |
| next step | Review and commit the re-pin, then freeze the separate T2 reward-study protocol. Do not dispatch the candidate call from the re-pin slice. |

## Purpose and strict scope

- One local subscription-authenticated initial parameter hypothesis; zero
  revision calls, retries, Python reward execution, simulator or training.
- Research input: supplied tracking-only baseline plus accepted T2 contracts.
  Output: alpha/beta hypothesis, rationale, predicted effect and falsifier.
- No protected evaluation or measured feedback exists. Do not invent it.
- This is a software integration check, not reward efficacy or model attestation.

## Preconditions

1. Independent verdict `ACCEPT_F3_STATIC_ONLY` closed F3-01 through F3-04 and
   the exact-baseline refresh. Separate verdict `APPROVE_F3_ONE_CALL_PROTOCOL`
   approved the one-call semantics. The Fable re-pin must be reviewed and
   committed before use. Any rejection or missing verdict stops dispatch.
2. The accepted Astra source is merged into `main`; this configuration is bound
   to checkout `/Users/samueldoane/Documents/ChatGPT/humanoid-harness`, branch
   `main`, and import origin
   `/Users/samueldoane/Documents/ChatGPT/humanoid-harness/src/oracle_composition/__init__.py`.
   These are expected local configuration, not served-model attestation. No
   source or accepted contract may change between re-pin review and use.
3. Baseline is the exact 1,773-byte no-newline blob at
   `8d617e30dd239529a42e3a0211314d6173ffeafd:experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json`,
   SHA-256 `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f`.
   It is byte-identical to the accepted binding at `87d2e39c47b6747b505bc2657d505aeda2265b5e`;
   FT2R3 did not change it. It remains declared configuration, not observed load.
4. The preparation and ingestion lease configuration is owner
   `fable-f3-prepare`, role `builder`, scope
   `t2-initial-packet-preparation-and-ingestion-only`. Own checkout only; no
   other local writer, heavy resource, or new shared scientific interface.

## Pinned initial dossier

- Task text: hold `3.0 m/s` COM forward speed from the expert start.
- F2 evaluator, recipe/parser, T2-input, and bounds SHA-256 values are
  `064393887bb4a7157d614981cc2000940252e12c31ad0aabc13109737fd94301`,
  `c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb`,
  `9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2`, and
  `2c6030264231a52320a44fe0f4d4ed519bc2e635b892f1b9c0d8929166106933`.
- Only bounded `alpha` and `beta` are authorable. No measured feedback exists.
- Canonical prompt: 6,011 bytes, SHA-256
  `4d1a29765d929472e2f2a346294f98ac08c0e35d3e18c147e761e46678fe2bbc`.
  It excludes simulator internals, corpus payload bytes, and a phase-B execution
  manifest.

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
- Before launch, call only `publish_initial_t2_dispatch_intent` to write the
  fixed `dispatch-intent.json`: maximum initial calls 1, attempts remaining 0,
  zero revision calls, zero retries, the 1,200-second deadline, and expected
  checkout/preparation/candidate configuration. The no-overwrite publisher
  refuses a second record at that path.
  An interrupted/uncertain launch must be reconciled from existing receipts;
  it is never permission to launch again.
- Existing `start-sol-worker` launcher only: `read-only`, owner
  `fable-f3-initial`, role `candidate`, scope
  `t2-initial-parameter-hypothesis-only`, requested `gpt-5.6-sol`, effort `max`.
  Use the exact published prompt SHA-256/count as launcher arguments.
- Attach the existing read-only exact-run watchdog in the same tool call as
  launch. Deadline: immutable request creation + 1,200 seconds. Preserve the
  watcher's start/terminal receipts and exact runner identity. Never infer a
  model/limit event from silence or substitute a model after a refusal.
- Store exact run directory/owner/packet and next action in the active pointer
  immediately. Stop rather than create a duplicate if any launch state is
  uncertain. The protocol authorizes one call total, not one per heartbeat.
- Release the preparation lease before waiting. Do not dispatch another Fable
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
- Record outcome and evidence in a new result document and the Fable-lane
  handoff. Notify Astra only for a shared scientific dependency. The next step needs its own matched,
  independently evaluated runtime protocol and resource agreement.
