# F3 one-call protocol

| status | current truth |
|---|---|
| progress | T2PAIRR1 re-seals the pending one-call inputs with integrated pairing receipt `1a2b7ece…2348` and repaired production-routing contracts. Preparation verifies the receipt before prompt rendering. |
| bottleneck | Dispatch remains withheld at `withheld_pending_dispatch_verdict`: the canonical candidate is TBD, T2PAIRR1 needs narrow independent review and commit, and no candidate call or behavioral result exists. |
| next step | Independently review and commit T2PAIRR1, then obtain a separate dispatch verdict. A future accepted candidate must pass the one-step final admission and clean-HEAD re-seal before report or runtime admission. |

## Purpose and strict scope

- One local subscription-authenticated initial parameter hypothesis; zero
  revision calls, retries, Python reward execution, simulator or training.
- Research input: supplied tracking-only baseline plus accepted T2 contracts.
  Output: alpha/beta hypothesis, rationale, predicted effect and falsifier.
- No protected evaluation or measured feedback exists. Do not invent it.
- This is a software integration check, not reward efficacy or model attestation.

## Preconditions

1. Independent verdict `ACCEPT_F3_STATIC_ONLY` closed the original F3 findings,
   while the Fable re-pin review and T2AR1 review required later repairs.
   T2PAIRR1 must be independently accepted and committed, and a separate dispatch
   verdict must exist; any rejection, missing verdict, or different source bytes
   stops dispatch.
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
4. Preparation receives and verifies the exact 1,671-byte
   `experiments/004_t2_reward_study/t2_seal_v1.json`, SHA-256
   `6ce006bb983611179ab9fc8bc47b7b01c74d6303a3f7f0a917bd35947d9d5a9d`,
   before it renders the prompt. The non-model-facing seal binds the expert-hold
   oracle, T2 training design, evaluator design, pending execution manifest,
   study manifest, tracking-only baseline, arm-invariant pairing key, and exact
   integrated pairing-receipt SHA-256. Every bound file is re-read and
   hash/count checked; the receipt is validated and the pairing key is
   recomputed from equal common arm fields. The current seal truth is
   `pairing_receipt: 1a2b7ece…2348` and `withheld_pending_dispatch_verdict`.
   Therefore dispatch remains prohibited.
5. The preparation and ingestion lease configuration is owner
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
- The seal SHA-256, evaluator-design SHA-256/content, study-manifest
  SHA-256/content, pairing key, and execution-manifest content are not placed in
  the prompt. Changing a valid non-model-facing seal changes the packet/call
  identity but leaves these 6,011 prompt bytes unchanged.

## Exact preparation and dispatch

- Use a fresh, initially absent local directory:
  `.orchestration/f3-initial-call-20260906/`. Never erase/reuse an existing one;
  inspect its saved state on resume instead of starting another call.
- Read the immutable baseline Git blob as bytes and verify SHA-256/count.
  Read `t2_seal_v1.json` as bytes. Use only reviewed
  `prepare_initial_t2_packet`, passing both exact byte strings, then
  `render_initial_t2_prompt` and `publish_initial_t2_packet`; retain
  `record.json` and `task-packet.md`.
  No prefix, suffix or handwritten prompt substitution is permitted.
- Retain a preparation record with baseline, reviewed source, final prompt and
  record hashes/counts, exact HEAD, and missing-evidence classification.
- Protocol identifier is fixed as `f3_initial_t2_one_call/v1`. Derive `call_id`
  as SHA-256 over `protocol_id || NUL || study_manifest_sha256 || NUL ||
  rendered_prompt_sha256`. The current value is
  `eeb24a672fd61fbf91d668cd2ca62629decf9f9ed12124f03a27925c01ee4aef`.
  The canonical call-identity record digest is
  `1ae0e967207c1dc2a2d9535e98b7ea6782dabccdf6afff6f518bb07242bc892d`.
- Before launch, call only `publish_initial_t2_dispatch_intent` with the exact
  canonical path `.orchestration/f3-initial-call-20260906/dispatch-intent.json`.
  It first claims `call-identity.json` at that call root using atomic
  no-overwrite publication, then publishes the intent with the call identity
  digest, T2 seal and study-manifest bindings, maximum initial calls 1,
  attempts remaining 0, zero revisions/retries, and the 1,200-second deadline.
  A sibling intent path, conflicting/pre-existing identity, or pre-existing
  canonical intent is refused with a retained protocol-refusal receipt.
  A second claim of the canonical call identity is refused; repository state
  cannot prove the absence of an unrecorded external call.
  An interrupted/uncertain launch must be reconciled from existing receipts;
  it is never permission to launch again.
- Existing `start-sol-worker` launcher only: `read-only`, owner
   `fable-f3-initial`, role `candidate`, scope
  `t2-initial-parameter-hypothesis-only;call_identity_sha256=1ae0e967207c1dc2a2d9535e98b7ea6782dabccdf6afff6f518bb07242bc892d`,
  requested `gpt-5.6-sol`, effort `max`. The existing launcher's exact `scope`
  field binds the digest into both retained `request.json` and `result.json`;
  changing or omitting it makes ingestion reject the envelope.
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
- Call only reviewed `ingest_initial_t2_sol_run` with retained record bytes and
  the exact run directory. Output is fixed at the canonical call root's
  `records/` directory; the caller cannot select a sibling output or intent.
  Retain request, packet, result, final response and an acceptance/rejection
  receipt as the implementation permits; missing bytes stay explicitly missing.
- Ingestion requires the exact canonical call-identity and dispatch-intent
  bytes. It atomically claims `ingestion-claim.json` before envelope/proposal
  validation. Missing or mismatched intent, a second valid run, or a replay of
  the same provider result is rejected with retained raw artifacts and a call
  receipt. Any first run—including malformed or failed output—consumes the
  ingestion claim; zero retry remains the rule.
- Accept only the fixed read-only envelope, exact prompt, bounded successful
  terminal, and valid response echoes/parameters. A parse or transport failure
  is a retained negative outcome; no repaired response or retry is authorized.
- If publication/ingestion fails, preserve its exact error and partial files;
  do not delete, overwrite or invent a receipt. A fresh reviewed repair decision
  is required before further processing. No automatic retry loop.
- Format acceptance yields a hypothesis-only proposal and trusted recipe.
  Record expected configuration separately from raw request facts; neither is
  authenticated served-model evidence.
- Candidate admission, observed COM certificate, protected evaluator results,
  candidate measurements, runtime/training authorization and improvement remain
  missing or unauthorized. Format acceptance alone does not admit a candidate.
  The reviewed admission step must registry-resolve both rewards, verify actual
  clean HEAD, and regenerate the final execution/study/seal chain together. Do
  not send a runnable candidate to the peer registry or call the broader Python
  reward runner from this protocol.
- Record outcome and evidence in a new result document and the Fable-lane
  handoff. Notify Astra only for a shared scientific dependency. The next step needs its own matched,
  independently evaluated runtime protocol and resource agreement.
