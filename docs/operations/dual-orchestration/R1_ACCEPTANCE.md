# R1 accepted: static reward-source validation

| status | current truth |
|---|---|
| progress | Independent review ACCEPTed static-only R1; parent verified the reviewed hashes and reproduced 144 passing checks after the requested documentation/test follow-up. |
| bottleneck | Sixteen runtime checks remain deferred. R2 containment/admission and R3 provenance still block generated reward execution and training. |
| next step | Prepare the smallest R2 runtime/host-fixture protocol, review it, and obtain any required shared resource agreement before execution. |

## Acceptance evidence

- Decision run: `20260905T135846Z-4adbb20a-a0ee-468a-a4ef-3dd4a807c825`.
- Final: `ACCEPT — static-only R1`; finished 2026-09-05 14:58:59 UTC.
- Requested: read-only `gpt-5.6-sol/max`. Sender names and requests are not
  provider identity attestation.
- All three previous findings closed: exact text/bytes ingress, proxy/callback
  rejection, and complete AST-derived source-envelope verification.
- Parent verified all 15 reviewed hashes before the non-blocking follow-up.
- Follow-up adds one regression test and this documentation; accepted
  production source bytes did not change.
- Retained history: `R1_RESULT.md`, `R1_REVIEW_FINDINGS.md`,
  `R1_INGRESS_REPAIR_RESULT.md`, and `R1_FINAL_REVIEW_PARTIAL.md`.
  Timed-out and rejected reviews remain recorded, not overwritten.

## Supported receipt copying

```python
clone = StaticAcceptanceReceiptV1.from_dict(receipt.to_dict())
```

- Receipt constructors accept plain JSON input. Their metadata output is a
  deeply immutable view, not accepted constructor input.
- `dataclasses.replace(receipt)` intentionally fails closed. The final
  reviewer classified this as a non-blocking API limitation; no production
  receipt-copy consumer requires it.
- No weaker mapping-proxy acceptance was added.

## Parent verification

- Focused suite: `144 passed, 16 skipped`; includes the 59 unchanged A1/mailbox
  checks. Same command recorded in `R1_COMPLETION_CHECKPOINT.md`.
- Ruff check/format, compile checks, and diff checks passed.
- Deferred: 13 host/subprocess cases and 3 dynamic scale cases.
- No candidate execution, live OS canary, simulator, training, paid API call,
  remote push, or historical receipt regeneration.

## Timing caveat and continuity repair

- The decision run took about 60 minutes of wall time, not the intended 20.
  Its watchdog was attached about 42 minutes after launch, following long gaps
  between parent tool calls. Stream idle-timeout warnings were also observed;
  their cause is not established.
- The old watcher measured its deadline from attachment. Both local one-shot
  watchers now measure from the immutable launch timestamp and act immediately
  when attached late. They still cannot run while the host is asleep.
- Future dispatch must launch and attach the watcher in one orchestration
  tool call, without waiting for another parent turn.
- No quota or model substitution is inferred from these gaps.
