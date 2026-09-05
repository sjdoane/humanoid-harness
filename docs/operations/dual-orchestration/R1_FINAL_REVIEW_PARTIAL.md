# R1 final review: partial evidence, no verdict

| status | current truth |
|---|---|
| progress | The reviewer explicitly reported all three prior ingress findings closed and verified all 15 hashes again. Parent verified the same hashes at collection. |
| bottleneck | Its 20-minute deadline expired before a final verdict. It also found that `dataclasses.replace(valid_receipt)` refuses the receipt's internal mapping proxy. |
| next step | Obtain a short independent acceptance decision, including the severity and disposition of the copy-API limitation. Do not repeat the broad review or enable runtime execution. |

- Run: `20260905T071423Z-bc589f15-291b-4341-94aa-730484edcd39`.
- Terminal: `INTERRUPTED_TERM`, exit 143, 07:35:25 UTC; no escalation,
  no `final.txt`. Watcher sent TERM at the configured deadline.
- Exact retained evidence: that run's `events.jsonl`, items 41, 45, 46, 49, 52.
- Item 49: reviewer reports the three prior findings closed on ordinary
  constructor/factory/JSON paths; dynamic refusal files are byte-identical.
- Item 52: all 15 snapshot hashes pass after review. Parent repeated the
  snapshot check at the 07:58 heartbeat; source remains unchanged.
- Item 41: selected no-temp pytest exited zero but has no captured output;
  no passing-test count is claimed from that event.
- Item 45: separate no-temp subtype/immutability/JSON clone probe passed.
- Item 46: no-argument `dataclasses.replace(receipt)` raises
  `StaticValidationError` because reconstruction receives `mappingproxy`.
- Parent reproduced both that refusal and a successful
  `StaticAcceptanceReceiptV1.from_dict(receipt.to_dict())` clone. No ordinary
  production caller of receipt-level `dataclasses.replace` was found in the
  adopted reward source; source-envelope replacement remains separately tested.

This is retained independent partial evidence, not an invented final verdict.
The timeout does not establish a quota or transport failure: no such event was
observed. The next packet narrows the decision and forbids another broad audit.
