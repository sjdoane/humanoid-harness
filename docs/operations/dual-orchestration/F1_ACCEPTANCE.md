# F1: accepted static formula plumbing

| status | current truth |
|---|---|
| progress | Independent closure accepts F1's data-only formula and synthetic proposal lineage. Parent reverified all ten reviewed file hashes. |
| bottleneck | F1 still uses the old B0 target set; no T2 compositor, authenticated model ingestion, or training connection is admitted. |
| next step | Map the smallest T2 formula handoff to Fable's committed Phase B registry before implementing that integration. |

## Decision and exact evidence

- Verdict: `ACCEPT_F1_STATIC_ONLY`; no actionable remaining findings.
- Review: `20260905T205043Z-470423ab-bf69-47dc-bc0e-d65f9ff01a74`.
- Requested leaf: `gpt-5.6-sol`, reasoning `max`, read-only, no delegation.
- Terminal receipt: `SUCCEEDED`, 2026-09-05 20:59:48 UTC; no writer lease
  required, no termination escalation. The deadline watcher observed completion.
- Reviewed base: `778f8ecadd14ade5bfdb5daf17274a981423c942`.
- Snapshot: local `.orchestration/f1-closure-20260905T2048.json`.
- Reviewer checked all ten file identities before and after review; parent
  repeated the check during the 21:25 UTC heartbeat. All matched.
- `uv.lock` remains
  `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40`.

| finding | closure |
|---|---|
| F1-01: missing raw response retention | Exact accepted/rejected bytes are retained, receipt-bound, rehashed and reparsed before revision. |
| F1-02: self-referential prompt identity | The prompt exposes a semantic request digest; its final bytes have a separate digest and byte count. |
| F1-03: forged schema/model ingress | Exact-class safe-tree validation and strict canonical copies precede public use. |
| Parent FIFO replay follow-up | Nonblocking open occurs before file-type inspection; regression checks ordering without risking a blocked OS call. |

## Checks and retained negative

- Reviewer actually ran **26 passed in 0.17 seconds**, plus an in-memory
  prompt-identity/validation probe. It did not run temporary-file tests.
- Earlier exact parent repair suite: **232 passed, 16 skipped**.
- Acceptance parent broadened the pure reward regression selection:
  **256 passed, 16 skipped, 1 deselected in 2.46 seconds**.

```bash
.venv/bin/python -m pytest -q -m 'not gym' tests/rewards tests/reward_search tests/integration/test_research_mailbox.py
```

- Retained selection error: the first invocation omitted `-m 'not gym'` and
  reported **1 failed, 256 passed, 16 skipped**. The marked real-Humanoid test
  failed on its initial `import gymnasium` because Astra has no simulator
  dependency. No simulator started. This is not a passed runtime test.
- Sixteen Python-worker/calibration tests remain explicitly deferred. The
  corrected invocation deselects the one simulator test; it does not certify it.
- Ruff and formatting pass for all five F1 Python source/test files.
- Tracked source diff was empty before acceptance; no Fable source was edited.

## Claim ceiling

- Accepted: bounded recipe math, data contracts, supplied-byte lineage and
  rejection/revision behavior under static/synthetic tests.
- Not accepted: arbitrary Python reward execution, authenticated model origin,
  a bound compositor, T2 runtime integration, learning, or humanoid improvement.
- `RESPONSE_ORIGIN` remains `unverified_synthetic_supplied_bytes`.
- Historical builder/repair receipts remain unchanged. This decision supersedes
  their pending-acceptance status, not their original observations.

