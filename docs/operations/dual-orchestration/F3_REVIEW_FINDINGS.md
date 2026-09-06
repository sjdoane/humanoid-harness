# F3 review: repair required

| status | current truth |
|---|---|
| progress | Independent reviewer ran 79 focused tests and seven defect/negative probes; parent reproduced all seven probes and verified the 18-file snapshot. |
| bottleneck | Three contract defects block F3 static acceptance. Receipt metadata also needs clarification; the baseline requires an explicit refresh. |
| next step | One bounded repair of the three F3 Python files, followed by targeted independent closure. No candidate call. |

## Decision and receipts

- Verdict: `REJECT_F3`; accept findings F3-01 through F3-04 for repair.
- Worker: `20260906T011040Z-3e33069f-5fd4-4de2-a9f1-9265da8367ec`.
- Terminal: `SUCCEEDED`, exit 0, 2026-09-06 01:21:49 UTC;
  lease `RELEASED`, watcher `terminal_observed`, no escalation.
- Independent report: `.orchestration/f3-independent-20260906/REVIEW.md`,
  SHA-256 `aea9436f48d7026a2dfd0a6e24b52ee5aefc849913eb867004eecf5a654ca9b6`.
- Independent probes: same directory, `test_f3_adversarial.py`, SHA-256
  `0f1109a4ae2c42f3dd1859e5d20e3bf35e7646f1518f23884606000de9268b2f`.
- Parent reproduced `7 passed in 0.10s`; these passes demonstrate defects or
  negative-path behavior, not correctness. Retained parent fixtures:
  `.orchestration/f3-parent-negative-sewc8e/pytest`.
- All 18 snapshot hashes still match; no reviewed source was modified.
- Parent correction to reviewer timing table: actual immutable request creation
  was 01:10:40 UTC and watcher deadline 01:30:40 UTC. The review used snapshot
  creation instead. Completion was before both times; original report retained.

## Confirmed findings

| ID | severity | reproduced consequence | required repair |
|---|---|---|---|
| F3-01 | high | Year-9999 timestamps overflow deadline addition after retaining four raw files; no outcome receipt is produced. | Compare elapsed duration without overflowing datetime addition; test extreme dates and duration boundaries. |
| F3-02 | medium | Rejected receipt/schema parses accept partial artifact or proposal/recipe identities. Production ingestion constructs complete pairs; no acceptance bypass was found. | Require every identity field absent on rejection and every field present on retention/acceptance; test each subset. |
| F3-03 | medium | Both path-taking APIs read three trusted sources before rejecting a native-Path subclass. No publication occurs. | Exact-check every public argument type before record reconstruction or any reads/callbacks. |
| F3-04 | low | A rejected Luna request is retained while receipt `requested_model` says Sol; that field actually means expected configuration. | Rename the fixed metadata fields to `expected_*`; keep observed request facts in retained bytes. |

## Bounded follow-up

- Preserve original four F3 files under
  `.orchestration/f3-pre-repair-20260906/` with the original relative paths.
  Parent verified all four archive hashes against the review snapshot.
- Do not modify the builder result, independent report, probes, or old negative
  fixtures. They describe the historical implementation.
- Refresh the unaccepted F3 protocol to the exact 1,773-byte tracking-only
  artifact at main object `87d2e39c47b6747b505bc2657d505aeda2265b5e`, SHA-256
  `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f`.
- Preserve `r_task=+0.0` as baseline, no parent recipe, no fabricated feedback.
  A declared admission contract is not an observed certificate or successful
  runtime load. Candidate admission and measurements remain missing.
- Existing A1/F1/F2/T2 sources, peer files and scientific interfaces stay fixed.
- Static closure is prerequisite work only. No generated reward execution,
  simulator use, training, improvement claim or shared heavy reservation.
