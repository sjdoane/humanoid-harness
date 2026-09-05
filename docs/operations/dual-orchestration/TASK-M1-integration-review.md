# M1: review the pinned-main merge only

- Read-only Sol/max leaf; no delegation, edits, tests, model calls, simulator,
  host probes, peer messages, or Git mutation. Maximum 20 minutes; final 500 words.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Review exact commit `8a48f322a4e3e14bb4467daddd709de1926d0e1d`, not an eventual
  working-tree snapshot. A separate builder may add new formula files.
- Read `AGENTS.md`, dual-orchestration `README.md`,
  `MAIN_INTEGRATION_20260905.md`, and `R1_ACCEPTANCE.md` at that commit.

## Verify, do not repeat old research reviews

1. Two parents are Astra `3cbdbdb344cb28546fda47ec44dab8dac07e4909` and agreed
   main `2fb31dcc85a23d1d1ed0b8ab3c86b30d460d7d51`; accepted R1/A1 history remains.
2. Inspect the merge diff relative to both parents. All conflicted reward
   source/tests and manifest are byte-identical to accepted R1 `754e869`.
   The peer's corresponding reward paths still matched donor `89d4c34`.
3. Both historical smoke receipts have the hashes/path provenance in the
   checkpoint. No regeneration, lost version, or current-runtime certification.
4. Check newly imported committed source for consumers of removed R1 names or
   a concrete cross-branch API regression. No broad hardening or runtime audit.
5. Source/status claims stay separate: parent reports 144 passed/16 deferred,
   9 pure oracle tests, format/lint/compile only, not a rerun of main's robots.

Return ACCEPT_INTEGRATION or REJECT with only actionable merge-induced findings.
Do not block on new formula files, pending training, or defects already declared
outside this exact merge. Do not read Fable's dirty source or claim new tests.
