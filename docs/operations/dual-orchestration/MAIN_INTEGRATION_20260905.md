# Pinned-main integration checkpoint

| status | current truth |
|---|---|
| progress | Main `2fb31dc` is merged into Astra with accepted R1 source/test bytes preserved. Parent focused checks passed. |
| bottleneck | Independent integration review and the new formula implementation remain; no reward training runtime is admitted. |
| next step | Review this exact merge and build the isolated data-only formula slice, then send Fable the reviewed result. |

## Exact inputs and permission

- Astra parent: `3cbdbdb344cb28546fda47ec44dab8dac07e4909`.
- Main parent: `2fb31dcc85a23d1d1ed0b8ab3c86b30d460d7d51`.
- Peer proposal: `20260905T183421.793603Z-8b619f3d56a84cbb81d31d39e11c36ef`.
- Acceptance: `20260905T185408.005663Z-ff80151886fc485782897509ffe98696`.
- Clean Astra worktree and unclaimed lease checked before acquiring the parent
  integration lease. Both read-only R2 workers had terminal receipts.
- No rebase, push, or edit to Fable's dirty checkout. Accepted history retained.

## Resolution and verification

- All 12 add/add conflicts matched the earlier merge-tree diagnostic.
- Eleven source/test conflicts resolved through `apply_patch` to exact R1
  `754e869` bytes. Main's corresponding paths still equaled original B0 donor
  `89d4c34`; no later peer reward-source repair was discarded.
- `git diff 754e869 --` across reward source/tests and the reward manifest/test
  is empty. No unresolved index entries or whitespace errors remain.
- Historical smoke receipt at its original path retains exact pinned-main
  bytes: SHA-256 `7412ecb7e73bb338f8c472232b401c6fd2161eaf886994ed6e37378640cc0e88`.
- Exact Astra predecessor bytes retained at
  `experiments/family_b_target_speed_v1/receipts/history/astra_754e869_builder_runtime_no_learning_smoke.json`:
  SHA-256 `2f6276011998c6105ca6714f7b380575f4a1c49e8497aac471c0e35e55629ea6`.
- Both receipts are historical, not certificates for the merged runtime.
  Neither was regenerated or reauthorized.
- Lock SHA-256 unchanged:
  `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40`.
- Own-worktree import verified. R1/A1/mailbox checks: **144 passed, 16 deferred**
  in 2.42 seconds. Deferred cases still require host/runtime approval.
- Newly imported pure oracle-contract tests: **9 passed** in 0.03 seconds.
- Fifteen reward/manifest source and test files: Ruff check/format pass.
  Reward, reward-search, and harness modules compile. Diff check passes.
- Simulator-dependent CLI/executor and complete suites were not run. Astra's
  environment remains dev-only; no dependency stubs or main environment used.

## Formula-path review collected

- Run: `.orchestration/sol-runs/20260905T182204Z-c7980f46-86fe-4d5a-88da-7b0479cd38bc`.
- Requested Sol/max, read-only; succeeded 18:28:40 UTC. Watcher observed terminal.
- Decision: new versioned JSON formula/parameter family; preserve A1 source
  schemas and old receipts. Full final is retained in the exact run.
- Review's missing committed runner was resolved by Fable's later `50fd60c`
  interface and `2fb31dc` integration. Fine-tuning/reward runtime still pending.
- New integration question found by parent: current composition task uses root
  speed and targets about 5.52/0.885 m/s; old B0 inputs use COM speed and only
  0.5/1.0/1.5 m/s. Do not silently widen or substitute those frozen semantics.
  Formula core can be built against B0 inputs while the peer resolves the
  actual reward-study task and adapter contract.

These checks establish an integration checkpoint, not humanoid improvement.
