# Resource slot 01: minimal cooperative heavy-job exclusion

## Authority and bounds

- Work only in `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`,
  branch `astra/reward-loop`; source base `6a39aaaf26455caa30e30cd527100ec7df375bc9`.
  The parent may add this packet/checkpoint as docs-only descendants.
- Explicit accepted scope: proposal
  `20260906T132352.198444Z-b86226c1a7b544e997ae9ce2324a4145`, acceptance
  `20260906T133444.291029Z-5169f292a3bd4f48a65d05a997521904`.
- Read AGENTS.md and the dual-orchestration README. One Sol/max builder, no
  nested agents. Use apply_patch. Preserve unrelated work and executable bits.
- Target 12 minutes. Stop at 18 minutes with an honest partial handoff; exact
  runner watchdog is 20 minutes. Do not broaden the task to finish everything.
- Authorized files only:
  1. `src/oracle_composition/harness/resource_slot.py` (new)
  2. `tests/harness/test_resource_slot.py` (new)
  3. `scripts/research-mailbox`
  4. `tests/integration/test_research_mailbox.py`
  5. `docs/operations/dual-orchestration/HEAVY_JOB_SLOT.md` (new)
- Do not edit `phase_b/supervision.py`, other runtime/reward/evaluator sources,
  dependencies, source-bound receipts, hooks, private logs or other checkouts.
  Fable owns supervisor integration and main promotion after independent review.
- Do not commit; leave the exact bounded diff for parent tests and review.
  The launcher releases your writer lease on terminal completion.

## Build the agreed token, not a scheduler

- One immutable `heavy-job.lock` at a supplied coordination root, matching the
  mailbox root shared by both worktrees. Publish by exclusive `os.link` from a
  fully written/fsynced temporary regular file in that directory.
- Persistent sibling guard file plus stdlib `flock` serializes reserve/release.
  Never delete the guard. Unique token identity and serialization must prevent
  a stale release from removing a newer reservation, even for the same owner.
- Versioned token binds unique `token_id`, exact owner, canonical argv, clean
  commit, proposal and acceptance identity, canonical reservation SHA-256,
  expected/hard wall seconds, and acceptance expiry. Keep a small explicit schema.
- Reserve accepts the existing v2 reservation format and verifies its actual
  bounded native-mailbox acceptance/acknowledgment artifacts and exact canonical
  acceptance body, not a caller's `accepted: true`. Compare canonical bytes,
  with strict relevant types, not Python bool/int or int/float alias equality.
- Existing authority contract to inspect read-only:
  `phase_b/supervision.py::validate_reservation` (558–777 in this base).
  Do not import that module: the new helper must import only stdlib and no
  Gymnasium, MuJoCo, Torch or training stack. Reuse this new helper at future
  integration rather than adding another validator to the CLI.
- Expose a documented read-only helper to validate a token against an already
  validated reservation immediately before spawn: complete binding, not only
  presence or owner. Do not implement that spawn integration in this slice.
- Release requires exact owner plus unique token identity. No force delete,
  expiry-based steal, implicit replacement, automatic retries or process killing.
- An expired slot remains occupied until explicit owner cleanup. Expiry rejects
  new dispatch; it does not transfer ownership. Missing versus malformed slot
  must remain distinct. Failure must not alter another owner's record.
- Bounded regular-file reads and safe root/path handling: reject symlinks,
  oversized/malformed/duplicate-key/nonfinite/contradictory records and unsafe
  authority paths. Fail closed on unsupported locking platforms.
- Cooperative coordination is not OS isolation or authentication. No claim
  that this prevents a same-user non-cooperating process from running.

## CLI and compatibility

- Thin `reserve`, `release`, and `board` status adapters in the existing CLI.
  Prefer `reserve --owner OWNER --reservation PATH --expected-wall-seconds N`,
  `release --owner OWNER --token-id ID`, plus the existing JSON error convention.
- Preserve existing send/inbox/ack/status semantics and worktree root resolution.
- Script still runs under macOS system Python 3.9. Keep helper stdlib-only and
  compatible; use `timezone.utc`, future annotations, and no new dependency.
- Resolve the helper from the script's own checkout, never another installed
  checkout. If the real-worktree test fixture needs the helper, copy that exact
  source into the fixture; no hidden cross-checkout import fallback.
- Runbook: one short state sequence (free -> held -> owner release -> free;
  expired -> held pending cleanup), examples with placeholders, integration
  helper signature and explicit not-yet-wired/no-training boundary.

## Focused verification only

- Before tests, prove imports resolve inside this checkout. Tests may create
  isolated temporary fake mailboxes/acceptance fixtures and small subprocesses.
- Run only the two authorized test files, then Ruff/format/diff checks on the
  changed files. No full suite, simulator, fake PPO loop, training, network,
  source-receipt regeneration, real coordination token acquisition, or API call.
- Cover: successful reserve/read/validate/release; real two-process contention
  yields one winner; loser preserves winner; wrong owner/token and stale release
  after reacquisition cannot remove it; expiry does not free/steal; acceptance,
  command/commit/digest mismatch; missing/malformed/symlink/oversized slot;
  invalid guard; bounded CLI errors and existing mailbox behavior.
- Use actual tempfile operations and real process coordination, not mocked final
  statuses. Retain incomplete tests honestly; do not weaken a test to pass.

## Final handoff

Start with progress/bottleneck/next step rows. List exact checkout, branch,
starting commit/dirty state, changed paths, lock-file digest, test commands and
results, helper API, and remaining integration/verification limits. Do not call
the token accepted, deployed or a training result before independent review.
