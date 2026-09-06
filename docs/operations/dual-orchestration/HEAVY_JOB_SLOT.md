# Cooperative heavy-job slot

| status | current truth |
|---|---|
| progress | A stdlib helper and mailbox adapters implement one atomic `heavy-job.lock`. |
| bottleneck | Supervisor pre-spawn validation and terminal cleanup are not wired in this slice. |
| next step | Independently review this diff, then integrate the helper at the Fable-owned spawn boundary. |

## Contract

- Coordination root: the same `harness-coordination/` directory used by both
  worktrees' mailbox commands.
- Token: one canonical, immutable `heavy-job.lock`, published by exclusive
  `os.link` from a fully written and fsynced temporary file in that directory.
- Serialization: `heavy-job.guard` is an empty persistent regular file guarded
  with stdlib `flock`. Never delete it.
- Authority: reserve validates the v2 reservation's bounded acceptance message,
  exact canonical acceptance body, digest, and read acknowledgment.
- Release: exact owner and unique token ID only. There is no force release,
  expiry steal, replacement, retry loop, process kill, or scheduler.
- Boundary: this is cooperative same-user coordination, not OS isolation,
  authentication, or protection from a non-cooperating process.

## State sequence

| event | resulting state | dispatch meaning |
|---|---|---|
| initial | free | a validated reservation may reserve |
| successful reserve | held | only the completely bound, unexpired token may dispatch |
| exact owner release | free | another validated reservation may reserve |
| acceptance expires while held | held pending cleanup | reject dispatch and stealing; exact owner release remains valid |

Malformed, symlinked, duplicate-key, nonfinite, or oversized token data is not
reported as free. It fails closed and remains untouched.

## Mailbox commands

Use placeholders until a separately accepted heavy run exists:

```bash
./scripts/research-mailbox reserve \
  --owner OWNER \
  --reservation /ABSOLUTE/PATH/reservation-v2.json \
  --expected-wall-seconds N

./scripts/research-mailbox board

./scripts/research-mailbox release \
  --owner OWNER \
  --token-id TOKEN_ID
```

Commands emit the mailbox's existing JSON success/error format. `board` adds a
`heavy_job_slot` object whose state is `free` or `held`; held state includes the
validated token and an `expired` flag.

## Future spawn integration

After the existing v2 reservation validator succeeds, and immediately before
spawn, call the read-only helper:

```python
validate_slot_for_reservation(
    coordination_root,
    validated_reservation,
    expected_wall_seconds=expected_wall_seconds,
)
```

This compares owner, canonical argv, commit, proposal and acceptance IDs,
canonical reservation SHA-256, expected/hard wall seconds, and acceptance
expiry. Fable owns wiring this call and exact terminal cleanup into
`phase_b/supervision.py`; neither is implemented here. These interface tests do
not run training or establish a humanoid-performance result.
