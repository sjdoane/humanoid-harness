# A2 transport interruption: host evidence

| status | finding |
|---|---|
| progress | Local authentication succeeds. macOS power logs show the host slept 16 seconds after the A2 launch, followed by repeated sleep/dark-wake cycles. |
| bottleneck | Sleep is a concrete execution confounder; it does not by itself prove the only cause of the websocket timeout. No reward proposal exists from A2. |
| next step | Run one explicitly bounded replacement canary with a temporary idle-sleep assertion; retain the failed attempt and stop if the replacement also fails. |

## Checks

- CLI: `codex login status` reports ChatGPT login; version `0.153.2`.
- Current account snapshot: 43% of the reported seven-day Codex window used;
  no reached-limit type reported. This snapshot does not establish historical
  quota at the failure time; a missing secondary window is unavailable, not zero.
- `pmset -g log`, September 4 local time (UTC minus seven hours):

| local time | UTC | event |
|---|---|---|
| 15:54:03 | 22:54:03 | Initial A2 launcher started |
| 15:54:19 | 22:54:19 | Idle Sleep, battery power |
| 15:57:23–16:22:16 | 22:57:23–23:22:16 | Repeated dark wakes and maintenance sleeps |
| 16:20:57 | 23:20:57 | CLI websocket idle timeout |
| 16:22:35 | 23:22:35 | Full wake due to user-activity assertion |
| 16:23:40 | 23:23:40 | Clamshell sleep |
| 16:23:57 | 23:23:57 | Lid/HID wake |

- At the diagnostic snapshot, the clamshell was open and battery was 100%.
- A temporary 900-second `caffeinate -i` assertion was started at the resumed
  00:45 UTC checkpoint and verified in `pmset -g assertions`. No global power
  setting, display setting, or lid-close behavior was changed.
- This assertion prevents idle sleep, not a guarantee against lid closure,
  shutdown, network loss, app exit, or quota exhaustion.
- Official OpenAI documentation says local scheduled work needs the computer
  on and the app running: [Scheduled tasks](https://learn.chatgpt.com/docs/automations).

## Interpretation

The original 31-minute interval was not an uninterrupted model run. Sleeping
overlaps the failed call and is a plausible contributor to the timeout.
Authentication and present quota do not show a current access blocker. This
warrants a single changed-condition follow-up, not an unbounded retry loop or
an assertion that the provider/model caused the failure.
