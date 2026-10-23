# Evidence UI IPv6 correction · October 22, 2026

| progress | Corrected the admitted IPv6 loopback bind and printed/browser URL; local fixture checks pass. |
|---|---|
| bottleneck | Previously, `ui --host ::1` passed the loopback allowlist but failed while binding an IPv4 socket. |
| next step | Independently review the correction and verify hosted software checks. |

At base `eacd9cb83b6fbcde3c02c487160ad0d38859e99f`, a native IPv6 loopback
socket could bind locally, but `humanoid-harness ui --host ::1 --no-open`
exited with a socket-address error. The URL builder would also produce
`http://::1:8765` instead of `http://[::1]:8765`.

The [server](../../src/oracle_composition/ui/server.py) now selects an IPv6
socket for the already-admitted `::1` host and brackets IPv6 addresses in the
URL shared by the printed message and browser callback. The bind allowlist,
HTTP Host validation, read-only routes and evidence validators are unchanged.
IPv4 and `localhost` remain supported.

Local macOS verification with the unchanged lockfile:

| Check | Observed result |
|---|---|
| Expanded evidence UI tests against original server | 2 failed, 25 passed: IPv6 bind and URL failures |
| Evidence UI and unified CLI tests | 37 passed |
| Other G1/TQC evidence UI fixtures | 22 passed |
| Portable selection, Python 3.12.14 / 3.13.15 | 261 passed / 261 passed; IPv6 available, no skips |
| Repository lint and changed Python-file formatting | Passed |
| Migration verification | 120/120 files; 14 archived snapshots; zero errors |
| Source distribution and wheel build | Passed |

The [regressions](../../tests/ui/test_evidence_ui.py) exercise local HTTP health
responses on all admitted bind hosts, hostile Host rejection, IPv4/IPv6 browser
URLs and non-loopback bind rejection. The IPv6 network case skips explicitly
only if a separate native socket probe finds no usable IPv6 loopback; server
construction errors remain failures. Browser callbacks are mocked.

No training, simulator rollout, provider call or new research evaluation was
performed. Study definitions and their recorded identities, archived receipts,
manifests and dependency lock are unchanged. This is a local inspection fix; it
establishes no robotics performance or experiment-readiness claim. Hosted results
and the full research suite are separate from these local checks.
