# A2R: live proposal and feedback-linked revision

| status | result |
|---|---|
| progress | Two real Sol/max responses passed A1 format and lineage admission. The second packet retained the first proposal and predeclared synthetic feedback. |
| bottleneck | Neither candidate was executed or semantically validated. No policy, robot, or protected evaluator ran. |
| next step | Repair the transferred B0 validation boundary before connecting proposals to execution. No more A2/A2R candidate calls. |

## What this demonstrates

- Subscription-authenticated model call → JSON proposal → immutable receipt.
- Prior proposal + exact feedback dossier → second call → linked revision.
- Missing real measurements and synthetic provenance survive both iterations.
- This is live software plumbing, not evidence of a useful reward or an
  improvement driven by measured failures. The `0.75` feedback was fabricated
  before either call as a test fixture, not measured from either candidate.
- A1 code remained at `76a0fe6e86f5c4e5b52231d3b69df3d0c9b317d2`.
- Parent reproduction: `59 passed in 2.39s` across reward-search and mailbox tests.

## All attempts retained

| attempt | start → finish (UTC) | terminal | admission |
|---|---|---|---|
| A2 initial | Sep 4 22:54:03 → 23:25:32 | Interrupted after transport timeout | Rejected: missing response; no revision |
| A2R initial | Sep 5 00:47:14 → 00:47:44 | Succeeded, 30 seconds | Format/lineage accepted |
| A2R revision | Sep 5 02:20:08 → 02:20:59 | Succeeded, 51 seconds | Format/lineage accepted |

- The original failure remains in `A2_RESULT.md`; host evidence is in
  `A2_HOST_DIAGNOSIS.md`. Successful awake-host calls do not prove sleep was
  the sole cause of the first failure.
- Each A2R watcher observed terminal completion without sending a signal.
  Its temporary idle-sleep assertion ended; no global power setting changed.
- Requested configuration: `gpt-5.6-sol`, `max`, read-only, candidate role.
  Receipts describe requested settings, not independent served-model attestation.
- Both event streams contain one final agent message and two configuration
  warnings (experimental feature and shortened skill descriptions), with no
  recorded tool-execution events. No warning was hidden or configuration changed.

## Exact retained evidence

Paths below are local to this checkout and excluded from Git.

| artifact | location or SHA-256 |
|---|---|
| Original input directory | `.orchestration/a2-canary/20260904T2250/` |
| A2R records directory | `.orchestration/a2-canary/awake-20260905/` |
| Initial run | `.orchestration/sol-runs/20260905T004714Z-3da777e6-e299-498e-91bb-c0cbdae07ed2` |
| Revision run | `.orchestration/sol-runs/20260905T022008Z-8e85a1cf-da38-4ec0-a823-ab70c8ba8a5d` |
| Initial packet, 5,925 bytes | `b07d7a87a916e39aad7c3a6172bfa9fd8b1bf5944626c30e7665e5fec85f0f71` |
| Revision packet, 6,637 bytes | `3098f591b76c57b0b605326883ae075e7d4930e90db2cdc26116926715766136` |
| Synthetic feedback file | `050b1581c8a95203eeb7879d19baa225ef53f25bf15632d1e6f0b2437553700e` |
| Initial candidate source | `7f8bdcfa9e0a373f7f3297e29b8b49037715c5d28a860a0803c8aed645a94875` |
| Revised candidate source | `4e7fa742d600162e7a4de12103dd84f4958b8a8271412171bc9178eb6f83a942` |
| Initial receipt | `118058736f26bda47e0de73608318c56c14bd5cbc2c293f568017255666ed870` |
| Initial iteration | `899fe1ef68e02418e6b7dbe08a4ff45e92b01c1548543ca231367a60271698d5` |
| Revision receipt | `bada6786074c5bab3845841d8e61b7baf30866be147931f6828191c4c7040743` |
| Revision iteration | `61861c6e455161312f17c7a1445523feb8e5648581d7210fa8f8889402519870` |

Receipt and iteration filenames are their kind plus the digest above. Proposal
filenames include the exact run ID and proposal digest. Each receipt binds its
request, retained packet, result, and final response bytes.

## Candidate meaning and limits

- Initial: symmetric target-speed error, bounded by a rational transform.
- Revision: changes the shaping curve to an arctangent transform; adds a math
  import. The synthetic interface does not establish whether B0 admits it.
- Both give explicit predictions and rejection conditions; neither supplies
  behavioral evidence. Both remain `awaiting_B0_validation`.
- The new curve is an untested hypothesis, not a demonstrated response to a
  diagnosed robot failure. Synthetic feedback cannot justify preferring it.
- No generated source was executed, training authorized, or result promoted.
