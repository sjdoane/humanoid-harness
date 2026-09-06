# F3 initial call result (Fable reward lane, 2026-09-06T15:35Z)

| status | current truth |
|---|---|
| progress | The single F3 hypothesis call ran under the reviewed one-call protocol after `APPROVE_DISPATCH` (readback run `20260906T144450Z-48272687`): one read-only Sol/max candidate worker, `28 s` wall, terminal `SUCCEEDED`; ingestion accepted the response. |
| bottleneck | The hypothesis is not admitted: candidate admission (registry resolution, final-ready re-seal at a verified clean HEAD) has not run; no baseline measurement exists; no training. |
| next step | Candidate admission slice `T2C1`, then the cycle-0 reservation and Samuel's authorization. |

## Identities

| item | value |
|---|---|
| dispatch base commit | `d97f8da8a72311e627ba7e51601fb93cb022ae10` (clean tree; equals the readback target d97f8da) |
| baseline | `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f` (1,773 bytes) |
| T2 seal | `6ce006bb983611179ab9fc8bc47b7b01c74d6303a3f7f0a917bd35947d9d5a9d` |
| canonical record | `b76c73cd1ef027b54760a324a17de61cfd300b53f334c2d8bb6a8c7fde1b5220` (28461 bytes) |
| prompt | `4d1a29765d929472e2f2a346294f98ac08c0e35d3e18c147e761e46678fe2bbc` (6011 bytes) |
| call ID | `e7f3ca013343a5f82ce6161e1667a52f00f7f4b019c329ecf6fe4820bc60312a` |
| identity digest (launcher scope suffix) | `b53841bc8e0dc60fbd3b5710c89e3a903c4acc82a25cc655a164f0671e40b375` |
| dispatch intent | `ad16687b76404517e3680f8f17de9ccea9fdc8dcad14f6557a8b0db44bdb105e` |
| candidate run | `20260906T150709Z-f0ac77c5-7773-48ac-a32e-80499f5ca161`; requested `gpt-5.6-sol` at `max`, runner `screen`, `codex-cli 0.153.2`; finished `2026-09-06T15:07:37Z`; status `SUCCEEDED` (expected configuration, not served-model attestation) |
| watchdog | armed for `1,200 s`; the worker was terminal `28 s` after request creation; deadline check found no live process |

## Hypothesis (retained bytes; hypothesis only, not admitted)

| field | value |
|---|---|
| parameters | alpha `1.0`, beta `0.0` |
| rationale | This conservative interior candidate gives the speed-target term unit-scale influence while avoiding a constant return offset. It is an initial unmeasured hypothesis, not the tracking-only baseline. The baseline load and binding are unobserved, and all candidate admission, adapter, manifest, protected-evaluator, measurement, execution, and training evidence remains missing. |
| predicted effect | If subsequently admitted and evaluated under the frozen comparison, the candidate may increase pressure to hold the target COM forward speed without adding a constant offset. No improvement or successful execution is claimed. |
| falsifier | The hypothesis is falsified if a predetermined independent evaluation shows no improvement in target-speed error, or shows a correctness or safety regression, relative to the declared tracking-only configuration. No such comparison is valid unless the stock-COM adapter first verifies the inclusive [-25,25] m/s gate, COM origin, and 0.015 s cadence, and all required evidence gates are satisfied. |
| status | `hypothesis_only_not_admitted` |
| echoed digests | request `40e3306985e28296…`, baseline `eea2b6a9893e6e4c…`, contract `b5f9cd440b269090…`, dossier `32cdcc548cefedca…` |

Trusted recipe constructed by the parent, not the model: `{"alpha":1.0,"beta":0.0,"formula_id":"target_speed_triangular_affine/v1"}` (`078fabc83ff3d572789b268c05ccff5a08050e00d0fd104399e2294d3b91ec28`, 73 bytes).

## Ingestion outcome

| artifact | SHA-256 | bytes |
|---|---|---|
| proposal | `060762c7873adb1c4cc3c6d883554cd9272668dd75755b3a45516e47f9ebe07b` | 1606 |
| recipe | `078fabc83ff3d572789b268c05ccff5a08050e00d0fd104399e2294d3b91ec28` | 73 |
| model-call receipt | `1a8d0e78de9436d1499b023455c69ea9c94dd9143704d9c88ba34d978e54a40c` | 4651 |
| retained request.json | `961d048800eac0861b89e97ab7df4c2dc400c7277b12e08c8bc8764be8635adf` | 659 |
| retained task-packet.md | `4d1a29765d929472e2f2a346294f98ac08c0e35d3e18c147e761e46678fe2bbc` | 6,011 |
| retained result.json | `ad3e50d882f02802eac6b30789064a45e6b3ba12f0e59a957dde9313b5e0d788` | 608 |
| retained final.txt | `062b1b713c498c542336971073783688c6d2eef6ab9dc27aec2d4a4979fa6476` | 1,605 |

`accepted: true`, `rejection_reason: null`. The call root `.orchestration/f3-initial-call-20260906/` is ignored and local; these hashes are its committed ledger.

## Claim ceiling

One retained model response with in-bounds parameters. It establishes the
harness's proposal plumbing end to end on a real model call. It establishes no
candidate admission, no runtime load, no training, no evaluation, no reward
effect, and no humanoid behavior. Repository state cannot prove the absence of
an unrecorded external call. The protocol authorized exactly one call; zero
attempts remain.
