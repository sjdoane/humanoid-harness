# After-reference crop: backtracking removed, candidate rejected

| progress | The actual LLM oracle revision survives 20 s and executes walk → crouch → rise → walk. |
|---|---|
| bottleneck | Heading and lateral screens fail; the original task still fails depth and lateral error. |
| next step | Test a separately versioned state-feedback oracle. Do not train or promote O7b from this result. |

## Matched result

- [Predata protocol](PROTOCOL.md): one after-only crop, zero residual, seed
  `20260906`, 1,000 steps. No training or reward change.
- Source `3cb3102d5cb2e3a8603663df13ed50b5d6cad0cb`; both fresh runs use
  the pinned 194-file executable tree `3ba7fe0ab72374e0667402848460c43915024fbcd2fe44f9ce5de791f0ab8b8e`.
- Fresh O7 control reproduces all three retained evidence files byte-for-byte.
- Candidate preserves all first 263 frame records, including rewards, 263
  actions and 264 qpos/qvel rows. First changed action is 263; state is 264.

| Measure | O7 control | O7b after-crop candidate |
|---|---:|---:|
| Duration / falls / switches | 20 s / 0 / 3 | 20 s / 0 / 3 |
| First-visit samples / compliant | 74 / 56 | 74 / 56 |
| Later region samples | 146 | 0 |
| All-visits posture compliance | 25.45% | 75.68% |
| Minimum inside root height | 0.515773 m | 0.515773 m |
| Maximum lateral error | 5.659565 m | **10.939883 m** |
| Final progress | 1.915612 m | 12.946624 m |
| Mean issued after yaw rate | 0.465845 rad/s | 0.062696 rad/s |
| Original task gates passed | 6 / 11 | 9 / 11 |

- Candidate overall speed MAE: `0.241548 m/s`; after-only: `0.214348 m/s`.
  Inside mean-speed target deviation: `0.032602 m/s`.
- Joint / roll-pitch p95: `0.239778 / 0.183717 rad`.
- **Both primary outcome screens fail:** after unwrapped heading maximum
  `1.396641 rad` exceeds `<1 rad`; lateral maximum exceeds `<2.5 m`.
- **Separate manipulation check fails:** mean issued yaw `0.062696 rad/s`
  exceeds `<=0.06 rad/s`. It was reduced, not removed.
- After-entry boundary heading is `0.092443 rad` at qpos row 263;
  maximum entry-relative excursion is `1.304198 rad`. Row 264 is post-action
  and is not the entry boundary. Crouch-exit heading is `0.857727 rad`.
- All region visits remain authoritative. The higher compliance fraction comes
  from eliminating upright revisits, not a deeper or better first crouch.
- **Reject adoption.** The robot veers sideways despite no longer backtracking.
  The crop changes multiple reference channels; this is not a yaw-only causal
  experiment. One failure does not rule out other crops or feedback laws.

## Receipts

Sibling runs: `humanoid-harness-probe-runs/`.

| Artifact | SHA-256 |
|---|---|
| Candidate config | `ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a` |
| Fresh control manifest | `3e2085e4e16b76b87d81303ee364cb8121c2cc0a76cfc7f466305003ef3c5da5` |
| Candidate manifest | `92e2a72f743bded8b2694760c7492f8d35f0829489415a05c3afff6ec459944a` |
| Candidate resource receipt | `ea00072569e159b8f04b1b7c801296b19297559b0ed91f04ec4077551cb429b8` |
| Independent score | `6933921fe8c7ddecefdbc654c3c0092681986afc51a97948976cd6469d3545d8` |
| Rebuilt feedback | `6ea27ad8e8faa2aa0cfb6a5397d9d06aa58b55bd7913704a8cbbcd2d9837c795` |
| Recorded-state GIF | `e133dcedfee7e9a897afe1a85d10418eba4c05a41e9f0494d9672f3c4e805a90` |

- Candidate run: `gmt_course_o7b_study015_candidate_20260907/`.
- Score: `artifacts/gmt/course_configs/study015_o7b_after_score_20260907.json`.
- GIF: sibling `gmt_course_o7b_recorded_20260907.gif`, 401 recorded frames.
  Rendering uses retained qpos, not new dynamics; this is not a trained policy.
- Native reservation/acceptance/acknowledgment chains reverified by the scorer.
  Both workers completed and released resources; source freeze ended.
- Fable independently confirmed the negative result in
  `20260907T064222.951303Z-e5fbbf3be13443c28b9a2e0b86635286`.
  Entry-boundary correction accepted in
  `20260907T064917.062827Z-f1e40f52b9af4c0fa791d615b41439f3`.
- Project totals remain **37 training runs / 2,293,760 transitions / zero
  full-task passes**. These two zero-training probes do not increase that count.
