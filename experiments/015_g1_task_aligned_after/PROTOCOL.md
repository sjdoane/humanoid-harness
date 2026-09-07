# Use a walking reference compatible with the straight course

| progress | O7 can execute walk → crouch → rise → walk, but the last clip commands a turn. |
|---|---|
| bottleneck | Backtracking adds 146 upright region samples; lateral drift reaches 5.66 m. |
| next step | One matched zero-residual after-crop probe. No training or learner change. |

## Decision

- Prioritize reference/task compatibility over more O2 reward sweeps, feature
  extractor changes, or phase-rate integration. Those remain separate studies.
- Retained O7's first spatial visit: 56/74 compliant, mean speed 0.6826 m/s,
  but lateral already about 0.964 m. These are diagnostic, not new task gates.
- Its issued `after` references have mean local yaw rate 0.4658 rad/s over
  737 intervals. This is a command summary, not executed orientation or proof
  that a particular channel causes the drift.
- Fable's actual proposal
  `20260907T054830.745021Z-6afaef0122314104b2106ef62d515c87` selects O7b:
  add `walk_after = basic_walk[30.62,31.56)` at native cadence with wrapping,
  map only the `after` state to it, and assign a new oracle ID.
- Preserve all existing segments, guards, dwell, reset, reward, actor, task,
  observations/actions, training budget and evaluator. The crop is kinematic;
  lower command yaw does not imply dynamically safe transitions.

## Exact pair

- Retained O7 parent config:
  `430c6674485a4ec9ca7546389a71b485c7670f11d08a8b911b48758ee6a51ff4`.
- Parent manifest/resource:
  `6141e860a018fcb0dfac89444234fe987351f188f29e14d3400d0a78db9fe34c` /
  `6fa316fc8b15eccf29e42920ac48914cf1baf67f885af0e93e2b1916d06db466`.
- Candidate config:
  `ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a`;
  revision receipt `c8350d35bd6b09984c94ab7eea932e89bf154f8875343a53363a01679d2de804`.
- Proposal raw body plus newline SHA:
  `0118cff0c816c65884d30ff06993abfc05a81b83d71400ef1805de4a28dcb6c5`.
  `g1 revise` rebuilt parent feedback before admitting it.
- Source has changed since O7: **first reproduce its three outputs byte-for-byte**
  at the new committed source, then run the candidate at that same source.
  No metadata exception is allowed; any mismatch stops the candidate.
- Both zero residual, seed `20260906`, four-state/2,172-observation probe profile,
  1,000 steps, no training. Each needs its own exact reservation, one heavy job.
- Candidate must preserve the first 263 actions and 264 state rows. Changed
  after behavior first executes at command 263; its resulting state may differ.
- Rebuild independent feedback/objective from retained raw states. Verify
  all input/output hashes, resource/source binding and zero residuals.

## Pre-data screen

| Requirement | Criterion |
|---|---|
| Survival | 20 s without falling |
| Composition | Three switches; rise and after actually executed |
| Completion | Finish reached at 3.5 m |
| No upright backtracking | No physical `[1,2)` region visit after after-mode entry |
| No large turn | Maximum absolute unwrapped heading after entry <1 rad relative to reset heading |
| Reduced drift | Maximum lateral error <2.5 m |
| Tracking | Joint p95 <=0.35 rad; roll-pitch p95 <=0.25 rad |

- Unwrap the full executed heading series before selecting after rows. A
  wrapped final heading near zero cannot hide a full turn. Also report maximum
  excursion relative to after-entry heading; it does not replace the fixed gate.
- Report all eleven original task gates unchanged, first-crossing and later
  exposure separately, after speed error, commanded yaw and progress.
  Also report executed heading at crouch exit and after entry.
- **All visits remain authoritative.** First-crossing metrics never erase
  later failures. Unchanged early lateral/depth errors mean this probe is not
  expected to pass the complete task even if the narrower screen succeeds.
- Any failed screen rejects adoption here. An early fall implicates this exact
  handover; it does not prove the crop generally unusable. A failed yaw/drift
  prediction does not establish channel-level causal ineffectiveness.
- No automatic training, crop search, yaw injection, phase scheduler, feature
  extractor change or evaluator relaxation follows. Decide the next study from
  this result and retain both traces.
