# Legacy Gym studies: retained negative evidence

## Verdict

Seven retained RL-Sculptor Gym runs verified bundle, command, checkpoint, and
evaluator plumbing. They did **not** test whether an oracle improves training.

```text
legacy path
reference summary -> phase-window selector -> appended observation -> new PPO
generated task reward ---------------------------------------------> new PPO

required path
controller-native reference -> frozen tracker + tracking reward -> adapted policy
oracle chooses reference ---------------------------------------> adapted policy
```

## Observations

| Scenario | Retained outcome | Valid interpretation |
|---|---|---|
| Humanoid forward | Both policies fell; best mean velocity `0.133 m/s` | No valid improvement |
| Humanoid stand-up | Completion stayed `0`; one mode-dispatch change | No recovery behavior |
| Walker2d | Both policies fell; completion could occur while moving backward | Completion semantics invalid |

- All seven objective fitness values were zero.
- Each iteration used 8,192 environment steps.
- The policy was a fresh SB3 MLP, not a fixed/pretrained tracker.
- The 8-D root/clock summary was not a retargeted Humanoid joint reference.
- Generated rewards ignored the appended command.

## Regression requirement

The new harness must reject an `oracle_training` label unless a frozen tracker
and tracking reward consume the exact reference, train/evaluation manifests
match, and exact/zero/shuffle/time-shift causal ablations pass.

## Provenance

Derived from the untracked old-worktree file
`RewardSculptor/docs/ORACLE_FAILURE_AUDIT.md` on 2026-09-02. The migration
manifest records its byte hash separately from old base commit
`c1779779f20f418a5ddf78576fdb689895f1e108`.

