# GMT closed-loop reference ablation

| state | evidence |
|---|---|
| progress | A fixed five-arm workload, immutable evidence format, and strict data-only validator are implemented. |
| bottleneck | No authorized simulator run has produced behavioral evidence yet. |
| next step | Run once from a clean reviewed commit, validate the retained artifact, then interpret only the tested interventions. |

## Fixed workload

| arm | actor input | objective target |
|---|---|---|
| exact | original `20 x 30` window | original post-step reference |
| zero | literal positive float32 zero | original post-step reference |
| current-frame repeated | current original frame in all 20 slots | original post-step reference |
| shuffled | one fixed PCG64 row permutation | original post-step reference |
| shifted | active segment at `phase + 5.0 s` | original post-step reference |

- Fresh plant, actor session, and oracle per arm.
- Literal zero residual in every arm.
- Stop at the fixed task horizon or first fall.
- The exact arm uses the ordinary course rollout writer; its frame, trajectory,
  and evaluation bytes can be compared directly with a same-config zero-residual run.
- Preserve the original command, actual 2,154-value actor input, base action,
  executed action, trajectory, and both target-error definitions.
- The actor-window first-row error is supplemental. A shuffled first row has no
  privileged objective meaning.

## Run

```bash
./scripts/run_gmt_development.py request \
  --workload reference-ablation \
  --repository-root /absolute/path/humanoid-harness \
  --python /absolute/path/.venv/bin/python \
  --config /absolute/path/probe-config.json \
  --output /fresh/path/reference-ablation \
  --owner astra-reference-ablation
```

- Use the returned exact binding in the existing reviewed reservation flow.
- Run the same command with subcommand `run`, plus the accepted
  `--coordination-root` and `--reservation` paths.
- The config must be `mode=probe`, `training_steps=0`, and carry no trainer.
- The launcher permits no arbitrary child command or caller-selected treatment.

## Claim ceiling

- Practical divergence is predeclared as any common-boundary difference of at
  least 1 mm root translation, 1 mrad joint angle, or 1 mrad sign-invariant root
  orientation. Report the first 50 steps and full common observed horizon.
- A difference supports closed-loop dependence on the actor reference input for
  this config and these interventions. It does not prove good tracking,
  composition quality, task success, learning, or generalization.
- No difference means no detected effect under these interventions. It does not
  prove that the controller never uses references.
- Measurements after the shorter arm falls are unavailable, not zero.
- Ordinary course feedback, study scoring, and run registries reject this
  distinct evidence class.
