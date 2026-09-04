# ADR 0005: public expert base controller and receipted bootstrap

| | |
|---|---|
| progress | A hash-pinned public `Humanoid-v5` TQC expert exists as a Stable-Baselines3 artifact; the local one-attempt supervisor is parked; the E1-E5 tracker gates from ADR 0004 are retained unchanged. |
| bottleneck | The data-only import path has no implementation, no security review, and no human answer on loading external checkpoint bytes. |
| next step | Sol read-only scientific and adversarial reviews of this route, Samuel's gate answer, then one builder slice that imports the actor data-only, proves equivalence, and runs the 20-reset development gate. |

**Status:** accepted by the verified Fable strategy session on 2026-09-04; not yet reviewed by Sol; one human gate open
**Date:** 2026-09-04
**Supersedes:** the local 1M/20M TQC execution plan and the one-attempt parent supervisor in
[`../../experiments/bootstrap_tqc_humanoid/DEV_1M_BASE_CONTROLLER_V2.md`](../../experiments/bootstrap_tqc_humanoid/DEV_1M_BASE_CONTROLLER_V2.md).
ADR 0004's tracker-family order, E1-E5 gates, and frozen boundary after selection remain in force.

## Decision

| item | decision |
|---|---|
| Base controller source | Public artifact `farama-minari/Humanoid-v5-TQC-expert`, Hugging Face commit `e5a86ffdb70e6f4750f39c0464ac026a8437001a`; `humanoid-v5-TQC-expert.zip` SHA-256 `c0675e01b4efd26d9c773de3e9b4defb40301b5fbdac9f0f31517156fac59fe3` (7,377,061 bytes); `policy.pth` SHA-256 `1e64e56288155087089214548b6a634f332a41955a0d22629efd5ff2240e495c` (3,321,462 bytes). Farama's description: SB3 TQC, `20 x 10^6` steps, "runs without falling over"; model card `mean_reward 10370.61 +/- 1542.02` over `1,000` deterministic episodes. Run facts read as plain JSON and text from the artifact: SB3 `2.4.1`, PyTorch `2.5.1+cu124`, Gymnasium `1.0.0`, seed `0`, `5` environments, `20,000,000` total timesteps, `policy_kwargs {"use_sde": false}` (default `[256, 256]` actor), `top_quantiles_to_drop_per_net 2`. Local verified copy: `artifacts/external/farama-minari-humanoid-v5-tqc-expert/` (ignored). |
| Import path | Data-only: verify pinned SHA-256 bytes; read the zip with bounded, symlink-free extraction; load `policy.pth` tensors with `torch.load(weights_only=True)`; parse the `data` member as JSON only and never decode any `:serialized:` cloudpickle field; reconstruct the actor from tensor shapes; write the repository's strict actor NPZ; run the existing actor-equivalence verifier on the fixed `[4,348]` batch. `TQC.load`, `custom_objects`, and every other loader are forbidden. Critic and target-critic tensors are deserialized transiently inside an isolated, resource-limited loader subprocess because the state dict is one mapping; they are never retained, exported, or value-hashed. Optimizer and replay files are never opened. |
| Authority | The import yields an `external_pretrained_artifact` authority that no local-training, persistence, strict-reload, or worker-manifest API may accept. Equivalence and the development screen run through new external-authority entry points that reuse only committed pure capabilities: the NPZ format, metric code, trace contract, instrumented environment, and all-substep contacts. The parked v2 supervisor, worker, training, persistence, and strict-reload paths are not reused. |
| Slices | 03A: preserve the WIP from an isolated worktree, register the artifact, import securely, prove equivalence. 03B: external evaluator adapter, the 20-reset screen, and a content-addressed replay bundle. Visuals come later from that bundle, not from live capture. |
| Development gate | The existing v2 protected evaluation: seeds `96001`-`96020`, `1,000` steps, instrumented `Humanoid-v5` with unhealthy termination off, healthy/upright/forward gates, reward fields never read. Same-rollout visual capture stays optional. Its receipt is labeled `external_base_import`, an `interface_check`; it is not E1. Reward is compared only as a plain-versus-instrumented equivalence canary and never enters locomotion metrics. Replayability is claimed only with the bound local bundle: source bytes, strict NPZ, the 20 canonical traces, action arrays, all-substep contacts, seed and reset order, the no-retry rule, importer, evaluator, and metric source hashes, `uv.lock`, platform and thread settings, Gymnasium and MuJoCo identity, model bytes, the termination flag, and a same-host determinism scope. A receipt without the bundle claims no replay. |
| Remaining admission chain | `external_base_import` (packet 03) -> construct both E1 contenders (zero-residual and expanded-TQC initializers) on the imported actor -> locally certified Tier-D references and E2 replay -> E3 identifiability corpus -> E4 tracker-family screen -> freeze the candidate -> E5 causal-use gate. No tracker-admission claim before E5. |
| Parked work | The uncommitted v2 attempt supervisor, worker, channel, resource, training, persistence, attempt, and visual modules are preserved byte-exact on branch `wip/tqc-v2-attempt-supervisor` and removed from the `main` working tree only after that commit is verified. Nothing is deleted. |
| Fallback | If the import is refused by the human gate, a hash mismatch, an ABI mismatch, or the development gate, run plain receipted local TQC training through the existing calibration runner extended with final-model persistence and strict actor export. Each attempt has a distinct excluded seed, its own receipt, and a predeclared rule: the first attempt in seed order that passes the development gate becomes the base; every attempt is reported. The current local-attempt budget is zero; receipts audit attempts, they do not authorize them, so any future training packet needs an atomic repository-global authorization ledger consumed before environment construction. Attempts require Samuel's attempt budget, and the finite ordered seed list, maximum attempt count, failure and missing-run rule, checkpoint rule, and gate are frozen before the first attempt. Adding seeds after observed failures is post-hoc selection; the rule earns no seed-robustness claim and feeds no formal inference. |
| Reference pool direction | Candidate non-self references pending E3: same-embodiment rollouts of the public `medium` and `simple` TQC policies and their Minari datasets (`mujoco/humanoid/medium-v0`, `simple-v0`), plus retimed variants with fresh Tier-D identities and feasibility evidence. Every clip needs Tier-D admission. E3 forks must be individually E2-certified, share byte-identical full integration, wrapper, clock, and non-reference policy state, differ in the numeric future window and executed controls above locked tolerances, include at least one non-expert-generated branch, expose no branch identity outside the window, and report every attempted fork. E5 stays mandatory. |

## Why

| route | science gained before E4/E5 | compute to a base controller | implementation risk | artifact availability | time to a falsifiable causal-use result |
|---|---|---|---|---|---|
| Finish the v2 one-attempt supervisor | none | `27 min` at the E0 rate, `3 h` at the floor, per attempt | high: `8` open P1 lifecycle findings, `4,499` uncommitted insertions, `3,835`-line supervisor | local | weeks: repairs, three review rounds, one attempt, then E1-E5 |
| Plain receipted local training | none | same per attempt; `9-48 h` for Farama's 20M steps | low to moderate: extend the working calibration runner | local | days to a base, then E1-E5 |
| Public expert import (this ADR) | none by itself; it enables E1 after contender construction | base acquisition only: download `7.4 MB` and one 20-reset evaluation; E2-E5 compute is unchanged and unmeasured | low to moderate: bounded loader plus security review | public, hash-pinned, same model and observation/action ABI as the local runtime (`348`-D observation, `17`-D `[-0.4, 0.4]` action) | unmeasured planning estimate: days to E1/E2; later gates not scheduled |

Schedule and risk entries are unmeasured planning estimates. Base acquisition is separate from the remaining admission compute, which this ADR does not change.

The one-attempt rule protected a job that costs about half an hour of CPU. That protection cost more engineering than the job it guarded, and it produced no controller.

## Evidence ceiling

- Passing the development gate supports only: the exact imported actor remained healthy and upright on all `20` predetermined resets and met the forward thresholds in the pinned local runtime. Its evidence label is `external_base_import`, an interface check; it is not E1.
- It does not establish tracker admission, reference use, oracle quality, naturalness, robustness, or training-seed uncertainty.
- The base controller is not the tracker. The tracker family, E1-E5, and the frozen boundary in ADR 0004 are unchanged.
- The Minari expert dataset was generated by this same policy. A zero-residual tracker can follow its own rollout while ignoring the reference. E3 must therefore admit forked continuations whose desired futures need different actions, and the reference pool must include non-self gaits.

## Consequences

- E1 initialization identity can run on integrity-verified external actor bytes, after bounded-loader review and contender construction, instead of a synthetic fixture. Hashing establishes identity and integrity, not trust.
- The attempt supervisor's `TQC-P1-01` to `TQC-P1-08` findings stop blocking the route because that code leaves the path. They remain open on the parked branch.
- External lineage enters the manifest through two separate fingerprints: the source-training fingerprint (Farama's default `Humanoid-v5` under Gymnasium `1.0.0` with `terminate_when_unhealthy=true`, SB3 `2.4.1`) and the project-execution fingerprint (`terminate_when_unhealthy=false`). Compatibility means the same model and observation/action ABI, not the same MDP. Every local reference, tracker-development run, admission gate, and later Family-A arm must use the same `terminate_when_unhealthy=false` manifest. The imported actor bytes, not the training run, are the frozen component.
- The checkpoint carries no explicit license on Hugging Face. Only code, public URLs, hashes, byte counts, provenance facts, and payload-free receipts enter git. The checkpoint, the derived NPZ, Minari data, and any publication candidate stay local and ignored, enforced by an index policy test keyed on byte hashes and actor-state fingerprints. Samuel's import approval is a technical gate, not a redistribution grant.
- `README.md`, `experiments/README.md`, and the E1 note about uploaded checkpoints must be updated by the builder slice after Samuel's gate answer.

## Risks retained

- Pickle surface: the loader must be reviewed for any path that executes serialized code; `weights_only=True` on the pinned Torch version is required, not assumed.
- Version drift between the artifact's SB3 `2.4.1` and the local `2.9.0` may change tensor names; the importer uses an exact versioned key mapping and fails closed on missing, extra, or renamed keys. The artifact records `num_timesteps 19,965,000` against `_total_timesteps 20,000,000`, an unknown SB3-Contrib version, and an unverified model-card metric; receipts carry all of these unreconciled.
- The expert gait is fast running; tracking slower references may need residual authority beyond the `0.08` scale in the exploration design. That is a tracker-development finding, not a reason to edit the reward.
- Rights to redistribute the checkpoint or its rollouts are unresolved.

## Sources

- [Hugging Face model record](https://huggingface.co/farama-minari/Humanoid-v5-TQC-expert)
- [Farama upload script at pinned commit](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/upload_model.py)
- [Farama Humanoid expert description](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/descriptions/Humanoid-expert.md)
- [Minari `mujoco/humanoid/expert-v0`](https://minari.farama.org/datasets/mujoco/humanoid/expert-v0/)
- [ADR 0004](0004_stable_tracker_bootstrap.md)
