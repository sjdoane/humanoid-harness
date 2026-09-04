# Farama Humanoid-v5 TQC expert registration

This directory registers provenance and integrity metadata only. It contains no
checkpoint, optimizer, replay, video, dataset, or derived actor payload.

| field | value |
|---|---|
| source | Hugging Face `farama-minari/Humanoid-v5-TQC-expert` |
| source commit | `e5a86ffdb70e6f4750f39c0464ac026a8437001a` |
| Farama script commit | `e74f9d0524c6df014c5a9985d0804001b9ce40dc` |
| source description | SB3 TQC, `20 x 10^6` steps, runs without falling |
| reported metric | `mean_reward 10370.61 +/- 1542.02`; 1,000 deterministic episodes; `verified: false` |

## Digest semantics

[`RECEIPT.json`](RECEIPT.json) preserves all 14 siblings from the captured
Hugging Face API response. LFS objects use the label `lfs.sha256` and retain the
API's `lfs` sub-object. Small Git objects use `git-blob-sha1`, sourced from the
API's `blobId`. These algorithms are not interchangeable. Local copies have a
separate locally computed SHA-256 and byte count.

The captured API record is bound at SHA-256
`69ccb043f76918fd601c71420ec6511a301193dd768aab5847a4beba3ec652f6`
and 3,670 bytes.

## Local inventory

The local, ignored subset contains the model card, `config.json`,
`results.json`, the source ZIP, and five unpacked members:
`_stable_baselines3_version`, `data`, `policy.pth`, `pytorch_variables.pth`, and
`system_info.txt`. `.gitattributes`, all optimizer files, and `replay.mp4` are
explicitly absent. The importer ignores `config.json`, opens only the unpacked
`policy.pth` and `data`, and never opens the source ZIP.

## Rights and claim boundary

The Hugging Face record specifies no license. Samuel's 2026-09-04 approval
authorizes the bounded local technical import; it is not a redistribution
grant. Local development use only. Payload bytes and the derived actor NPZ must
not enter Git.

The source description and metric are reported provenance, not locally verified
behavior. Registration alone establishes neither a trusted controller nor E1,
tracker admission, reference use, or Humanoid behavior.
