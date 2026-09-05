# Farama Humanoid-v5 TQC simple registration

This directory contains provenance and integrity metadata only. It contains no
checkpoint, optimizer, replay, video, dataset, or derived actor payload.

| field | value |
|---|---|
| source | Hugging Face `farama-minari/Humanoid-v5-TQC-simple` |
| source commit | `39e2954c193fc1352535935a7d71eca9e5745d9b` |
| Farama script commit | `e74f9d0524c6df014c5a9985d0804001b9ce40dc` |
| source progress | `num_timesteps 1,965,000` of `2,000,000`; SB3 `2.4.1`; seed `0`; `policy_kwargs {"use_sde": false}` |
| reported metric | `mean_reward 5543.17 +/- 825.31`; 1,000 deterministic episodes; `verified: false` |

## Digest semantics

[`RECEIPT.json`](RECEIPT.json) preserves all 14 siblings from the captured
Hugging Face API response. LFS objects retain the label `lfs.sha256`; small Git
objects use `git-blob-sha1` from `blobId`. Local SHA-256 values and byte counts
are recorded separately. Missing optimizer, replay, and `.gitattributes` files
are explicit absences.

## Rights and claim boundary

The API record specifies no license. Samuel's 2026-09-04 approval authorizes
the bounded local technical import, not redistribution. Payload bytes and the
derived strict NPZ must not enter Git. Source metrics are reported provenance,
not locally verified behavior; this registration grants no E1, tracker,
reference-use, or Humanoid-behavior claim.
