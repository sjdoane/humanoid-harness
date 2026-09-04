# Private-source manifest

| status | provenance state |
|---|---|
| progress | The exact transcript and proposal snapshots are hash-pinned, and stable evidence anchors are defined without committing the raw private files. |
| bottleneck | The transcript is a one-line automatic transcription without timestamps or speaker labels; speaker attribution remains inferential. |
| next step | Fable must verify the hashes, use anchor IDs in strategy changes, and ask Samuel when speaker identity or wording changes a decision. |

Raw paths stay in ignored `CLAUDE.local.md`. This file contains provenance, not
permission to publish either source.

## Source records

| source id | format | bytes/pages | SHA-256 | handling |
|---|---|---|---|---|
| `SRC-LOKESH-TRANSCRIPT-01` | one-line UTF-8 auto-transcript | `42,294` bytes | `2b0c8af95e51e86db4785d3a67fb7d4fc0f4c6b2ef848814f09cfdbaaaa6220c` | private; no direct publication; no native diarization |
| `SRC-PROPOSAL-01` | PDF | `166,679` bytes / `2` pages | `d97df2c73e12d89c01850ab0dd5005887b85fb5a63664c340540327ae9b78013` | user-supplied research evidence |

## Transcript anchors

Offsets are zero-based byte offsets in `SRC-LOKESH-TRANSCRIPT-01`. Short anchor
fragments are lookup keys, not authoritative quotations.

| anchor id | byte offset | lookup fragment | topic | likely speaker | attribution confidence |
|---|---:|---|---|---|---|
| `LKS-A01` | `1219` | `weekly sink` | crisp progress, bottleneck, next-step updates | Lokesh | medium |
| `LKS-A02` | `19835` | `the only thing that your harness has to focus` | two scientific workstreams | Lokesh | medium |
| `LKS-A03` | `20990` | `you can intervene and steer` | human steering and agent questions | Lokesh | medium |
| `LKS-A04` | `23928` | `reference data set by itself` | supplied references are composed, not regenerated | Lokesh | medium |
| `LKS-A05` | `28379` | `parameterized MDP` | frozen policy-training interface and two inputs | Lokesh | medium |
| `LKS-A06` | `34828` | `the end goal for this project` | baseline-to-final improvement on several setups | Lokesh | medium |
| `LKS-A07` | `37262` | `a paper would be amazing` | paper/preprint is a desired outcome | Lokesh | medium |
| `LKS-A08` | `38152` | `good open source repo` | reusable open-source implementation is the primary impact path | Lokesh | medium |
| `LKS-A09` | `40646` | `points over Paris` | terse, structured research communication | Lokesh | medium |

## Proposal anchors

| anchor id | page | section | evidence role |
|---|---:|---|---|
| `PRP-A01` | 1 | Research objective | two authorable outputs and evaluator-guided iteration |
| `PRP-A02` | 1 | Figure 1 | inputs, outputs, frozen MDP, policy, evaluator, and feedback path |
| `PRP-A03` | 2 | Methods | matched trainer, seeds, protocol, and auditable `O_k`/`r_k` outputs |
| `PRP-A04` | 2 | Results and deliverables | multi-MDP comparison, staged tasks, open-source repo, and paper |

## Use rules

- Recompute the SHA-256 before relying on a raw file in a later session.
- If a hash differs, create a new source ID. Never update these bytes in place.
- Record source anchor, interpretation, confidence, and unresolved question
  separately.
- Treat the proposal as Samuel's written plan, not proof of a result.
- Treat every transcript speaker attribution as uncertain unless Samuel or
  Lokesh confirms it.
