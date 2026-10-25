# Study 012 source identities during CI maintenance

The current converter declares `study012-converter-source/v2`. This source epoch
updates the expected `course_task.py` digest after Ruff 0.16.5 formatting. Its
14 import paths and module names are unchanged. The converter still hashes the
actual imported files and its own producer bytes, checks exact source-root
containment, and requires a caller-supplied full reviewed commit equal to clean
HEAD, including staged, unstaged and untracked files. A version label alone is
not evidence of review: a current execution needs a separate completed source
review and records its actual producer, source-set digest and reviewed commit.

The scientific candidate ID and output schema version are unchanged; source
identity gains the additive `binding_version` provenance field. No
donor/model/mesh binding, retained receipt, reference or seal changed. Formatting preserves the Python AST; it does
change byte-level source identity. The new `source.binding_version` field is
additional provenance, not admission authority. The existing derived-reference
consumer still pins its prior manifest bytes and reviewed commit. It must not
admit a new output merely because a converter retains the same candidate ID.
This maintenance change generates no conversion output or research result.

## Historical bytes remain historical

The [public preservation snapshot](https://github.com/sjdoane/humanoid-harness/tree/c9ee2735829af75559fc066da6a0bd502e372667)
retains the exact old producer and declared source dependencies. The adjacent
[historical source index](study012-historical-source-index.json) records:

- The unchanged geometry audit receipt, its producer and all eight recorded
  project imports, whose canonical source-set hash matches the receipt.
- The old converter and all 14 of its bound project modules.
- The snapshot's dependency lockfile, without claiming it reconstructs the
  original execution environment.

The geometry receipt does not state its producer's original Git commit. The
preservation snapshot is a verified match for its recorded bytes, **not a claim
that the audit ran at that commit**. Input-run commits identify input executions;
they do not establish the analyzer's original source revision. The eight listed
audit imports are the complete declared set, not proof of every transitive or
external dependency used during the historical run.

For historical source inspection, resolve paths against the exact preservation
revision, for example:

```sh
git show c9ee2735829af75559fc066da6a0bd502e372667:scripts/convert_study012_reference_candidate.py
```

If that revision is unavailable locally, obtain the exact public revision before
inspection; do not substitute current modules. Historical execution, if separately
authorized, must use its own admitted source and dependencies. Old receipts and
seals do not certify current code. Current code never reports the archived
producer digest as its own identity or accepts an old/new digest union.
