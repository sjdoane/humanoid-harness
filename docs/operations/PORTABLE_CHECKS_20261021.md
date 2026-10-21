# Portable software checks · October 21, 2026

| progress | Added independent lint, portable test, migration-receipt and package-build feedback. |
|---|---|
| bottleneck | Existing full CI still fails its repository-wide formatting gate; full-suite and research-reproduction gaps remain. |
| next step | Review the additive workflow and verify its hosted Python 3.12/3.13 results without treating them as a full-CI or robotics result. |

## Why a separate workflow

At audited main `7c0303feffdcf22db868ce9c42fa57d19f9ab6d7`, the latest
[CI run](https://github.com/sjdoane/humanoid-harness/actions/runs/34527174416)
failed on both Python versions: Ruff reported 39 files requiring formatting
and 326 already formatted. Lint passed; tests, receipt verification and package
build were skipped. Local Ruff 0.16.5 reproduced the same formatting result.

The [September 10 record](SHARE_READINESS_20260910.md) already documents that
formatting backlog and the historical full-suite failures. The new
[Portable software checks](../../.github/workflows/portable-checks.yml) workflow
provides independent software feedback while those failures remain visible.
The existing `ci.yml`, runtime, tests, research source and evidence bytes are
unchanged. No failure is suppressed or reclassified as passing.

## Scope

- Same pinned actions, Python 3.12/3.13 matrix and locked optional dependencies
  as the existing CI workflow.
- Full repository lint, the [CONTRIBUTING portable selection](../../CONTRIBUTING.md#portable-checks)
  including migration-verifier regressions, repository migration verification,
  and source/wheel builds.
- Tests use committed or temporary fixtures and injected/mocked runtimes. They
  do not launch policy training or require downloaded controllers or the lab stack.
- No change to scientific admission, source identity, experiment seals, study
  protocols or licensing. Passing software checks do not establish simulator
  or robot performance, asset availability or complete reproducibility.

## Local verification

Checks run in a fresh public clone with the unchanged lockfile on macOS.
Hosted Linux results are separate and must be read from Actions after publication.

| Check | Observed result |
|---|---|
| Locked optional-dependency installation, Python 3.12.14 and 3.13.15 | Passed |
| Portable selection including migration regressions, Python 3.12.14 | 254 passed in 7.00 s |
| Same selection, Python 3.13.15 | 254 passed in 7.19 s |
| Repository migration verification, both Python versions | 120/120 files; 14 archived snapshots; zero errors |
| Repository Ruff lint | Passed |
| Source distribution and wheel build | Passed |
| Workflow YAML/structure, shell syntax and relative documentation links | Passed locally; hosted execution not yet verified |
| Original repository-wide formatting | Still fails: 39 files would be reformatted, 326 already formatted |

The full suite and simulator smoke were not run during this maintenance change.
The full suite contains host-, asset- and source-identity-dependent studies and
remains outside this portable signal. See the historical check record for its
last retained failures. No training, provider call or new research evaluation
was launched.
