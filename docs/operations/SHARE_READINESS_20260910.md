# Research-sharing check · September 10, 2026

| progress | Current research docs and status surfaces refreshed; main and Astra histories combined without changing experiment outcomes. |
|---|---|
| bottleneck | Research snapshot, not a reproducible release: full-suite/formatting failures, local-only assets and no project license remain. |
| next step | Review the share branch, approve publication, and separately resolve task/interface and reproduction dependencies. |

## Scope and provenance

- Integration base: `a83dd90`, with parents `2e5e868` (main) and `6536c4f`
  (Astra). An independent review verified the merged source, tests, scripts and
  experiments matched Astra exactly; main's additional strategy history remains.
- Work was isolated on `share/lokesh-readiness-20260910`. The main and active
  research checkouts, private files, local workers and experiment seals were not
  modified. A new share commit does not inherit old execution authorization.
- Changes: concise README and navigation, current task rationale/metrics,
  explicit history labels, CLI/UI status correction, and archived landing-page
  snapshots for immutable migration verification.
- No new research training, policy evaluation or provider call was made.
  Software tests and a read-only UI check are not new robotics evidence.

## Verification

Local macOS, Python 3.13.15, fresh worktree environment installed from the
unchanged lockfile. Python 3.12 and hosted CI were not run in this check.

| Check | Observed result |
|---|---|
| Core-only `uv sync --locked` | Passed; 7 packages installed |
| README doctor/status/build/query/G1-help commands | Passed; index built from 47 paper records, 46 active |
| Portable selection in CONTRIBUTING, with optional extras | 221 passed in 12.30 s |
| Final portable selection plus migration regressions | 254 passed in 7.47 s |
| Migration verification | 120/120 original files verified; 14 exact archived snapshots; no errors |
| Repository Ruff lint | Passed |
| Changed status code/tests formatting | Passed |
| Final status and UI regression selection | 41 passed in 5.48 s |
| Repository-wide formatting | 39 files would be reformatted; 326 already formatted; pre-existing research source left unchanged |
| Source distribution and wheel build | Passed |
| Read-only UI | Loads without a G1 registry; missing local runs shown explicitly; current status is separate from native receipt evidence |

- A first core+dev-only test attempt had three collection errors; a reduced
  attempt had five import failures. Gymnasium/Torch are imported by some tests
  and the UI. After installing the documented optional extras, all 221 selected
  tests passed. No tests or import gates were disabled to obtain that result.
- The broader suite was not rerun. Its latest retained result is **2,723 passed,
  89 failed, 40 skipped, 2 deselected** on September 8; see
  [the exact record](EFFORT_REPORTING.md). Host-bound native artifacts, import
  origin and historical source lineage remain unresolved. This is not a green-CI claim.
- Documentation check: 47 relative links across 12 current entry-point documents
  resolve. An independent writing/evidence review accepted the scope and claims;
  its three small clarity corrections were incorporated.
- Independent migration review reran all 33 focused tests and byte-compared the
  two new snapshots against their original commit. Missing/tampered archives and
  unauthorized redirects fail; the earlier schema's rules remain unchanged.
- A separately tasked Sol reviewer found no claim-invalidating content defect
  in the final public docs or status surfaces. Its remaining conditions were a
  clean committed snapshot and Samuel's publication approval.
- Archived `PUBLIC_CONTEXT_BOUNDARY.md` preserves its original Markdown
  hard-break spaces. Do not format those immutable bytes to remove a diff warning.

## Availability and sharing

- Git contains study reports and source records, not the large local videos,
  motion/controller assets or complete run receipts. There is no verified
  clean-clone G1 training recipe yet.
- Tracked-file review found no high-confidence credential/private-key pattern
  or raw private meeting transcript. This is a hygiene check, not a security audit.
- Historical host paths remain as provenance; the README uses portable commands.
- No project license was selected or granted during cleanup. Third-party
  licenses remain unchanged.
- Publication requires Samuel's approval. Preparing or committing this branch
  does not mean GitHub or the default branch has been updated.
