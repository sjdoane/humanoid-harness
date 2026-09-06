# Hardening backlog

Findings from independent reviews that do not change a scientific conclusion.
They stay here until a slice touches the file, or until a release gate needs
them. Source reviews: `sol-review-sci-20260905-03b` and `sol-review-adv-20260905-03b` on commit `41b2597`.

| id | file area | weakness | status |
|---|---|---|---|
| R03B-02 | corpus contract, bundle, certifier readers | archive member count, expansion, and NPY header not bounded before allocation; some JSON readers read before bounding | backlog |
| R03B-03 | corpus runner | certifier child has no deadline, resource limits, environment allowlist, or bounded IPC; crash and timeout collapse into one outcome | backlog |
| R03B-04 | validation manifest, runner | recorded commands are prose in two places; no strict validator recomputes E3 from bundles; E3 timing absent (added in v2) | backlog |
| R03B-06 | E3 certifier | cross-branch runtime identity not enforced | backlog |
| R03B-08, SCI-04 | corpus tests | wrapper-counter bit flip, ledger replacement and order, six receipts plus three NPZs before reset, child replay corruption negatives | backlog |
| R03B-09 | payload policy | enforcement is the git index only; no release or export scanner | backlog (release gate) |
| SCI-07 | E3 receipt | wall time (present from run v2) | done in v2 |
| T2C2-A01 | phase B supervision (`_supervise_seed`, `supervise_training_job`) | worker spawned before the monitor `try`; cleanup not in a `finally`, so `KeyboardInterrupt`/`SystemExit` or a post-spawn exception orphans the worker; wrapper `finally` releases the heavy-job token even after `cleanup_worker` fails, permitting duplicate heavy work | open blocker before any smoke or training (Astra owns the repair; found by Astra's review of `f4309e62…`, verified by Fable at `f3edcff`) |

Folded into the next touching slice because they change a claim's wording or
reproducibility: SCI-03 (admission map; rename the boolean), SCI-05 (screen
evidence label `interface_check`), R03B-05 (process IDs out of scientific
digests), R03B-07 (E3 manifest describes a proposed downstream tracker input
as measured policy input), SCI-06 (corpus wording).

Added 2026-09-05 from the Experiment 003 robustness review (`sol-review-adv-20260905-e003`), deferred because they do not change a claim: E003-R05 (supervised child process with limits and structured failure statuses for evaluation), E003-R06 (descriptor-relative no-follow reads for every ancestor, bounded reads, transactional publication of index, report, and Markdown), E003-R10 (git object size checks before streaming, provenance allowlist for binary blobs, pinned build backend, wheel inventory in CI). Folded into `E003R1`: R01, R02, R03, R04, R07, R08, R09 and SCI-001 to SCI-007.
