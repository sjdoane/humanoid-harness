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

Folded into the next touching slice because they change a claim's wording or
reproducibility: SCI-03 (admission map; rename the boolean), SCI-05 (screen
evidence label `interface_check`), R03B-05 (process IDs out of scientific
digests), R03B-07 (E3 manifest describes a proposed downstream tracker input
as measured policy input), SCI-06 (corpus wording).
