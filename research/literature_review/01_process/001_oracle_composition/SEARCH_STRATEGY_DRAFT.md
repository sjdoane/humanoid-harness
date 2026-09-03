# Search strategy — proposed for approval

**Status:** pilot discovery completed; formal search not started.
**Coverage:** 2002 through 2026-09-02. The 2001 composable-controller
foundation is a documented backward-chain candidate, not a reason to expand the
database date window.

## Formal discovery sources

- arXiv API: domain-specific preprints and version metadata.
- OpenAlex Works API: broad scholarly index plus backward/forward citation
  graph.

ACM Digital Library, IEEE Xplore, RSS proceedings, PMLR/CoRL, OpenReview, DOI
landing pages, author manuscripts, and official project/code pages are used
only to resolve and register exact primary sources returned by discovery or
citation chaining. They are not claimed as independently searched Boolean
databases. This avoids an unprovisioned IEEE API dependency and prohibited
automated use of IEEE's web interface. If either formal source is unavailable,
stop and obtain approval for a dated protocol amendment rather than silently
substituting another index.

The IEEE boundary follows its published
[bot policy](https://ieeexplore.ieee.org/Xplorehelp/overview-of-ieee-xplore/legal-information).

## Search lanes

| ID | Query concepts |
|---|---|
| S1 | reference motion + composition, sequencing, transition, or motion graph |
| S2 | motion phase, local phase, phase manifold, or time warping + humanoid/character control |
| S3 | oracle, scheduler, hybrid automaton, control graph, skill graph, policy gating, or motion matching |
| S4 | recovery, perturbation, resynchronization, or reacquisition + humanoid tracking |
| S5 | retargeting, dynamics feasibility, contact, cadence, or root frame + humanoid reference |
| S6 | Gymnasium/MuJoCo Humanoid + reference, imitation, tracking, phase, or recovery |
| S7 | skill chaining, policy switching, transition policy, or long-horizon composition |

`FORMAL_SEARCH_QUERIES_DRAFT.md` contains the literal query for every lane, the
source-specific translation, filters, pagination, and export receipt. No query
in that ledger has been run. Search results remain non-evidence until exact
primary-source bytes are registered.

## Deduplication

1. Exact DOI.
2. Exact arXiv identifier, retaining one cutoff-valid version.
3. Title and author comparison for conference/preprint duplicates.
4. Preserve metadata disagreements in notes; never silently merge authors.

## Proposed stopping rule

Run all seven lanes in both formal sources, deduplicate the union, register
exact sources, and apply the approved screening criteria. Use OpenAlex for both
directions of citation chaining from every `direct_fit` source. A citation
round is complete only when every seed's incoming and outgoing records has been
fully paginated, deduplicated, source-registered where available, and screened.

Stop after two consecutive complete citation rounds yield zero new
`direct_fit` primary sources. Preserve each zero-yield round. A database,
literal-query, filter, or rule change requires a dated human-approved amendment
before execution continues.

## Pilot boundary

`matrices/search_log.csv` records only the 2026-09-02 pilot discovery routes and
retained candidate counts. Those counts are not formal database result counts.
