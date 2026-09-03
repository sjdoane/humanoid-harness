# Formal search-query ledger — proposed for approval

**State:** draft only. None of these queries has been executed as a formal
search. The 2026-09-02 pilot registry is not a result set from this ledger.

The syntax and paging rules follow the official
[arXiv API manual](https://info.arxiv.org/help/api/user-manual.html),
[OpenAlex search documentation](https://help.openalex.org/api/searching/), and
[OpenAlex citation recipes](https://help.openalex.org/how-to/api-recipes/).

## Literal Boolean queries

The following strings are the exact `search` value for OpenAlex. Quotation
marks, capitalization, and parentheses are part of the query.

### S1 — reference composition and transition

```text
("reference motion" OR "motion reference" OR "reference trajectory" OR "reference state" OR "reference playback" OR "reference-conditioned" OR "reference conditioned") AND (composition OR composable OR sequencing OR transition OR "motion graph" OR "reference selection" OR "adaptive reference" OR "state-conditioned reference" OR "state conditioned reference") AND (humanoid OR biped OR bipedal OR "bipedal robot" OR "physics-based character" OR "physics based character" OR "articulated character" OR "virtual character")
```

### S2 — phase and resynchronization

```text
("motion phase" OR "local phase" OR "latent phase" OR "phase manifold" OR "phase function" OR "phase-functioned" OR "phase variable" OR "phase estimator" OR "phase estimation" OR "phase-aware" OR "phase aware" OR "phase-conditioned" OR "phase conditioned" OR "time warping" OR resynchronization OR resynchronisation OR reacquisition) AND (control OR tracking) AND (humanoid OR biped OR bipedal OR "bipedal robot" OR "physics-based character" OR "physics based character" OR "articulated character" OR "virtual character")
```

### S3 — scheduling and switching

```text
(oracle OR scheduler OR "mode scheduler" OR "skill scheduler" OR "reference generator" OR "finite state machine" OR "hybrid automaton" OR "hybrid automata" OR "control graph" OR "control graphs" OR "skill graph" OR "skill graphs" OR gating OR "policy gating" OR "policy switching" OR "motion matching") AND (control OR controller OR policy OR reference) AND (humanoid OR biped OR bipedal OR "bipedal robot" OR "physics-based character" OR "physics based character" OR "articulated character" OR "virtual character")
```

### S4 — perturbation and recovery

```text
(recovery OR recover OR "fall recovery" OR "push recovery" OR perturbation OR perturbations OR disturbance OR disturbances OR push OR pushes OR fall OR falls OR "tracking deviation" OR "phase reset" OR resynchronization OR resynchronisation OR reacquisition) AND ("reference tracking" OR "motion tracking" OR "motion imitation" OR imitation) AND (humanoid OR biped OR bipedal OR "bipedal robot")
```

### S5 — reference feasibility and admission

```text
(retargeting OR "motion retargeting" OR kinodynamic OR feasibility OR "dynamic feasibility" OR "dynamics feasibility" OR "dynamically feasible" OR "dynamics-aware" OR "dynamics aware" OR "physics-aware" OR "physics aware" OR "controller-aware" OR "controller aware" OR "control-aware" OR "control aware" OR contact OR "contact-aware" OR "contact aware" OR cadence OR retiming OR "time scaling" OR "root frame" OR "root-frame" OR "root motion" OR "contact schedule" OR "contact sequence") AND (reference OR motion) AND (humanoid OR biped OR bipedal OR "bipedal robot" OR "physics-based character" OR "physics based character" OR "articulated character" OR "virtual character")
```

### S6 — target environment

```text
(Gymnasium OR MuJoCo OR "Humanoid-v5" OR "MuJoCo Humanoid") AND (reference OR imitation OR tracking OR phase OR recovery OR transition)
```

### S7 — long-horizon skill composition

```text
("skill chaining" OR "skill composition" OR "policy switching" OR "mode switching" OR "transition policy" OR "skill sequencing" OR "long horizon" OR "long-horizon" OR "multi skill" OR "multi-skill" OR multimodal OR "multi-modal" OR multimode OR "multi-mode") AND (control OR controller OR policy) AND (humanoid OR biped OR bipedal OR "bipedal robot" OR "physics-based character" OR "physics based character" OR "articulated character" OR "virtual character")
```

## arXiv translations

Use the following exact `search_query` values. The client percent-encodes them
without changing their decoded text.

```text
S1=((all:"reference motion" OR all:"motion reference" OR all:"reference trajectory" OR all:"reference state" OR all:"reference playback" OR all:"reference-conditioned" OR all:"reference conditioned") AND (all:composition OR all:composable OR all:sequencing OR all:transition OR all:"motion graph" OR all:"reference selection" OR all:"adaptive reference" OR all:"state-conditioned reference" OR all:"state conditioned reference") AND (all:humanoid OR all:biped OR all:bipedal OR all:"bipedal robot" OR all:"physics-based character" OR all:"physics based character" OR all:"articulated character" OR all:"virtual character")) AND submittedDate:[200201010000 TO 202609022359]
S2=((all:"motion phase" OR all:"local phase" OR all:"latent phase" OR all:"phase manifold" OR all:"phase function" OR all:"phase-functioned" OR all:"phase variable" OR all:"phase estimator" OR all:"phase estimation" OR all:"phase-aware" OR all:"phase aware" OR all:"phase-conditioned" OR all:"phase conditioned" OR all:"time warping" OR all:resynchronization OR all:resynchronisation OR all:reacquisition) AND (all:control OR all:tracking) AND (all:humanoid OR all:biped OR all:bipedal OR all:"bipedal robot" OR all:"physics-based character" OR all:"physics based character" OR all:"articulated character" OR all:"virtual character")) AND submittedDate:[200201010000 TO 202609022359]
S3=((all:oracle OR all:scheduler OR all:"mode scheduler" OR all:"skill scheduler" OR all:"reference generator" OR all:"finite state machine" OR all:"hybrid automaton" OR all:"hybrid automata" OR all:"control graph" OR all:"control graphs" OR all:"skill graph" OR all:"skill graphs" OR all:gating OR all:"policy gating" OR all:"policy switching" OR all:"motion matching") AND (all:control OR all:controller OR all:policy OR all:reference) AND (all:humanoid OR all:biped OR all:bipedal OR all:"bipedal robot" OR all:"physics-based character" OR all:"physics based character" OR all:"articulated character" OR all:"virtual character")) AND submittedDate:[200201010000 TO 202609022359]
S4=((all:recovery OR all:recover OR all:"fall recovery" OR all:"push recovery" OR all:perturbation OR all:perturbations OR all:disturbance OR all:disturbances OR all:push OR all:pushes OR all:fall OR all:falls OR all:"tracking deviation" OR all:"phase reset" OR all:resynchronization OR all:resynchronisation OR all:reacquisition) AND (all:"reference tracking" OR all:"motion tracking" OR all:"motion imitation" OR all:imitation) AND (all:humanoid OR all:biped OR all:bipedal OR all:"bipedal robot")) AND submittedDate:[200201010000 TO 202609022359]
S5=((all:retargeting OR all:"motion retargeting" OR all:kinodynamic OR all:feasibility OR all:"dynamic feasibility" OR all:"dynamics feasibility" OR all:"dynamically feasible" OR all:"dynamics-aware" OR all:"dynamics aware" OR all:"physics-aware" OR all:"physics aware" OR all:"controller-aware" OR all:"controller aware" OR all:"control-aware" OR all:"control aware" OR all:contact OR all:"contact-aware" OR all:"contact aware" OR all:cadence OR all:retiming OR all:"time scaling" OR all:"root frame" OR all:"root-frame" OR all:"root motion" OR all:"contact schedule" OR all:"contact sequence") AND (all:reference OR all:motion) AND (all:humanoid OR all:biped OR all:bipedal OR all:"bipedal robot" OR all:"physics-based character" OR all:"physics based character" OR all:"articulated character" OR all:"virtual character")) AND submittedDate:[200201010000 TO 202609022359]
S6=((all:Gymnasium OR all:MuJoCo OR all:"Humanoid-v5" OR all:"MuJoCo Humanoid") AND (all:reference OR all:imitation OR all:tracking OR all:phase OR all:recovery OR all:transition)) AND submittedDate:[200201010000 TO 202609022359]
S7=((all:"skill chaining" OR all:"skill composition" OR all:"policy switching" OR all:"mode switching" OR all:"transition policy" OR all:"skill sequencing" OR all:"long horizon" OR all:"long-horizon" OR all:"multi skill" OR all:"multi-skill" OR all:multimodal OR all:"multi-modal" OR all:multimode OR all:"multi-mode") AND (all:control OR all:controller OR all:policy) AND (all:humanoid OR all:biped OR all:bipedal OR all:"bipedal robot" OR all:"physics-based character" OR all:"physics based character" OR all:"articulated character" OR all:"virtual character")) AND submittedDate:[200201010000 TO 202609022359]
```

## Source-specific execution contract

| Source | Exact wrapper and filters | Exhaustion rule |
|---|---|---|
| arXiv API | `GET https://export.arxiv.org/api/query`; use the S1–S7 value above as `search_query`; `start=0`; `max_results=100`; `sortBy=submittedDate`; `sortOrder=ascending` | Advance `start` by returned items until `opensearch:totalResults` is exhausted; wait at least 3 seconds between calls. |
| OpenAlex Works API | `GET https://api.openalex.org/works`; use the literal Boolean query as `search`; `filter=from_publication_date:2002-01-01,to_publication_date:2026-09-02`; `corpus=core`; `sort=publication_date:asc`; `per_page=100`; `cursor=*` | Follow `meta.next_cursor` until null; retain all results, with no score cutoff. Use free unauthenticated access only; stop if it is unavailable rather than incur cost. |

Apply no language, open-access, citation-count, topic, semantic-search, or
relevance-score filter. “Available by cutoff” means an exact primary-source
version was public by `2026-09-02T23:59:59Z`; it does not mean today's latest
version. For each arXiv hit, use versioned `id_list` requests and retain the
highest `vN` whose `updated` timestamp is no later than the cutoff. Register and
hash those exact PDF bytes.

ACM Digital Library and IEEE Xplore are not independently searched because no
authorized automated route is provisioned in this project. OpenReview's search
endpoint does not provide a documented equivalent for this Boolean fielded
strategy. These sites, RSS, and PMLR/CoRL may resolve exact venue records
discovered by the two formal routes; the resolution URL and exact downloaded
bytes are recorded. This coverage limitation belongs in the final report.

## Request and export receipt

For every source, lane, and page, record:

- source and lane ID;
- UTC execution timestamp and cutoff;
- decoded literal query and final encoded request URL or UI query;
- all filters, sort, page/start/cursor, reported total, and returned count;
- raw Atom/JSON response or database export path and SHA-256;
- pre-deduplication IDs, post-deduplication IDs, and every merge decision;
- tool/API version when exposed, access failure, and any deviation.
- response headers, retry count, parser version, and query-renderer source
  hash;
- per-lane raw count, per-source union count, cross-source deduplicated count,
  and every zero-result page or query.

The page/request ledger uses these fields in this order:

```text
request_id,database,lane_id,query_version,expanded_query_sha256,
canonical_redacted_url,cutoff_start_utc,cutoff_end_utc,requested_at_utc,
page_start_or_cursor_in,cursor_out,http_status,declared_total,
page_record_count,raw_response_path,raw_response_bytes,raw_response_sha256,
parser_version,parsed_output_path,parsed_output_rows,parsed_output_sha256,
retry_count,notes
```

Never store API keys, session cookies, or subscription-only full text in the
repository. Metadata hits are candidates, not evidence.

## Citation chaining through OpenAlex

For each source screened as `direct_fit`:

1. Resolve its OpenAlex Work with
   `GET https://api.openalex.org/works/doi:<DOI>`. For an arXiv-only record,
   first try `10.48550/arXiv.<base-id>`. Store the DOI-to-Work-ID response, UTC
   timestamp, and SHA-256; confirm title and authors before accepting the
   mapping.
2. **Backward:** read `referenced_works` from that Work and batch-fetch those
   IDs with `filter=openalex:W1|W2|...&corpus=all` in groups of at most 100. Do
   not apply the 2002 lower bound here; an older work can enter only as a
   documented foundational backward-chain candidate.
3. **Forward:** request
   `filter=cites:W_SEED,from_publication_date:2002-01-01,to_publication_date:2026-09-02&corpus=all&per_page=100&cursor=*`
   and follow every `meta.next_cursor`.
4. Union both directions, preserve raw responses, deduplicate by DOI, then
   arXiv ID, then adjudicated title/authors. Register exact source bytes before
   screening the new candidate.
5. Independently inspect the bibliography of every acquired `direct_fit` PDF
   and record any cited work absent from OpenAlex's `referenced_works`; the
   index alone does not prove backward-chain completeness.
6. Only newly screened `direct_fit` papers seed the next round. Preserve a
   round ledger even when its yield is zero.

Two consecutive fully exhausted rounds with zero new `direct_fit` sources are
the proposed stopping condition. Changing a query, source, filter, cutoff,
classification definition, or stopping rule requires a dated amendment and
fresh human approval before the next request.

## Known coverage risks

- The pilot seed set influenced the vocabulary. Keep pilot provenance separate
  from formal-query yield to expose anchoring.
- arXiv and OpenAlex search different fields; their result counts are not
  comparable.
- OpenAlex is a live, merged index. Frozen raw responses—not the query alone—
  preserve what was observed at execution time.
- Publisher/preprint records remain separate until exact version and content
  equivalence is established.
- Not independently searching ACM, IEEE, or OpenReview may leave an indexing
  gap. Report that limitation and resolve publisher records surfaced by the
  two formal indexes and citation chains.

## Approval effect

Approval authorizes this exact ledger for formal searching only after all other
pending pre-search gates are approved. It does not approve any candidate for
inclusion, extraction, synthesis, or implementation.
