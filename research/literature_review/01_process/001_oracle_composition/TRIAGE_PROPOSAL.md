# Candidate follow-up queues — proposed for human approval

**State:** metadata-only proposal. This is not source triage, screening,
inclusion, or a `fit_class` assignment. The queues use only discovery-route and
candidate-title metadata to order later source acquisition.

| Queue | Count | Acquisition purpose after approval |
|---|---:|---|
| A — mechanism-keyword priority | 42 | Acquire first because metadata contains reference composition, phase/transition, switching, motion matching, sequencing, or recovery terms. This is not a relevance judgment. |
| B — adjacent-keyword priority | 34 | Acquire second because metadata contains potentially transferable control, phase, retargeting, or feasibility terms. This is not a transferability judgment. |
| C — context-keyword priority | 3 | Acquire after A/B because metadata points to benchmark, runtime, or evaluation context. This is not a `background_only` decision. |
| H — administrative hold | 1 | Preserve the metadata record; do not acquire or screen while the withdrawal remains active. |

## A — mechanism-keyword priority

`arxiv:1804.02717v3`, `arxiv:1905.09808v1`,
`arxiv:2205.01906v2`, `arxiv:2303.05711v2`,
`arxiv:2305.02195v1`, `arxiv:2305.06456v3`,
`arxiv:2309.11351v1`, `arxiv:2310.04582v2`,
`arxiv:2310.10198v3`, `arxiv:2403.04205v3`,
`arxiv:2406.06005v2`, `arxiv:2408.00776v2`,
`arxiv:2409.14393v1`, `arxiv:2410.01030v3`,
`arxiv:2508.08241v4`, `arxiv:2509.22442v1`,
`arxiv:2510.05070v2`, `arxiv:2510.14454v1`,
`arxiv:2510.22632v1`, `arxiv:2601.23080v1`,
`arxiv:2602.13656v1`, `arxiv:2602.15060v2`,
`arxiv:2602.15827v2`, `arxiv:2602.20375v1`,
`arxiv:2603.03279v1`, `arxiv:2603.27756v2`,
`arxiv:2604.01064v1`, `arxiv:2604.14834v1`,
`arxiv:2604.17335v2`, `arxiv:2604.21355v2`,
`arxiv:2604.22911v1`, `arxiv:2606.10340v1`,
`arxiv:2606.12814v1`, `arxiv:2606.20705v1`,
`arxiv:2606.22998v2`, `arxiv:2606.27581v1`,
`arxiv:2607.12114v1`, `arxiv:2607.24083v1`,
`arxiv:2608.02385v1`, `arxiv:2608.02653v1`,
`arxiv:2608.07746v1`, `arxiv:2608.18234v2`.

## B — adjacent-keyword priority

`arxiv:2104.02180v2`, `arxiv:2201.04439v1`,
`arxiv:2208.07363v3`, `arxiv:2308.12751v1`,
`arxiv:2406.08858v1`, `arxiv:2407.18946v1`,
`arxiv:2410.21229v2`, `arxiv:2412.13196v2`,
`arxiv:2506.14770v2`, `arxiv:2507.07356v3`,
`arxiv:2509.13833v3`, `arxiv:2509.15443v2`,
`arxiv:2510.02252v1`, `arxiv:2511.07820v4`,
`arxiv:2512.09423v2`, `arxiv:2601.12799v1`,
`arxiv:2603.09956v2`, `arxiv:2603.22201v3`,
`arxiv:2606.03476v1`, `arxiv:2607.06052v1`,
`doi:10.1111/cgf.12607`, `doi:10.1145/1230100.1230123`,
`doi:10.1145/1276377.1276510`, `doi:10.1145/2366145.2366173`,
`doi:10.1145/2893476`, `doi:10.1145/3072959.3073663`,
`doi:10.1145/3083723`, `doi:10.1145/3386569.3392440`,
`doi:10.1145/3386569.3392450`, `doi:10.1145/3528223.3530178`,
`doi:10.1145/3528233.3530735`, `doi:10.1145/383259.383287`,
`doi:10.1145/566654.566605`, `doi:10.1145/566654.566607`.

## C — context-keyword priority

`arxiv:2108.13264v4`, `arxiv:2403.10506v2`,
`arxiv:2502.08844v1`.

## H — administrative hold

`arxiv:2607.04837v3` (Athena-WBC). The arXiv record displays a withdrawal
notice and no current PDF. Retention here is provenance, not eligibility.

## Approval effect

Approval authorizes only the order for follow-up source acquisition. It does
not authorize reading-based classification or populate `source_triage.csv`.
After exact sources are registered, formal screening must apply the approved
criteria source by source and return proposed `fit_class` assignments and the
included set for a separate human gate.
