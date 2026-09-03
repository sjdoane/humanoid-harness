from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROCESS_ROOT = ROOT / "research/literature_review/01_process/001_oracle_composition"
MATRICES = PROCESS_ROOT / "matrices"
SCRIPT_PATH = ROOT / "scripts/build_candidate_registry.py"
SPEC = importlib.util.spec_from_file_location("build_candidate_registry", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
REGISTRY_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REGISTRY_MODULE
SPEC.loader.exec_module(REGISTRY_MODULE)
ARXIV_ID = re.compile(r"arxiv:(\d{4}\.\d{4,5})v(\d+)")
ASSESSMENT_FIELDS = (
    "candidate_relevance",
    "workflow_stage_fit",
    "grounding_relevance",
    "conceptual_synthesis_relevance",
    "empirical_evaluation",
)
DECISION_TABLES = (
    "citation_ledger.csv",
    "coding_taxonomy.csv",
    "extraction_table.csv",
    "reference_table.csv",
    "screening_table.csv",
    "selection_decisions.csv",
    "source_triage.csv",
    "synthesis_matrix.csv",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_candidate_registry_is_deduplicated_and_unassessed() -> None:
    rows = _rows(MATRICES / "candidate_papers.csv")
    assert len(rows) == 80
    assert len({row["candidate_id"] for row in rows}) == len(rows)
    assert all(row[field] == "not_assessed" for row in rows for field in ASSESSMENT_FIELDS)

    for row in rows:
        match = ARXIV_ID.fullmatch(row["candidate_id"])
        if match is None:
            assert row["candidate_id"].startswith("doi:")
            assert row["url"] == f"https://doi.org/{row['doi']}"
            continue
        arxiv_id, version = match.groups()
        assert row["url"] == f"https://arxiv.org/abs/{arxiv_id}v{version}"
        assert row["doi"] == f"10.48550/arXiv.{arxiv_id}"


def test_targeted_discovery_records_pin_supplementary_code_without_claiming_evidence() -> None:
    rows = {
        row["candidate_id"]: row
        for row in _rows(MATRICES / "candidate_papers.csv")
        if row["search_string_id"] == "D5_TARGETED_WEB_DISCOVERY"
    }

    assert set(rows) == {"arxiv:2506.14770v2", "arxiv:2509.13833v3"}
    expected_repositories = {
        "arxiv:2506.14770v2": (
            "https://github.com/zixuan417/humanoid-general-motion-tracking/"
            "tree/2a590de25a1eb08e47491977a738549c22f16e1f"
        ),
        "arxiv:2509.13833v3": (
            "https://github.com/GalaxyGeneralRobotics/OpenTrack/"
            "tree/cb9b751993a2483e5d1805a2565ddbfe950c04c9"
        ),
    }
    for candidate_id, row in rows.items():
        assert f"supplementary_source={expected_repositories[candidate_id]}" in row["notes"]
        assert "supplementary_source_role=official_code_only" in row["notes"]
        assert (
            "not_registered_screened_extracted_or_evidence_of_harness_implementation"
            in row["notes"]
        )


def test_arxiv_metadata_lookup_is_version_pinned_and_politely_throttled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []
    body = b"""<html><head>
<meta name="citation_title" content="Fixture paper">
<meta name="citation_date" content="2024/03/07">
<meta name="citation_author" content="Fixture Author">
<meta name="citation_pdf_url" content="https://arxiv.org/pdf/2403.04205v3">
<meta property="og:url" content="https://arxiv.org/abs/2403.04205v3">
</head><body></body></html>"""

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return body

    def fake_urlopen(request: object, *, timeout: int) -> Response:
        assert timeout == 30
        requested_urls.append(request.full_url)  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr(REGISTRY_MODULE.urllib.request, "urlopen", fake_urlopen)
    seed = REGISTRY_MODULE.CandidateSeed("2403.04205", 3, "fixture", "oracle", "fixture")
    row = REGISTRY_MODULE._fetch(seed, attempts=1)

    assert requested_urls == ["https://arxiv.org/abs/2403.04205v3"]
    assert row["candidate_id"] == "arxiv:2403.04205v3"
    assert REGISTRY_MODULE.REQUEST_INTERVAL_SECONDS >= 3.0


@pytest.mark.parametrize(
    ("observed_page", "observed_pdf"),
    [
        (
            "https://arxiv.org/abs/9999.99999v3",
            "https://arxiv.org/pdf/2403.04205",
        ),
        (
            "https://arxiv.org/abs/2403.04205v3",
            "https://arxiv.org/pdf/9999.99999v3",
        ),
    ],
)
def test_arxiv_metadata_lookup_rejects_wrong_id_with_same_version(
    monkeypatch: pytest.MonkeyPatch,
    observed_page: str,
    observed_pdf: str,
) -> None:
    body = f"""<html><head>
<meta name="citation_title" content="Fixture paper">
<meta name="citation_date" content="2024/03/07">
<meta name="citation_author" content="Fixture Author">
<meta name="citation_pdf_url" content="{observed_pdf}">
<meta property="og:url" content="{observed_page}">
</head><body></body></html>""".encode()

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return body

    monkeypatch.setattr(
        REGISTRY_MODULE.urllib.request,
        "urlopen",
        lambda _request, timeout: Response(),
    )
    seed = REGISTRY_MODULE.CandidateSeed("2403.04205", 3, "fixture", "oracle", "fixture")

    with pytest.raises(RuntimeError, match=r"metadata identity|mismatched PDF"):
        REGISTRY_MODULE._fetch(seed, attempts=1)


def test_withdrawn_metadata_may_expose_same_id_pdf_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"""<html><head>
<meta name="citation_title" content="Withdrawn fixture">
<meta name="citation_date" content="2026/07/07">
<meta name="citation_author" content="Fixture Author">
<meta name="citation_pdf_url" content="https://arxiv.org/pdf/2607.04837">
<meta property="og:url" content="https://arxiv.org/abs/2607.04837v3">
</head><body><p>This paper has been withdrawn.</p></body></html>"""

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return body

    monkeypatch.setattr(
        REGISTRY_MODULE.urllib.request,
        "urlopen",
        lambda _request, timeout: Response(),
    )
    seed = REGISTRY_MODULE.CandidateSeed(
        "2607.04837",
        3,
        "fixture",
        "administrative_hold",
        "fixture",
        "no",
        "withdrawn fixture",
        True,
    )

    row = REGISTRY_MODULE._fetch(seed, attempts=1)
    assert row["candidate_id"] == "arxiv:2607.04837v3"
    assert row["full_text_available"] == "no"


def test_withdrawn_record_is_preserved_but_not_acquired() -> None:
    candidates = {row["candidate_id"]: row for row in _rows(MATRICES / "candidate_papers.csv")}
    withdrawn = candidates["arxiv:2607.04837v3"]
    assert withdrawn["full_text_available"] == "no"
    assert "withdrawal banner" in withdrawn["notes"]

    batch_ids = {row["candidate_id"] for row in _rows(PROCESS_ROOT / "acquisition_batch_01.csv")}
    assert "arxiv:2607.04837v3" not in batch_ids


def test_pilot_search_counts_reconcile_to_registry_without_becoming_screening() -> None:
    candidates = _rows(MATRICES / "candidate_papers.csv")
    expected = Counter(row["search_string_id"] for row in candidates)
    logged = {
        row["search_string_id"]: int(row["result_count"])
        for row in _rows(MATRICES / "search_log.csv")
    }
    assert logged == expected
    assert all("formal" in row["notes"] for row in _rows(MATRICES / "search_log.csv"))
    for file_name in DECISION_TABLES:
        assert _rows(MATRICES / file_name) == []


def test_formal_query_ledger_is_complete_but_unexecuted() -> None:
    queries = (PROCESS_ROOT / "FORMAL_SEARCH_QUERIES_DRAFT.md").read_text(encoding="utf-8")
    strategy = (PROCESS_ROOT / "SEARCH_STRATEGY_DRAFT.md").read_text(encoding="utf-8")
    normalized_queries = " ".join(queries.split())
    normalized_strategy = " ".join(strategy.split())

    assert "None of these queries has been executed as a formal search" in normalized_queries
    assert "arXiv API" in strategy and "OpenAlex Works API" in strategy
    assert "not claimed as independently searched Boolean databases" in normalized_strategy
    for lane in range(1, 8):
        assert f"### S{lane} —" in queries
        assert f"S{lane}=" in queries
    assert "Two consecutive fully exhausted rounds" in queries


def test_acquisition_batch_is_a_registered_candidate_subset() -> None:
    candidate_ids = {row["candidate_id"] for row in _rows(MATRICES / "candidate_papers.csv")}
    batch = _rows(PROCESS_ROOT / "acquisition_batch_01.csv")
    assert len(batch) == 31
    assert {row["candidate_id"] for row in batch} < candidate_ids
    assert all(row["registration_status"] == "registered_not_screened" for row in batch)


def test_committed_acquisition_receipt_binds_every_batch_record() -> None:
    candidates = {row["candidate_id"]: row for row in _rows(MATRICES / "candidate_papers.csv")}
    batch_ids = {row["candidate_id"] for row in _rows(PROCESS_ROOT / "acquisition_batch_01.csv")}
    manifest = json.loads(
        (PROCESS_ROOT / "source_acquisition_manifest.json").read_text(encoding="utf-8")
    )
    sources = manifest["sources"]
    assert manifest["evidence_status"] == "registered_not_screened"
    assert "not screened" in manifest["claim_boundary"]
    assert {source["source_id"] for source in sources} == batch_ids
    assert all(source["candidate_id"] == source["source_id"] for source in sources)
    assert all(source["title"] == candidates[source["source_id"]]["title"] for source in sources)
    assert all(re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) for source in sources)


def test_triage_proposal_partitions_every_candidate_without_deciding_inclusion() -> None:
    candidate_ids = {row["candidate_id"] for row in _rows(MATRICES / "candidate_papers.csv")}
    text = (PROCESS_ROOT / "TRIAGE_PROPOSAL.md").read_text(encoding="utf-8")
    section_markers = (
        ("## A — mechanism-keyword priority", "## B — adjacent-keyword priority", 42),
        ("## B — adjacent-keyword priority", "## C — context-keyword priority", 34),
        ("## C — context-keyword priority", "## H — administrative hold", 3),
        ("## H — administrative hold", "## Approval effect", 1),
    )
    proposed_ids: list[str] = []
    for start, end, expected_count in section_markers:
        section = text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]
        section_ids = re.findall(r"`((?:arxiv|doi):[^`]+)`", section)
        assert len(section_ids) == expected_count
        proposed_ids.extend(section_ids)
    assert len(proposed_ids) == len(set(proposed_ids))
    assert set(proposed_ids) == candidate_ids
    assert "This is not source triage, screening" in text
    assert "not authorize reading-based classification" in text
