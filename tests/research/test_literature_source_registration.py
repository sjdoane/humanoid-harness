from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts/register_literature_sources.py"
SPEC = importlib.util.spec_from_file_location("register_literature_sources", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SOURCE_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SOURCE_MODULE
SPEC.loader.exec_module(SOURCE_MODULE)

REGISTERED_STATUS = SOURCE_MODULE.REGISTERED_STATUS
SourceRegistrationError = SOURCE_MODULE.SourceRegistrationError
_regular_pdf = SOURCE_MODULE._regular_pdf
_safe_local_path = SOURCE_MODULE._safe_local_path
_validate_arxiv_binding = SOURCE_MODULE._validate_arxiv_binding
_download_pdf = SOURCE_MODULE._download_pdf
register = SOURCE_MODULE.register
verify = SOURCE_MODULE.verify

CANDIDATE_FIELDS = (
    "candidate_id",
    "database",
    "search_string_id",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "abstract_available",
    "full_text_available",
    "candidate_relevance",
    "workflow_stage_fit",
    "grounding_relevance",
    "conceptual_synthesis_relevance",
    "empirical_evaluation",
    "notes",
)


def _write_fixture_tables(root: Path) -> tuple[Path, Path]:
    candidates = root / "candidate_papers.csv"
    with candidates.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": "arxiv:2403.04205v3",
                "database": "arXiv",
                "search_string_id": "fixture",
                "title": "Fixture paper",
                "authors": "Fixture Author",
                "year": "2024",
                "venue": "arXiv",
                "doi": "10.48550/arXiv.2403.04205",
                "url": "https://arxiv.org/abs/2403.04205v3",
                "abstract_available": "yes",
                "full_text_available": "yes",
                "candidate_relevance": "not_assessed",
                "workflow_stage_fit": "not_assessed",
                "grounding_relevance": "not_assessed",
                "conceptual_synthesis_relevance": "not_assessed",
                "empirical_evaluation": "not_assessed",
                "notes": "fixture",
            }
        )
    batch = root / "batch.csv"
    with batch.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["candidate_id", "source_url", "local_path", "registration_status"])
        writer.writerow(
            [
                "arxiv:2403.04205v3",
                "https://arxiv.org/pdf/2403.04205v3",
                "research/raw_sources/oracle_composition/2403.04205v3.pdf",
                REGISTERED_STATUS,
            ]
        )
    return candidates, batch


def test_exact_arxiv_identity_requires_versioned_url_and_matching_path() -> None:
    with pytest.raises(SourceRegistrationError, match="version-pinned"):
        _validate_arxiv_binding(
            "arxiv:2403.04205v3",
            "https://arxiv.org/pdf/2403.04205",
            "research/raw_sources/oracle_composition/2403.04205v3.pdf",
        )


def test_source_path_rejects_escape_and_wrong_directory(tmp_path: Path) -> None:
    with pytest.raises(SourceRegistrationError, match="unsafe"):
        _safe_local_path(tmp_path, "../2403.04205v3.pdf")
    with pytest.raises(SourceRegistrationError, match="directly below"):
        _safe_local_path(tmp_path, "research/raw_sources/2403.04205v3.pdf")
    alternate = tmp_path / "alternate"
    alternate.mkdir()
    (tmp_path / "research").symlink_to(alternate, target_is_directory=True)
    with pytest.raises(SourceRegistrationError, match="crosses a symlink"):
        _safe_local_path(
            tmp_path,
            "research/raw_sources/oracle_composition/2403.04205v3.pdf",
        )


def test_registration_hashes_exact_local_pdf_and_reverifies(tmp_path: Path) -> None:
    source = tmp_path / "research/raw_sources/oracle_composition/2403.04205v3.pdf"
    source.parent.mkdir(parents=True)
    raw = b"%PDF-1.7\nfixture exact bytes\n%%EOF\n"
    source.write_bytes(raw)
    candidates, batch = _write_fixture_tables(tmp_path)
    manifest = tmp_path / "manifest.json"
    inventory = tmp_path / "inventory.csv"

    result = register(
        repository_root=tmp_path,
        candidates_path=candidates,
        batch_path=batch,
        manifest_path=manifest,
        inventory_path=inventory,
        download=False,
    )

    assert result["evidence_status"] == REGISTERED_STATUS
    assert result["sources"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "not screened" in result["claim_boundary"]
    verify_args = {
        "repository_root": tmp_path,
        "candidates_path": candidates,
        "batch_path": batch,
        "manifest_path": manifest,
        "inventory_path": inventory,
    }
    assert verify(**verify_args) == {
        "source_count": 1,
        "total_bytes": len(raw),
        "verified": True,
    }

    source.write_bytes(raw.replace(b"fixture", b"fixturE"))
    with pytest.raises(SourceRegistrationError, match="do not match"):
        verify(**verify_args)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("review_id", "unrelated_review"),
        ("claim_boundary", "These sources are admissible evidence."),
        ("registered_at_utc", "2026-09-03T02:22:04"),
    ],
)
def test_verifier_rejects_tampered_manifest_contract(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    source = tmp_path / "research/raw_sources/oracle_composition/2403.04205v3.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.7\nfixture exact bytes\n%%EOF\n")
    candidates, batch = _write_fixture_tables(tmp_path)
    manifest = tmp_path / "manifest.json"
    inventory = tmp_path / "inventory.csv"
    register(
        repository_root=tmp_path,
        candidates_path=candidates,
        batch_path=batch,
        manifest_path=manifest,
        inventory_path=inventory,
        download=False,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = replacement
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SourceRegistrationError):
        verify(
            repository_root=tmp_path,
            candidates_path=candidates,
            batch_path=batch,
            manifest_path=manifest,
            inventory_path=inventory,
        )


def test_verifier_rejects_partial_manifest_against_batch(tmp_path: Path) -> None:
    source = tmp_path / "research/raw_sources/oracle_composition/2403.04205v3.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.7\nfixture exact bytes\n%%EOF\n")
    candidates, batch = _write_fixture_tables(tmp_path)
    manifest = tmp_path / "manifest.json"
    inventory = tmp_path / "inventory.csv"
    register(
        repository_root=tmp_path,
        candidates_path=candidates,
        batch_path=batch,
        manifest_path=manifest,
        inventory_path=inventory,
        download=False,
    )
    with candidates.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS, lineterminator="\n")
        writer.writerow(
            {
                "candidate_id": "arxiv:1905.09808v1",
                "database": "arXiv",
                "search_string_id": "fixture",
                "title": "Second fixture paper",
                "authors": "Fixture Author",
                "year": "2019",
                "venue": "arXiv",
                "doi": "10.48550/arXiv.1905.09808",
                "url": "https://arxiv.org/abs/1905.09808v1",
                "abstract_available": "yes",
                "full_text_available": "yes",
                "candidate_relevance": "not_assessed",
                "workflow_stage_fit": "not_assessed",
                "grounding_relevance": "not_assessed",
                "conceptual_synthesis_relevance": "not_assessed",
                "empirical_evaluation": "not_assessed",
                "notes": "fixture",
            }
        )
    with batch.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "arxiv:1905.09808v1",
                "https://arxiv.org/pdf/1905.09808v1",
                "research/raw_sources/oracle_composition/1905.09808v1.pdf",
                REGISTERED_STATUS,
            ]
        )

    with pytest.raises(SourceRegistrationError, match="exactly match"):
        verify(
            repository_root=tmp_path,
            candidates_path=candidates,
            batch_path=batch,
            manifest_path=manifest,
            inventory_path=inventory,
        )


def test_verifier_rejects_inventory_tampering(tmp_path: Path) -> None:
    source = tmp_path / "research/raw_sources/oracle_composition/2403.04205v3.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.7\nfixture exact bytes\n%%EOF\n")
    candidates, batch = _write_fixture_tables(tmp_path)
    manifest = tmp_path / "manifest.json"
    inventory = tmp_path / "inventory.csv"
    register(
        repository_root=tmp_path,
        candidates_path=candidates,
        batch_path=batch,
        manifest_path=manifest,
        inventory_path=inventory,
        download=False,
    )
    inventory.write_text(
        inventory.read_text(encoding="utf-8").replace(
            "version-pinned primary-paper PDF", "untrusted document"
        ),
        encoding="utf-8",
    )

    with pytest.raises(SourceRegistrationError, match="inventory disagrees"):
        verify(
            repository_root=tmp_path,
            candidates_path=candidates,
            batch_path=batch,
            manifest_path=manifest,
            inventory_path=inventory,
        )


def test_registration_rejects_candidate_without_full_text(tmp_path: Path) -> None:
    source = tmp_path / "research/raw_sources/oracle_composition/2403.04205v3.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.7\nfixture exact bytes\n%%EOF\n")
    candidates, batch = _write_fixture_tables(tmp_path)
    candidates.write_text(
        candidates.read_text(encoding="utf-8").replace(
            ",yes,yes,not_assessed,", ",yes,no,not_assessed,"
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    inventory = tmp_path / "inventory.csv"

    with pytest.raises(SourceRegistrationError, match="not marked full-text available"):
        register(
            repository_root=tmp_path,
            candidates_path=candidates,
            batch_path=batch,
            manifest_path=manifest,
            inventory_path=inventory,
            download=False,
        )
    assert not manifest.exists()
    assert not inventory.exists()


def test_pdf_reader_rejects_symlink(tmp_path: Path) -> None:
    real_source = tmp_path / "real.pdf"
    real_source.write_bytes(b"%PDF-1.7\nfixture\n%%EOF\n")
    linked_source = tmp_path / "linked.pdf"
    linked_source.symlink_to(real_source)
    with pytest.raises(SourceRegistrationError, match="non-symlink"):
        _regular_pdf(linked_source)
    with pytest.raises(SourceRegistrationError, match="destination must not be a symlink"):
        _download_pdf("https://arxiv.org/pdf/2403.04205v3", linked_source)


def test_downloader_rejects_redirect_to_wrong_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "2403.04205v3.pdf"

    class Response:
        def __init__(self) -> None:
            self.headers = {"Content-Length": "32"}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://arxiv.org/pdf/9999.99999v3"

        def read(self, _size: int) -> bytes:
            raise AssertionError("wrong-identity response must not be read")

    monkeypatch.setattr(
        SOURCE_MODULE.urllib.request,
        "urlopen",
        lambda _request, timeout: Response(),
    )

    with pytest.raises(SourceRegistrationError, match="changed exact source identity"):
        _download_pdf("https://arxiv.org/pdf/2403.04205v3", destination)
    assert not destination.exists()
