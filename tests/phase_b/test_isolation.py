from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b.isolation import (
    seal_artifact,
    sealed_input_lineage_sha256,
    validate_executing_modules,
    verify_sealed_inputs,
)


def test_replaced_input_fails_before_construction_or_success_receipt(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    artifact = root / "oracle.json"
    artifact.write_bytes(b'{"oracle":1}\n')
    sealed = (seal_artifact(root, artifact, roles=("oracle",)),)
    lineage_sha256 = sealed_input_lineage_sha256(sealed)
    artifact.write_bytes(b'{"oracle":2}\n')
    construction_count = 0

    with pytest.raises(ExperimentContractError, match="changed after preflight"):
        verify_sealed_inputs(root, sealed, expected_lineage_sha256=lineage_sha256)
        construction_count += 1

    assert construction_count == 0
    assert not list(root.glob("*success_receipt*"))


def test_switched_import_root_fails_before_construction_or_success_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "declared"
    source = root / "src/oracle_composition/phase_b/example.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")
    switched = tmp_path / "switched/example.py"
    switched.parent.mkdir(parents=True)
    switched.write_bytes(source.read_bytes())
    relative = source.relative_to(root).as_posix()
    source_hashes = {relative: hashlib.sha256(source.read_bytes()).hexdigest()}
    construction_count = 0

    with pytest.raises(ExperimentContractError, match="outside the declared checkout"):
        validate_executing_modules(
            root,
            source_hashes,
            module_files={"oracle_composition.phase_b.example": switched},
        )
        construction_count += 1

    assert construction_count == 0
    assert not list(root.glob("*success_receipt*"))


def test_module_digest_must_match_the_recorded_checkout_source(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    source = root / "src/oracle_composition/example.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")
    relative = source.relative_to(root).as_posix()
    source_hashes = {relative: hashlib.sha256(source.read_bytes()).hexdigest()}
    source.write_bytes(b"VALUE = 2\n")

    with pytest.raises(ExperimentContractError, match="digest differs"):
        validate_executing_modules(
            root,
            source_hashes,
            module_files={"oracle_composition.example": source},
        )
