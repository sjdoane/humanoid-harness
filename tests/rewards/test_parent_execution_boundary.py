from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

from oracle_composition.rewards import _sandbox_worker
from oracle_composition.rewards.sandbox import (
    RUNTIME_ADMISSION_REFUSAL,
    RewardSandboxError,
    RewardSandboxWorkerV1,
    assert_source_snapshot_unchanged,
    capture_candidate_source,
)
from oracle_composition.rewards.static_validation import (
    StaticAcceptanceReceiptV1,
    StaticallyAcceptedTaskTermSourceV1,
    StaticValidationError,
    statically_validate_task_term_source,
)

SOURCE = b'CANDIDATE_ID = "safe-static-fixture"\n\ndef task_term(x):\n    return 0.0\n'


class HostileMapping(Mapping[str, object]):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def _touch(self, method: str) -> None:
        self.calls.append(method)
        raise AssertionError(f"hostile mapping method {method} was invoked")

    def __getitem__(self, _key: str) -> object:
        self._touch("getitem")

    def __iter__(self):
        self._touch("iter")

    def __len__(self) -> int:
        self._touch("len")

    def items(self):
        self._touch("items")

    def __eq__(self, _other: object) -> bool:
        self._touch("eq")

    def __ne__(self, _other: object) -> bool:
        self._touch("ne")


def test_parent_validation_and_capture_never_invoke_child_installer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer_calls: list[bytes] = []

    def forbidden_installer(source: bytes) -> object:
        installer_calls.append(source)
        raise AssertionError("parent invoked the child-only candidate installer")

    monkeypatch.setattr(_sandbox_worker, "_install_task_term", forbidden_installer)
    accepted = statically_validate_task_term_source(SOURCE)
    candidate = tmp_path / "candidate.py"
    candidate.write_bytes(SOURCE)
    snapshot = capture_candidate_source(candidate)
    assert_source_snapshot_unchanged(snapshot)

    assert accepted.receipt.source_sha256 == snapshot.source_sha256
    assert installer_calls == []


def test_parent_facing_source_and_receipt_are_data_only() -> None:
    accepted = statically_validate_task_term_source(SOURCE)
    assert type(accepted) is StaticallyAcceptedTaskTermSourceV1
    assert tuple(field.name for field in dataclasses.fields(accepted)) == (
        "source_bytes",
        "receipt",
    )
    assert not callable(accepted)
    assert all(
        not callable(getattr(accepted, field.name)) for field in dataclasses.fields(accepted)
    )
    assert StaticAcceptanceReceiptV1.from_dict(accepted.receipt.to_dict()) == accepted.receipt


def test_exact_builtin_bytes_and_text_pass_but_text_subclass_is_untouched() -> None:
    calls: list[str] = []

    class HostileText(str):
        def encode(self, *_args: object, **_kwargs: object) -> bytes:
            calls.append("encode")
            raise AssertionError("text subclass encode was invoked")

    for source in (SOURCE, SOURCE.decode("utf-8")):
        assert statically_validate_task_term_source(source).source_bytes == SOURCE
    with pytest.raises(StaticValidationError, match="exact bytes or text"):
        statically_validate_task_term_source(HostileText(SOURCE.decode("utf-8")))
    assert calls == []


@pytest.mark.parametrize("version", [True, False, 1.0, "1", None, 2])
def test_static_receipt_requires_exact_integer_version(version: object) -> None:
    payload = statically_validate_task_term_source(SOURCE).receipt.to_dict()
    payload["schema_version"] = version
    with pytest.raises(StaticValidationError, match="keys or version"):
        StaticAcceptanceReceiptV1.from_dict(payload)


def test_receipt_metadata_rejects_serializer_callbacks_without_invocation() -> None:
    calls: list[str] = []

    class Callback:
        def to_dict(self) -> object:
            calls.append("serialized")
            return 0

    receipt = statically_validate_task_term_source(SOURCE).receipt
    payload = receipt.to_dict()
    payload["metadata"] = {"nested": [Callback()]}
    with pytest.raises(StaticValidationError, match="metadata"):
        StaticAcceptanceReceiptV1.from_dict(payload)
    with pytest.raises(StaticValidationError, match="metadata"):
        dataclasses.replace(receipt, metadata={"nested": [Callback()]})
    assert calls == []


def test_nested_receipt_metadata_has_a_plain_json_round_trip() -> None:
    source = b'META = {"values": [1, True, None, "text"]}\n\ndef task_term(x):\n    return 0.0\n'
    receipt = statically_validate_task_term_source(source).receipt
    payload = receipt.to_dict()
    assert type(payload["metadata"]["META"]) is dict
    assert type(payload["metadata"]["META"]["values"]) is list
    restored = StaticAcceptanceReceiptV1.from_dict(json.loads(json.dumps(payload)))
    assert restored.canonical_bytes == receipt.canonical_bytes


def test_receipt_copy_requires_plain_json_not_dataclass_replacement() -> None:
    receipt = statically_validate_task_term_source(SOURCE).receipt
    with pytest.raises(StaticValidationError, match="plain JSON object"):
        dataclasses.replace(receipt)
    clone = StaticAcceptanceReceiptV1.from_dict(receipt.to_dict())
    assert clone.canonical_bytes == receipt.canonical_bytes


def test_valid_direct_factory_and_json_source_construction_round_trip() -> None:
    factory = statically_validate_task_term_source(SOURCE)
    payload = factory.receipt.to_dict()
    direct_receipt = StaticAcceptanceReceiptV1(
        source_sha256=payload["source_sha256"],  # type: ignore[arg-type]
        source_bytes=payload["source_bytes"],  # type: ignore[arg-type]
        ast_nodes=payload["ast_nodes"],  # type: ignore[arg-type]
        read_set=tuple(payload["read_set"]),  # type: ignore[arg-type]
        metadata=payload["metadata"],  # type: ignore[arg-type]
        validation_scope=payload["validation_scope"],  # type: ignore[arg-type]
        static_accepted=payload["static_accepted"],  # type: ignore[arg-type]
        dynamic_validation_status=payload["dynamic_validation_status"],  # type: ignore[arg-type]
    )
    json_receipt = StaticAcceptanceReceiptV1.from_dict(
        json.loads(json.dumps(factory.receipt.to_dict()))
    )
    direct = StaticallyAcceptedTaskTermSourceV1(
        source_bytes=SOURCE,
        receipt=direct_receipt,
    )
    payload["metadata"]["CANDIDATE_ID"] = "caller-mutated"  # type: ignore[index]
    restored = StaticallyAcceptedTaskTermSourceV1(
        source_bytes=SOURCE,
        receipt=json_receipt,
    )
    assert direct.receipt.canonical_bytes == factory.receipt.canonical_bytes
    assert restored.receipt.canonical_bytes == factory.receipt.canonical_bytes
    assert direct.metadata["CANDIDATE_ID"] == "safe-static-fixture"
    assert direct.metadata is direct.receipt.metadata
    with pytest.raises(TypeError):
        direct.metadata["CANDIDATE_ID"] = "changed"  # type: ignore[index]


def test_public_metadata_ingress_rejects_hostile_mappings_without_callbacks() -> None:
    calls: list[str] = []
    hostile = HostileMapping(calls)
    hostile_proxy = MappingProxyType(hostile)
    receipt = statically_validate_task_term_source(SOURCE).receipt

    for metadata in (hostile, hostile_proxy):
        with pytest.raises(StaticValidationError, match="plain JSON object"):
            dataclasses.replace(receipt, metadata=metadata)

        payload = receipt.to_dict()
        payload["metadata"] = {"nested": metadata}
        with pytest.raises(StaticValidationError, match="metadata"):
            StaticAcceptanceReceiptV1.from_dict(payload)

    for payload in (hostile, hostile_proxy):
        with pytest.raises(StaticValidationError, match="keys or version"):
            StaticAcceptanceReceiptV1.from_dict(payload)

    with pytest.raises(TypeError, match="metadata"):
        StaticallyAcceptedTaskTermSourceV1(  # type: ignore[call-arg]
            source_bytes=SOURCE,
            receipt=receipt,
            metadata=hostile,
        )
    assert calls == []


def test_serialized_metadata_rejects_custom_containers_without_traversal() -> None:
    calls: list[str] = []

    class CustomDict(dict):
        def items(self):
            calls.append("items")
            raise AssertionError("custom mapping was traversed")

    class CustomList(list):
        def __iter__(self):
            calls.append("iter")
            raise AssertionError("custom sequence was traversed")

    for value in (CustomDict(), CustomList()):
        payload = statically_validate_task_term_source(SOURCE).receipt.to_dict()
        payload["metadata"] = {"nested": value}
        with pytest.raises(StaticValidationError, match="metadata"):
            StaticAcceptanceReceiptV1.from_dict(payload)
    assert calls == []


def test_accepted_source_revalidates_complete_ast_receipt() -> None:
    accepted = statically_validate_task_term_source(SOURCE)

    invalid_source = b"import os\n\ndef task_term(x):\n    return 0.0\n"
    fabricated = StaticAcceptanceReceiptV1(
        source_sha256=hashlib.sha256(invalid_source).hexdigest(),
        source_bytes=len(invalid_source),
        ast_nodes=accepted.receipt.ast_nodes,
        read_set=accepted.receipt.read_set,
        metadata={},
    )
    with pytest.raises(StaticValidationError):
        StaticallyAcceptedTaskTermSourceV1(
            source_bytes=invalid_source,
            receipt=fabricated,
        )

    altered_payload = accepted.receipt.to_dict()
    altered_payload["ast_nodes"] = accepted.receipt.ast_nodes + 1
    altered_ast = StaticAcceptanceReceiptV1.from_dict(altered_payload)
    with pytest.raises(StaticValidationError, match="stale"):
        StaticallyAcceptedTaskTermSourceV1(
            source_bytes=SOURCE,
            receipt=altered_ast,
        )

    with pytest.raises(StaticValidationError, match="stale"):
        StaticallyAcceptedTaskTermSourceV1(
            source_bytes=SOURCE + b"\n",
            receipt=accepted.receipt,
        )


@pytest.mark.parametrize("name", ["stock_r0", "manual_target_speed_v1"])
def test_donor_candidate_fixture_is_static_only_without_simulator_import(name: str) -> None:
    root = Path(__file__).parents[2]
    source = root / "experiments" / "family_b_target_speed_v1" / "candidates" / f"{name}.py"
    accepted = statically_validate_task_term_source(source.read_bytes())
    assert accepted.receipt.metadata["CANDIDATE_ID"] == name
    assert accepted.receipt.static_accepted is True
    assert accepted.receipt.dynamic_validation_status == "not_performed"
    assert not hasattr(accepted, "task_term")


def test_parent_modules_contain_no_in_process_candidate_execution_primitive() -> None:
    package = Path(__file__).parents[2] / "src" / "oracle_composition" / "rewards"
    for name in ("static_validation.py", "sandbox.py"):
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        forbidden = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"compile", "exec", "eval"}
        }
        assert forbidden == set()


def test_public_worker_start_refuses_before_any_runtime_setup() -> None:
    uninitialized_worker = object.__new__(RewardSandboxWorkerV1)
    with pytest.raises(RewardSandboxError, match="runtime admission") as refused:
        uninitialized_worker.start()
    assert str(refused.value) == RUNTIME_ADMISSION_REFUSAL


def test_capture_rejects_invalid_source_without_child_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched = False

    def forbidden_installer(_source: bytes) -> object:
        nonlocal touched
        touched = True
        raise AssertionError("installer must remain child-only")

    monkeypatch.setattr(_sandbox_worker, "_install_task_term", forbidden_installer)
    candidate = tmp_path / "candidate.py"
    candidate.write_bytes(b"import os\n\ndef task_term(x):\n    return 0.0\n")
    with pytest.raises(StaticValidationError):
        capture_candidate_source(candidate)
    assert touched is False


def test_source_snapshot_drift_still_fails_closed(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_bytes(SOURCE)
    snapshot = capture_candidate_source(candidate)
    candidate.write_bytes(SOURCE + b"\n")
    with pytest.raises(RewardSandboxError, match="drifted"):
        assert_source_snapshot_unchanged(snapshot)
