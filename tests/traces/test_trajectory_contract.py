from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.traces import (
    BEHAVIORAL_BINDING_ROLES,
    REQUIRED_DIAGNOSTIC_ROLES,
    REQUIRED_DIAGNOSTIC_SIGNALS,
    ArtifactBinding,
    EvidenceClass,
    MissingReason,
    MissingSignal,
    NumericSignalSpec,
    TraceContractError,
    TraceRecorder,
    TraceRole,
    load_trace,
)

SHA = "a" * 64


def _signals() -> tuple[NumericSignalSpec, ...]:
    return (
        NumericSignalSpec(
            "robot.qpos",
            TraceRole.ROBOT,
            (3,),
            "mixed",
            "world_and_joint",
            "adapter.qpos",
        ),
        NumericSignalSpec(
            "oracle.phase",
            TraceRole.ORACLE,
            (1,),
            "1",
            "mode_local",
            "oracle.runtime",
        ),
    )


def _missing() -> tuple[MissingSignal, ...]:
    recorded = {signal.name for signal in _signals()}
    return tuple(
        MissingSignal(name, MissingReason.NOT_IMPLEMENTED, "Not emitted by this fixture.")
        for name in sorted(REQUIRED_DIAGNOSTIC_SIGNALS - recorded)
    )


def _trace():
    recorder = TraceRecorder(
        trace_id="trace/test/v1",
        evidence_class=EvidenceClass.INTERFACE_CHECK,
        control_period_seconds=0.02,
        numeric_signals=_signals(),
        missing_signals=_missing(),
        artifact_bindings=(ArtifactBinding("runtime", "runtime/test", SHA),),
    )
    recorder.add_sample(
        {
            "robot.qpos": np.asarray([0.0, 1.0, 2.0]),
            "oracle.phase": 0.0,
        }
    )
    recorder.add_sample(
        {
            "robot.qpos": [0.1, 1.1, 2.1],
            "oracle.phase": [0.25],
        }
    )
    recorder.add_event(
        sample_index=1,
        event_type="oracle.transition",
        value="stand_to_walk",
        source="oracle.runtime",
    )
    return recorder.finish()


def test_trace_round_trip_preserves_content_identity(tmp_path: Path) -> None:
    trace = _trace()
    path = trace.write(tmp_path / "trace.json")
    loaded = load_trace(path)

    assert loaded == trace
    assert loaded.sha256 == trace.sha256
    assert hashlib.sha256(path.read_bytes()).hexdigest() == trace.sha256
    assert loaded.diagnostic_coverage["recorded"] == ["oracle.phase", "robot.qpos"]
    assert loaded.diagnostic_coverage["missing"] == sorted(
        item.name for item in _missing()
    )
    assert loaded.to_dict()["samples"][1]["time_seconds"] == 0.02


def test_trace_write_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    _trace().write(path)

    with pytest.raises(TraceContractError, match="already exists"):
        _trace().write(path)


def test_missing_diagnostic_signal_fails_closed() -> None:
    trace = _trace()

    with pytest.raises(TraceContractError, match="diagnostic signal coverage"):
        replace(trace, missing_signals=trace.missing_signals[:-1])


@pytest.mark.parametrize(
    "bad_qpos",
    ([1.0, 2.0], [1.0, 2.0, float("nan")], [1.0, 2.0, float("inf")]),
)
def test_sample_shape_and_finiteness_fail_closed(bad_qpos: list[float]) -> None:
    recorder = TraceRecorder(
        trace_id="trace/test/v1",
        evidence_class=EvidenceClass.INTERFACE_CHECK,
        control_period_seconds=0.02,
        numeric_signals=_signals(),
        missing_signals=_missing(),
        artifact_bindings=(ArtifactBinding("runtime", "runtime/test", SHA),),
    )

    with pytest.raises(TraceContractError):
        recorder.add_sample({"robot.qpos": bad_qpos, "oracle.phase": 0.0})


def test_behavioral_trace_requires_complete_artifact_lineage() -> None:
    trace = _trace()

    with pytest.raises(TraceContractError, match="must record every required"):
        replace(trace, evidence_class=EvidenceClass.BEHAVIORAL_EVALUATION)


def test_event_must_point_inside_trace() -> None:
    trace = _trace()

    with pytest.raises(TraceContractError, match="outside the trace"):
        replace(
            trace,
            events=(
                *trace.events,
                replace(trace.events[0], sample_index=2),
            ),
        )


def _write_canonical_json(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def test_load_rejects_unknown_fields_and_undeclared_sample_signals(
    tmp_path: Path,
) -> None:
    payload = _trace().to_dict()
    payload["unexpected"] = "ignored by a permissive loader"
    root_extra = tmp_path / "root-extra.json"
    _write_canonical_json(root_extra, payload)

    with pytest.raises(TraceContractError, match="fields mismatch"):
        load_trace(root_extra)

    payload = _trace().to_dict()
    payload["samples"][0]["signals"]["undeclared.signal"] = [0.0]
    signal_extra = tmp_path / "signal-extra.json"
    _write_canonical_json(signal_extra, payload)

    with pytest.raises(TraceContractError, match="fields mismatch"):
        load_trace(signal_extra)


def test_load_rejects_dtype_bool_and_noncanonical_bytes(tmp_path: Path) -> None:
    payload = _trace().to_dict()
    payload["numeric_signals"][0]["dtype"] = "float32"
    dtype_path = tmp_path / "dtype.json"
    _write_canonical_json(dtype_path, payload)
    with pytest.raises(TraceContractError, match="dtype must be 'float64'"):
        load_trace(dtype_path)

    payload = _trace().to_dict()
    payload["samples"][0]["signals"]["robot.qpos"][0] = True
    bool_path = tmp_path / "bool.json"
    _write_canonical_json(bool_path, payload)
    with pytest.raises(TraceContractError, match="must be numeric"):
        load_trace(bool_path)

    pretty_path = tmp_path / "pretty.json"
    pretty_path.write_text(json.dumps(_trace().to_dict(), indent=2), encoding="utf-8")
    with pytest.raises(TraceContractError, match="bytes are not canonical"):
        load_trace(pretty_path)

    payload = _trace().to_dict()
    payload["schema_version"] = True
    schema_path = tmp_path / "schema-bool.json"
    _write_canonical_json(schema_path, payload)
    with pytest.raises(TraceContractError, match="schema_version"):
        load_trace(schema_path)


def test_required_diagnostic_signal_role_is_fixed() -> None:
    trace = _trace()
    bad_signal = replace(trace.numeric_signals[0], role=TraceRole.TASK)

    with pytest.raises(TraceContractError, match="roles are invalid"):
        replace(trace, numeric_signals=(bad_signal, *trace.numeric_signals[1:]))


def test_required_diagnostic_role_contract_is_immutable() -> None:
    with pytest.raises(TypeError):
        REQUIRED_DIAGNOSTIC_ROLES["robot.qpos"] = TraceRole.TASK


def test_reset_sample_requires_zero_controller_action() -> None:
    signals = (
        *_signals(),
        NumericSignalSpec(
            "controller.action",
            TraceRole.CONTROLLER,
            (2,),
            "1",
            "actuator",
            "controller.output",
        ),
    )
    missing = tuple(
        MissingSignal(name, MissingReason.NOT_IMPLEMENTED, "Fixture omission.")
        for name in sorted(
            REQUIRED_DIAGNOSTIC_SIGNALS - {signal.name for signal in signals}
        )
    )
    recorder = TraceRecorder(
        trace_id="trace/nonzero-reset/v1",
        evidence_class=EvidenceClass.INTERFACE_CHECK,
        control_period_seconds=0.02,
        numeric_signals=signals,
        missing_signals=missing,
        artifact_bindings=(ArtifactBinding("runtime", "runtime/test", SHA),),
    )
    recorder.add_sample(
        {
            "robot.qpos": [0.0, 1.0, 2.0],
            "oracle.phase": [0.0],
            "controller.action": [0.0, 0.1],
        }
    )

    with pytest.raises(TraceContractError, match="zero at the reset-state"):
        recorder.finish()


def test_behavioral_trace_names_every_required_artifact_role() -> None:
    signals = tuple(
        NumericSignalSpec(name, role, (1,), "1", "declared", "fixture")
        for name, role in sorted(REQUIRED_DIAGNOSTIC_ROLES.items())
    )
    recorder = TraceRecorder(
        trace_id="trace/behavioral/v1",
        evidence_class=EvidenceClass.BEHAVIORAL_EVALUATION,
        control_period_seconds=0.02,
        numeric_signals=signals,
        missing_signals=(),
        artifact_bindings=(ArtifactBinding("runtime", "runtime/test", SHA),),
    )
    recorder.add_sample({signal.name: [0.0] for signal in signals})

    with pytest.raises(
        TraceContractError,
        match=r"execution_manifest.*oracle.*policy_checkpoint",
    ):
        recorder.finish()


def test_behavioral_mode_change_requires_transition_reason_event() -> None:
    signals = tuple(
        NumericSignalSpec(name, role, (1,), "1", "declared", "fixture")
        for name, role in sorted(REQUIRED_DIAGNOSTIC_ROLES.items())
    )
    recorder = TraceRecorder(
        trace_id="trace/transition/v1",
        evidence_class=EvidenceClass.BEHAVIORAL_EVALUATION,
        control_period_seconds=0.02,
        numeric_signals=signals,
        missing_signals=(),
        artifact_bindings=tuple(
            ArtifactBinding(role, f"artifact/{role}", SHA)
            for role in sorted(BEHAVIORAL_BINDING_ROLES)
        ),
    )
    first = {signal.name: [0.0] for signal in signals}
    second = {signal.name: [0.0] for signal in signals}
    second["oracle.mode"] = [1.0]
    recorder.add_sample(first)
    recorder.add_sample(second)

    with pytest.raises(TraceContractError, match="mode changes require"):
        recorder.finish()

    recorder.add_event(
        sample_index=1,
        event_type="oracle.transition",
        value="guard contact_ready fired",
        source="oracle.runtime",
    )
    assert recorder.finish().events[0].sample_index == 1


def test_trace_file_size_is_checked_before_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oracle_composition.traces import contract

    path = tmp_path / "large.json"
    path.write_bytes(b"{}")
    monkeypatch.setattr(contract, "MAX_TRACE_BYTES", 1)

    with pytest.raises(TraceContractError, match="exceeds"):
        load_trace(path)


def test_trace_write_obeys_the_same_file_size_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oracle_composition.traces import contract

    path = tmp_path / "too-large.json"
    monkeypatch.setattr(contract, "MAX_TRACE_BYTES", 1)

    with pytest.raises(TraceContractError, match="exceeds"):
        _trace().write(path)
    assert not path.exists()


def test_missing_signals_and_artifact_bindings_are_bounded() -> None:
    trace = _trace()
    excess_missing = tuple(
        MissingSignal(f"extra.signal.{index}", MissingReason.NOT_APPLICABLE, "Fixture.")
        for index in range(257)
    )
    with pytest.raises(TraceContractError, match="missing signal count"):
        replace(trace, missing_signals=excess_missing)

    excess_bindings = (
        ArtifactBinding("runtime", "runtime/test", SHA),
        *(
            ArtifactBinding(f"extra.{index}", f"artifact/{index}", SHA)
            for index in range(64)
        ),
    )
    with pytest.raises(TraceContractError, match="artifact binding count"):
        replace(trace, artifact_bindings=excess_bindings)
