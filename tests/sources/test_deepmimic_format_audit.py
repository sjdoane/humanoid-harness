from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import oracle_composition.sources.deepmimic as deepmimic
from oracle_composition.sources.deepmimic import (
    DeepMimicFormatError,
    audit_source_tree,
    inspect_motion_bytes,
    parse_motion_bytes,
    render_audit_json,
    verify_audit_payload_hash,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "research/source_motions/deepmimic_humanoid3d"
AUDIT_PATH = SOURCE_ROOT / "source_format_audit.json"
IDENTITY = [1.0, 0.0, 0.0, 0.0]


def _frame(duration: float) -> list[float]:
    frame = [
        duration,
        0.0,
        1.0,
        0.0,
        *IDENTITY,  # root
        *IDENTITY,  # chest
        *IDENTITY,  # neck
        *IDENTITY,  # right hip
        0.0,  # right knee
        *IDENTITY,  # right ankle
        *IDENTITY,  # right shoulder
        0.0,  # right elbow
        *IDENTITY,  # left hip
        0.0,  # left knee
        *IDENTITY,  # left ankle
        *IDENTITY,  # left shoulder
        0.0,  # left elbow
    ]
    assert len(frame) == deepmimic.HUMANOID3D_FRAME_WIDTH
    return frame


def _raw(
    *,
    frames: list[list[object]] | None = None,
    loop: object = "none",
    extra: dict[str, object] | None = None,
) -> bytes:
    payload: dict[str, object] = {
        "Loop": loop,
        "Frames": [_frame(0.1), _frame(0.0)] if frames is None else frames,
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def test_official_sources_regenerate_exact_content_addressed_audit() -> None:
    before = {path.name: path.read_bytes() for path in sorted((SOURCE_ROOT / "raw").glob("*.txt"))}
    audit = audit_source_tree(SOURCE_ROOT)

    assert render_audit_json(audit).encode("utf-8") == AUDIT_PATH.read_bytes()
    assert verify_audit_payload_hash(audit)
    assert audit["audit_payload_sha256"] == (
        "dd6584597b5a192ebb4562c6adbcf02d8874b6fd721b0836f97c25559ac4ddaf"
    )
    manifest_bytes = (SOURCE_ROOT / "source_manifest.json").read_bytes()
    assert audit["source_manifest"]["sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert audit["summary"] == {
        "motion_count": 4,
        "source_format_pass_count": 3,
        "source_format_fail_count": 1,
        "failed_paths": ["raw/humanoid3d_getup_facedown.txt"],
    }
    assert audit["admission"] == {
        "source_bytes_transformed": False,
        "normalized_motion_emitted": False,
        "gymnasium_joint_mapping_established": False,
        "retargeting_performed": False,
        "controller_compatibility_established": False,
        "dynamics_feasibility_established": False,
        "training_admission_granted": False,
    }
    after = {path.name: path.read_bytes() for path in sorted((SOURCE_ROOT / "raw").glob("*.txt"))}
    assert after == before


def test_official_facedown_failure_is_precisely_located() -> None:
    motions = {item["path"]: item for item in audit_source_tree(SOURCE_ROOT)["motions"]}
    facedown = motions.pop("raw/humanoid3d_getup_facedown.txt")

    assert not facedown["source_format_pass"]
    assert not facedown["checks"]["quaternion_norm_pass"]
    assert not facedown["checks"]["quaternion_sign_continuity_pass"]
    fields = {item["field"]: item for item in facedown["quaternion_fields"]}
    assert len(fields) == 9
    assert fields["neck"]["maximum_absolute_norm_error"] == pytest.approx(0.023408610683459008)
    assert fields["neck"]["maximum_absolute_norm_error_frame"] == 152
    assert fields["right_ankle"]["maximum_absolute_norm_error"] == pytest.approx(
        0.02119944079116698
    )
    assert fields["right_ankle"]["maximum_absolute_norm_error_frame"] == 151
    assert fields["left_shoulder"]["minimum_adjacent_normalized_dot"] == pytest.approx(
        -0.9997382202702
    )
    assert fields["left_shoulder"]["negative_adjacent_normalized_dot_pairs"] == [[4, 5]]
    assert all(item["source_format_pass"] for item in motions.values())


def test_diagnostic_normalization_does_not_change_or_admit_source_values() -> None:
    frames = [_frame(0.1), _frame(0.0)]
    frames[0][4:8] = [1.01, 0.0, 0.0, 0.0]
    raw = _raw(frames=frames)

    parsed = parse_motion_bytes(raw)
    result = inspect_motion_bytes(raw)

    assert parsed.frames[0][4:8] == (1.01, 0.0, 0.0, 0.0)
    assert parsed.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert result["source_sha256"] == parsed.source_sha256
    assert result["source_format_pass"]
    root = next(item for item in result["quaternion_fields"] if item["field"] == "root")
    assert root["maximum_absolute_norm_error"] == pytest.approx(0.01)


def test_negative_adjacent_quaternion_dot_is_reported_without_sign_flipping() -> None:
    frames = [_frame(0.1), _frame(0.0)]
    frames[1][4:8] = [-1.0, 0.0, 0.0, 0.0]
    result = inspect_motion_bytes(_raw(frames=frames))
    root = next(item for item in result["quaternion_fields"] if item["field"] == "root")

    assert root["norm_tolerance_pass"]
    assert not root["sign_continuity_pass"]
    assert root["minimum_adjacent_normalized_dot"] == -1.0
    assert root["negative_adjacent_normalized_dot_pairs"] == [[0, 1]]
    assert not result["source_format_pass"]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"\xff", "valid UTF-8"),
        (b"[]", "top-level value must be an object"),
        (b'{"Loop":"none","Frames":NaN}', "non-finite JSON constant"),
        (b'{"Loop":"none","Loop":"wrap","Frames":[]}', "duplicate JSON object key"),
        (_raw(extra={"unexpected": 1}), "unknown top-level fields"),
        (_raw(loop=True), "Loop must be exactly"),
    ],
)
def test_rejects_noncanonical_json_envelopes(raw: bytes, message: str) -> None:
    with pytest.raises(DeepMimicFormatError, match=message):
        parse_motion_bytes(raw)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frames: frames.__setitem__(slice(None), [frames[0]]), "frame count 1"),
        (lambda frames: frames[0].pop(), "width 43"),
        (lambda frames: frames[0].append(0.0), "width 45"),
        (lambda frames: frames[0].__setitem__(0, 0.0), "nonterminal frame 0 duration"),
        (lambda frames: frames[-1].__setitem__(0, -0.1), "terminal frame duration"),
        (lambda frames: frames[0].__setitem__(0, 61.0), "duration exceeds"),
        (lambda frames: frames[0].__setitem__(1, True), "must be a JSON number"),
        (lambda frames: frames[0].__setitem__(1, "0"), "must be a JSON number"),
        (lambda frames: frames[0].__setitem__(1, 1e309), "non-finite"),
        (lambda frames: frames[0].__setitem__(1, 1_000_001), "exceeds absolute bound"),
    ],
)
def test_rejects_bad_frame_count_width_types_bounds_and_durations(mutate, message: str) -> None:
    frames: list[list[object]] = [_frame(0.1), _frame(0.0)]
    mutate(frames)
    with pytest.raises(DeepMimicFormatError, match=message):
        parse_motion_bytes(_raw(frames=frames))


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"RightJoints": [3]}, "supplied together"),
        ({"RightJoints": [3], "LeftJoints": [True]}, "exact JSON integer"),
        ({"RightJoints": [3, 3], "LeftJoints": [9]}, "must be unique"),
        ({"RightJoints": [44], "LeftJoints": [9]}, "less than 44"),
    ],
)
def test_rejects_invalid_optional_joint_index_metadata(
    extra: dict[str, object], message: str
) -> None:
    with pytest.raises(DeepMimicFormatError, match=message):
        parse_motion_bytes(_raw(extra=extra))


def test_rejects_oversized_source_before_json_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _raw()
    monkeypatch.setattr(deepmimic, "MAX_SOURCE_BYTES", len(raw) - 1)
    with pytest.raises(DeepMimicFormatError, match="bytes; limit"):
        parse_motion_bytes(raw)


def test_rejects_frame_count_above_frozen_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deepmimic, "MAX_FRAMES", 1)
    with pytest.raises(DeepMimicFormatError, match="frame count 2"):
        parse_motion_bytes(_raw())


def test_source_tree_fails_closed_on_manifest_byte_or_hash_mismatch(tmp_path: Path) -> None:
    copied_root = tmp_path / "deepmimic_humanoid3d"
    shutil.copytree(SOURCE_ROOT, copied_root)
    source = copied_root / "raw/humanoid3d_walk.txt"
    source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(DeepMimicFormatError, match="byte count does not match"):
        audit_source_tree(copied_root)


def test_source_tree_fails_closed_on_same_length_source_mutation(tmp_path: Path) -> None:
    copied_root = tmp_path / "deepmimic_humanoid3d"
    shutil.copytree(SOURCE_ROOT, copied_root)
    source = copied_root / "raw/humanoid3d_walk.txt"
    raw = source.read_bytes()
    mutated = raw.replace(b"0.0333320000", b"0.0333320001", 1)
    assert len(mutated) == len(raw)
    assert mutated != raw
    source.write_bytes(mutated)

    with pytest.raises(DeepMimicFormatError, match="SHA-256 does not match"):
        audit_source_tree(copied_root)


def test_source_manifest_parser_rejects_duplicate_keys(tmp_path: Path) -> None:
    copied_root = tmp_path / "deepmimic_humanoid3d"
    shutil.copytree(SOURCE_ROOT, copied_root)
    manifest_path = copied_root / "source_manifest.json"
    original = manifest_path.read_text(encoding="utf-8").lstrip()
    manifest_path.write_text(
        '{"schema_version":1,"schema_version":1,' + original[1:], encoding="utf-8"
    )

    with pytest.raises(DeepMimicFormatError, match="duplicate JSON object key"):
        audit_source_tree(copied_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_repository", "https://example.com/xbpeng/DeepMimic", "source_repository"),
        (
            "source_url",
            "https://example.com/xbpeng/DeepMimic/"
            "1f915c52fcd4b95b5f5f15b759ae91bd81e9a801/data/motions/"
            "humanoid3d_getup_facedown.txt",
            "exact pinned official",
        ),
    ],
)
def test_source_manifest_rejects_lookalike_provenance_urls(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    copied_root = tmp_path / "deepmimic_humanoid3d"
    shutil.copytree(SOURCE_ROOT, copied_root)
    manifest_path = copied_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "source_repository":
        manifest[field] = value
    else:
        manifest["files"][0][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DeepMimicFormatError, match=message):
        audit_source_tree(copied_root)


def test_source_tree_rejects_linked_raw_directory(tmp_path: Path) -> None:
    copied_root = tmp_path / "deepmimic_humanoid3d"
    copied_root.mkdir()
    shutil.copy2(SOURCE_ROOT / "source_manifest.json", copied_root)
    (copied_root / "raw").symlink_to(SOURCE_ROOT / "raw", target_is_directory=True)

    with pytest.raises(DeepMimicFormatError, match="raw root must be a real directory"):
        audit_source_tree(copied_root)


def test_payload_hash_detects_audit_tampering() -> None:
    audit = audit_source_tree(SOURCE_ROOT)
    audit["summary"]["source_format_pass_count"] = 4

    assert not verify_audit_payload_hash(audit)
    with pytest.raises(DeepMimicFormatError, match="audit_payload_sha256"):
        render_audit_json(audit)
