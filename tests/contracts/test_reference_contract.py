from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from oracle_composition.contracts import (
    OracleContractError,
    ReferenceArtifact,
    ReferenceSchema,
    assert_exact_schema,
)


def _schema() -> ReferenceSchema:
    return ReferenceSchema(
        feature_names=("root.z", "hip.angle"),
        feature_units=("m", "rad"),
        root_frame="world",
        cadence_hz=50.0,
    )


def _artifact() -> ReferenceArtifact:
    return ReferenceArtifact.create(
        artifact_id="humanoid/walk/v1",
        schema=_schema(),
        values=((0.8, 0.0), (0.81, 0.1), (0.82, 0.2)),
    )


def test_reference_copies_mutable_inputs_and_is_frozen() -> None:
    rows = [[0.8, 0.0], [0.81, 0.1]]
    artifact = ReferenceArtifact.create(
        artifact_id="humanoid/walk/v1",
        schema=_schema(),
        values=rows,
    )
    rows[0][0] = 99.0

    assert artifact.values[0][0] == 0.8
    with pytest.raises(FrozenInstanceError):
        artifact.values = ((1.0, 2.0),)  # type: ignore[misc]


def test_reference_accepts_numpy_arrays_without_retaining_aliases() -> None:
    values = np.asarray([[0.8, 0.0], [0.81, 0.1]], dtype=np.float32)
    artifact = ReferenceArtifact.create(
        artifact_id="humanoid/walk/numpy",
        schema=_schema(),
        values=values,
    )
    values[0, 0] = 99.0

    assert artifact.values[0][0] == pytest.approx(0.8)


def test_content_hash_binds_samples_order_units_frame_and_cadence() -> None:
    base = _artifact()
    variants = (
        ReferenceArtifact.create(
            artifact_id=base.identity.artifact_id,
            schema=base.schema,
            values=((0.8, 0.0), (0.81, 0.1), (0.82, 0.3)),
        ),
        ReferenceArtifact.create(
            artifact_id=base.identity.artifact_id,
            schema=ReferenceSchema(
                feature_names=("hip.angle", "root.z"),
                feature_units=("rad", "m"),
                root_frame="world",
                cadence_hz=50.0,
            ),
            values=((0.0, 0.8), (0.1, 0.81), (0.2, 0.82)),
        ),
        ReferenceArtifact.create(
            artifact_id=base.identity.artifact_id,
            schema=replace(base.schema, feature_units=("cm", "rad")),
            values=base.values,
        ),
        ReferenceArtifact.create(
            artifact_id=base.identity.artifact_id,
            schema=replace(base.schema, root_frame="pelvis"),
            values=base.values,
        ),
        ReferenceArtifact.create(
            artifact_id=base.identity.artifact_id,
            schema=replace(base.schema, cadence_hz=60.0),
            values=base.values,
        ),
    )

    hashes = {base.identity.content_sha256}
    hashes.update(item.identity.content_sha256 for item in variants)
    assert len(hashes) == 1 + len(variants)


@pytest.mark.parametrize(
    "observed",
    [
        ReferenceSchema(("hip.angle", "root.z"), ("rad", "m"), "world", 50.0),
        ReferenceSchema(("root.z", "hip.angle"), ("cm", "rad"), "world", 50.0),
        ReferenceSchema(("root.z", "hip.angle"), ("m", "rad"), "pelvis", 50.0),
        ReferenceSchema(("root.z", "hip.angle"), ("m", "rad"), "world", 60.0),
    ],
)
def test_exact_schema_rejects_every_semantic_mismatch(
    observed: ReferenceSchema,
) -> None:
    with pytest.raises(OracleContractError, match="schema mismatch"):
        assert_exact_schema(_schema(), observed)


@pytest.mark.parametrize(
    "values",
    [
        (),
        ((1.0,),),
        ((1.0, 2.0), (3.0,)),
        ((float("nan"), 0.0),),
        ((float("inf"), 0.0),),
        ((True, 0.0),),
    ],
)
def test_reference_rejects_empty_ragged_nonfinite_or_boolean_data(values: tuple) -> None:
    with pytest.raises(OracleContractError):
        ReferenceArtifact.create(
            artifact_id="bad/reference",
            schema=_schema(),
            values=values,
        )


def test_forged_identity_is_rejected() -> None:
    artifact = _artifact()
    with pytest.raises(OracleContractError, match="content hash"):
        ReferenceArtifact(
            identity=artifact.identity,
            schema=artifact.schema,
            values=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        )


def test_window_is_finite_h_by_d_and_holds_only_within_boundary() -> None:
    window = _artifact().window(
        start_frame=1,
        horizon_steps=4,
        end_exclusive=3,
    )

    assert window.frame_indices == (1, 2, 2, 2)
    assert window.horizon == 4
    assert window.width == 2
    assert len(window.values) == 4
    assert all(len(row) == 2 for row in window.values)


@pytest.mark.parametrize(
    ("start", "horizon", "stop"),
    [(-1, 1, 3), (3, 1, 3), (0, 0, 3), (0, 1, 4)],
)
def test_window_rejects_invalid_bounds(start: int, horizon: int, stop: int) -> None:
    with pytest.raises(OracleContractError):
        _artifact().window(
            start_frame=start,
            horizon_steps=horizon,
            end_exclusive=stop,
        )
