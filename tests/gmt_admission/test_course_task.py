from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from oracle_composition.adapters.gmt.course_task import (
    ALLOWED_GROUND_CONTACT_BODIES,
    TASK_FEATURE_NAMES,
    TASK_REWARD_DEPTH_SCALE_M,
    TASK_REWARD_RECIPE_ID,
    TASK_REWARD_RECIPE_V2_ID,
    CourseTaskSpec,
    TaskFrame,
    TaskRewardRecipe,
    evaluate_step,
    reward,
)


def _quaternion_wxyz(*, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _qpos(
    x: float,
    y: float,
    z: float = 0.55,
    *,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    result = np.zeros(30, dtype=np.float64)
    result[:3] = (x, y, z)
    result[3:7] = _quaternion_wxyz(roll=roll, pitch=pitch, yaw=yaw)
    return result


def _reference(*, height: float = 0.55, roll: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    result = np.zeros(30, dtype=np.float64)
    result[:3] = (height, roll, pitch)
    return result


def _spec() -> CourseTaskSpec:
    return CourseTaskSpec(
        region_entry_distance_m=1.0,
        region_exit_distance_m=2.0,
        finish_distance_m=3.0,
        target_speed_outside_m_s=1.0,
        target_speed_inside_m_s=0.5,
        posture_band_low_m=0.45,
        posture_band_high_m=0.62,
        horizon_steps=500,
    )


def _metrics(
    *,
    spec: CourseTaskSpec | None = None,
    before: np.ndarray | None = None,
    after: np.ndarray | None = None,
    contacts: tuple[str, ...] = (),
    reference: np.ndarray | None = None,
    control_step: int = 10,
):
    task = spec or _spec()
    initial = _qpos(0.0, 0.0)
    return evaluate_step(
        spec=task,
        frame=TaskFrame.initialize(initial[:2], initial[3:7]),
        before_qpos=_qpos(1.48, 0.0) if before is None else before,
        after_qpos=_qpos(1.50, 0.0) if after is None else after,
        ground_contact_bodies=list(contacts),
        current_reference=_reference() if reference is None else reference,
        control_step=control_step,
    )


def test_task_frame_projection_is_invariant_to_world_yaw_rotation() -> None:
    base = TaskFrame.initialize(np.asarray([1.0, 2.0]), _quaternion_wxyz())
    rotated = TaskFrame.initialize(np.asarray([-2.0, 1.0]), _quaternion_wxyz(yaw=math.pi / 2))

    base_projection = base.project(np.asarray([2.0, 2.5]), _quaternion_wxyz(yaw=0.2))
    rotated_projection = rotated.project(
        np.asarray([-2.5, 2.0]), _quaternion_wxyz(yaw=math.pi / 2 + 0.2)
    )

    assert rotated_projection.progress_m == pytest.approx(base_projection.progress_m)
    assert rotated_projection.lateral_m == pytest.approx(base_projection.lateral_m)
    assert rotated_projection.heading_error_rad == pytest.approx(base_projection.heading_error_rad)


def test_forward_progress_cannot_hide_lateral_drift() -> None:
    centered = _metrics(after=_qpos(1.50, 0.0))
    drifting = _metrics(after=_qpos(1.50, 1.25))

    assert drifting.progress_m == pytest.approx(centered.progress_m)
    assert drifting.forward_speed_m_s == pytest.approx(centered.forward_speed_m_s)
    assert centered.lateral_error_m == 0.0
    assert drifting.lateral_error_m == pytest.approx(1.25)


def test_low_robot_inside_posture_band_is_fallen_not_posture_success() -> None:
    spec = replace(_spec(), posture_band_low_m=0.20, posture_band_high_m=0.40)
    metrics = _metrics(spec=spec, after=_qpos(1.50, 0.0, 0.25))

    assert metrics.posture_band_error_m == 0.0
    assert metrics.fallen is True
    assert metrics.posture_success is False
    assert metrics.failure_reasons == ("root_height_below_0.30_m",)


def test_only_xml_ankle_roll_bodies_are_allowed_ground_contacts() -> None:
    feet = _metrics(contacts=tuple(sorted(ALLOWED_GROUND_CONTACT_BODIES)))
    torso = _metrics(contacts=("left_ankle_roll_link", "torso_link"))
    toppled = _metrics(after=_qpos(1.50, 0.0, pitch=math.pi / 2))

    assert feet.fallen is False
    assert torso.fallen is True
    assert torso.failure_reasons == ("non_foot_ground_contact",)
    assert toppled.fallen is True
    assert toppled.failure_reasons == ("torso_up_below_0.5",)


def test_zero_task_recipe_leaves_fixed_tracking_reward_only() -> None:
    metrics = _metrics(reference=_reference(height=0.60, roll=0.1, pitch=-0.1))
    recipe = TaskRewardRecipe(0.0, 0.0, 0.0, 0.0, 0.0)
    result = reward(spec=_spec(), recipe=recipe, metrics=metrics)

    assert 0.0 < result.tracking_reward <= 1.0
    assert result.task_reward == 0.0
    assert result.total_reward == result.tracking_reward
    assert all(
        value >= 0.0
        for value in (
            result.speed_component_reward,
            result.posture_component_reward,
            result.lateral_component_reward,
            result.heading_component_reward,
        )
    )
    assert result.failure_penalty == 0.0


def test_fall_gates_all_positive_components_and_tracking_reward() -> None:
    metrics = _metrics(contacts=("torso_link",))
    recipe = TaskRewardRecipe(1.0, 2.0, 3.0, 4.0, 5.0)
    result = reward(spec=_spec(), recipe=recipe, metrics=metrics)

    assert result.tracking_reward == 0.0
    assert result.speed_component_reward == 0.0
    assert result.posture_component_reward == 0.0
    assert result.lateral_component_reward == 0.0
    assert result.heading_component_reward == 0.0
    assert result.failure_penalty == -5.0
    assert result.task_reward == result.total_reward == -5.0


def test_satisfied_posture_component_has_no_region_occupancy_bonus() -> None:
    recipe = TaskRewardRecipe(0.0, 5.0, 0.0, 0.0, 0.0)
    inside = reward(spec=_spec(), recipe=recipe, metrics=_metrics())
    outside_metrics = _metrics(
        before=_qpos(0.48, 0.0),
        after=_qpos(0.50, 0.0),
    )
    outside = reward(spec=_spec(), recipe=recipe, metrics=outside_metrics)

    assert inside.posture_component_reward == 1.0
    assert outside_metrics.inside_posture_region is False
    assert outside.posture_component_reward == inside.posture_component_reward
    assert outside.task_reward == inside.task_reward


def test_reward_v2_disabled_is_numerically_exact_v1() -> None:
    v1 = TaskRewardRecipe(1.0, 2.0, 3.0, 4.0, 5.0)
    v2 = TaskRewardRecipe(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        recipe_version=2,
        depth_strength=0.0,
        ceiling_fraction=0.6,
    )
    fields = (
        "tracking_reward",
        "task_reward",
        "total_reward",
        "speed_component_reward",
        "posture_component_reward",
        "lateral_component_reward",
        "heading_component_reward",
        "failure_penalty",
    )
    cases = (
        _metrics(after=_qpos(1.50, 0.0, 0.55)),
        _metrics(after=_qpos(1.50, 0.0, 0.62)),
        _metrics(before=_qpos(0.48, 0.0), after=_qpos(0.50, 0.0)),
        _metrics(contacts=("torso_link",)),
    )
    for metrics in cases:
        old = reward(spec=_spec(), recipe=v1, metrics=metrics)
        migrated = reward(spec=_spec(), recipe=v2, metrics=metrics)
        assert tuple(getattr(migrated, field) for field in fields) == tuple(
            getattr(old, field) for field in fields
        )
        assert migrated.task_reward_recipe_sha256 != old.task_reward_recipe_sha256


def test_reward_v2_depth_ceiling_only_reduces_inside_posture_component() -> None:
    spec = _spec()
    v1 = TaskRewardRecipe(0.0, 2.0, 0.0, 0.0, 0.0)
    v2 = TaskRewardRecipe(
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        recipe_version=2,
        depth_strength=1.0,
        ceiling_fraction=0.6,
    )
    ceiling = spec.posture_band_low_m + 0.6 * (
        spec.posture_band_high_m - spec.posture_band_low_m
    )
    below = _metrics(spec=spec, after=_qpos(1.50, 0.0, ceiling - 0.01))
    above = _metrics(spec=spec, after=_qpos(1.50, 0.0, ceiling + 0.05))
    outside = _metrics(
        spec=spec,
        before=_qpos(0.48, 0.0, ceiling + 0.05),
        after=_qpos(0.50, 0.0, ceiling + 0.05),
    )

    below_result = reward(spec=spec, recipe=v2, metrics=below)
    above_result = reward(spec=spec, recipe=v2, metrics=above)
    outside_old = reward(spec=spec, recipe=v1, metrics=outside)
    outside_new = reward(spec=spec, recipe=v2, metrics=outside)

    assert below_result.posture_component_reward == 1.0
    assert above_result.posture_component_reward == pytest.approx(
        1.0 / (1.0 + (0.05 / TASK_REWARD_DEPTH_SCALE_M) ** 2)
    )
    assert 0.0 <= above_result.posture_component_reward < 1.0
    assert outside_new.posture_component_reward == outside_old.posture_component_reward == 1.0
    assert above_result.tracking_reward == reward(spec=spec, recipe=v1, metrics=above).tracking_reward


def test_reward_v2_depth_multiplier_is_monotone_bounded_and_fall_safe() -> None:
    spec = _spec()
    recipe = TaskRewardRecipe(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        recipe_version=2,
        depth_strength=1.0,
        ceiling_fraction=0.6,
    )
    heights = (0.56, 0.58, 0.60, 0.62)
    components = [
        reward(
            spec=spec,
            recipe=recipe,
            metrics=_metrics(spec=spec, after=_qpos(1.50, 0.0, height)),
        ).posture_component_reward
        for height in heights
    ]
    fallen = reward(spec=spec, recipe=recipe, metrics=_metrics(contacts=("torso_link",)))

    assert all(0.0 <= value <= 1.0 for value in components)
    assert components == sorted(components, reverse=True)
    assert fallen.tracking_reward == 0.0
    assert fallen.posture_component_reward == 0.0
    assert fallen.task_reward == fallen.total_reward == -5.0


def test_task_metrics_and_targets_do_not_depend_on_oracle_reference() -> None:
    matching = _metrics(reference=_reference())
    unrelated = np.full(30, 4.0, dtype=np.float64)
    different_reference = _metrics(reference=unrelated)
    task_fields = (
        "progress_m",
        "lateral_error_m",
        "heading_error_rad",
        "root_height_m",
        "forward_speed_m_s",
        "target_speed_m_s",
        "speed_error_m_s",
        "posture_band_error_m",
        "inside_posture_region",
        "posture_success",
        "fallen",
    )

    assert matching.target_speed_m_s == _spec().target_speed_inside_m_s
    assert tuple(getattr(matching, name) for name in task_fields) == tuple(
        getattr(different_reference, name) for name in task_fields
    )
    assert matching.joint_position_rmse_rad != different_reference.joint_position_rmse_rad
    assert matching.root_height_abs_error_m != different_reference.root_height_abs_error_m


def test_task_features_have_one_fixed_actor_and_critic_layout() -> None:
    spec = _spec()
    metrics = _metrics(spec=spec)
    features = metrics.task_features(spec)

    assert features.shape == (len(TASK_FEATURE_NAMES),)
    assert features.dtype == np.dtype("<f4")
    assert features[7] == 1.0
    with pytest.raises(ValueError, match="exact task specification"):
        metrics.task_features(replace(spec, finish_distance_m=3.5))

    reset = _metrics(
        spec=spec,
        before=_qpos(0.0, 0.0),
        after=_qpos(0.0, 0.0),
        control_step=0,
    )
    assert reset.task_features(spec)[-1] == 1.0


def test_finish_is_only_a_step_condition_not_an_episode_success_claim() -> None:
    metrics = _metrics(
        before=_qpos(2.98, 1.5),
        after=_qpos(3.01, 1.5),
    )

    assert metrics.finish_condition_met is True
    assert metrics.lateral_error_m == pytest.approx(1.5)
    assert metrics.episode_success is None


def test_task_and_recipe_identities_are_canonical_and_strict() -> None:
    spec = _spec()
    recipe = TaskRewardRecipe(1.0, 2.0, 3.0, 4.0, 5.0)

    assert CourseTaskSpec.from_dict(spec.to_dict()) == spec
    assert TaskRewardRecipe.from_dict(recipe.to_dict()) == recipe
    assert len(spec.sha256) == len(recipe.sha256) == 64


def test_v1_serialization_is_unchanged_and_v2_round_trips_exactly() -> None:
    v1 = TaskRewardRecipe(1.0, 2.0, 3.0, 4.0, 5.0)
    assert v1.to_dict() == {
        "speed_weight": 1.0,
        "posture_weight": 2.0,
        "lateral_weight": 3.0,
        "heading_weight": 4.0,
        "failure_weight": 5.0,
        "schema_id": TASK_REWARD_RECIPE_ID,
        "schema_version": 1,
    }
    v2 = TaskRewardRecipe(
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        recipe_version=2,
        depth_strength=0.75,
        ceiling_fraction=0.6,
    )
    assert v2.to_dict()["schema_id"] == TASK_REWARD_RECIPE_V2_ID
    assert v2.to_dict()["schema_version"] == 2
    assert TaskRewardRecipe.from_dict(v2.to_dict()) == v2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("region_entry_distance_m", 2.0),
        ("finish_distance_m", 101.0),
        ("target_speed_inside_m_s", 5.1),
        ("posture_band_high_m", 0.29),
        ("horizon_steps", True),
        ("target_speed_outside_m_s", math.nan),
        ("posture_band_low_m", False),
    ],
)
def test_invalid_task_spec_values_fail_closed(field: str, value: object) -> None:
    raw = _spec().to_dict()
    raw[field] = value
    with pytest.raises(ValueError):
        CourseTaskSpec.from_dict(raw)


def test_task_spec_rejects_unknown_or_missing_posture_fields() -> None:
    unknown = _spec().to_dict()
    unknown["oracle_mode"] = "crouch"
    missing = _spec().to_dict()
    del missing["posture_band_high_m"]
    with pytest.raises(ValueError, match="fields differ"):
        CourseTaskSpec.from_dict(unknown)
    with pytest.raises(ValueError, match="fields differ"):
        CourseTaskSpec.from_dict(missing)


@pytest.mark.parametrize("value", [-0.1, 5.1, math.inf, math.nan, True])
def test_invalid_task_reward_weights_fail_closed(value: object) -> None:
    raw = TaskRewardRecipe(0.0, 0.0, 0.0, 0.0, 0.0).to_dict()
    raw["speed_weight"] = value
    with pytest.raises(ValueError):
        TaskRewardRecipe.from_dict(raw)


def test_task_reward_recipe_rejects_unknown_and_boolean_schema_version() -> None:
    unknown = TaskRewardRecipe(0.0, 0.0, 0.0, 0.0, 0.0).to_dict()
    unknown["tracking_weight"] = 1.0
    with pytest.raises(ValueError, match="fields differ"):
        TaskRewardRecipe.from_dict(unknown)
    wrong_version = TaskRewardRecipe(0.0, 0.0, 0.0, 0.0, 0.0).to_dict()
    wrong_version["schema_version"] = True
    with pytest.raises(ValueError, match="schema identity"):
        TaskRewardRecipe.from_dict(wrong_version)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("depth_strength", -0.1),
        ("depth_strength", 1.1),
        ("depth_strength", math.nan),
        ("depth_strength", True),
        ("ceiling_fraction", -0.1),
        ("ceiling_fraction", 1.1),
        ("ceiling_fraction", math.inf),
        ("ceiling_fraction", False),
    ],
)
def test_reward_v2_rejects_unbounded_or_nonfloat_depth_settings(
    field: str, value: object
) -> None:
    raw = TaskRewardRecipe(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        recipe_version=2,
        depth_strength=0.0,
        ceiling_fraction=0.6,
    ).to_dict()
    raw[field] = value
    with pytest.raises(ValueError, match=r"depth|finite"):
        TaskRewardRecipe.from_dict(raw)


def test_reward_versions_reject_cross_schema_or_hidden_fields() -> None:
    v1 = TaskRewardRecipe(0.0, 0.0, 0.0, 0.0, 0.0).to_dict()
    v1["depth_strength"] = 0.0
    with pytest.raises(ValueError, match="fields differ"):
        TaskRewardRecipe.from_dict(v1)

    v2 = TaskRewardRecipe(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        recipe_version=2,
        depth_strength=0.0,
        ceiling_fraction=0.6,
    ).to_dict()
    del v2["ceiling_fraction"]
    with pytest.raises(ValueError, match="fields differ"):
        TaskRewardRecipe.from_dict(v2)
