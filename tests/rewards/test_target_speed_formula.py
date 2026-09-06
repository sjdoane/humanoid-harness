from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from oracle_composition.rewards.contract import CandidateTaskInputsV1, RewardContractError
from oracle_composition.rewards.target_speed_formula import (
    FORMULA_ID,
    MAX_RECIPE_BYTES,
    TargetSpeedFormulaError,
    TargetSpeedFormulaRecipeV1,
    evaluate_target_speed_formula,
    parse_target_speed_formula_recipe,
)

ROOT = Path(__file__).parents[2]


def _recipe(alpha: float = 1.0, beta: float = 0.0) -> TargetSpeedFormulaRecipeV1:
    return TargetSpeedFormulaRecipeV1(formula_id=FORMULA_ID, alpha=alpha, beta=beta)


@pytest.mark.parametrize(
    ("velocity", "expected"),
    [(-1.0, 0.0), (0.0, 0.0), (1.0, 1.25), (2.0, 0.0)],
)
def test_trusted_unit_evidence_at_one_metre_per_second(velocity: float, expected: float) -> None:
    inputs = CandidateTaskInputsV1(velocity, 1.0)
    assert evaluate_target_speed_formula(_recipe(), inputs) == expected


@pytest.mark.parametrize("target", [0.5, 1.0, 1.5])
def test_all_and_only_frozen_b0_targets_use_com_x_velocity(target: float) -> None:
    assert evaluate_target_speed_formula(_recipe(), CandidateTaskInputsV1(target, target)) == 1.25
    with pytest.raises(RewardContractError, match="one of"):
        CandidateTaskInputsV1(5.520768616125457, 5.520768616125457)


def test_affine_boundaries_and_finite_output_envelope() -> None:
    at_target = CandidateTaskInputsV1(0.5, 0.5)
    assert evaluate_target_speed_formula(_recipe(0.25, -10.0), at_target) == -9.6875
    assert evaluate_target_speed_formula(_recipe(4.0, 10.0), at_target) == 15.0
    assert (
        evaluate_target_speed_formula(_recipe(4.0, -10.0), CandidateTaskInputsV1(1e308, 0.5))
        == -10.0
    )


@pytest.mark.parametrize(
    "encoded",
    [
        b"{}",
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":1.0}',
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":1.0,"beta":0.0,"x":1}',
        b'{"formula_id":"unknown/v1","alpha":1.0,"beta":0.0}',
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":true,"beta":0.0}',
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":0.249,"beta":0.0}',
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":4.001,"beta":0.0}',
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":1.0,"beta":-10.01}',
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":1.0,"beta":10.01}',
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":NaN,"beta":0.0}',
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":Infinity,"beta":0.0}',
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":1.0,"alpha":2.0,"beta":0.0}',
        b"not-json",
        b"\xff",
    ],
)
def test_recipe_json_rejects_wrong_shape_identity_type_bounds_and_encoding(
    encoded: bytes,
) -> None:
    with pytest.raises(TargetSpeedFormulaError):
        parse_target_speed_formula_recipe(encoded)


def test_recipe_is_bounded_and_canonicalized_without_source_or_expression_fields() -> None:
    recipe = parse_target_speed_formula_recipe(
        b'{ "beta": 0, "alpha": 1, "formula_id": "target_speed_triangular_affine/v1" }'
    )
    assert recipe.canonical_bytes == (
        b'{"alpha":1.0,"beta":0.0,"formula_id":"target_speed_triangular_affine/v1"}'
    )
    assert set(json.loads(recipe.canonical_bytes)) == {"formula_id", "alpha", "beta"}
    with pytest.raises(TargetSpeedFormulaError, match="1,024"):
        parse_target_speed_formula_recipe(b" " * (MAX_RECIPE_BYTES + 1))
    deep = b"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
    with pytest.raises(TargetSpeedFormulaError):
        parse_target_speed_formula_recipe(deep)


def test_recipe_and_input_callback_canaries_are_rejected_without_touching_callbacks() -> None:
    calls: list[str] = []

    class HostileBytes(bytes):
        def decode(self, *_args: object, **_kwargs: object) -> str:
            calls.append("decode")
            raise AssertionError("hostile bytes callback invoked")

    class HostileNumber:
        def __float__(self) -> float:
            calls.append("float")
            raise AssertionError("hostile number callback invoked")

    class HostileInputs:
        @property
        def com_x_velocity_m_s(self) -> float:
            calls.append("velocity")
            raise AssertionError("hostile input property invoked")

    with pytest.raises(TargetSpeedFormulaError, match="exact nonempty bytes"):
        parse_target_speed_formula_recipe(HostileBytes(b"{}"))
    with pytest.raises(TargetSpeedFormulaError, match="exact JSON number"):
        TargetSpeedFormulaRecipeV1(FORMULA_ID, HostileNumber(), 0.0)  # type: ignore[arg-type]
    with pytest.raises(TargetSpeedFormulaError, match="exact CandidateTaskInputsV1"):
        evaluate_target_speed_formula(_recipe(), HostileInputs())  # type: ignore[arg-type]

    forged = object.__new__(CandidateTaskInputsV1)
    object.__setattr__(forged, "com_x_velocity_m_s", HostileNumber())
    object.__setattr__(forged, "target_speed_m_s", 1.0)
    with pytest.raises(TargetSpeedFormulaError, match="exact finite float"):
        evaluate_target_speed_formula(_recipe(), forged)
    assert calls == []


def test_subclasses_and_forged_recipe_are_refused() -> None:
    class InputSubclass(CandidateTaskInputsV1):
        pass

    class RecipeSubclass(TargetSpeedFormulaRecipeV1):
        pass

    with pytest.raises(TargetSpeedFormulaError, match="exact CandidateTaskInputsV1"):
        evaluate_target_speed_formula(_recipe(), InputSubclass(1.0, 1.0))
    with pytest.raises(TargetSpeedFormulaError, match="exact TargetSpeedFormulaRecipeV1"):
        evaluate_target_speed_formula(
            RecipeSubclass(FORMULA_ID, 1.0, 0.0), CandidateTaskInputsV1(1, 1)
        )

    forged = object.__new__(TargetSpeedFormulaRecipeV1)
    object.__setattr__(forged, "formula_id", FORMULA_ID)
    object.__setattr__(forged, "alpha", math.nan)
    object.__setattr__(forged, "beta", 0.0)
    with pytest.raises(TargetSpeedFormulaError, match="finite"):
        evaluate_target_speed_formula(forged, CandidateTaskInputsV1(1.0, 1.0))


def test_config_records_incompatible_root_speed_profile_and_absent_compositor() -> None:
    config = json.loads(
        (ROOT / "experiments/family_b_target_speed_formula_v1/config.json").read_bytes()
    )
    assert config["read_contract"]["speed_measurement"] == "center_of_mass_x_velocity_m_s"
    assert config["read_contract"]["target_speeds_m_s"] == [0.5, 1.0, 1.5]
    assert config["composition_profile"]["speed_measurement"] == "root_x_velocity_m_s"
    assert config["composition_profile"]["state"] == "incompatible_pending_peer_resolution"
    assert config["compositor"]["state"] == "explicitly_absent_pending_peer_resolution"


def test_core_has_no_dynamic_code_or_heavy_runtime_imports() -> None:
    source = (ROOT / "src/oracle_composition/rewards/target_speed_formula.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not imported_roots & {"gymnasium", "mujoco", "torch"}
    assert not called_names & {"compile", "exec", "eval", "__import__"}
