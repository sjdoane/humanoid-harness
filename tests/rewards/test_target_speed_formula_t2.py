from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from oracle_composition.rewards.contract import CandidateTaskInputsV1
from oracle_composition.rewards.target_speed_formula import (
    FORMULA_ID,
    TargetSpeedFormulaError,
    TargetSpeedFormulaRecipeV1,
    parse_target_speed_formula_recipe,
)
from oracle_composition.rewards.target_speed_formula_t2 import (
    FORMULA_RUNTIME_ID,
    PARSER_ID,
    TargetSpeedFormulaT2Error,
    evaluate_target_speed_formula_t2,
    target_speed_formula_t2_bounds,
)
from oracle_composition.rewards.task_inputs_v2 import CandidateTaskInputsV2

ROOT = Path(__file__).parents[2]


def _recipe(alpha: float = 1.0, beta: float = 0.0) -> TargetSpeedFormulaRecipeV1:
    return TargetSpeedFormulaRecipeV1(FORMULA_ID, alpha, beta)


@pytest.mark.parametrize(
    ("velocity", "expected"),
    [
        (3.0, 1.25),
        (1.5, 0.625),
        (0.0, 0.0),
        (-1.0, 0.0),
        (1e308, 0.0),
        (-1e308, 0.0),
    ],
)
def test_fixed_target_triangular_values_are_exact_and_finite(
    velocity: float, expected: float
) -> None:
    result = evaluate_target_speed_formula_t2(_recipe(), CandidateTaskInputsV2(velocity, 3.0))
    assert result == expected
    assert type(result) is float
    assert math.isfinite(result)
    assert -10.0 <= result <= 15.0


@pytest.mark.parametrize("alpha", [0.25, 4.0])
@pytest.mark.parametrize("beta", [-10.0, 10.0])
def test_all_parameter_endpoint_combinations(alpha: float, beta: float) -> None:
    at_target = evaluate_target_speed_formula_t2(
        _recipe(alpha, beta), CandidateTaskInputsV2(3.0, 3.0)
    )
    saturated = evaluate_target_speed_formula_t2(
        _recipe(alpha, beta), CandidateTaskInputsV2(1e308, 3.0)
    )
    assert at_target == alpha * 1.25 + beta
    assert saturated == beta
    assert type(at_target) is float
    assert type(saturated) is float
    assert math.isfinite(at_target)
    assert math.isfinite(saturated)
    assert -10.0 <= at_target <= 15.0
    assert -10.0 <= saturated <= 15.0


def test_wrong_duplicate_v1_subclass_and_incomplete_inputs_are_refused() -> None:
    class DuplicateInputs:
        def __init__(self) -> None:
            self.com_x_velocity_m_s = 3.0
            self.target_speed_m_s = 3.0

    class InputSubclass(CandidateTaskInputsV2):
        pass

    invalid_inputs: list[object] = [
        CandidateTaskInputsV1(1.0, 1.0),
        DuplicateInputs(),
        object.__new__(InputSubclass),
        object.__new__(CandidateTaskInputsV2),
    ]
    wrong_target = object.__new__(CandidateTaskInputsV2)
    object.__setattr__(wrong_target, "com_x_velocity_m_s", 3.0)
    object.__setattr__(wrong_target, "target_speed_m_s", 1.0)
    invalid_inputs.append(wrong_target)

    for inputs in invalid_inputs:
        with pytest.raises(TargetSpeedFormulaT2Error):
            evaluate_target_speed_formula_t2(_recipe(), inputs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("com_x_velocity_m_s", True),
        ("com_x_velocity_m_s", math.nan),
        ("com_x_velocity_m_s", math.inf),
        ("com_x_velocity_m_s", 10**400),
        ("target_speed_m_s", "3.0"),
        ("target_speed_m_s", 3.0001),
    ],
)
def test_forged_input_fields_are_normalized_to_t2_errors(field: str, value: object) -> None:
    inputs = object.__new__(CandidateTaskInputsV2)
    object.__setattr__(inputs, "com_x_velocity_m_s", 3.0)
    object.__setattr__(inputs, "target_speed_m_s", 3.0)
    object.__setattr__(inputs, field, value)
    with pytest.raises(TargetSpeedFormulaT2Error):
        evaluate_target_speed_formula_t2(_recipe(), inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("formula_id", "unknown/v1"),
        ("alpha", True),
        ("alpha", math.nan),
        ("alpha", math.inf),
        ("alpha", 0.249),
        ("beta", -math.inf),
        ("beta", 10.001),
    ],
)
def test_forged_recipe_fields_are_normalized_to_t2_errors(field: str, value: object) -> None:
    recipe = object.__new__(TargetSpeedFormulaRecipeV1)
    object.__setattr__(recipe, "formula_id", FORMULA_ID)
    object.__setattr__(recipe, "alpha", 1.0)
    object.__setattr__(recipe, "beta", 0.0)
    object.__setattr__(recipe, field, value)
    with pytest.raises(TargetSpeedFormulaT2Error):
        evaluate_target_speed_formula_t2(recipe, CandidateTaskInputsV2(3.0, 3.0))


def test_recipe_subclass_and_incomplete_recipe_are_refused() -> None:
    class RecipeSubclass(TargetSpeedFormulaRecipeV1):
        pass

    with pytest.raises(TargetSpeedFormulaT2Error, match="exact"):
        evaluate_target_speed_formula_t2(
            RecipeSubclass(FORMULA_ID, 1.0, 0.0),
            CandidateTaskInputsV2(3.0, 3.0),
        )
    with pytest.raises(TargetSpeedFormulaT2Error, match="incomplete"):
        evaluate_target_speed_formula_t2(
            object.__new__(TargetSpeedFormulaRecipeV1),
            CandidateTaskInputsV2(3.0, 3.0),
        )


def test_hostile_values_and_objects_never_receive_callbacks() -> None:
    callbacks: list[str] = []

    class HostileFloat(float):
        def __float__(self) -> float:
            callbacks.append("float")
            raise AssertionError("numeric conversion callback ran")

    class HostileStr(str):
        def __eq__(self, other: object) -> bool:
            callbacks.append("eq")
            raise AssertionError("string comparison callback ran")

    class HostileInputs:
        @property
        def com_x_velocity_m_s(self) -> float:
            callbacks.append("velocity")
            raise AssertionError("input property callback ran")

    class HostileInputSubclass(CandidateTaskInputsV2):
        def __getattribute__(self, name: str) -> object:
            callbacks.append(name)
            raise AssertionError("input attribute callback ran")

    class HostileRecipeSubclass(TargetSpeedFormulaRecipeV1):
        def __getattribute__(self, name: str) -> object:
            callbacks.append(name)
            raise AssertionError("recipe attribute callback ran")

    forged_input = object.__new__(CandidateTaskInputsV2)
    object.__setattr__(forged_input, "com_x_velocity_m_s", HostileFloat(3.0))
    object.__setattr__(forged_input, "target_speed_m_s", 3.0)

    forged_unknown_recipe = object.__new__(TargetSpeedFormulaRecipeV1)
    object.__setattr__(forged_unknown_recipe, "formula_id", "unknown/v1")
    object.__setattr__(forged_unknown_recipe, "alpha", HostileFloat(1.0))
    object.__setattr__(forged_unknown_recipe, "beta", 0.0)

    forged_number_recipe = object.__new__(TargetSpeedFormulaRecipeV1)
    object.__setattr__(forged_number_recipe, "formula_id", FORMULA_ID)
    object.__setattr__(forged_number_recipe, "alpha", HostileFloat(1.0))
    object.__setattr__(forged_number_recipe, "beta", 0.0)

    forged_string_recipe = object.__new__(TargetSpeedFormulaRecipeV1)
    object.__setattr__(forged_string_recipe, "formula_id", HostileStr(FORMULA_ID))
    object.__setattr__(forged_string_recipe, "alpha", 1.0)
    object.__setattr__(forged_string_recipe, "beta", 0.0)

    cases = [
        (_recipe(), HostileInputs()),
        (forged_unknown_recipe, HostileInputs()),
        (_recipe(), object.__new__(HostileInputSubclass)),
        (forged_number_recipe, CandidateTaskInputsV2(3.0, 3.0)),
        (forged_string_recipe, CandidateTaskInputsV2(3.0, 3.0)),
        (
            object.__new__(HostileRecipeSubclass),
            CandidateTaskInputsV2(3.0, 3.0),
        ),
        (_recipe(), forged_input),
    ]
    for recipe, inputs in cases:
        with pytest.raises(TargetSpeedFormulaT2Error):
            evaluate_target_speed_formula_t2(recipe, inputs)  # type: ignore[arg-type]
    assert callbacks == []


def test_existing_parser_feeds_t2_and_runtime_id_is_not_a_recipe_id() -> None:
    recipe = parse_target_speed_formula_recipe(
        b'{"formula_id":"target_speed_triangular_affine/v1","alpha":2,"beta":-1}'
    )
    assert evaluate_target_speed_formula_t2(recipe, CandidateTaskInputsV2(3.0, 3.0)) == 1.5
    assert FORMULA_RUNTIME_ID == "target_speed_triangular_affine_t2_adapter/v1"
    assert PARSER_ID == "target_speed_triangular_affine_recipe/v1"
    assert FORMULA_RUNTIME_ID != FORMULA_ID
    with pytest.raises(TargetSpeedFormulaError):
        parse_target_speed_formula_recipe(
            json.dumps(
                {"formula_id": FORMULA_RUNTIME_ID, "alpha": 2, "beta": -1},
                separators=(",", ":"),
            ).encode("utf-8")
        )


def test_bounds_are_exact_fresh_and_deterministic() -> None:
    expected = {
        "parameters": {
            "alpha": {"minimum": 0.25, "maximum": 4.0},
            "beta": {"minimum": -10.0, "maximum": 10.0},
        },
        "r_task": {"minimum": -10.0, "maximum": 15.0},
    }
    first = target_speed_formula_t2_bounds()
    second = target_speed_formula_t2_bounds()
    assert first == expected
    assert second == expected
    assert first is not second
    assert first["parameters"] is not second["parameters"]
    assert json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False) == (
        '{"parameters":{"alpha":{"maximum":4.0,"minimum":0.25},'
        '"beta":{"maximum":10.0,"minimum":-10.0}},'
        '"r_task":{"maximum":15.0,"minimum":-10.0}}'
    )
    first["parameters"]["alpha"]["minimum"] = 99  # type: ignore[index]
    assert target_speed_formula_t2_bounds() == expected


def test_module_uses_only_fixed_trusted_imports_and_no_dynamic_execution() -> None:
    source = (ROOT / "src/oracle_composition/rewards/target_speed_formula_t2.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add("." * node.level + (node.module or ""))
    assert imported_modules == {
        "__future__",
        "math",
        ".target_speed_formula",
        ".task_inputs_v2",
    }

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called_names & {
        "__import__",
        "compile",
        "eval",
        "exec",
        "open",
    }
    assert not called_attributes & {
        "import_module",
        "read_bytes",
        "read_text",
        "run",
        "system",
        "write_bytes",
        "write_text",
    }
