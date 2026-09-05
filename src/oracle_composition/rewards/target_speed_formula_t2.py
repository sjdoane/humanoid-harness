"""Pure target-speed formula adapter for accepted T2 inputs."""

from __future__ import annotations

import math

from .target_speed_formula import (
    ALPHA_MAX,
    ALPHA_MIN,
    BASE_SCALE,
    BETA_MAX,
    BETA_MIN,
    FORMULA_ID,
    OUTPUT_MAX,
    OUTPUT_MIN,
    TargetSpeedFormulaError,
    TargetSpeedFormulaRecipeV1,
)
from .task_inputs_v2 import (
    CandidateTaskInputsV2,
    TaskInputsV2Error,
    validate_task_inputs_v2,
)

FORMULA_RUNTIME_ID = "target_speed_triangular_affine_t2_adapter/v1"
PARSER_ID = "target_speed_triangular_affine_recipe/v1"


class TargetSpeedFormulaT2Error(ValueError):
    """A recipe or T2 input violates the pure adapter boundary."""


def _validated_recipe_copy(
    recipe: TargetSpeedFormulaRecipeV1,
) -> TargetSpeedFormulaRecipeV1:
    if type(recipe) is not TargetSpeedFormulaRecipeV1:
        raise TargetSpeedFormulaT2Error("recipe must be an exact TargetSpeedFormulaRecipeV1")
    try:
        formula_id = object.__getattribute__(recipe, "formula_id")
        alpha = object.__getattribute__(recipe, "alpha")
        beta = object.__getattribute__(recipe, "beta")
    except AttributeError as exc:
        raise TargetSpeedFormulaT2Error("recipe slots are incomplete") from exc
    if type(formula_id) is not str or formula_id != FORMULA_ID:
        raise TargetSpeedFormulaT2Error("recipe has an unknown formula identifier")
    try:
        return TargetSpeedFormulaRecipeV1(
            formula_id=formula_id,
            alpha=alpha,
            beta=beta,
        )
    except TargetSpeedFormulaError as exc:
        raise TargetSpeedFormulaT2Error("recipe fields are invalid or forged") from exc


def evaluate_target_speed_formula_t2(
    recipe: TargetSpeedFormulaRecipeV1,
    inputs: CandidateTaskInputsV2,
) -> float:
    """Evaluate the accepted affine recipe against fixed-target T2 inputs."""

    validated_recipe = _validated_recipe_copy(recipe)
    try:
        velocity, target = validate_task_inputs_v2(inputs)
    except TaskInputsV2Error as exc:
        raise TargetSpeedFormulaT2Error("T2 inputs are invalid or forged") from exc

    error = abs(velocity - target)
    normalized_error = 1.0 if error >= target else error / target
    base = BASE_SCALE * (1.0 - min(1.0, normalized_error))
    output = validated_recipe.alpha * base + validated_recipe.beta
    if (
        type(output) is not float
        or not math.isfinite(base)
        or not 0.0 <= base <= BASE_SCALE
        or not math.isfinite(output)
        or not OUTPUT_MIN <= output <= OUTPUT_MAX
    ):
        raise TargetSpeedFormulaT2Error("trusted T2 formula output violates its finite envelope")
    return output


def target_speed_formula_t2_bounds() -> dict[str, object]:
    """Return fresh data-only bounds for the fixed T2 formula family."""

    return {
        "parameters": {
            "alpha": {"minimum": ALPHA_MIN, "maximum": ALPHA_MAX},
            "beta": {"minimum": BETA_MIN, "maximum": BETA_MAX},
        },
        "r_task": {"minimum": OUTPUT_MIN, "maximum": OUTPUT_MAX},
    }


__all__ = [
    "FORMULA_RUNTIME_ID",
    "PARSER_ID",
    "TargetSpeedFormulaT2Error",
    "evaluate_target_speed_formula_t2",
    "target_speed_formula_t2_bounds",
]
