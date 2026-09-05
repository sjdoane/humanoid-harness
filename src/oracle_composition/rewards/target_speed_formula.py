"""Trusted data-only target-speed formula."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .contract import TARGET_SPEEDS_M_S, CandidateTaskInputsV1, canonical_json_bytes

FORMULA_ID = "target_speed_triangular_affine/v1"
MAX_RECIPE_BYTES = 1_024
ALPHA_MIN = 0.25
ALPHA_MAX = 4.0
BETA_MIN = -10.0
BETA_MAX = 10.0
BASE_SCALE = 1.25
OUTPUT_MIN = -10.0
OUTPUT_MAX = 15.0
RECIPE_KEYS = frozenset(("formula_id", "alpha", "beta"))


class TargetSpeedFormulaError(ValueError):
    """A recipe or trusted input violates the formula boundary."""


def _safe_float(value: object, *, field: str) -> float:
    # Exact built-ins exclude bools and numeric subclasses with conversion hooks.
    if type(value) not in (int, float):
        raise TargetSpeedFormulaError(f"{field} must be an exact JSON number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise TargetSpeedFormulaError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise TargetSpeedFormulaError(f"{field} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class TargetSpeedFormulaRecipeV1:
    formula_id: str
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if type(self.formula_id) is not str or self.formula_id != FORMULA_ID:
            raise TargetSpeedFormulaError(f"formula_id must equal {FORMULA_ID!r}")
        alpha = _safe_float(self.alpha, field="alpha")
        beta = _safe_float(self.beta, field="beta")
        if not ALPHA_MIN <= alpha <= ALPHA_MAX:
            raise TargetSpeedFormulaError("alpha is outside [0.25, 4.0]")
        if not BETA_MIN <= beta <= BETA_MAX:
            raise TargetSpeedFormulaError("beta is outside [-10.0, 10.0]")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    def to_dict(self) -> dict[str, object]:
        return {"formula_id": self.formula_id, "alpha": self.alpha, "beta": self.beta}

    @property
    def canonical_bytes(self) -> bytes:
        encoded = canonical_json_bytes(self.to_dict())
        if len(encoded) > MAX_RECIPE_BYTES:
            raise TargetSpeedFormulaError("canonical recipe exceeds 1,024 UTF-8 bytes")
        return encoded


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TargetSpeedFormulaError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise TargetSpeedFormulaError(f"non-finite JSON constant {value!r}")


def parse_target_speed_formula_recipe(encoded: bytes) -> TargetSpeedFormulaRecipeV1:
    """Parse bounded UTF-8 JSON without accepting executable or callback-bearing values."""

    if type(encoded) is not bytes or not encoded or len(encoded) > MAX_RECIPE_BYTES:
        raise TargetSpeedFormulaError("recipe must be exact nonempty bytes at most 1,024 bytes")
    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except TargetSpeedFormulaError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TargetSpeedFormulaError("recipe is not valid bounded UTF-8 JSON") from exc
    if type(value) is not dict or set(value) != RECIPE_KEYS:
        raise TargetSpeedFormulaError("recipe keys must be exactly formula_id, alpha, and beta")
    return TargetSpeedFormulaRecipeV1(
        formula_id=value["formula_id"],  # type: ignore[arg-type]
        alpha=value["alpha"],  # type: ignore[arg-type]
        beta=value["beta"],  # type: ignore[arg-type]
    )


def evaluate_target_speed_formula(
    recipe: TargetSpeedFormulaRecipeV1,
    inputs: CandidateTaskInputsV1,
) -> float:
    """Evaluate the fixed triangular base and its bounded affine parameters."""

    if type(recipe) is not TargetSpeedFormulaRecipeV1:
        raise TargetSpeedFormulaError("recipe must be an exact TargetSpeedFormulaRecipeV1")
    if type(recipe.formula_id) is not str or recipe.formula_id != FORMULA_ID:
        raise TargetSpeedFormulaError("recipe was forged or has an unknown formula identifier")
    alpha = _safe_float(recipe.alpha, field="alpha")
    beta = _safe_float(recipe.beta, field="beta")
    if not ALPHA_MIN <= alpha <= ALPHA_MAX or not BETA_MIN <= beta <= BETA_MAX:
        raise TargetSpeedFormulaError("recipe was forged outside the parameter envelope")

    if type(inputs) is not CandidateTaskInputsV1:
        raise TargetSpeedFormulaError("inputs must be an exact CandidateTaskInputsV1")
    velocity = inputs.com_x_velocity_m_s
    target = inputs.target_speed_m_s
    if type(velocity) is not float or not math.isfinite(velocity):
        raise TargetSpeedFormulaError("com_x_velocity_m_s must be an exact finite float")
    if type(target) is not float or not math.isfinite(target) or target not in TARGET_SPEEDS_M_S:
        raise TargetSpeedFormulaError("target_speed_m_s differs from the frozen B0 targets")

    error = abs(velocity - target)
    # Saturating before division preserves the declared formula without overflowing at large v.
    normalized_error = 1.0 if error >= target else error / target
    base = BASE_SCALE * (1.0 - min(1.0, normalized_error))
    output = alpha * base + beta
    if (
        not math.isfinite(base)
        or not 0.0 <= base <= BASE_SCALE
        or not math.isfinite(output)
        or not OUTPUT_MIN <= output <= OUTPUT_MAX
    ):
        raise TargetSpeedFormulaError("trusted formula output violates its finite envelope")
    return output


__all__ = [
    "ALPHA_MAX",
    "ALPHA_MIN",
    "BASE_SCALE",
    "BETA_MAX",
    "BETA_MIN",
    "FORMULA_ID",
    "MAX_RECIPE_BYTES",
    "OUTPUT_MAX",
    "OUTPUT_MIN",
    "TargetSpeedFormulaError",
    "TargetSpeedFormulaRecipeV1",
    "evaluate_target_speed_formula",
    "parse_target_speed_formula_recipe",
]
