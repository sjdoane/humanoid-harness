"""Load the immutable artifact helpers without simulator package side effects."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

_ALIAS = "oracle_composition._reward_search_experiments"
_EXPECTED_EXPERIMENTS = (Path(__file__).resolve().parents[1] / "experiments").resolve(strict=True)
_EXPECTED_ARTIFACT_IO = (_EXPECTED_EXPERIMENTS / "artifact_io.py").resolve(strict=True)


def _resolved_origin(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ImportError(f"{label} has no filesystem origin")
    try:
        return Path(value).resolve(strict=True)
    except OSError as exc:
        raise ImportError(f"{label} has an unreadable filesystem origin") from exc


if _ALIAS not in sys.modules:
    package = ModuleType(_ALIAS)
    # The canonical experiments initializer imports Gym; A1 intentionally has no simulator extra.
    package.__path__ = [str(_EXPECTED_EXPERIMENTS)]  # type: ignore[attr-defined]
    package.__package__ = _ALIAS
    sys.modules[_ALIAS] = package
else:
    package = sys.modules[_ALIAS]

package_paths = getattr(package, "__path__", None)
try:
    resolved_package_paths = tuple(Path(item).resolve(strict=True) for item in package_paths)
except (OSError, TypeError) as exc:
    raise ImportError("reward-search publication alias has an unexpected origin") from exc
if resolved_package_paths != (_EXPECTED_EXPERIMENTS,):
    raise ImportError("reward-search publication alias has an unexpected origin")

_artifact_io = importlib.import_module(f"{_ALIAS}.artifact_io")
module_spec = getattr(_artifact_io, "__spec__", None)
if (
    _resolved_origin(getattr(_artifact_io, "__file__", None), label="artifact helper")
    != _EXPECTED_ARTIFACT_IO
    or _resolved_origin(getattr(module_spec, "origin", None), label="artifact helper spec")
    != _EXPECTED_ARTIFACT_IO
):
    raise ImportError("reward-search artifact helper has an unexpected origin")

PublishedArtifact = _artifact_io.PublishedArtifact
finite_pretty_json = _artifact_io.finite_pretty_json
publish_bytes_without_overwrite = _artifact_io.publish_bytes_without_overwrite
publish_json_without_overwrite = _artifact_io.publish_json_without_overwrite

__all__ = [
    "PublishedArtifact",
    "finite_pretty_json",
    "publish_bytes_without_overwrite",
    "publish_json_without_overwrite",
]
