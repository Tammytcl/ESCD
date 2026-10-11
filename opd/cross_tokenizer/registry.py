"""Registry for cross-tokenizer algorithms."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from importlib import metadata
from pathlib import Path
import pkgutil
from typing import Any, TypeVar

from .base import CrossTokenizerAlgorithm, CrossTokenizerCapabilities


AlgorithmType = TypeVar("AlgorithmType", bound=type[CrossTokenizerAlgorithm])
_ALGORITHMS: dict[str, type[CrossTokenizerAlgorithm]] = {}
_DISCOVERY_COMPLETE = False
_BUILTIN_MODULE_EXCLUSIONS = {"base", "registry", "runtime", "sglang_teacher"}
_ENTRY_POINT_GROUP = "multi_teacher_opd.cross_tokenizer"


def _load_entry_point(entry_point: Any) -> None:
    loaded = entry_point.load()
    if not isinstance(loaded, type) or not issubclass(loaded, CrossTokenizerAlgorithm):
        raise TypeError(
            f"Entry point {_ENTRY_POINT_GROUP}:{entry_point.name} ({entry_point.value}) must load "
            "a CrossTokenizerAlgorithm class"
        )
    register_cross_tokenizer(entry_point.name)(loaded)


def register_cross_tokenizer(name: str) -> Callable[[AlgorithmType], AlgorithmType]:
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Cross-tokenizer algorithm name cannot be empty")

    def decorator(algorithm_cls: AlgorithmType) -> AlgorithmType:
        if not issubclass(algorithm_cls, CrossTokenizerAlgorithm):
            raise TypeError(f"{algorithm_cls!r} must inherit CrossTokenizerAlgorithm")
        existing = _ALGORITHMS.get(normalized_name)
        if existing is not None and existing is not algorithm_cls:
            raise ValueError(f"Cross-tokenizer algorithm {normalized_name!r} is already registered")
        algorithm_cls.name = normalized_name
        _ALGORITHMS[normalized_name] = algorithm_cls
        return algorithm_cls

    return decorator


def discover_cross_tokenizers() -> None:
    """Import built-in modules and installed algorithm entry points once."""

    global _DISCOVERY_COMPLETE
    if _DISCOVERY_COMPLETE:
        return
    # Set the guard before imports because imported modules may ask the
    # registry for its current contents while registering themselves.
    _DISCOVERY_COMPLETE = True
    package_name = __package__ or "opd.cross_tokenizer"
    package_dir = Path(__file__).resolve().parent
    module_names = sorted(
        module.name
        for module in pkgutil.iter_modules([str(package_dir)])
        if module.name not in _BUILTIN_MODULE_EXCLUSIONS and not module.name.startswith("_")
    )
    for module_name in module_names:
        importlib.import_module(f"{package_name}.{module_name}")

    discovered = metadata.entry_points()
    entry_points = (
        discovered.select(group=_ENTRY_POINT_GROUP)
        if hasattr(discovered, "select")
        else discovered.get(_ENTRY_POINT_GROUP, ())
    )
    for entry_point in sorted(entry_points, key=lambda item: (item.name, item.value)):
        _load_entry_point(entry_point)


def available_cross_tokenizers() -> tuple[str, ...]:
    discover_cross_tokenizers()
    return tuple(sorted(_ALGORITHMS))


def cross_tokenizer_capabilities(name: str) -> CrossTokenizerCapabilities:
    """Return the immutable launch/runtime contract for one registered plugin."""

    discover_cross_tokenizers()
    normalized_name = name.strip().lower()
    algorithm_cls = _ALGORITHMS.get(normalized_name)
    if algorithm_cls is None:
        available = ", ".join(sorted(_ALGORITHMS)) or "<none>"
        raise ValueError(f"Unknown cross-tokenizer algorithm {name!r}; available algorithms: {available}")
    capabilities = algorithm_cls.capabilities
    if not isinstance(capabilities, CrossTokenizerCapabilities):
        raise TypeError(
            f"Cross-tokenizer algorithm {normalized_name!r} has an invalid capabilities contract"
        )
    return capabilities


def build_cross_tokenizer(name: str, **kwargs: Any) -> CrossTokenizerAlgorithm:
    discover_cross_tokenizers()
    normalized_name = name.strip().lower()
    algorithm_cls = _ALGORITHMS.get(normalized_name)
    if algorithm_cls is None:
        available = ", ".join(sorted(_ALGORITHMS)) or "<none>"
        raise ValueError(f"Unknown cross-tokenizer algorithm {name!r}; available algorithms: {available}")
    return algorithm_cls(**kwargs)
