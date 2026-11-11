"""Explicit model selection and call-time overrides for experiment launchers."""

from dataclasses import replace


def select_runs(runs, *, models, overrides):
    if not isinstance(models, (list, tuple)) or not models or any(not isinstance(x, str) for x in models):
        raise ValueError("MODELS must be a nonempty list of model names")
    if len(set(models)) != len(models):
        raise ValueError("MODELS must not contain duplicate names")
    available = dict(runs)
    if len(available) != len(runs):
        raise ValueError("Available runs must have unique names")
    unknown = set(models) - available.keys()
    if unknown:
        raise ValueError(f"Unknown MODELS: {sorted(unknown)}; available: {list(available)}")
    unknown_overrides = set(overrides) - available.keys()
    if unknown_overrides:
        raise ValueError(f"Overrides contain unknown run names: {sorted(unknown_overrides)}")
    return [(name, replace(available[name], **overrides[name]) if name in overrides else available[name])
            for name in models]
