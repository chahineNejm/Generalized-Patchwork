"""Explicit model selection and call-time overrides for experiment launchers."""

from dataclasses import replace


def bind_patchwork_reference(runs, reference):
    """Bind matched baselines to the explicitly selected GPW initialization."""
    expected = {key: getattr(reference, key) for key in ("inducing_points", "patches", "seed", "threads")}
    result = []
    for name, config in runs:
        if config.model == "patchwork" and config.exact_training_points == "gpw_initial":
            if config.exact_gpw_reference is not None and config.exact_gpw_reference != expected:
                raise ValueError("Standard Patchwork reference differs from the requested GPW initialization")
            config = replace(config, exact_gpw_reference=expected.copy())
        result.append((name, config))
    return result


def resolve_patchwork_references(runs):
    if not any(config.model == "patchwork" and config.exact_training_points == "gpw_initial" for _, config in runs):
        return runs
    references = [config for _, config in runs if config.model == "gpw"]
    if len(references) > 1:
        raise ValueError("Matched comparison requires one unambiguous GPW reference")
    if references:
        return bind_patchwork_reference(runs, references[0])
    for _, config in runs:
        if config.model == "patchwork" and config.exact_training_points == "gpw_initial" and config.exact_gpw_reference is None:
            raise ValueError("Matched baseline needs GPW in the comparison or an explicit initialization reference")
    return runs


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
    selected = [(name, replace(available[name], **overrides[name]) if name in overrides else available[name])
                for name in models]
    if any(config.model == "patchwork" and config.exact_training_points == "gpw_initial" for _, config in selected):
        if "GPW" not in available:
            raise ValueError("Available configurations must include GPW to define matched initial points")
        reference = replace(available["GPW"], **overrides["GPW"]) if "GPW" in overrides else available["GPW"]
        return bind_patchwork_reference(selected, reference)
    return selected
