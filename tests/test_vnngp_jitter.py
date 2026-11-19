from dataclasses import replace

import pytest

from experiments.default_hyperparameters import benchmark_repeat_defaults
from experiments.multidataset_worker import make_plan
from experiments.ukhousing_benchmark import UKConfig, model_numerical
from experiments.vnngp_cpu_worker import make_plan as cpu_plan


@pytest.mark.parametrize("value", [-1e-3, float("nan"), float("inf"), True, None, "0.001"])
def test_invalid_jitter_rejected_before_model_build(value):
    with pytest.raises(ValueError, match="vnngp_jitter"):
        UKConfig(vnngp_jitter=value)
    with pytest.raises(ValueError, match="vnngp_jitter"):
        cpu_plan({"dataset": "oisst", "seeds": [31], "input_root": "unused",
                  "data_overrides": {}, "overrides": {"vnngp_jitter": value}})


def test_regular_notebook_override_is_saved_and_only_changes_vnngp():
    request = {**benchmark_repeat_defaults(device="cpu", input_root="unused"),
        "study": "benchmark_repeats", "device": "cpu", "input_root": "unused",
        "datasets": ["oisst"], "seeds": [31], "data_overrides": {},
        "models": {"oisst": ["VNNGP", "SVGP"]}, "excluded_model_types": [],
        "save_inducing_positions": True, "model_overrides": {}}
    before = make_plan(request)["plan"]
    after = make_plan({**request, "model_overrides": {
        "oisst": {"VNNGP": {"vnngp_jitter": 1e-3}}}})["plan"]
    for old, new in zip(before["runs"], after["runs"], strict=True):
        expected = dict(old["config"])
        if expected["model"] == "vnngp":
            expected["vnngp_jitter"] = 1e-3
        assert new["config"] == expected
        config = UKConfig(**new["config"])
        assert model_numerical(config).prior_jitter == (1e-3 if config.model == "vnngp" else 1e-4)
    base = UKConfig()
    assert model_numerical(replace(base, vnngp_jitter=1e-3)) == model_numerical(base)
