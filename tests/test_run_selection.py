import pytest

from experiments.default_hyperparameters import oisst_defaults
from experiments.run_selection import select_runs


def test_subset_order_and_overrides_preserve_defaults():
    _, runs = oisst_defaults(device="cpu")
    selected = select_runs(runs, models=["ProSpar", "GPW"],
                           overrides={"GPW": {"epochs": 2, "warmup_epochs": 1},
                                      "SVGP": {"epochs": 50}})
    assert [name for name, _ in selected] == ["ProSpar", "GPW"]
    assert selected[1][1].epochs == 2 and selected[1][1].warmup_epochs == 1
    assert selected[0][1].epochs == 100
    assert dict(runs)["GPW"].epochs == 100
    assert dict(runs)["SVGP"].epochs == 100


@pytest.mark.parametrize("models,match", [([], "nonempty"), ("GPW", "nonempty"),
    (["GPW", "GPW"], "duplicate"), (["unknown"], "Unknown MODELS"), ([1], "nonempty")])
def test_invalid_model_selection_is_not_silently_filtered(models, match):
    _, runs = oisst_defaults(device="cpu")
    with pytest.raises(ValueError, match=match):
        select_runs(runs, models=models, overrides={})


def test_unknown_override_and_unknown_parameter_raise():
    _, runs = oisst_defaults(device="cpu")
    with pytest.raises(ValueError, match="unknown run names"):
        select_runs(runs, models=["GPW"], overrides={"typo": {}})
    with pytest.raises(TypeError, match="unknown_parameter"):
        select_runs(runs, models=["GPW"], overrides={"GPW": {"unknown_parameter": 1}})
