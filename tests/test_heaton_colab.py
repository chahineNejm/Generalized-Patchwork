import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_noise_sensitivity_reuses_saved_predictions(tmp_path) -> None:
    import numpy as np
    import pytest

    notebook = json.loads((ROOT / "notebooks/GPW_Heaton_Colab.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def noise_nlpd_sensitivity" in "".join(cell["source"])
    )
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    namespace = {"np": np, "json": json, "pd": __import__("pandas")}
    # Execute only the diagnostic definition from the repository-owned notebook.
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "diagnostic", "exec"), namespace)  # noqa: S102
    (tmp_path / "metrics.json").write_text(json.dumps({"noise_std_celsius": 2.0}))
    np.savez(
        tmp_path / "predictions.npz", test_mask=np.array([True, False, True]),
        truth=np.array([1.0, np.nan, 3.0]), mean=np.array([0.0, 0.0, 1.0]),
        latent_variance=np.array([1.0, 2.0, 2.0]),
        observation_variance=np.array([5.0, 6.0, 6.0]),
    )
    diagnostic = namespace["noise_nlpd_sensitivity"]
    result = diagnostic([tmp_path], [0.0, 1.0, 2.0])
    for multiplier, actual in zip([0.0, 1.0, 2.0], result["NLPD"], strict=True):
        variance = np.array([1.0, 2.0]) + multiplier * 4.0
        expected = np.mean(0.5 * (np.log(2 * np.pi * variance) + np.array([1.0, 4.0]) / variance))
        np.testing.assert_allclose(actual, expected)
    assert result.iloc[1]["delta_vs_saved_NLPD"] == 0.0
    with pytest.raises(ValueError, match="nonnegative"):
        diagnostic([tmp_path], [-1.0])
    with pytest.raises(ValueError, match="No completed runs"):
        diagnostic([], [1.0])


def test_colab_bootstrap_imports_github_package_in_active_interpreter(tmp_path, github_checkout) -> None:
    notebook = json.loads((ROOT / "notebooks/GPW_Heaton_Colab.ipynb").read_text())
    script = (
        github_checkout("".join(notebook["cells"][2]["source"]))
        + """
import importlib.util
spec = importlib.util.find_spec("pwgp")
assert spec is not None, "GitHub pwgp package is not importable"
assert Path(spec.origin).resolve() == (ROOT / "src/pwgp/__init__.py").resolve(), spec.origin
from experiments.heaton_benchmark import HeatonConfig, run_heaton
from pwgp import GeneralizedPatchworkGP, SVGP
assert HeatonConfig().epochs == 100
assert HeatonConfig().warmup_epochs == 20
assert SOURCE_PROVENANCE['commit'] == COMMIT
assert len(COMMIT) == 40
print("GitHub runner and models imported without a kernel restart")
"""
    )
    process = subprocess.run(
        [sys.executable, "-I", "-"],
        input=script,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_colab_calls_github_instead_of_embedding_source() -> None:
    notebook = json.loads((ROOT / "notebooks/GPW_Heaton_Colab.ipynb").read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["nbformat"] == 4
    source = "".join(notebook["cells"][2]["source"])
    assert "SOURCE_B64" not in source
    assert '"clone"' in source and '"sparse-checkout"' in source
    assert notebook["metadata"]["gpw_source"]["branch"] == "main"


def test_colab_cells_compile_and_keep_requested_schedule() -> None:
    notebook = json.loads((ROOT / "notebooks/GPW_Heaton_Colab.ipynb").read_text())
    configurations = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%pip "):
            assert "/content/gpw_heaton[visual,benchmark]" in source
            continue
        tree = ast.parse(source)
        compile(tree, "colab_cell", "exec")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "HeatonConfig"
            ):
                configurations.append(
                    {
                        item.arg: ast.literal_eval(item.value)
                        for item in node.keywords
                        if isinstance(item.value, ast.Constant)
                    }
                )
    from experiments.default_hyperparameters import heaton_defaults

    assert not configurations
    config = heaton_defaults(device="cuda:0")
    assert (config.inducing_points, config.epochs, config.warmup_epochs) == (480, 100, 20)
    assert "heaton_defaults(device=DEVICE)" in "".join(notebook["cells"][5]["source"])


def test_colab_preflight_covers_batched_solver_shapes() -> None:
    notebook = json.loads((ROOT / "notebooks/GPW_Heaton_Colab.ipynb").read_text())
    setup = ast.parse("".join(notebook["cells"][5]["source"]))
    probes = next(
        node.value
        for node in setup.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "GPU_PROBES"
    )
    assert ast.literal_eval(probes) == [(8, 60, 96), (4, 120, 48), (2048, 7, 1), (1, 480, 16)]
    function = next(
        node
        for node in setup.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_batched_solve"
    )
    calls = [node.func for node in ast.walk(function) if isinstance(node, ast.Call)]
    assert any(isinstance(node, ast.Name) and node.id == "cholesky_solve" for node in calls)
    assert any(isinstance(node, ast.Attribute) and node.attr == "backward" for node in calls)
