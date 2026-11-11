import ast
import base64
import hashlib
import io
import json
import subprocess
import sys
import zipfile
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


def test_colab_bootstrap_imports_embedded_package_in_active_interpreter(tmp_path) -> None:
    notebook = json.loads((ROOT / "notebooks/GPW_Heaton_Colab.ipynb").read_text())
    tree = ast.parse("".join(notebook["cells"][2]["source"]))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "ROOT"
    )
    assignment.value.args[0] = ast.Constant(str(tmp_path / "snapshot"))
    ast.fix_missing_locations(tree)
    script = (
        ast.unparse(tree)
        + """
import importlib.util
spec = importlib.util.find_spec("pwgp")
assert spec is not None, "Embedded pwgp package is not importable"
assert Path(spec.origin).resolve() == (ROOT / "src/pwgp/__init__.py").resolve(), spec.origin
from experiments.heaton_benchmark import HeatonConfig, run_heaton
from pwgp import GeneralizedPatchworkGP, SVGP
assert HeatonConfig().epochs == 100
assert HeatonConfig().warmup_epochs == 20
print("Embedded runner and models imported without a kernel restart")
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


def test_colab_source_snapshot_matches_current_code() -> None:
    notebook = json.loads((ROOT / "notebooks/GPW_Heaton_Colab.ipynb").read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["nbformat"] == 4
    source = "".join(notebook["cells"][2]["source"])
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "SOURCE_B64"
    )
    payload = base64.b64decode(ast.literal_eval(assignment.value))
    assert hashlib.sha256(payload).hexdigest() == notebook["metadata"]["gpw_source_sha256"]
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        for name in bundle.namelist():
            assert bundle.read(name) == (ROOT / name).read_bytes(), f"Rebuild snapshot: {name}"
            if name.endswith(".py"):
                compile(bundle.read(name), name, "exec")
        assert "src/pwgp/models/patchwork.py" in bundle.namelist()
        assert "src/pwgp/models/svgp.py" in bundle.namelist()
        assert "experiments/heaton_benchmark.py" in bundle.namelist()


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
    assert len(configurations) == 1
    assert configurations[0]["inducing_points"] == 480
    assert configurations[0]["epochs"] == 100
    assert configurations[0]["warmup_epochs"] == 20


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
