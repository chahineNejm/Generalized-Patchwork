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


def test_heat_colab_embeds_current_sources_and_compiles_cells():
    notebook = json.loads((ROOT / "notebooks/GPW_Periodic_Heat_Colab.ipynb").read_text())
    bootstrap = next(c for c in notebook["cells"] if "SOURCE_B64 =" in "".join(c["source"]))
    tree = ast.parse("".join(bootstrap["source"]))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "SOURCE_B64")
    payload = base64.b64decode(ast.literal_eval(assignment.value))
    assert hashlib.sha256(payload).hexdigest() == notebook["metadata"]["gpw_source_sha256"]
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        for name in bundle.namelist():
            assert bundle.read(name) == (ROOT / name).read_bytes(), name
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and not source.startswith("%pip"):
            compile(source, "periodic_heat_colab", "exec")
    settings = next("".join(c["source"]) for c in notebook["cells"] if "STITCH_LEARNING_RATES =" in "".join(c["source"]))
    assert 'DEVICE = "cuda:0"' in settings
    assert "STITCH_LEARNING_RATES = [0.003]" in settings
    run_cell = next("".join(c["source"]) for c in notebook["cells"] if "summaries = []" in "".join(c["source"]))
    assert "learning_rate_line=rate" in run_cell
    assert "capture_git=False" in run_cell
    assert "on_snapshot=show_progress, on_complete=completed" in run_cell
    assert 'folder / "kernel_evolution.png"' in run_cell


def test_embedded_heat_imports_without_git_in_fresh_interpreter(tmp_path):
    notebook = json.loads((ROOT / "notebooks/GPW_Periodic_Heat_Colab.ipynb").read_text())
    bootstrap = next(c for c in notebook["cells"] if "SOURCE_B64 =" in "".join(c["source"]))
    tree = ast.parse("".join(bootstrap["source"]))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "ROOT")
    assignment.value.args[0] = ast.Constant(str(tmp_path / "embedded"))
    ast.fix_missing_locations(tree)
    script = ast.unparse(tree) + """
import pwgp
from dataclasses import replace
from experiments.stitch_line_heat import RunConfig, build_model
assert Path(pwgp.__file__).resolve().is_relative_to(ROOT / 'src')
config = RunConfig()
assert config.learning_rate_line == 0.003
assert config.device == 'cpu'
model = build_model(config, 0)
assert model.stitches().shape == (1, 10, 2)
assert model.stitch.raw.shape == (1, 10, 2)
"""
    result = subprocess.run([sys.executable, "-I", "-"], input=script, text=True,
                            capture_output=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
