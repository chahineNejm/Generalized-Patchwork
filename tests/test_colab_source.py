import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.build_heaton_colab import build_notebook as build_heaton
from experiments.build_oisst_colab import build_notebook as build_oisst
from experiments.build_stitch_line_colab import build_notebook as build_heat
from experiments.build_ukhousing_colab import build_notebook as build_uk
from experiments.colab_source import BRANCH, REPOSITORY, github_bootstrap
from experiments.default_hyperparameters import (
    heaton_defaults,
    periodic_heat_defaults,
    ukhousing_defaults,
)

ROOT = Path(__file__).resolve().parents[1]


def normalized_code(source):
    tree = ast.parse(source)
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("OVERRIDES", "DATA_OVERRIDES", "MODELS", "DATA_ACTION",
                    "PREPARED_PATH", "ANALYSIS_SOURCE", "SAVED_RUNS", "DOWNLOAD_SOURCE", "SAVED_DIRECTORY")):
            node.value = ast.Dict(keys=[], values=[])
    return ast.dump(tree, include_attributes=False)


@pytest.mark.parametrize("builder", [build_oisst, build_heaton, build_heat, build_uk])
def test_notebook_calls_github_and_matches_generator_except_explicit_overrides(builder, tmp_path):
    (tmp_path / "notebooks").mkdir()
    path = builder(tmp_path)
    notebook = json.loads(path.read_text())
    saved = json.loads((ROOT / "notebooks" / path.name).read_text())
    assert len(saved["cells"]) == len(notebook["cells"])
    for generated, actual in zip(notebook["cells"], saved["cells"], strict=True):
        assert actual["cell_type"] == generated["cell_type"]
        left, right = "".join(generated["source"]), "".join(actual["source"])
        if actual["cell_type"] == "code":
            assert normalized_code(left) == normalized_code(right)
        else:
            assert left == right
    assert notebook["metadata"]["gpw_source"]["repository"] == REPOSITORY
    assert notebook["metadata"]["gpw_source"]["branch"] == "main"
    for cell in saved["cells"]:
        assert not cell.get("outputs")
        source = "".join(cell["source"])
        assert "SOURCE_B64" not in source
        if cell["cell_type"] == "code":
            tree = ast.parse(source)
            assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
            compile(tree, path.name, "exec")


def test_checkout_records_resolved_commit_and_excludes_datasets(github_checkout, tmp_path):
    source = github_checkout(github_bootstrap("/content/test"))
    source += '''
import io, zipfile
assert not (ROOT / 'airline.pickle').exists()
assert SOURCE_PROVENANCE['branch'] == 'main'
assert len(SOURCE_PROVENANCE['commit']) == 40
with zipfile.ZipFile(io.BytesIO(SOURCE_ZIP)) as bundle:
    assert 'src/pwgp/models/prospar.py' in bundle.namelist()
    assert 'experiments/default_hyperparameters.py' in bundle.namelist()
    assert 'airline.pickle' not in bundle.namelist()
from experiments.default_hyperparameters import oisst_defaults
assert oisst_defaults(device='cpu')[1][2][1].prospar.prior_jitter == 0
'''
    process = subprocess.run([sys.executable, "-I", "-"], input=source, text=True,
                             capture_output=True, cwd=tmp_path, check=False)
    assert process.returncode == 0, process.stdout + process.stderr


def test_latest_main_updates_are_resolved_at_runtime(github_checkout, tmp_path):
    remote = github_checkout.remote
    (remote / "README.md").write_text("changed source fixture")
    for arguments in (["add", "README.md"], ["-c", "user.name=Fixture", "-c",
            "user.email=fixture@example.org", "commit", "-m", "latest main"]):
        subprocess.run(["git", "-C", str(remote), *arguments], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(remote), "rev-parse", "HEAD"], text=True).strip()
    source = github_checkout(github_bootstrap("/content/test"))
    source += f"\nassert COMMIT == {commit!r}\n"
    process = subprocess.run([sys.executable, "-I", "-"], input=source, text=True,
                             capture_output=True, cwd=tmp_path, check=False)
    assert process.returncode == 0, process.stdout + process.stderr


def test_historical_notebook_defaults_preserved():
    h = heaton_defaults(device="cpu")
    u = ukhousing_defaults(device="cpu")
    p = periodic_heat_defaults(device="cpu")
    assert (h.epochs, h.warmup_epochs, h.inducing_points) == (100, 20, 480)
    assert (u.epochs, u.warmup_epochs, u.inducing_points) == (100, 20, 1024)
    assert (p.epochs, p.warmup_epochs, p.layouts.heat.supports_per_patch) == (350, 40, 28)
    assert p.layouts.line_angles == (90, 90, 45, 0)
    assert BRANCH == "main"
