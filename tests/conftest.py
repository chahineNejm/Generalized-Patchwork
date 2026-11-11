import ast
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def github_checkout(tmp_path):
    """Run notebook Git operations against a real local repository, not the network."""
    root = Path(__file__).resolve().parents[1]
    remote = tmp_path / "remote"
    remote.mkdir()
    for name in ("src", "experiments"):
        shutil.copytree(root / name, remote / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(root / name, remote / name)
    (remote / "airline.pickle").write_bytes(b"not real data; tests sparse checkout exclusion")
    for arguments in (["init", "-b", "main"], ["add", "."],
                      ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.org",
                       "commit", "-m", "source fixture"]):
        subprocess.run(["git", "-C", str(remote), *arguments], check=True, capture_output=True)

    def prepare(source):
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "ROOT":
                    node.value.args[0] = ast.Constant(str(tmp_path / "checkout"))
                elif node.targets[0].id == "REPOSITORY":
                    node.value = ast.Constant(str(remote))
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    prepare.remote = remote
    return prepare
