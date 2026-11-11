"""Generate GitHub-only Colab setup without embedding model or dataset code."""

REPOSITORY = "https://github.com/chahineNejm/Generalized-Patchwork.git"
BRANCH = "main"


def github_bootstrap(root):
    return f'''import hashlib, importlib, json, os, subprocess, sys
from pathlib import Path

REPOSITORY = {REPOSITORY!r}
BRANCH = {BRANCH!r}
ROOT = Path({root!r})
if ROOT.exists():
    raise FileExistsError("Checkout already exists; use a fresh Colab runtime")
if any(name == "pwgp" or name.startswith("pwgp.") or name == "experiments"
       or name.startswith("experiments.") for name in sys.modules):
    raise RuntimeError("Restart the runtime before loading a new source version")
os.environ["GIT_TERMINAL_PROMPT"] = "0"
subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--branch", BRANCH, "--single-branch",
                REPOSITORY, str(ROOT)], check=True)
subprocess.run(["git", "-C", str(ROOT), "sparse-checkout", "set", "--no-cone",
                "/src/", "/experiments/", "/pyproject.toml", "/README.md"], check=True)
subprocess.run(["git", "-C", str(ROOT), "checkout", BRANCH], check=True)
COMMIT = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
SOURCE_ZIP = subprocess.check_output(["git", "-C", str(ROOT), "archive", "--format=zip",
                                      COMMIT, "pyproject.toml", "README.md", "src", "experiments"])
SOURCE_PROVENANCE = {{"repository": REPOSITORY, "branch": BRANCH, "commit": COMMIT,
                     "source_sha256": hashlib.sha256(SOURCE_ZIP).hexdigest()}}
os.environ["MPLCONFIGDIR"] = str(ROOT / "results/.matplotlib")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
importlib.invalidate_caches()
print(SOURCE_PROVENANCE)
'''


def source_metadata():
    return {"repository": REPOSITORY, "branch": BRANCH, "mode": "github_latest_branch"}
