"""Generate GitHub-only Colab setup without embedding model or dataset code."""

REPOSITORY = "https://github.com/chahineNejm/Generalized-Patchwork.git"
BRANCH = "main"


def github_bootstrap(root, *, isolated=False):
    paths = '''for path in reversed((str(ROOT / "src"), str(ROOT))):
    if path not in sys.path:
        sys.path.insert(0, path)''' if isolated else 'sys.path[:0] = [str(ROOT / "src"), str(ROOT)]'
    if isolated:
        checkout = '''os.environ["GIT_TERMINAL_PROMPT"] = "0"
if ROOT.exists():
    top = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(top).resolve() != ROOT.resolve():
        raise ValueError("Existing checkout must be its own Git repository")
    remote = subprocess.check_output(["git", "-C", str(ROOT), "remote", "get-url", "origin"], text=True).strip()
    branch = subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()
    if remote != REPOSITORY or branch != BRANCH:
        raise ValueError("Existing checkout repository or branch does not match the requested source")
    subprocess.run(["git", "-C", str(ROOT), "diff", "--exit-code", "HEAD"], check=True)
    untracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--others",
        "--exclude-standard", "--", ":(glob)src/**/*.py", ":(glob)experiments/**/*.py"], text=True).strip()
    if untracked:
        raise ValueError(f"Untracked source files prevent verified reuse: {untracked}")
    subprocess.run(["git", "-C", str(ROOT), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "HEAD", "origin/" + BRANCH], check=True)
    subprocess.run(["git", "-C", str(ROOT), "merge", "--ff-only", "origin/" + BRANCH], check=True)
else:
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--branch", BRANCH,
                    "--single-branch", REPOSITORY, str(ROOT)], check=True)
subprocess.run(["git", "-C", str(ROOT), "sparse-checkout", "set", "--no-cone",
                "/src/", "/experiments/", "/pyproject.toml", "/README.md"], check=True)
subprocess.run(["git", "-C", str(ROOT), "checkout", BRANCH], check=True)
'''
    else:
        checkout = '''if ROOT.exists():
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
'''
    return f'''import hashlib, importlib, json, os, subprocess, sys
from pathlib import Path

REPOSITORY = {REPOSITORY!r}
BRANCH = {BRANCH!r}
ROOT = Path({root!r})
{checkout}
COMMIT = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
SOURCE_ZIP = subprocess.check_output(["git", "-C", str(ROOT), "archive", "--format=zip",
                                      COMMIT, "pyproject.toml", "README.md", "src", "experiments"])
SOURCE_PROVENANCE = {{"repository": REPOSITORY, "branch": BRANCH, "commit": COMMIT,
                     "source_sha256": hashlib.sha256(SOURCE_ZIP).hexdigest()}}
os.environ["MPLCONFIGDIR"] = str(ROOT / "results/.matplotlib")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.chdir(ROOT)
{paths}
importlib.invalidate_caches()
print(SOURCE_PROVENANCE)
'''


def source_metadata():
    return {"repository": REPOSITORY, "branch": BRANCH, "mode": "github_latest_branch"}
