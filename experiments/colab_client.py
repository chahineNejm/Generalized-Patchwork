"""Standard-library-only client; experiment imports remain in fresh workers."""

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


def run_worker(root, request, *, on_event=None):
    root = Path(root).resolve()
    jobs = root / "results/worker_jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="job_", dir=jobs))
    request_path = job / "request.json"
    request_path.write_text(json.dumps(request, indent=2, allow_nan=False), encoding="utf-8")
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
                   "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                   "MPLCONFIGDIR": str(root / "results/.matplotlib"),
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    process = None
    try:
        with (job / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.oisst_worker", str(request_path)],
                cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="strict", bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("GPW_EVENT "):
                    event = json.loads(line[len("GPW_EVENT "):])
                    if on_event is not None:
                        on_event(event)
                    process.stdin.write("continue\n")
                    process.stdin.flush()
                else:
                    print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, process.args)
        return json.loads((job / "result.json").read_text(encoding="utf-8"))
    except BaseException:
        (job / "client_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stdout.close()
            process.stdin.close()
