"""Record run failures without swallowing or replacing their exceptions."""

import json
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


@contextmanager
def record_failures(directory: Path, *, status_key: str):
    """Persist the full traceback and re-raise, including notebook interrupts."""
    try:
        yield
    except BaseException as error:
        failure = {
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
            "failed_at_utc": datetime.now(UTC).isoformat(),
        }
        (directory / "failure.json").write_text(
            json.dumps(failure, indent=2, allow_nan=False), encoding="utf-8"
        )
        (directory / "status.json").write_text(
            json.dumps({status_key: "failed", **failure}, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        raise
