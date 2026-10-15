"""Cooperative device serialization; file locks also survive in the evaluator child."""
import fcntl
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def acquire(path, timeout=180):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open("a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("cooperative resource lock wait expired") from None
                time.sleep(.05)
        try:
            yield lock, time.monotonic() - started
        finally:
            # Closing instead of explicit LOCK_UN lets a still-running child retain
            # the inherited open-file-description lock after parent interruption.
            pass
