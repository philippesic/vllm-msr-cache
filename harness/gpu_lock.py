"""One process-wide GPU reservation across cases; compatible with frozen leaf locks."""

import fcntl
from functools import wraps
from pathlib import Path

GPU_LOCK = Path.home() / ".cache/semantic-offload/gpu-0.lock"
LEGACY_LOCK = Path("/tmp/vllm-semantic-cache-gpu.lock")


class GPULock:
    def __init__(self, path=None, *, legacy=True):
        self.paths = (
            [Path(path)]
            if path is not None
            else [GPU_LOCK] + ([LEGACY_LOCK] if legacy else [])
        )
        self.files = []

    def __enter__(self):
        try:
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open("a+")
                self.files.append(stream)
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as exc:
            self.__exit__()
            if isinstance(exc, BlockingIOError):
                raise RuntimeError("GPU benchmark lock held by another suite") from exc
            raise
        return self

    def __exit__(self, *args):
        for stream in reversed(self.files):
            stream.close()
        self.files.clear()


def gpu_reserved(function):
    """Supervisors hold the common reservation; frozen children take the legacy lock.

    Never pass the reservation descriptor to children or unlink a lock file.
    Holding the supervisor reservation also protects idle gaps between cases.
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        with GPULock(legacy=False):
            return function(*args, **kwargs)

    return wrapped
