"""Materialize an immutable source closure for independently running probes."""
from __future__ import annotations

import fcntl
import shutil
import tempfile
from pathlib import Path

from .campaign_contracts import atomic_json, digest, file_hash, local_path


def materialize(root, cache, identity):
    root, cache = Path(root), Path(cache)
    if identity['digest'] != digest(identity['files']):
        raise ValueError('Source manifest digest mismatch')
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / identity['digest']
    with (cache / '.snapshot.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            for relative, expected in identity['files'].items():
                if file_hash(local_path(target, relative)) != expected:
                    raise ValueError('Existing source snapshot changed')
            return target
        temporary = Path(tempfile.mkdtemp(prefix='.snapshot-', dir=cache))
        try:
            for relative, expected in identity['files'].items():
                source, destination = local_path(root, relative), local_path(temporary, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if file_hash(destination) != expected:
                    raise ValueError('Source changed during snapshot')
            atomic_json(temporary/'source-closure.json', identity)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target
