from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def apply_runtime_patches(
    root: Path, python: Path, names: list[str]
) -> list[dict[str, Any]]:
    location = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    base = Path(location.stdout.strip())
    records = json.loads(
        (root / "benchmarks/runtime-patches/manifest.json").read_text()
    )
    selected = [record for record in records if record["name"] in names]
    if {record["name"] for record in selected} != set(names):
        raise ValueError("unknown runtime patch")
    prepared = []
    for record in selected:
        path = base / record["target"]
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest == record["patched_sha256"]:
            continue
        if digest != record["original_sha256"]:
            raise ValueError(f"runtime source differs: {path}")
        content = data.decode()
        for edit in record["edits"]:
            if content.count(edit["old"]) != 1:
                raise ValueError("runtime patch context is not unique")
            content = content.replace(edit["old"], edit["new"], 1)
        patched = content.encode()
        if hashlib.sha256(patched).hexdigest() != record["patched_sha256"]:
            raise ValueError("runtime patch hash differs")
        backup = path.with_suffix(path.suffix + ".benchmark-original")
        if backup.exists() and backup.read_bytes() != data:
            raise ValueError("runtime backup differs")
        prepared.append((path, data, patched))
    for path, data, patched in prepared:
        backup = path.with_suffix(path.suffix + ".benchmark-original")
        if not backup.exists():
            backup.write_bytes(data)
        if path.read_bytes() != data:
            raise ValueError("runtime source changed during patch preparation")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(patched)
            os.chmod(temporary, path.stat().st_mode)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return selected
