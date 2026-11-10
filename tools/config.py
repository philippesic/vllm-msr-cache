from __future__ import annotations

import os
from pathlib import Path


def check_environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def proof_environment(root: Path, project: Path) -> dict[str, str]:
    env = check_environment(root)
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(project)])
    env["MYPY_CACHE_DIR"] = str(root.parent.parent / "mypy-cache")
    return env
