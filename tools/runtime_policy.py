from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.config import check_environment


def runtime_interface_check(root: Path) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "tools.runtime_verify"],
        cwd=root,
        env=check_environment(root),
        text=True,
        capture_output=True,
        timeout=10,
    )
    if (
        result.returncode != 2
        or result.stdout
        != "UNAVAILABLE: supply --out outside the repository using pinned runtime Python.\n"
        or result.stderr
    ):
        return [
            "runtime.claim: no-argument interface must exit 2 without claiming qualification"
        ]
    return []
