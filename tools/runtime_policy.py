from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.config import check_environment


def runtime_stub_check(root: Path) -> list[str]:
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
        != "NOT IMPLEMENTED: runtime verification is not implemented until source port.\n"
        or result.stderr
    ):
        return [
            "runtime.claim: stub must exit 2 and report not implemented until source port"
        ]
    return []
