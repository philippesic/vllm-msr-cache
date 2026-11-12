from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


def added_json(patch: Path, filename: str) -> dict[str, Any]:
    checked = subprocess.run(
        ["git", "apply", "--numstat", str(patch.resolve())],
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = [line.split("\t") for line in checked.stdout.splitlines()]
    text = patch.read_text()
    if (
        checked.returncode
        or len(rows) != 1
        or len(rows[0]) != 3
        or rows[0][2] != filename
        or not rows[0][0].isdigit()
        or rows[0][1] != "0"
        or "new file mode 100644\n" not in text
    ):
        raise ValueError("structured submission must only add its JSON answer")
    sections = re.split(r"^@@[^\n]*\n", text, flags=re.MULTILINE)
    if len(sections) != 2:
        raise ValueError("structured answer must be a single added file")
    lines = sections[1].splitlines()
    if lines and lines[-1] == "\\ No newline at end of file":
        lines.pop()
    if any(not line.startswith("+") for line in lines):
        raise ValueError("invalid structured answer patch")
    answer: dict[str, Any] = json.loads("\n".join(line[1:] for line in lines))
    if not isinstance(answer, dict):
        raise ValueError("structured answer must be an object")
    return answer
