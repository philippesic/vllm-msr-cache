from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from tools.agent_policy import exceptions


def code_check(root: Path) -> list[str]:
    entries, errors = exceptions(root)
    allowed = {(e.path, e.match) for e in entries if e.rule == "unused"}
    used = set()
    result = subprocess.run(
        [
            "vulture",
            "wait_hint_offload",
            "benchmarks",
            "tools",
            "tests",
            "--min-confidence",
            "60",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in {0, 3}:
        return errors + ["unused.tool: vulture failed to inspect project"]
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"(.+?):\d+: (.+)", line)
        if not match:
            errors.append("unused.tool: unrecognized vulture diagnostic")
            continue
        key = (match[1], match[2])
        if key in allowed:
            used.add(key)
        else:
            errors.append(f"unused.code: {match[1]}: {match[2]}")
    errors.extend(f"exceptions.stale: {path} unused" for path, _match in allowed - used)
    return errors


def dependency_command(root: Path) -> list[str]:
    entries, errors = exceptions(root)
    if errors:
        raise ValueError(errors[0])
    command = ["deptry", ".", "--no-ansi", "--exclude", r"(?:\.venv|\.git)(?:/|$)"]
    modules_path = root / "runtime/modules.json"
    if modules_path.exists():
        modules = json.loads(modules_path.read_text())
        requirements = (root / "runtime/requirements.in").read_text()
        for module, distribution in modules.items():
            if not re.fullmatch(r"[a-z_]+", module) or not re.search(
                rf"^{re.escape(distribution)}(?:==| @ )", requirements, re.MULTILINE
            ):
                raise ValueError("runtime import is not pinned")
        command.extend(
            [
                "--per-rule-ignores",
                ",".join("DEP001=" + module for module in sorted(modules)),
            ]
        )
    for entry in entries:
        if entry.rule == "development-root" and entry.match == "deptry":
            command.extend(["--extend-exclude", f"^{re.escape(entry.path)}(?:/|$)"])
    return command
