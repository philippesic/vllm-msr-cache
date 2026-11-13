from __future__ import annotations

import ast
from pathlib import Path

ENTRY_POINTS = (
    (
        "Benchmark qualification",
        "uv run --locked python -m tools.benchmark",
        "tools/benchmark.py",
        "main",
        "model/coding probes, CPU trace-replay simulator and hash-checked runtime patches implemented; comparison campaigns unqualified",
    ),
    (
        "CPU checks",
        "uv run --locked python tools/check.py",
        "tools/check.py",
        "main",
        "implemented",
    ),
    (
        "Runtime verify",
        "uv run --locked python -m tools.runtime_verify",
        "tools/runtime_verify.py",
        "main",
        "implemented; explicit runtime and output required",
    ),
)


def rows() -> list[str]:
    return [
        f"| {name} | `{command}` | `{path}:{symbol}` | {state} |"
        for name, command, path, symbol, state in ENTRY_POINTS
    ]


def check(root: Path) -> list[str]:
    errors = []
    file = root / "docs/system-map.md"
    if not file.is_file():
        return ["docs.map: system map is required"]
    text = file.read_text()
    if any(row not in text.splitlines() for row in rows()):
        errors.append(
            "docs.map: commands, targets and availability must match the system map"
        )
    for _name, _command, path, symbol, _state in ENTRY_POINTS:
        target = root / path
        if not target.is_file() or not any(
            isinstance(node, ast.FunctionDef) and node.name == symbol
            for node in ast.parse(target.read_text()).body
        ):
            errors.append(f"docs.entry: {path}:{symbol} disappeared")
    return errors
