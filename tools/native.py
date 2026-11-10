from __future__ import annotations

import json
from pathlib import Path

NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh"}


def check(root: Path, paths: list[Path]) -> list[str]:
    config = json.loads((root / "tools/native.json").read_text())
    errors: list[str] = []
    for architecture in ("75", "120"):
        if architecture not in config["architectures"]:
            errors.append(f"native.sm_{architecture}: required target missing")
    sources = [
        p
        for p in paths
        if p.suffix in NATIVE_SUFFIXES
        or p.name == "CMakeLists.txt"
        or p.suffix == ".cmake"
    ]
    if sources:
        errors.append(
            "native.unqualified: native sources require qualified clang-format/clang-tidy/CMake recipes and their deliberate-failure tests for sm_75 and sm_120; see tools/native.json"
        )
    return errors
