from __future__ import annotations

import ast
import io
import re
import tokenize
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExceptionEntry:
    rule: str
    path: str
    match: str
    reason: str


def exceptions(root: Path) -> tuple[list[ExceptionEntry], list[str]]:
    file = root / "tools/exceptions.toml"
    try:
        data = tomllib.loads(file.read_text())
        if set(data) != {"exception"} or not isinstance(data["exception"], list):
            raise ValueError
        entries = []
        for item in data["exception"]:
            if set(item) != {"rule", "path", "match", "reason"} or any(
                not isinstance(v, str) or not v.strip() for v in item.values()
            ):
                raise ValueError
            entry = ExceptionEntry(**item)
            if (
                entry.rule
                not in {
                    "directive",
                    "script",
                    "unused",
                    "development-root",
                    "adapter-import",
                }
                or Path(entry.path).is_absolute()
                or ".." in Path(entry.path).parts
                or any(c in entry.path for c in "*?[")
                or not (root / entry.path).exists()
            ):
                raise ValueError
            entries.append(entry)
        keys = [(e.rule, e.path, e.match) for e in entries]
        if len(keys) != len(set(keys)):
            raise ValueError
        return entries, []
    except (OSError, ValueError, TypeError):
        return [], [
            "exceptions.invalid: require exact existing paths, known rules, unique entries and reasons"
        ]


def comment_errors(
    root: Path, paths: list[Path], entries: list[ExceptionEntry]
) -> list[str]:
    errors = []
    used: set[tuple[str, str]] = set()
    directives = {(e.path, e.match) for e in entries if e.rule == "directive"}
    for path in paths:
        file = root / path
        if not file.is_file() or file.is_symlink():
            continue
        label = path.as_posix()
        text = file.read_text(errors="replace")
        if path.suffix == ".py":
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type != tokenize.COMMENT:
                    continue
                value = token.string
                if token.start[0] == 1 and value.startswith("#!"):
                    continue
                if token.start[0] <= 2 and re.fullmatch(
                    r"#.*coding[:=]\s*[-\w.]+\s*", value
                ):
                    continue
                if token.start[0] <= 2 and value.startswith(
                    "# SPDX-License-Identifier: "
                ):
                    continue
                key = (label, value)
                if (
                    re.match(r"#\s*(?:noqa\b|type:\s*ignore\b|.*\bsuppress\b)", value)
                    and key in directives
                ):
                    used.add(key)
                    continue
                errors.append(f"comments.banned: {label}:{token.start[0]}")
            tree = ast.parse(text, filename=label)
            if any(
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
                for n in ast.walk(tree)
            ):
                errors.append(f"comments.docstring: {label}")
        elif path.suffix in {".toml", ".yml", ".yaml", ".sh"} or path.name in {
            "Dockerfile.checks",
            "pre-commit",
            ".gitignore",
            ".dockerignore",
        }:
            for number, line in enumerate(text.splitlines(), 1):
                quote = ""
                escaped = False
                comment = False
                for char in line:
                    if escaped:
                        escaped = False
                    elif char == "\\" and quote == '"':
                        escaped = True
                    elif quote:
                        if char == quote:
                            quote = ""
                    elif char in {'"', "'"}:
                        quote = char
                    elif char == "#":
                        comment = True
                        break
                if comment:
                    if number == 1 and line.startswith("#!"):
                        continue
                    errors.append(f"comments.banned: {label}:{number}")
    errors.extend(f"exceptions.stale: {p} directive" for p, m in directives - used)
    return errors


def check(root: Path, paths: list[Path]) -> list[str]:
    entries, errors = exceptions(root)
    errors.extend(comment_errors(root, paths, entries))
    return sorted(set(errors))
