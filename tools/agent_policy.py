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
                or not entry.reason.strip()
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
    errors.extend(f"exceptions.stale: {p} directive" for p, _match in directives - used)
    return errors


def qualified_names(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = (
                    item.name if item.asname else item.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = qualified(node.value, aliases)
            for target in node.targets:
                if isinstance(target, ast.Name) and value:
                    aliases[target.id] = value
    return aliases


def qualified(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{qualified(node.value, aliases)}.{node.attr}"
    return ""


def python_errors(path: Path, text: str, entries: list[ExceptionEntry]) -> list[str]:
    label = path.as_posix()
    tree = ast.parse(text, filename=label)
    aliases = qualified_names(tree)
    errors = []
    library = path.parts[0] in {"semantic_offload", "benchmarks"}
    script_allowed = any(
        e.rule == "script" and e.path == label and e.match == "entry" for e in entries
    )
    main_guard = any(
        isinstance(n, ast.Compare)
        and isinstance(n.left, ast.Name)
        and n.left.id == "__name__"
        and any(
            isinstance(x, ast.Constant) and x.value == "__main__" for x in n.comparators
        )
        for n in ast.walk(tree)
    )
    if (len(path.parts) == 1 or main_guard) and not script_allowed:
        errors.append(f"layout.script: {label}")
    top = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not script_allowed and any(
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        for node in top
        for n in ast.walk(node)
    ):
        errors.append(f"layout.script: {label}")
    for node in ast.walk(tree):
        name = qualified(node, aliases)
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            errors.append(f"footgun.bare-except: {label}")
        if name in {
            "os.environ",
            "os.getenv",
            "os.putenv",
            "os.unsetenv",
        } and label not in {"tools/config.py", "semantic_offload/config.py"}:
            errors.append(f"footgun.environment: {label}")
        if name == "time.sleep" and path.parts[0] == "tests":
            errors.append(f"footgun.test-sleep: {label}")
        if isinstance(node, ast.Call):
            call = qualified(node.func, aliases)
            if library and call in {"print", "builtins.print"}:
                errors.append(f"footgun.print: {label}")
            if path.parts[0] == "benchmarks":
                random_call = call.startswith(("random.", "numpy.random."))
                torch_call = call in {
                    "torch.rand",
                    "torch.randn",
                    "torch.randint",
                    "torch.randperm",
                    "torch.normal",
                    "torch.multinomial",
                    "torch.bernoulli",
                }
                if random_call:
                    seed = (
                        node.args[0]
                        if node.args
                        else next(
                            (k.value for k in node.keywords if k.arg in {"seed", "x"}),
                            None,
                        )
                    )
                    seeded = isinstance(seed, ast.Constant) and isinstance(
                        seed.value, int
                    )
                    if (
                        call
                        not in {
                            "random.Random",
                            "numpy.random.default_rng",
                            "numpy.random.RandomState",
                        }
                        or not seeded
                    ):
                        errors.append(f"footgun.rng: {label}")
                elif torch_call:
                    errors.append(f"footgun.rng: {label}")
        if (
            library
            and isinstance(node, ast.Dict)
            and any(
                isinstance(value, (ast.Name, ast.Attribute)) for value in node.values
            )
        ):
            errors.append(f"layout.registry: {label}")
        if isinstance(node, ast.Name) and (
            node.id.upper() == "REGISTRY" or node.id.upper().endswith("_REGISTRY")
        ):
            errors.append(f"layout.registry: {label}")
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith(
                ("benchmarks.workloads.", "semantic_offload.policies.")
            )
        ):
            errors.append(f"layout.registry: {label}")
        if isinstance(node, ast.Import):
            if any(
                n.name.startswith(
                    ("benchmarks.workloads.", "semantic_offload.policies.")
                )
                for n in node.names
            ):
                errors.append(f"layout.registry: {label}")
        if library and isinstance(node, ast.Constant) and isinstance(node.value, str):
            if (
                Path(node.value).name
                in {"result.json", "results.json", "summary.json", "results.jsonl"}
                and label != "benchmarks/runner.py"
            ):
                errors.append(f"ownership.result-path: {label}")
    if path.stem == "registry":
        errors.append(f"layout.registry: {label}")
    if (
        path.parts[:2]
        in {("benchmarks", "workloads"), ("semantic_offload", "policies")}
        and path.name != "__init__.py"
    ):
        suffix = "_workload.py" if path.parts[0] == "benchmarks" else "_policy.py"
        if not path.name.endswith(suffix):
            errors.append(f"layout.extension-name: {label}")
    return errors


def check(root: Path, paths: list[Path]) -> list[str]:
    entries, errors = exceptions(root)
    errors.extend(comment_errors(root, paths, entries))
    for path in paths:
        file = root / path
        if not file.is_file() or file.is_symlink():
            continue
        if path.suffix == ".py":
            errors.extend(python_errors(path, file.read_text(), entries))
        if (
            path.suffix in {".sh", ".bash"}
            or (file.read_bytes().startswith(b"#!") and path.suffix != ".py")
        ) and not any(
            e.rule == "script" and e.path == path.as_posix() and e.match == "entry"
            for e in entries
        ):
            errors.append(f"layout.script: {path.as_posix()}")
    config = tomllib.loads((root / "pyproject.toml").read_text()).get("tool", {})
    if any(name in config for name in ("vulture", "deptry")) or any(
        key in config.get("ruff", {}).get("lint", {})
        for key in (
            "ignore",
            "per-file-ignores",
            "extend-ignore",
            "extend-per-file-ignores",
        )
    ):
        errors.append("exceptions.config: tool suppressions belong in exceptions.toml")
    imports = [
        item
        for contract in config.get("importlinter", {}).get("contracts", [])
        for item in contract.get("ignore_imports", [])
    ]
    allowed_imports = [
        e.match
        for e in entries
        if e.rule == "adapter-import" and e.path == "pyproject.toml"
    ]
    if sorted(imports) != sorted(allowed_imports):
        errors.append(
            "exceptions.imports: import-linter exceptions must match reasoned entries"
        )
    return sorted(set(errors))
