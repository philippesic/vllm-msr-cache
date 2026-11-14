from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from tools import agent_policy, system_map

SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".import_linter_cache",
    "build",
    "dist",
    "artifacts",
}
RAW_DIRS = {
    "output",
    "outputs",
    "runs",
    "artifacts",
    "raw",
    "vendor",
    "plans",
    "scratch",
}
DIARIES = {"STATUS.md", "PROJECT_HISTORY.md", "PORTING_PLAN.md", "plan.json"}
TASK_SECTIONS = ["## Goal", "## Authorized", "## Done when", "## State", "## Next"]
SECRET = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{30,}|https://hooks\.slack\.com/services/[^\s\"<>]+|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)
LOCK_PATTERN = re.compile(
    r"(?:\bgpu_?(?:lock|reserved|reservation)\b|\b(?:GPU|VLLM|BENCHMARK)_(?:LOCK|RESERVATION)(?:_\w+)?\b|\bLOCK_\w*_GPU\b|gpu[-_]lock|gpu[-_]reservation)",
    re.IGNORECASE,
)


def git_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return sorted({Path(p) for p in result.stdout.decode().split("\0") if p})


def visible_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for directory, dirs, files in root.walk(follow_symlinks=False):
        dirs[:] = [
            d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")
        ]
        found.extend((directory / f).relative_to(root) for f in files)
    return sorted(found)


def boundary_policy(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted((root / "wait_hint_offload").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(), filename=relative)
        adapter = relative.startswith("wait_hint_offload/vllm_adapter/")
        pure = not adapter
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.level and node.module == "vllm_adapter":
                    names = ["wait_hint_offload.vllm_adapter"]
            for name in names:
                if name.split(".")[0] == "vllm" and not adapter:
                    errors.append(f"boundary.vllm: {relative}")
                if name.split(".")[0] in {
                    "tools",
                    "tests",
                    "harness",
                    "output",
                    "vendor",
                }:
                    errors.append(f"boundary.development: {relative}")
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                if call in {"__import__", "importlib.import_module"} or call.endswith(
                    ".import_module"
                ):
                    errors.append(f"boundary.dynamic: {relative}")
            if isinstance(node, ast.ImportFrom) and node.module == "importlib":
                if any(n.name == "import_module" for n in node.names):
                    errors.append(f"boundary.dynamic: {relative}")
        if pure:
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        [n.name for n in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or ""]
                    )

                    if any(
                        n.split(".")[0]
                        not in sys.stdlib_module_names | {"wait_hint_offload"}
                        or n.startswith("wait_hint_offload.vllm_adapter")
                        for n in names
                    ):
                        errors.append(f"boundary.pure: {relative}")
                elif not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if any(isinstance(n, ast.Call) for n in ast.walk(node)):
                        errors.append(f"boundary.side_effect: {relative}")
    return sorted(set(errors))


def check(root: Path) -> list[str]:
    paths = git_paths(root)
    errors = agent_policy.check(root, paths)
    errors.extend(system_map.check(root))
    visible = visible_files(root)
    if [p.as_posix() for p in visible if p.name == "AGENTS.md"] != ["AGENTS.md"]:
        errors.append("repo.agents: require exactly one root AGENTS.md")
    pointer = root / "CLAUDE.md"
    if (
        [p.as_posix() for p in visible if p.name == "CLAUDE.md"] != ["CLAUDE.md"]
        or pointer.is_symlink()
        or not pointer.is_file()
        or pointer.read_text() != "@AGENTS.md\n"
    ):
        errors.append("repo.claude: CLAUDE.md must contain @AGENTS.md and a newline")
    for path in paths:
        label = path.as_posix()
        file = root / path
        if file.is_symlink():
            errors.append(f"repo.symlink: {label}")
            continue
        if not file.is_file():
            errors.append(f"repo.missing: {label}")
            continue
        if (
            set(path.parts) & RAW_DIRS
            or path.name in DIARIES
            or path.suffix in {".log", ".jsonl", ".whl", ".so", ".pt", ".safetensors"}
        ):
            errors.append(f"repo.artifacts: {label}")
        if file.stat().st_size > 256 * 1024:
            errors.append(f"repo.size: {label} exceeds compact-file budget (256 KiB)")
        data = file.read_bytes()
        if b"\0" in data:
            errors.append(f"repo.binary: {label}")
            continue
        text = data.decode("utf-8", errors="replace")
        if (
            path.parts[0] == "tasks"
            and path.name != "README.md"
            and (
                len(path.parts) != 2
                or path.suffix != ".md"
                or not text.startswith("# ")
                or re.findall(r"^## .*$", text, re.MULTILINE) != TASK_SECTIONS
            )
        ):
            errors.append(
                f"docs.task: {label} must be tasks/NAME.md with a title and sections "
                + ", ".join(TASK_SECTIONS)
            )
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                if re.match(
                    r"https://github\.com/philippesic/(?:wait-hint-kv|vllm-msr-cache)/",
                    target,
                ):
                    if not re.search(r"/(?:blob|tree)/[0-9a-f]{40}/", target):
                        errors.append(f"docs.immutable: {label}")
                elif not re.match(r"[a-z]+://", target) and not target.startswith("#"):
                    local = target.split("#", 1)[0]
                    if not (file.parent / local).exists():
                        errors.append(f"docs.link: {label}")
        if SECRET.search(text):
            errors.append(f"repo.secret: {label} (redacted)")
        if path.suffix == ".py":
            allowed, _ = agent_policy.exceptions(root)
            if re.search(r"#\s*(?:type:\s*ignore|mypy:)", text) and not any(
                e.rule == "directive" and e.path == label and e.match in text
                for e in allowed
            ):
                errors.append(f"types.suppression: {label}")

            tree = ast.parse(text, filename=label)
            for node in ast.walk(tree):
                token = ""
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    token = node.name
                elif isinstance(node, ast.Name):
                    token = node.id
                elif isinstance(node, ast.Attribute):
                    token = node.attr
                elif isinstance(node, ast.alias):
                    token = node.name
                elif isinstance(node, ast.ImportFrom):
                    token = node.module or ""
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    token = node.value if re.fullmatch(r"[\w./-]+", node.value) else ""
                if LOCK_PATTERN.search(token):
                    errors.append(f"repo.gpu_lock: {label}")
            if LOCK_PATTERN.search(path.stem):
                errors.append(f"repo.gpu_lock: {label}")
    authored = [
        p
        for p in visible
        if p.parts[0]
        in {
            "tools",
            "tests",
            ".github",
            ".githooks",
            "docs",
            "wait_hint_offload",
            "tasks",
            "results",
            "skills",
        }
        or p.name in {"AGENTS.md", "CLAUDE.md", "pyproject.toml", "uv.lock"}
    ]
    if authored:
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin"],
            cwd=root,
            input="\n".join(p.as_posix() for p in authored),
            text=True,
            capture_output=True,
        )
        if ignored.returncode not in {0, 1}:
            raise RuntimeError("git check-ignore failed")
        errors.extend(
            f"repo.ignored_workflow: {p}" for p in ignored.stdout.splitlines()
        )
    config = root / "pyproject.toml"
    if config.exists():
        mypy = tomllib.loads(config.read_text()).get("tool", {}).get("mypy", {})
        if (
            mypy.get("strict") is not True
            or any(
                mypy.get(n)
                for n in (
                    "ignore_missing_imports",
                    "ignore_errors",
                    "disable_error_code",
                )
            )
            or "overrides" in mypy
            or any(
                pattern not in {"wait_hint_offload/vllm_adapter/", "tests/runtime/"}
                for pattern in mypy.get("exclude", [])
            )
        ):
            errors.append("types.config: require strict without broad exemptions")
    return sorted(set(errors))
