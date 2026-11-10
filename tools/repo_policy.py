"""Small, fail-closed repository checks. Diagnostics never echo file contents."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

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
SECRET = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{30,}|https://hooks\.slack\.com/services/[^\s\"<>]+|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)
LOCK_PATTERN = re.compile(
    r"(?:gpu_?(?:lock|reserved|reservation)|(?:GPU|VLLM|BENCHMARK)_\w*(?:LOCK|RESERV)\w*|(?:LOCK|RESERV)\w*_GPU|gpu[-_]lock|gpu[-_]reservation)",
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
    """Include ignored instruction/workflow files, excluding dependency/cache trees."""
    found: list[Path] = []
    for directory, dirs, files in root.walk(follow_symlinks=False):
        dirs[:] = [
            d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")
        ]
        found.extend((directory / f).relative_to(root) for f in files)
    return sorted(found)


def boundary_policy(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted((root / "semantic_offload").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(), filename=relative)
        adapter = relative.startswith("semantic_offload/vllm_adapter/")
        pure = relative == "semantic_offload/contracts.py"
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.level and node.module == "vllm_adapter":
                    names = ["semantic_offload.vllm_adapter"]
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
                        n.split(".")[0] not in sys.stdlib_module_names for n in names
                    ):
                        errors.append(f"boundary.pure: {relative}")
                elif not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    if any(isinstance(n, ast.Call) for n in ast.walk(node)):
                        errors.append(f"boundary.side_effect: {relative}")
    return sorted(set(errors))


def check(root: Path) -> list[str]:
    errors: list[str] = []
    paths = git_paths(root)
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
    agents = root / "AGENTS.md"
    if agents.is_file() and (
        len(agents.read_bytes()) > 8192 or len(agents.read_text().splitlines()) > 80
    ):
        errors.append("repo.budget: AGENTS.md exceeds 80 lines or 8192 bytes")
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
        if ".claude" in path.parts or "skills" in path.parts or path.name == "SKILL.md":
            errors.append(f"repo.skills: {label}")
        if file.stat().st_size > 256 * 1024:
            errors.append(f"repo.size: {label} exceeds compact-file budget (256 KiB)")
        data = file.read_bytes()
        if b"\0" in data:
            errors.append(f"repo.binary: {label}")
            continue
        text = data.decode("utf-8", errors="replace")
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                if target.startswith("https://github.com/philippesic/vllm-msr-cache/"):
                    if not re.search(r"/(?:blob|tree)/[0-9a-f]{40}/", target):
                        errors.append(f"docs.immutable: {label}")
                elif not re.match(r"[a-z]+://", target) and not target.startswith("#"):
                    local = target.split("#", 1)[0]
                    if not (file.parent / local).exists():
                        errors.append(f"docs.link: {label}")
        if SECRET.search(text):
            errors.append(f"repo.secret: {label} (redacted)")
        if path.suffix == ".py":
            if re.search(r"#\s*(?:type:\s*ignore|mypy:)", text):
                errors.append(f"types.suppression: {label}")
            # Inspect Python tokens via AST, so policy patterns in the checker itself
            # and specimens encoded in JSON do not become false lock detections.
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
                    # Match complete lock identifiers/paths, not explanatory prose/regex.
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
            "semantic_offload",
            "benchmarks",
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
        ):
            errors.append("types.config: require strict without broad exemptions")
    return sorted(set(errors))
