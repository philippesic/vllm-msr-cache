# SPDX-License-Identifier: Apache-2.0
"""Stage and verify the pinned native child runtime's compiled overlay."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.eviction_prefetch_campaign import digest, file_digest
from benchmarks.wait_hint_serving_diagnostic import verify_binary_inventory

SCHEMA = "ep-native-child-binary-overlay-v1"
PARENT_REVISION = "149cb6d864c46a88f38ab8602c7e3071034f8512"
PARENT_TREE = "adb805d0bd51486cde394f06c36b72a933617f2e"
CHILD_REVISION = "f3f036805c886c562d1b5397da40a898f163e93a"
CHILD_TREE = "c4b6cd9a94426340256f0957215e7bbb03ff1bc9"
FLASH_BINARIES = (
    Path("vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"),
    Path("vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so"),
)


class NativeChildError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeChildError(message)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def _git_identity(path: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "path": str(path.resolve()),
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_clean": not git("status", "--porcelain", "--untracked-files=no"),
    }


def _validate_source_identities(parent_root: Path, child_root: Path) -> None:
    parent = _git_identity(parent_root)
    child = _git_identity(child_root)
    _require(
        parent["revision"] == PARENT_REVISION
        and parent["tree"] == PARENT_TREE
        and parent["tracked_clean"] is True,
        "native parent identity",
    )
    _require(
        child["revision"] == CHILD_REVISION
        and child["tree"] == CHILD_TREE
        and child["tracked_clean"] is True,
        "native child identity",
    )


def _file_entry(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"native overlay target is not a file: {path}")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_digest(resolved),
        "symlink": path.is_symlink(),
    }


def _overlay_entry(path: Path, child_root: Path) -> dict[str, object]:
    entry = _file_entry(path)
    entry["relative_path"] = str(path.relative_to(child_root))
    return entry


def verify_child_inventory(
    inventory_path: Path,
    expected_sha256: str,
    *,
    run_import_probe: bool = True,
) -> dict[str, object]:
    _require(
        inventory_path.is_file() and file_digest(inventory_path) == expected_sha256,
        "native child inventory artifact hash",
    )
    with inventory_path.open(encoding="utf-8") as source:
        inventory = json.load(source)
    body = dict(inventory)
    inventory_digest = body.pop("inventory_digest", None)
    _require(
        inventory.get("schema") == SCHEMA and inventory_digest == digest(body),
        "native child inventory schema/digest",
    )
    parent_root = Path(str(inventory["parent"]["path"]))
    child_root = Path(str(inventory["child"]["path"]))
    _validate_source_identities(parent_root, child_root)
    base = inventory.get("base_inventory")
    _require(isinstance(base, Mapping), "native child base inventory")
    base_result = verify_binary_inventory(
        Path(str(base.get("path"))), str(base.get("sha256"))
    )
    entries = inventory.get("files")
    _require(isinstance(entries, list), "native child inventory files")
    observed = []
    for expected in entries:
        _require(isinstance(expected, Mapping), "native child inventory entry")
        path = Path(str(expected.get("path")))
        actual = _overlay_entry(path, child_root)
        _require(actual == expected, f"native child binary mismatch: {path}")
        parent_entry = _file_entry(parent_root / str(actual["relative_path"]))
        _require(
            actual["bytes"] == parent_entry["bytes"]
            and actual["sha256"] == parent_entry["sha256"],
            f"native child binary differs from parent: {path}",
        )
        observed.append(actual)
    _require(
        {entry["relative_path"] for entry in observed}
        == {
            str(path.relative_to(parent_root))
            for path in parent_root.rglob("*.so")
            if path.is_file()
        }
        and {entry["relative_path"] for entry in observed}
        >= {str(path) for path in FLASH_BINARIES}
        and all(entry["symlink"] is False for entry in observed),
        "native child source-package binary closure",
    )
    import_probe = inventory.get("import_probe")
    _require(isinstance(import_probe, Mapping), "native child import probe")
    if run_import_probe:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(child_root)
        completed = subprocess.run(
            list(import_probe["argv"]),
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
            check=False,
        )
        _require(
            completed.returncode == 0
            and "native_child_flash_import=PASS" in completed.stdout,
            "native child exact import probe",
        )
    return {
        "schema": "ep-native-child-binary-overlay-verification-v1",
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": expected_sha256,
        "base_verified_file_count": base_result["verified_file_count"],
        "child_verified_file_count": len(observed),
        "child_verified_entries_digest": digest(observed),
        "import_probe_executed": run_import_probe,
        "verified": True,
    }


def stage_child_inventory(
    parent_root: Path,
    child_root: Path,
    base_inventory_path: Path,
    python_path: Path,
    output_path: Path,
) -> dict[str, object]:
    _validate_source_identities(parent_root, child_root)
    base_sha256 = file_digest(base_inventory_path)
    verify_binary_inventory(base_inventory_path, base_sha256)
    actions = []
    parent_libraries = sorted(
        path for path in parent_root.rglob("*.so") if path.is_file()
    )
    _require(
        {path.relative_to(parent_root) for path in parent_libraries}
        >= set(FLASH_BINARIES),
        "native parent flash binary closure",
    )
    for source in parent_libraries:
        relative = source.relative_to(parent_root)
        destination = child_root / relative
        if destination.is_file() and not destination.is_symlink():
            _require(
                file_digest(destination) == file_digest(source)
                and destination.stat().st_size == source.stat().st_size,
                f"native child binary differs: {destination}",
            )
            action = "existing_exact_copy"
        else:
            _require(
                not destination.exists(),
                f"native child path already exists: {destination}",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            action = "created_exact_copy"
        actions.append(
            {
                "action": action,
                "source": str(source),
                "destination": str(destination),
            }
        )
    _validate_source_identities(parent_root, child_root)
    files = [
        _overlay_entry(path, child_root)
        for path in sorted(child_root.rglob("*.so"))
        if path.is_file()
    ]
    import_probe = {
        "environment": {"PYTHONPATH": str(child_root)},
        "argv": [
            str(python_path),
            "-c",
            (
                "import vllm.vllm_flash_attn; "
                "import vllm._C_stable_libtorch; "
                "import vllm.vllm_flash_attn._vllm_fa2_C; "
                "print('native_child_flash_import=PASS')"
            ),
        ],
    }
    inventory = {
        "schema": SCHEMA,
        "parent": _git_identity(parent_root),
        "child": _git_identity(child_root),
        "base_inventory": {
            "path": str(base_inventory_path),
            "sha256": base_sha256,
        },
        "staging_actions": actions,
        "files": files,
        "binary_files_count": len(files),
        "import_probe": import_probe,
    }
    inventory["inventory_digest"] = digest(inventory)
    _write_json(output_path, inventory)
    verify_child_inventory(output_path, file_digest(output_path))
    return inventory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--parent-root", required=True, type=Path)
    stage.add_argument("--child-root", required=True, type=Path)
    stage.add_argument("--base-inventory", required=True, type=Path)
    stage.add_argument("--python", required=True, type=Path)
    stage.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--inventory", required=True, type=Path)
    verify.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "stage":
        stage_child_inventory(
            args.parent_root,
            args.child_root,
            args.base_inventory,
            args.python,
            args.output,
        )
        return 0
    verify_child_inventory(args.inventory, args.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
