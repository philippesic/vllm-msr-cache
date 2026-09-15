# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path

from benchmarks import native_child_runtime


def test_stage_child_inventory_copies_exact_parent_source_libraries(
    tmp_path, monkeypatch
):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    for root in (parent, child):
        (root / "vllm/vllm_flash_attn").mkdir(parents=True)
    for index, relative in enumerate(native_child_runtime.FLASH_BINARIES):
        (parent / relative).write_bytes(f"binary-{index}".encode())
    extra = parent / "vllm/_C_stable_libtorch.abi3.so"
    extra.write_bytes(b"core")
    base_inventory = tmp_path / "base.json"
    base_inventory.write_text("{}")
    output = tmp_path / "child-inventory.json"
    identities = {
        str(parent): {
            "path": str(parent),
            "revision": native_child_runtime.PARENT_REVISION,
            "tree": native_child_runtime.PARENT_TREE,
            "tracked_clean": True,
        },
        str(child): {
            "path": str(child),
            "revision": native_child_runtime.CHILD_REVISION,
            "tree": native_child_runtime.CHILD_TREE,
            "tracked_clean": True,
        },
    }
    monkeypatch.setattr(
        native_child_runtime,
        "_git_identity",
        lambda path: copy.deepcopy(identities[str(path)]),
    )
    monkeypatch.setattr(
        native_child_runtime,
        "verify_binary_inventory",
        lambda *_args: {"verified_file_count": 298, "verified": True},
    )
    monkeypatch.setattr(
        native_child_runtime,
        "verify_child_inventory",
        lambda *_args, **_kwargs: {"verified": True},
    )

    inventory = native_child_runtime.stage_child_inventory(
        parent,
        child,
        base_inventory,
        Path("/usr/bin/python3"),
        output,
    )

    assert inventory["binary_files_count"] == 3
    assert {entry["relative_path"] for entry in inventory["files"]} == {
        str(path) for path in native_child_runtime.FLASH_BINARIES
    } | {"vllm/_C_stable_libtorch.abi3.so"}
    for relative in (
        *native_child_runtime.FLASH_BINARIES,
        Path("vllm/_C_stable_libtorch.abi3.so"),
    ):
        destination = child / relative
        assert destination.is_file()
        assert not destination.is_symlink()
        assert destination.read_bytes() == (parent / relative).read_bytes()
    assert output.is_file()
