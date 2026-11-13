from __future__ import annotations

import hashlib
from typing import Any

import torch


class KVProbe:
    model_runner: Any
    kv_snapshot: dict[int, torch.Tensor]

    def kv_storages(self) -> dict[int, torch.Tensor]:
        blocks = int(self.model_runner.kv_cache_config.num_blocks)
        stores: dict[int, torch.Tensor] = {}
        for tensor in self.model_runner.kv_caches:
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in stores:
                raw = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                stores[storage.data_ptr()] = raw.set_(storage).view(blocks, -1)
        return stores

    def snapshot_kv(self) -> int:
        self.kv_snapshot = {
            pointer: rows.cpu() for pointer, rows in self.kv_storages().items()
        }
        return len(self.kv_snapshot)

    def clear_kv(self) -> int:
        stores = self.kv_storages()
        for rows in stores.values():
            rows.zero_()
        torch.cuda.synchronize()
        return len(stores)

    def compare_kv(self) -> dict[str, int]:
        written = matched = 0
        for pointer, rows in self.kv_storages().items():
            before = self.kv_snapshot[pointer]
            after = rows.cpu()
            stored: dict[bytes, int] = {}
            for block in range(before.shape[0]):
                digest = hashlib.blake2b(before[block].numpy().tobytes()).digest()
                stored.setdefault(digest, block)
            for block in torch.nonzero(after.any(dim=1)).flatten().tolist():
                written += 1
                digest = hashlib.blake2b(after[block].numpy().tobytes()).digest()
                source = stored.get(digest)
                matched += int(
                    source is not None and torch.equal(before[source], after[block])
                )
        self.kv_snapshot = {}
        return {
            "blocks_written_after_clear": written,
            "written_blocks_identical_to_a_stored_block": matched,
        }
