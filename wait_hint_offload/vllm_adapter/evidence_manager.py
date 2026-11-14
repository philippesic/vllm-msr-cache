from __future__ import annotations

from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

from vllm.v1.kv_offload import base
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from wait_hint_offload.evidence_events import EventJournal
from wait_hint_offload.vllm_adapter.policy_context import prefix_from_context


def key_records(keys: Iterable[bytes], context: Any = None) -> list[dict[str, Any]]:
    group_index = getattr(base, "get_offload_group_idx")
    return [
        {
            "key": key.hex(),
            "group": group_index(key),
            "position": context.get_offload_key_position(key) if context else None,
        }
        for key in keys
    ]


class ObservedManager:
    def __init__(self, native: Any, journal: EventJournal):
        self.native = native
        self.journal = journal
        self.last_policy: object = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def prepare_store(
        self, keys: Iterable[bytes], context: Any, *, trigger: Any
    ) -> Any:
        offered = tuple(keys)
        result = self.native.prepare_store(offered, context, trigger=trigger)
        self.journal.emit(
            "prepare_store",
            {
                "request_id": context.req_id,
                "offered": key_records(offered, context),
                "trigger": str(trigger),
                "accepted": result is not None,
                "stored": key_records(result.keys_to_store, context) if result else [],
                "evicted": key_records(result.evicted_keys) if result else [],
                "cpu_block_ids": result.store_spec.block_ids.tolist() if result else [],
            },
        )
        return result

    def complete_store(
        self,
        keys: Collection[bytes],
        context: Any,
        *,
        trigger: Any,
        success: bool = True,
    ) -> None:
        self.native.complete_store(keys, context, trigger=trigger, success=success)
        self.journal.emit(
            "complete_store",
            {
                "request_id": context.req_id,
                "keys": key_records(keys, context),
                "success": success,
                "ready": [
                    key.hex()
                    for key in keys
                    if (chunk := self.native._policy.get(key)) is not None
                    and chunk.is_ready
                ],
            },
        )

    def prepare_load(self, keys: Collection[bytes], context: Any) -> Any:
        result = self.native.prepare_load(keys, context)
        self.journal.emit(
            "prepare_load",
            {
                "request_id": context.req_id,
                "keys": key_records(keys, context),
                "cpu_block_ids": result.block_ids.tolist(),
            },
        )
        return result

    def complete_load(self, keys: Collection[bytes], context: Any) -> None:
        self.native.complete_load(keys, context)
        self.journal.emit(
            "complete_load",
            {"request_id": context.req_id, "keys": key_records(keys, context)},
        )

    def on_request_finished(self, context: Any) -> None:
        self.native.on_request_finished(context)
        prefix = prefix_from_context(context)
        self.journal.emit(
            "request_finished",
            {
                "request_id": context.req_id,
                "complete_tokens": prefix.complete_tokens,
                "alignment_tokens": prefix.alignment_tokens,
                "tokens_per_hash": prefix.tokens_per_hash,
                "partial_tail": prefix.partial_tail,
                "groups": [
                    {
                        "group": group.group_id,
                        "tokens_per_chunk": group.tokens_per_chunk,
                        "window_chunks": group.window_chunks,
                        "window_tokens": group.window_tokens,
                        "keys": [key.hex() for key in group.keys],
                        "boundary_keys": [key.hex() for key in group.boundary_keys],
                    }
                    for group in prefix.groups
                ],
            },
        )
        self.record_policy()

    def record_policy(self) -> None:
        policy = self.native._policy
        book = getattr(policy, "book", None)
        signature = (
            tuple(getattr(policy, "outcomes", {}).items()),
            tuple(
                (owner, lease.generation, lease.deadline_ns, len(lease.keys))
                for owner, lease in (book.leases.items() if book else ())
            ),
            book.yields if book else 0,
            len(book.references) if book else 0,
        )
        if signature == self.last_policy:
            return
        self.last_policy = signature
        self.journal.emit(
            "policy",
            {
                "outcomes": dict(getattr(policy, "outcomes", {})),
                "physical_held_chunks": len(book.references) if book else 0,
                "leases": {
                    owner: {
                        "generation": lease.generation,
                        "deadline_ns": lease.deadline_ns,
                        "keys": key_records(sorted(lease.keys)),
                    }
                    for owner, lease in (book.leases.items() if book else ())
                },
                "yields": book.yields if book else 0,
                "metadata_bytes": (
                    book.metadata_bytes + policy.observation_bytes if book else 0
                ),
            },
        )

    def reset_cache(self) -> None:
        self.native.reset_cache()
        self.journal.emit("reset", {})


class ObservedCPUOffloadingSpec(CPUOffloadingSpec):
    def get_manager(self) -> Any:
        observed = getattr(self, "_evidence_manager", None)
        if observed is None:
            native = super().get_manager()
            journal = EventJournal(
                Path(self.extra_config["evidence_directory"]), "scheduler"
            )
            observed = ObservedManager(native, journal)
            self._evidence_manager = observed
            journal.emit(
                "cpu_config",
                {
                    "num_chunks": getattr(self, "num_chunks"),
                    "kv_bytes_per_chunk": self.kv_bytes_per_chunk,
                    "cpu_page_size_per_worker": self.cpu_page_size_per_worker,
                    "policy": self.eviction_policy,
                    "policy_module": self.cache_policy_module_path,
                },
            )
        return observed
