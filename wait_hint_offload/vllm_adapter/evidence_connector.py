from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_offload.base import make_offload_key

from wait_hint_offload.evidence_events import EventJournal
from wait_hint_offload.vllm_adapter.evidence_manager import ObservedManager, key_records
from wait_hint_offload.vllm_adapter.policy_context import PolicyGroup


def reachable_keys(request: Any, config: Any) -> list[dict[str, Any]]:
    alignment = math.lcm(*(group.tokens_per_chunk for group in config.kv_group_configs))
    step = config.tokens_per_hash if config.supports_partial_tail else alignment
    boundary = max(0, request.num_prompt_tokens - 1) // step * step
    if boundary == 0:
        return []
    keys: list[bytes] = []
    for group in config.kv_group_configs:
        params = group.manager_config.params
        geometry = PolicyGroup(
            group.group_idx,
            group.tokens_per_chunk,
            group.sliding_window_size_in_chunks,
            params.sliding_window - 1 + params.extra_retained_tokens
            if hasattr(params, "sliding_window")
            else None,
            tuple(
                make_offload_key(key, group.group_idx)
                for key in request.block_hashes[
                    group.hashes_per_chunk - 1 :: group.hashes_per_chunk
                ]
            ),
            tuple(
                make_offload_key(key, group.group_idx) for key in request.block_hashes
            ),
        )
        keys.extend(
            geometry.required_keys(
                boundary, boundary % alignment != 0, config.tokens_per_hash
            )
        )
    return key_records(keys)


class EvidenceConnector(OffloadingConnector):
    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any):
        super().__init__(vllm_config, role, kv_cache_config)
        native: Any = self
        if native.connector_scheduler is not None:
            manager = native.connector_scheduler.manager
            if not isinstance(manager, ObservedManager):
                raise ValueError("evidence connector requires observed CPU spec")
            self.journal = manager.journal
            config = native.connector_scheduler.config
            self.journal.emit(
                "group_config",
                {
                    "gpu_blocks": kv_cache_config.num_blocks,
                    "tokens_per_hash": config.tokens_per_hash,
                    "blocks_per_chunk": config.blocks_per_chunk,
                    "partial_tail": config.supports_partial_tail,
                    "cache_hit_alignment_tokens": config.cache_hit_alignment_tokens,
                    "groups": [
                        {
                            "group": group.group_idx,
                            "tokens_per_block": group.tokens_per_block,
                            "tokens_per_chunk": group.tokens_per_chunk,
                            "window_chunks": group.sliding_window_size_in_chunks,
                            "window_tokens": (
                                group.manager_config.params.sliding_window
                                - 1
                                + group.manager_config.params.extra_retained_tokens
                                if hasattr(
                                    group.manager_config.params, "sliding_window"
                                )
                                else None
                            ),
                            "manager": group.manager_cls.__name__,
                            "layers": list(
                                kv_cache_config.kv_cache_groups[
                                    group.group_idx
                                ].layer_names
                            ),
                        }
                        for group in config.kv_group_configs
                    ],
                },
            )
        else:
            self.journal = EventJournal(
                Path(
                    vllm_config.kv_transfer_config.kv_connector_extra_config[
                        "evidence_directory"
                    ]
                ),
                "worker",
            )

    def on_new_request(self, request: Any) -> None:
        getattr(super(), "on_new_request")(request)
        native: Any = self
        state = native.connector_scheduler._req_status[request.request_id]
        self.journal.emit(
            "request_new",
            {
                "request_id": request.request_id,
                "external_request_id": (request.kv_transfer_params or {}).get(
                    "pr_evidence_request_id"
                ),
                "prompt_tokens": request.num_prompt_tokens,
                "reachable_keys": reachable_keys(request, state.config),
                "session": (request.kv_transfer_params or {}).get("wait_hint_policy"),
                "groups": [
                    key_records(
                        make_offload_key(key, group.group_idx)
                        for key in request.block_hashes[
                            group.hashes_per_chunk - 1 :: group.hashes_per_chunk
                        ]
                    )
                    for group in state.config.kv_group_configs
                ],
                "block_hashes": [key.hex() for key in request.block_hashes],
            },
        )

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> Any:
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        self.journal.emit(
            "lookup",
            {
                "request_id": request.request_id,
                "supplied_gpu_tokens": num_computed_tokens,
                "offered_cpu_tokens": result[0],
                "asynchronous": result[1],
            },
        )
        return result

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        native: Any = self
        state = native.connector_scheduler._req_status[request.request_id]
        self.journal.emit(
            "allocated",
            {
                "request_id": request.request_id,
                "cpu_tokens": num_external_tokens,
                "load_jobs": sorted(state.transfer_jobs) if num_external_tokens else [],
                "supplied_gpu_tokens": state.num_locally_computed_tokens,
                "load_ranges": [
                    {"start": lo, "end": hi, **key_records([key])[0]}
                    for lo, hi, key in (
                        state.load_key_ranges if num_external_tokens else ()
                    )
                ],
                "blocks": [
                    [
                        {
                            "id": block.block_id,
                            "null": block.is_null,
                            "hashed": block.block_hash is not None,
                        }
                        for block in group
                    ]
                    for group in blocks.blocks
                ],
            },
        )

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        new = scheduler_output.scheduled_new_reqs
        cached = scheduler_output.scheduled_cached_reqs
        computed = {request.req_id: request.num_computed_tokens for request in new}
        computed.update(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
        native: Any = self
        scheduled = []
        for request_id, tokens in scheduler_output.num_scheduled_tokens.items():
            state = native.connector_scheduler._req_status[request_id]
            scheduled.append(
                {
                    "request_id": request_id,
                    "start": computed[request_id],
                    "tokens": tokens,
                    "prompt_tokens": state.req.num_prompt_tokens,
                    "preemptions": state.req.num_preemptions,
                }
            )
        result: Any = super().build_connector_meta(scheduler_output)
        if (
            scheduled
            or result.load_jobs
            or result.store_jobs
            or scheduler_output.preempted_req_ids
        ):
            self.journal.emit(
                "scheduled",
                {
                    "requests": scheduled,
                    "new_ids": [request.req_id for request in new],
                    "preempted_ids": sorted(scheduler_output.preempted_req_ids or ()),
                    "load_jobs": {
                        str(job): transfer.req_id
                        for job, transfer in result.load_jobs.items()
                    },
                    "store_jobs": {
                        str(job): transfer.req_id
                        for job, transfer in result.store_jobs.items()
                    },
                },
            )
        native.connector_scheduler.manager.record_policy()
        return result

    def build_connector_worker_meta(self) -> Any:
        result = getattr(super(), "build_connector_worker_meta")()
        if result is not None:
            stats = result.transfer_stats
            if result.completed_jobs or result.failed_jobs or not stats.is_empty():
                self.journal.emit(
                    "worker_transfers",
                    {
                        "completed": result.completed_jobs,
                        "failed": sorted(result.failed_jobs),
                        "load": {
                            "bytes": stats.load.bytes,
                            "seconds": stats.load.time,
                            "sizes": stats.load.sizes,
                        },
                        "store": {
                            "bytes": stats.store.bytes,
                            "seconds": stats.store.time,
                            "sizes": stats.store.sizes,
                        },
                    },
                )
        return result

    def register_kv_caches(self, kv_caches: Any) -> None:
        super().register_kv_caches(kv_caches)
        self.journal.emit(
            "gpu_kv_storage",
            {
                "layers": {
                    name: {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "bytes": tensor.numel() * tensor.element_size(),
                    }
                    for name, tensor in kv_caches.items()
                }
            },
        )

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            self.journal.close()
