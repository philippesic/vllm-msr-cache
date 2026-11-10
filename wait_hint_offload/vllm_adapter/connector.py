from __future__ import annotations

import sys
import time
import uuid
from dataclasses import replace
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from wait_hint_offload.advice import advise
from wait_hint_offload.config import Settings
from wait_hint_offload.contracts import Hint, Registration, prefix_digest
from wait_hint_offload.observed import Observation, ObservedStore
from wait_hint_offload.vllm_adapter.arc import ARCRecency
from wait_hint_offload.vllm_adapter.binding import require_binding
from wait_hint_offload.vllm_adapter.transfers import TransferMetadata, TransferWorker


class WaitHintScheduler(OffloadingConnectorScheduler):
    def __init__(self, spec: Any, vllm_config: Any, kv_cache_config: Any) -> None:
        if sys.flags.optimize:
            raise RuntimeError("optimized pinned runtime is unsupported")
        require_binding()
        super().__init__(spec, vllm_config, kv_cache_config)
        self.settings = Settings.parse(spec.extra_config)
        if (
            len(self.config.kv_group_configs) != 1
            or self.config.kv_group_configs[0].sliding_window_size_in_chunks is not None
        ):
            raise ValueError("dense single KV group required")
        if vllm_config.parallel_config.world_size != 1:
            raise ValueError("TP1 required")
        if not isinstance(self.manager, CPUOffloadingManager):
            raise ValueError("native CPU manager required")
        if not isinstance(self.manager._policy, ARCCachePolicy):
            raise ValueError("native ARC required")
        self.recency = ARCRecency(self.manager._policy)
        self.observed = ObservedStore(
            self.settings.max_sessions, self.settings.max_events
        )
        self.clock_domain = "engine-" + uuid.uuid4().hex

    def _calc_num_offloadable_tokens(
        self, req_status: Any, num_computed_tokens: int
    ) -> int:
        request = req_status.req
        if request.is_finished():
            computed = request.num_computed_tokens
            inflight = request.num_in_flight_tokens
            if (
                type(computed) is not int
                or type(inflight) is not int
                or min(computed, inflight) < 0
            ):
                return 0
            num_computed_tokens = min(num_computed_tokens, max(0, computed - inflight))
        return int(
            super()._calc_num_offloadable_tokens(req_status, num_computed_tokens)
        )

    def request_finished(self, request: Any) -> tuple[bool, dict[str, Any] | None]:
        status = self._req_status.get(request.request_id)
        parameters = request.kv_transfer_params or {}
        registration = parameters.get("wait_hint")
        result = super().request_finished(request)
        if registration is None or status is None:
            return result
        try:
            registration = Registration.parse(registration)
        except ValueError as error:
            return result[0], {"wait_hint": {"registered": False, "reason": str(error)}}
        count = self._calc_num_offloadable_tokens(status, request.num_computed_tokens)
        chunk = self.config.kv_group_configs[0].tokens_per_chunk
        boundary = count // chunk * chunk
        if not boundary:
            return result
        status.update_offload_keys()
        keys = tuple(
            bytes(key)
            for key in status.group_states[0].offload_keys[: boundary // chunk]
        )
        if len(keys) != boundary // chunk:
            return result
        digest = prefix_digest(
            list(request.all_token_ids[:boundary]), self.settings.namespace
        )
        observation = Observation(
            registration.incarnation,
            registration.generation,
            digest,
            boundary,
            self.settings.namespace,
            time.monotonic_ns(),
            keys,
        )
        accepted = self.observed.register(
            registration.scope, registration.session, observation
        )
        return result[0], {
            "wait_hint": {
                "registered": accepted,
                "session_retired_total": self.observed.retired,
                "prefix": digest,
                "boundary": boundary,
                "observed_ns": observation.observed_ns,
                "clock_domain": self.clock_domain,
            }
        }

    def publish_wait_hint(self, raw: dict[str, Any]) -> dict[str, Any]:
        hint = Hint.parse(raw)
        session = self.observed.sessions.get((hint.scope, hint.session))
        previous = session.events.get(hint.event) if session is not None else None
        decision, count = advise(
            self.observed,
            self.recency,
            hint,
            time.monotonic_ns(),
            self.clock_domain,
            self.settings.advice_enabled,
        )
        session = self.observed.sessions.get((hint.scope, hint.session))
        replayed = previous is not None and previous == (hint, decision, count)
        ready = None
        if session is not None and decision in {
            "disabled",
            "refreshed",
            "no_ready_prefix",
            "event_capacity",
            "event_conflict",
        }:
            ready = len(self.recency.ready_prefix(session.observation.keys))
        return {
            "schema": "wait-hint-ack-v2",
            "ready_cpu_prefix_keys": ready,
            "event": hint.event,
            "replayed": replayed,
            "newly_refreshed_keys": 0 if replayed else count,
            "decision": decision,
            "refreshed_keys": count,
            "mechanisms": self.settings.mechanisms(),
        }

    def update_connector_output(self, connector_output: Any) -> None:
        meta = connector_output.kv_connector_worker_meta
        if meta is not None and not isinstance(meta, TransferMetadata):
            raise TypeError("incompatible transfer metadata")
        if isinstance(meta, TransferMetadata):
            if set(meta.failed_jobs) & set(meta.completed_jobs):
                raise ValueError("conflicting transfer result")
            for job_id, count in meta.failed_jobs.items():
                if job_id < self._stale_job_threshold:
                    continue
                job = self._jobs[job_id]
                if count != 1 or job.pending_count != 1:
                    raise ValueError("TP1 transfer result count")
                status = self._req_status[job.req_id]
                if job.is_store:
                    self.manager.complete_store(
                        job.keys, status.req_context, success=False
                    )
                else:
                    self.manager.complete_load(job.keys, status.req_context)
                    if self._chunks_being_loaded is not None:
                        self._chunks_being_loaded.difference_update(job.keys)
                    failed = set(meta.failed_blocks[job_id])
                    for group in status.group_states:
                        group.block_ids[:] = [
                            0 if i in failed else i for i in group.block_ids
                        ]
                for block_id in (job.sliding_window_block_ids or []) + (
                    job.non_sliding_window_block_ids or []
                ):
                    pending = self._block_id_to_pending_jobs.get(block_id)
                    if pending is not None:
                        pending.discard(job_id)
                        if not pending:
                            del self._block_id_to_pending_jobs[block_id]
                status.transfer_jobs.remove(job_id)
                del self._jobs[job_id]
                if status.finished_signaled and not status.transfer_jobs:
                    del self._req_status[job.req_id]
        super().update_connector_output(connector_output)

    def reset_cache(self) -> None:
        super().reset_cache()
        self.observed.clear()
        self.clock_domain = "engine-" + uuid.uuid4().hex


class WaitHintConnector(OffloadingConnector):
    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any) -> None:
        if sys.flags.optimize:
            raise RuntimeError("optimized pinned runtime is unsupported")
        require_binding()
        if vllm_config.kv_transfer_config.kv_load_failure_policy != "recompute":
            raise ValueError("kv_load_failure_policy must be recompute")
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1 or type(groups[0].kv_cache_spec) is not FullAttentionSpec:
            raise ValueError("single dense full-attention group required")
        if str(groups[0].kv_cache_spec.dtype) != "torch.float16":
            raise ValueError("FP16 KV required")
        if (
            vllm_config.parallel_config.world_size != 1
            or getattr(vllm_config.parallel_config, "data_parallel_size", 1) != 1
            or vllm_config.speculative_config is not None
            or getattr(vllm_config, "lora_config", None) is not None
        ):
            raise ValueError("TP1 without speculative decoding or LoRA required")
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        configuration = build_offloading_config(vllm_config, kv_cache_config)
        Settings.parse(configuration.extra_config)
        configuration = replace(
            configuration,
            extra_config={**configuration.extra_config, "eviction_policy": "arc"},
        )
        spec = CPUOffloadingSpec(configuration)
        spec.offload_prompt_only = False
        self.connector_scheduler = None
        self.connector_worker = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = WaitHintScheduler(
                spec, vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = TransferWorker(spec, kv_cache_config)
        else:
            raise ValueError("unsupported connector role")

    def publish_wait_hint(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(self.connector_scheduler, WaitHintScheduler):
            raise RuntimeError("hint publication requires scheduler ownership")
        return self.connector_scheduler.publish_wait_hint(raw)

    def get_block_ids_with_load_errors(self) -> set[int]:
        if not isinstance(self.connector_worker, TransferWorker):
            raise RuntimeError("load errors require worker ownership")
        return self.connector_worker.get_block_ids_with_load_errors()
