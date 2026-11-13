from __future__ import annotations

import math
import time
import uuid
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
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from wait_hint_offload.advice import advise
from wait_hint_offload.config import Settings
from wait_hint_offload.contracts import MAX_BOUNDARY, Hint, Registration, prefix_digest
from wait_hint_offload.observed import Observation, ObservedGroup, ObservedStore
from wait_hint_offload.vllm_adapter import engine_rpc
from wait_hint_offload.vllm_adapter.arc import (
    HOLD_SHARE,
    ARCHolds,
    ARCPrefix,
    GroupARCPrefix,
    ReachableHold,
)
from wait_hint_offload.vllm_adapter.binding import require_runtime
from wait_hint_offload.vllm_adapter.transfers import TransferMetadata, TransferWorker


class WaitHintScheduler(OffloadingConnectorScheduler):
    def __init__(self, spec: Any, vllm_config: Any, kv_cache_config: Any) -> None:
        require_runtime()
        engine_rpc.install()
        if spec.eviction_policy != "arc":
            raise ValueError("native ARC required")
        super().__init__(spec, vllm_config, kv_cache_config)
        self.settings = Settings.parse(spec.extra_config)
        if not any(
            group.sliding_window_size_in_chunks is None
            for group in self.config.kv_group_configs
        ):
            raise ValueError("full attention anchor required")
        if vllm_config.parallel_config.world_size != 1:
            raise ValueError("TP1 required")
        if not isinstance(self.manager, CPUOffloadingManager):
            raise ValueError("native CPU manager required")
        if not isinstance(self.manager._policy, ARCCachePolicy):
            raise ValueError("native ARC required")
        self.prefix = ARCPrefix(self.manager._policy)
        self.observed = ObservedStore(
            self.settings.max_sessions,
            self.settings.max_events,
            self.settings.metadata_bytes,
        )
        self.holds = ARCHolds(
            self.manager._policy,
            HOLD_SHARE,
            time.monotonic_ns,
            lambda owner: self.observed.observation(*owner),
        )
        self.clock_domain = "engine-" + uuid.uuid4().hex
        self.cpu_chunk_bytes = int(spec.kv_bytes_per_chunk)
        self.cpu_chunks = int(spec.num_blocks)

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

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        registration = (request.kv_transfer_params or {}).get("wait_hint")
        if isinstance(registration, dict):
            scope, session = registration.get("scope"), registration.get("session")
            if isinstance(scope, str) and isinstance(session, str):
                self.holds.release((scope, session))
        return result

    def request_finished(self, request: Any) -> tuple[bool, dict[str, Any] | None]:
        status = self._req_status.get(request.request_id)
        parameters = request.kv_transfer_params or {}
        registration = parameters.get("wait_hint")
        result = super().request_finished(request)
        if registration is None:
            return result
        if status is None:
            return result[0], {
                "wait_hint": {"registered": False, "reason": "untracked_request"}
            }
        try:
            registration = Registration.parse(registration)
        except ValueError as error:
            return result[0], {"wait_hint": {"registered": False, "reason": str(error)}}
        count = self._calc_num_offloadable_tokens(status, request.num_computed_tokens)
        chunk = math.lcm(
            *(group.tokens_per_chunk for group in self.config.kv_group_configs)
        )
        boundary = count // chunk * chunk
        if not boundary:
            return result[0], {
                "wait_hint": {"registered": False, "reason": "no_complete_prefix"}
            }
        if boundary > MAX_BOUNDARY:
            return result[0], {
                "wait_hint": {"registered": False, "reason": "boundary_out_of_range"}
            }
        status.update_offload_keys()
        if any(
            len(state.offload_keys) < boundary // group.tokens_per_chunk
            for group, state in zip(
                self.config.kv_group_configs, status.group_states, strict=True
            )
        ):
            return result[0], {
                "wait_hint": {"registered": False, "reason": "incomplete_prefix_keys"}
            }
        groups = tuple(
            ObservedGroup(
                group.tokens_per_chunk,
                group.sliding_window_size_in_chunks,
                tuple(
                    bytes(key)
                    for key in state.offload_keys[: boundary // group.tokens_per_chunk]
                ),
            )
            for group, state in zip(
                self.config.kv_group_configs, status.group_states, strict=True
            )
        )
        keys = next(group.keys for group in groups if group.window_chunks is None)
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
            groups if len(groups) > 1 else (),
        )
        accepted = self.observed.register(
            registration.scope, registration.session, observation
        )
        return result[0], {
            "wait_hint": {
                "registered": accepted.accepted,
                "reason": accepted.decision.value,
                "session_retired_total": self.observed.retired,
                "metadata": self.observed.metrics(),
                "cache_layout": {
                    "cpu_chunk_bytes": self.cpu_chunk_bytes,
                    "cpu_chunks": self.cpu_chunks,
                    "minimum_full_prefix_chunks": sum(
                        boundary // group.tokens_per_chunk
                        for group in self.config.kv_group_configs
                        if group.sliding_window_size_in_chunks is None
                    ),
                },
                "prefix": digest,
                "boundary": boundary,
                "observed_ns": observation.observed_ns,
                "clock_domain": self.clock_domain,
            }
        }

    def publish_wait_hint(self, raw: dict[str, Any]) -> dict[str, Any]:
        hint = Hint.parse(raw)
        observation = self.observed.observation(hint.scope, hint.session)
        prefix = (
            GroupARCPrefix(self.prefix.policy, observation.groups)
            if observation is not None and observation.groups
            else self.prefix
        )
        target = ReachableHold(
            prefix,
            self.holds,
            (hint.scope, hint.session),
            hint.deadline_ns,
            observation,
        )
        result = advise(
            self.observed,
            target,
            hint,
            time.monotonic_ns(),
            self.clock_domain,
            self.settings.advice_enabled,
        )
        return {
            "schema": "wait-hint-ack-v3",
            "ready_cpu_prefix_keys": result.ready_cpu_prefix_keys,
            "event": hint.event,
            "replayed": result.replayed,
            "newly_refreshed_keys": 0 if result.replayed else result.refreshed_keys,
            "decision": result.decision.value,
            "refreshed_keys": result.refreshed_keys,
            "held_keys": 0 if result.replayed else target.held,
            "mechanisms": self.settings.mechanisms(),
            "metadata": self.observed.metrics(),
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
        self.holds.clear()
        self.clock_domain = "engine-" + uuid.uuid4().hex


class WaitHintConnector(OffloadingConnector):
    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any) -> None:
        require_runtime()
        if vllm_config.kv_transfer_config.kv_load_failure_policy != "recompute":
            raise ValueError("kv_load_failure_policy must be recompute")
        groups = kv_cache_config.kv_cache_groups
        if not groups or not any(
            type(group.kv_cache_spec) is FullAttentionSpec for group in groups
        ):
            raise ValueError("full-attention group required")
        for group in groups:
            specification = group.kv_cache_spec
            if type(specification) in (FullAttentionSpec, SlidingWindowSpec):
                if str(specification.dtype) != "torch.float16":
                    raise ValueError("FP16 KV required")
            elif type(specification) is MambaSpec:
                if specification.mamba_cache_mode != "align":
                    raise ValueError("aligned recurrent checkpoints required")
            else:
                raise ValueError("unsupported cache group")
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
        spec = CPUOffloadingSpec(configuration)
        spec.eviction_policy = "arc"
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
