# SPDX-License-Identifier: Apache-2.0
"""SemanticOffloadingSpec: CPUOffloadingSpec with a SemanticOffloadingManager.

Register via kv_connector_extra_config={"spec_name": "SemanticOffloadingSpec",
"spec_module_path": "semantic_offload.spec", "cpu_bytes_to_use": ...}. See
.claude/docs/semantic-eviction-plan.md, Step 1.1.
"""

import json

from typing_extensions import override
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingManager
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from semantic_offload._debug import record_runtime_resolution
from semantic_offload._vllm_compat import spec_blocks_per_chunk
from semantic_offload.manager import (
    PreservationOffloadingManager,
    SemanticOffloadingManager,
    validate_ema_rank_weight_floor,
)
from semantic_offload.policy import DEFAULT_POLICY_ALPHA
from semantic_offload.worker import (
    SemanticOffloadingWorker,
    validate_rank_ema_mode,
    validate_scoring_mode,
)


def validate_preservation_audit(extra_config) -> bool:
    """Keep full evidence by default; allow a narrow operational ARC profile.

    Disabling audit removes request histories, not transfer failure handling.
    It is deliberately unavailable to semantic scoring, staging and lifecycle
    policies whose decisions or qualifications depend on those histories.
    """
    audit = extra_config.get("preservation_audit", True)
    if not isinstance(audit, bool):
        raise TypeError("preservation_audit must be a JSON boolean")
    if audit:
        return True
    required = {
        "preservation_events": True,
        "instrumentation_only": True,
        "eviction_policy": "arc",
        "prefetch_trigger": "known_prefix_wait_hint",
        "wait_hint_diagnostic_no_dispatch": True,
    }
    for name, expected in required.items():
        value = extra_config.get(name)
        if value != expected or (isinstance(expected, bool) and type(value) is not bool):
            raise ValueError(f"preservation_audit=false requires {name}={expected!r}")
    defaults = {
        "shared_prefetch_scorer": False,
        "prefix_mean_prefetch_scoring": False,
        "lifecycle_release_enabled": False,
        "decision_feature_trace": False,
        "bounded_prefill_cuda_timing": False,
        "scoring_mode": "synchronous",
        "scoring_schedule": "continuous",
        "admission_mode": "disabled",
        "soft_retention": "off",
    }
    for name, expected in defaults.items():
        value = extra_config.get(name, expected)
        if value != expected or (isinstance(expected, bool) and type(value) is not bool):
            raise ValueError(f"preservation_audit=false requires {name}={expected!r}")
    if (extra_config.get("wait_hint_cpu_retention_only") is not True
            and extra_config.get("prefetch_budget_fraction") != 0):
        raise ValueError("preservation_audit=false requires CPU retention-only or a zero prefetch budget")
    size = extra_config.get("observed_session_registry_size", 4096)
    if type(size) is not int or size <= 0:
        raise ValueError("observed_session_registry_size must be a positive JSON integer")
    return False


class SemanticOffloadingSpec(CPUOffloadingSpec):
    def __init__(self, config):
        super().__init__(config)
        rank = self.extra_config.get("kv_svd_rank", 0)
        expand = self.extra_config.get("kv_svd_expand_capacity", False)
        if type(rank) is not int or not 0 <= rank <= 15:
            raise ValueError("kv_svd_rank must be an integer in 0..15")
        if type(expand) is not bool:
            raise TypeError("kv_svd_expand_capacity must be a JSON boolean")
        if rank and expand and self.kv_bytes_per_chunk:
            # FP16 16x256 -> 16xr plus rx256. Include the ready bitmap;
            # Python policy/summary metadata is separately measured as RSS.
            factor_bytes = self.kv_bytes_per_chunk * rank * 272 // 4096
            self.num_blocks = int(self.extra_config["cpu_bytes_to_use"]) // (factor_bytes + 1)

    @classmethod
    def build_metric_definitions(cls, extra_config):
        from semantic_offload.benchmark_metrics import definitions
        return {**super().build_metric_definitions(extra_config), **definitions(extra_config)}

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> SemanticOffloadingWorker:
        preservation_audit = validate_preservation_audit(self.extra_config)
        # Worker-side scoring only ever needs the one configured method
        # (SemanticPolicy only consults its own method's relevance EMA) --
        # same extra_config key get_manager() already reads below.
        method = str(self.extra_config.get("method", "mean"))
        # Only every capture_stride-th eligible query-capture step actually
        # scores/updates relevance (default 1: unchanged, every step) -- a
        # TTFT-tax follow-up knob, see query_capture.py's
        # `_should_sample_step` and semantic-eviction-plan.md's post-fix
        # optimization pass. Relies on the manager's EMA staleness-tolerance
        # to carry signal across skipped steps.
        capture_stride = int(self.extra_config.get("capture_stride", 1))
        instrumentation_only = self.extra_config.get("instrumentation_only", False)
        if not isinstance(instrumentation_only, bool):
            raise TypeError("instrumentation_only must be a JSON boolean")
        shared_prefetch_scorer = self.extra_config.get(
            "shared_prefetch_scorer", False
        )
        if not isinstance(shared_prefetch_scorer, bool):
            raise TypeError("shared_prefetch_scorer must be a JSON boolean")
        if shared_prefetch_scorer and not instrumentation_only:
            raise ValueError(
                "shared_prefetch_scorer is only for instrumentation-only controls"
            )
        probe_layer = (
            None
            if instrumentation_only and not shared_prefetch_scorer
            else self.extra_config.get("probe_layer", "middle")
        )
        head_aggregation = str(self.extra_config.get("head_aggregation", "mean"))
        rank_ema_mode = str(self.extra_config.get("rank_ema_mode", "default"))
        validate_rank_ema_mode(rank_ema_mode)
        scoring_mode = validate_scoring_mode(
            self.extra_config.get("scoring_mode", "synchronous")
        )
        def positive_int(name: str, default: int) -> int:
            value = self.extra_config.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive JSON integer")
            return value

        async_scoring_max_queue_items = positive_int(
            "async_scoring_max_queue_items", 2
        )
        async_scoring_max_snapshot_bytes = positive_int(
            "async_scoring_max_snapshot_bytes", 64 * 1024 * 1024
        )
        async_scoring_workers = positive_int("async_scoring_workers", 2)
        async_scoring_max_snapshot_age_ms = positive_int(
            "async_scoring_max_snapshot_age_ms", 1000
        )
        if scoring_mode == "asynchronous" and len(self.tokens_per_block) != 1:
            raise ValueError("asynchronous scoring initially requires one dense KV group")
        ema_rank_weight_floor = validate_ema_rank_weight_floor(
            self.extra_config.get("ema_rank_weight_floor", 0.0)
        )
        summary_offload_overlap = self.extra_config.get(
            "summary_offload_overlap", False
        )
        if not isinstance(summary_offload_overlap, bool):
            raise TypeError("summary_offload_overlap must be a JSON boolean")
        preservation_events = self.extra_config.get("preservation_events", False)
        if not isinstance(preservation_events, bool):
            raise TypeError("preservation_events must be a JSON boolean")
        if scoring_mode == "asynchronous" and not preservation_events:
            raise ValueError("asynchronous scoring requires preservation_events")
        nvtx_ranges = self.extra_config.get("nvtx_ranges", False)
        if not isinstance(nvtx_ranges, bool):
            raise TypeError("nvtx_ranges must be a JSON boolean")
        lifecycle_release_enabled = self.extra_config.get(
            "lifecycle_release_enabled", False
        )
        if not isinstance(lifecycle_release_enabled, bool):
            raise TypeError("lifecycle_release_enabled must be a JSON boolean")
        if lifecycle_release_enabled and not preservation_events:
            raise ValueError("lifecycle_release_enabled requires preservation_events")
        decision_feature_trace = self.extra_config.get("decision_feature_trace", False)
        if not isinstance(decision_feature_trace, bool):
            raise TypeError("decision_feature_trace must be a JSON boolean")
        if decision_feature_trace and not preservation_events:
            raise ValueError(
                "decision_feature_trace requires preservation_events "
                "for provenance binding"
            )
        if preservation_events:
            scheduler_config = getattr(self.vllm_config, "scheduler_config", None)
            model_config = getattr(self.vllm_config, "model_config", None)
            cache_config = getattr(self.vllm_config, "cache_config", None)
            parallel_config = getattr(self.vllm_config, "parallel_config", None)
            async_scheduling = getattr(scheduler_config, "async_scheduling", None)
            scheduler_class = None
            get_scheduler_class = getattr(scheduler_config, "get_scheduler_cls", None)
            if callable(get_scheduler_class):
                resolved_class = get_scheduler_class()
                if isinstance(resolved_class, str):
                    scheduler_class = resolved_class
                else:
                    scheduler_class = (
                        f"{resolved_class.__module__}.{resolved_class.__qualname__}"
                    )
            dtype = getattr(model_config, "dtype", None)
            runtime_resolution = {
                "schema": "semantic-runtime-resolution-v1",
                "async_scheduling": async_scheduling,
                "scheduler_class": scheduler_class,
                "dtype": (
                    str(dtype).removeprefix("torch.") if dtype is not None else None
                ),
                "tensor_parallel_size": getattr(
                    parallel_config, "tensor_parallel_size", None
                ),
                "enforce_eager": getattr(model_config, "enforce_eager", None),
                "max_num_seqs": getattr(scheduler_config, "max_num_seqs", None),
                "max_num_batched_tokens": getattr(
                    scheduler_config, "max_num_batched_tokens", None
                ),
                "kv_block_size_tokens": getattr(cache_config, "block_size", None),
                "max_model_len": getattr(model_config, "max_model_len", None),
                "gpu_memory_utilization": getattr(
                    cache_config, "gpu_memory_utilization", None
                ),
                "num_gpu_blocks_override": getattr(
                    cache_config, "num_gpu_blocks_override", None
                ),
            }
            runtime_resolution["available"] = all(
                value is not None
                for key, value in runtime_resolution.items()
                if key not in {"schema", "available"}
            )
            if not preservation_audit:
                runtime_resolution["preservation_audit"] = False
                runtime_resolution["preservation_proofs_available"] = False
                runtime_resolution["observed_session_registry_size"] = (
                    self.extra_config.get("observed_session_registry_size", 4096)
                )
                runtime_resolution["session_registry_expiry"] = (
                    "none; capacity rejects new session advice until cache reset"
                )
            record_runtime_resolution(runtime_resolution)
        admission_mode = self.extra_config.get("admission_mode", "disabled")
        if admission_mode != "disabled" and not preservation_events:
            raise ValueError(
                "admission_mode requires preservation_events for causal tracing"
            )
        worker = SemanticOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=spec_blocks_per_chunk(self),
            num_cpu_blocks=self.num_blocks,
            vllm_config=self.vllm_config,
            method=method,
            capture_stride=capture_stride,
            probe_layer=probe_layer,
            head_aggregation=head_aggregation,
            rank_ema_mode=rank_ema_mode,
            scoring_mode=scoring_mode,
            async_scoring_max_queue_items=async_scoring_max_queue_items,
            async_scoring_max_snapshot_bytes=async_scoring_max_snapshot_bytes,
            async_scoring_workers=async_scoring_workers,
            async_scoring_max_snapshot_age_ms=async_scoring_max_snapshot_age_ms,
            ema_rank_weight_floor=ema_rank_weight_floor,
            summary_offload_overlap=summary_offload_overlap,
            preservation_events=preservation_events,
            nvtx_ranges=nvtx_ranges,
            scoring_schedule=self.extra_config.get("scoring_schedule", "continuous"),
            scoring_search=self.extra_config.get("scoring_search", "flat"),
            search_min_pool=positive_int("search_min_pool", 128),
            search_clusters=positive_int("search_clusters", 16),
            search_probes=positive_int("search_probes", 4),
            search_exploration=positive_int("search_exploration", 8),
            query_snapshot_max_age_ms=positive_int("query_snapshot_max_age_ms", 1000),
            query_snapshot_max_requests=positive_int("query_snapshot_max_requests", 256),
            kv_svd_rank=self.extra_config.get("kv_svd_rank", 0),
            kv_svd_trace=self.extra_config.get("kv_svd_trace"),
        )
        if self.extra_config.get("benchmark_telemetry") is True:
            from semantic_offload.benchmark_metrics import allocation_receipt
            observed = allocation_receipt(worker, num_cpu_blocks=self.num_blocks,
                kv_bytes_per_chunk=self.kv_bytes_per_chunk,
                requested_cpu_bytes=self.extra_config.get("cpu_bytes_to_use"),
                blocks_per_chunk=spec_blocks_per_chunk(self),
                model_config=self.vllm_config.model_config)
            print("SEMANTIC_OFFLOAD_ALLOCATION " + json.dumps(observed), flush=True)
        return worker

    @override
    def get_manager(self) -> OffloadingManager:
        preservation_audit = validate_preservation_audit(self.extra_config)
        if not self._manager:
            preservation_events = self.extra_config.get("preservation_events", False)
            instrumentation_only = self.extra_config.get("instrumentation_only", False)
            if not isinstance(instrumentation_only, bool):
                raise TypeError("instrumentation_only must be a JSON boolean")
            shared_prefetch_scorer = self.extra_config.get(
                "shared_prefetch_scorer", False
            )
            if not isinstance(shared_prefetch_scorer, bool):
                raise TypeError("shared_prefetch_scorer must be a JSON boolean")
            if shared_prefetch_scorer and not instrumentation_only:
                raise ValueError(
                    "shared_prefetch_scorer is only for instrumentation-only controls"
                )
            if not isinstance(preservation_events, bool):
                raise TypeError("preservation_events must be a JSON boolean")
            lifecycle_release_enabled = self.extra_config.get(
                "lifecycle_release_enabled", False
            )
            if not isinstance(lifecycle_release_enabled, bool):
                raise TypeError("lifecycle_release_enabled must be a JSON boolean")
            if lifecycle_release_enabled and not preservation_events:
                raise ValueError(
                    "lifecycle_release_enabled requires preservation_events"
                )
            decision_feature_trace = self.extra_config.get(
                "decision_feature_trace", False
            )
            if not isinstance(decision_feature_trace, bool):
                raise TypeError("decision_feature_trace must be a JSON boolean")
            if decision_feature_trace and not preservation_events:
                raise ValueError(
                    "decision_feature_trace requires preservation_events "
                    "for provenance binding"
                )
            admission_mode = self.extra_config.get("admission_mode", "disabled")
            admission_cost_per_byte = self.extra_config.get(
                "admission_cost_per_byte"
            )
            admission_default_cost_per_byte = self.extra_config.get(
                "admission_default_cost_per_byte", 1.0
            )
            admission_namespace = self.extra_config.get("admission_namespace")
            chain_admission_max_prefix_blocks = self.extra_config.get(
                "chain_admission_max_prefix_blocks", 0
            )
            if admission_mode != "disabled" and not preservation_events:
                raise ValueError(
                    "admission_mode requires preservation_events for causal tracing"
                )
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))
            if instrumentation_only:
                eviction_policy = str(self.extra_config.get("eviction_policy", "lru"))
                if eviction_policy not in ("lru", "arc"):
                    raise ValueError(
                        "instrumentation-only preservation supports only lru/arc"
                    )
                # The instrumentation-only path deliberately keeps vLLM's
                # stock cache policy and block/refcount behavior. The small
                # subclass only exposes the stock manager's already-authoritative
                # evictions to the shared request-event seam.
                self._manager = PreservationOffloadingManager(
                    num_blocks=self.num_blocks,
                    cache_policy=eviction_policy,
                    enable_events=(
                        self.kv_events_config.enable_kv_cache_events
                        or (preservation_events and preservation_audit)
                    ),
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                    admission_mode=admission_mode,
                    admission_cost_per_byte=admission_cost_per_byte,
                    admission_default_cost_per_byte=admission_default_cost_per_byte,
                    admission_namespace=admission_namespace,
                    chain_admission_max_prefix_blocks=(
                        chain_admission_max_prefix_blocks
                    ),
                    shared_prefetch_scorer=shared_prefetch_scorer,
                    method=str(self.extra_config.get("method", "mean")),
                    rank_ema_mode=str(
                        self.extra_config.get("rank_ema_mode", "default")
                    ),
                    ema_rank_weight_floor=validate_ema_rank_weight_floor(
                        self.extra_config.get("ema_rank_weight_floor", 0.0)
                    ),
                    decision_feature_trace=decision_feature_trace,
                )
                return self._manager
            grace_window_blocks = int(self.extra_config.get("grace_window_blocks", 0))
            eviction_mode = str(self.extra_config.get("eviction_mode", "blend"))
            # Structural reuse bonuses (SemanticPolicy's chain/session logic).
            # Read here so the benchmark grid can actually exercise them --
            # previously they defaulted off with no launch-time knob, so every
            # Step 1.6 cell ran with both fully disabled (issues log entry #66
            # follow-up).
            chain_aware = bool(self.extra_config.get("chain_aware", False))
            session_aware = bool(self.extra_config.get("session_aware", False))
            session_bonus_half_life = int(
                self.extra_config.get("session_bonus_half_life", 0)
            )
            prefix_aware = self.extra_config.get("prefix_aware", False)
            if not isinstance(prefix_aware, bool):
                raise TypeError("prefix_aware must be a JSON boolean")
            if (
                "future_reuse_feedback" in self.extra_config
                or "future_reuse_weight" in self.extra_config
            ):
                raise ValueError(
                    "future_reuse_feedback is offline-only; use the future-window gate"
                )
            # Step 1.6's benchmark grid needs {minmax, mean, cuboid_mean} as
            # separate, selectable configs -- previously hardcoded to
            # manager.py's _DEFAULT_METHOD with no way to pick another at
            # launch time (issues log entry #34).
            method = str(self.extra_config.get("method", "mean"))
            alpha_config = self.extra_config.get("alpha", DEFAULT_POLICY_ALPHA)
            if isinstance(alpha_config, bool) or not isinstance(
                alpha_config, (int, float)
            ):
                raise TypeError("alpha must be a JSON number")
            alpha = float(alpha_config)
            ema_rank_weight_floor = validate_ema_rank_weight_floor(
                self.extra_config.get("ema_rank_weight_floor", 0.0)
            )
            self._manager = SemanticOffloadingManager(
                num_blocks=self.num_blocks,
                enable_events=(
                    self.kv_events_config.enable_kv_cache_events
                    or (preservation_events and preservation_audit)
                ),
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
                grace_window_blocks=grace_window_blocks,
                eviction_mode=eviction_mode,
                chain_aware=chain_aware,
                session_aware=session_aware,
                session_bonus_half_life=session_bonus_half_life,
                prefix_aware=prefix_aware,
                ema_rank_weight_floor=ema_rank_weight_floor,
                admission_mode=admission_mode,
                admission_cost_per_byte=admission_cost_per_byte,
                admission_default_cost_per_byte=admission_default_cost_per_byte,
                admission_namespace=admission_namespace,
                chain_admission_max_prefix_blocks=(
                    chain_admission_max_prefix_blocks
                ),
                decision_feature_trace=decision_feature_trace,
                method=method,
                alpha=alpha,
            )
        return self._manager
