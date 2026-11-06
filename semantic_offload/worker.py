# SPDX-License-Identifier: Apache-2.0
"""SemanticOffloadingWorker: builds per-block key summaries during offload.

Hook-point rationale (see .claude/docs/semantic-eviction-plan.md, Step 1.2 --
full investigation trail, including two dead ends, is in project memory):

By the time KV blocks reach CPUOffloadingWorker, they are opaque
(num_blocks, page_size_bytes) int8 byte pages -- vllm/v1/kv_offload/base.py's
CanonicalKVCacheTensor deliberately erases per-head/per-token structure so the
generic offloading connector works across attention backends without knowing
any backend's physical KV layout. Reconstructing that layout from raw bytes
would be backend-specific and fragile.

nn.Module forward hooks and TorchDispatchMode were both tried and both fail
under vLLM's default torch.compile + CUDA-graph execution mode (verified
empirically): compiled/graph-replayed execution never returns to the Python
interpreter for the replayed portion, so no Python-side interception can see
per-step values once graphs are captured.

What actually works: each Attention instance holds `self.kv_cache`, the real,
already-correctly-shaped per-layer KV cache tensor (populated by
`bind_kv_cache`, vllm/v1/worker/utils.py). This is persistent GPU memory, not
a transient forward-pass activation, so reading it has nothing to do with
hooks or graph capture -- it works identically whether the model ran eager or
compiled. Its shape is not reverse-engineered: every attention backend
implements a standard `get_kv_cache_shape()` classmethod
(vllm/v1/attention/backends/*.py) as part of the public AttentionBackend
interface. Verified on the real server (default compiled+cudagraph config):
kv_cache.shape matched get_kv_cache_shape()'s prediction exactly, with real
non-degenerate values.

Layout assumption (asserted at runtime, not just trusted): Triton,
FlashAttention, and FlashInfer's `get_kv_cache_shape` all document the same
logical shape `(num_blocks, num_kv_heads, block_size, 2*head_size)`, K and V
packed into the last dimension (first half K, second half V). This is the
convention this file relies on; the assert in `_check_layout` catches any
backend that doesn't match it, rather than silently producing wrong
summaries. Does not apply to MLA (no per-head keys exist there at all).

Experimental local screen: SEMANTIC_SUMMARY_FIELDS=required retains only mean
centroids for mean scoring with continuous or opportunity scheduling.
Default "all" retains BlockSummary's
complete payload. The reduced private payload is not a general BlockSummary;
consumers requesting other fields must fail rather than inventing values.
"""

import hashlib
import itertools
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadKey,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

from semantic_offload._debug import TIMING as _TIMING, ENABLED as _DEBUG
from semantic_offload._debug import (
    debug_print,
    record_count,
    record_gpu_memory,
    record_process_state,
    record_timing,
)
from semantic_offload._vllm_compat import init_cpu_offloading_worker_base
from semantic_offload.async_scoring import AsyncScoringController
from semantic_offload.index import (
    BlockSummary,
    build_summary,
    score_cuboid_mean_batch,
    score_mean_batch,
    score_minmax_batch,
)
from semantic_offload.manager import validate_ema_rank_weight_floor
from semantic_offload.profiling import PROFILE, profiled
from semantic_offload.scoring_search import ClusterSearch, SearchConfig
from semantic_offload.query_capture import install as install_query_capture
from semantic_offload.query_capture import preflight_install as preflight_query_capture

_SCORING_METHODS = ("minmax", "mean", "cuboid_mean")
_HEAD_AGGREGATIONS = ("mean", "max")
_RANK_EMA_MODES = ("default", "fast")
_SCORING_MODES = ("synchronous", "asynchronous")
SummaryPublicationId = tuple[int, int, int]
# Which BlockSummary fields each method's batched scorer actually needs --
# only these get stacked into the cache, not all four.
_METHOD_FIELDS = {
    "minmax": ("max", "min"),
    "mean": ("mean",),
    "cuboid_mean": ("mean", "mad"),
}


@dataclass(frozen=True, slots=True)
class _MeanOnlySummary:
    """Owned mean reduction, deliberately exposing no unused summary fields."""

    mean: torch.Tensor


_WorkerSummary = BlockSummary | _MeanOnlySummary


@dataclass(frozen=True, slots=True)
class _AsyncScorePayload:
    req_ids: tuple[str, ...]
    queries: torch.Tensor
    keys: tuple[OffloadKey, ...]
    summaries: tuple[tuple[_WorkerSummary, ...], ...]
    method: str
    head_aggregation: str
    request_versions: tuple[tuple[str, int], ...]
    residency_versions: tuple[tuple[OffloadKey, int], ...] = ()
    decision_leases: tuple[tuple[str, tuple[str, ...]], ...] = ()
    transport_epoch: str | None = None
    source_tokens: tuple[tuple[str, int | None], ...] = ()
    expires_at_ns: int | None = None


def validate_parallel_scoring(
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    total_num_kv_heads: int | None,
    method: str,
    head_aggregation: str,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    world_size: int | None = None,
    use_mla: bool = False,
) -> None:
    if pipeline_parallel_size != 1:
        raise ValueError(
            "Semantic offloading currently requires pipeline_parallel_size=1"
        )
    if prefill_context_parallel_size != 1:
        raise ValueError(
            "Semantic offloading currently requires prefill_context_parallel_size=1"
        )
    if decode_context_parallel_size != 1:
        raise ValueError(
            "Semantic offloading currently requires decode_context_parallel_size=1"
        )
    if world_size is not None and world_size != tensor_parallel_size:
        raise ValueError(
            "Semantic offloading worker aggregation requires world_size to equal "
            "tensor_parallel_size"
        )
    if use_mla:
        raise ValueError("Semantic offloading does not yet support MLA KV caches")
    replicated_kv_heads = (
        total_num_kv_heads is None or tensor_parallel_size > total_num_kv_heads
    )
    if (
        tensor_parallel_size > 1
        and replicated_kv_heads
        and (method != "mean" or head_aggregation != "mean")
    ):
        raise ValueError(
            "Replicated KV heads are only supported with method='mean' and "
            "head_aggregation='mean'"
        )


def validate_rank_ema_mode(mode: str) -> None:
    if mode not in _RANK_EMA_MODES:
        raise ValueError(
            f"Unknown rank_ema_mode {mode!r}. Supported: {_RANK_EMA_MODES}"
        )


def validate_scoring_mode(mode: object) -> str:
    if not isinstance(mode, str):
        raise TypeError("scoring_mode must be a string")
    if mode not in _SCORING_MODES:
        raise ValueError(f"Unknown scoring_mode {mode!r}. Supported: {_SCORING_MODES}")
    return mode


def select_probe_layer(
    layer_names, selection: str | int | None = "middle"
) -> str | None:
    """Select a probe layer by natural model order or exact name."""
    if selection is None:
        return None
    names = list(layer_names)
    if not names:
        return None

    def natural_key(name: str):
        return tuple(
            (1, int(part)) if part.isdigit() else (0, part)
            for part in re.split(r"(\d+)", name)
        )

    ordered = sorted(names, key=natural_key)
    if selection in names:
        return str(selection)
    if isinstance(selection, int) or str(selection).lstrip("-").isdigit():
        index = int(selection)
        try:
            return ordered[index]
        except IndexError as exc:
            raise ValueError(
                f"probe_layer index {index} outside 0..{len(ordered) - 1}"
            ) from exc
    positions = {"first": 0, "middle": len(ordered) // 2, "last": -1}
    if selection not in positions:
        raise ValueError(
            f"Unknown probe_layer {selection!r}; use first/middle/last, "
            "an integer index, or an exact layer name"
        )
    return ordered[positions[str(selection)]]


class SemanticOffloadingWorker(CPUOffloadingWorker):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_blocks: int,
        vllm_config: VllmConfig,
        method: str = "mean",
        capture_stride: int = 1,
        probe_layer: str | int | None = "middle",
        head_aggregation: str = "mean",
        rank_ema_mode: str = "default",
        scoring_mode: str = "synchronous",
        async_scoring_max_queue_items: int = 2,
        async_scoring_max_snapshot_bytes: int = 64 * 1024 * 1024,
        async_scoring_workers: int = 2,
        async_scoring_max_snapshot_age_ms: int = 1000,
        ema_rank_weight_floor: float = 0.0,
        summary_offload_overlap: bool = False,
        preservation_events: bool = False,
        nvtx_ranges: bool = False,
        scoring_schedule: str = "continuous",
        scoring_search: str = "flat",
        search_min_pool: int = 128,
        search_clusters: int = 16,
        search_probes: int = 4,
        search_exploration: int = 8,
        query_snapshot_max_age_ms: int = 1000,
        query_snapshot_max_requests: int = 256,
        kv_svd_rank: int = 0,
        kv_svd_trace: str | None = None,
    ):
        if method not in _SCORING_METHODS:
            raise ValueError(
                f"Unknown scoring method: {method!r}. Supported: {_SCORING_METHODS}"
            )
        if capture_stride < 1:
            raise ValueError("capture_stride must be >= 1")
        if head_aggregation not in _HEAD_AGGREGATIONS:
            raise ValueError(
                f"Unknown head_aggregation: {head_aggregation!r}. "
                f"Supported: {_HEAD_AGGREGATIONS}"
            )
        validate_rank_ema_mode(rank_ema_mode)
        scoring_mode = validate_scoring_mode(scoring_mode)
        if scoring_schedule not in ("continuous", "opportunity", "capture_only"):
            raise ValueError("scoring_schedule must be continuous, opportunity or capture_only")
        summary_fields = os.environ.get("SEMANTIC_SUMMARY_FIELDS", "all")
        if summary_fields not in ("all", "required"):
            raise ValueError("SEMANTIC_SUMMARY_FIELDS must be all or required")
        if summary_fields == "required" and (
            method != "mean" or scoring_schedule not in ("continuous", "opportunity")
        ):
            raise ValueError("required summary fields require mean scoring with continuous or opportunity schedule")
        search_config = SearchConfig(scoring_search, search_min_pool, search_clusters,
                                     search_probes, search_exploration)
        for value in (query_snapshot_max_age_ms, query_snapshot_max_requests):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("query snapshot bounds must be positive integers")
        if (scoring_schedule != "continuous" or scoring_search != "flat") and (
            method != "mean" or head_aggregation != "mean"
            or vllm_config.parallel_config.tensor_parallel_size != 1
            or (scoring_mode != "synchronous" and not (scoring_schedule == "opportunity" and scoring_search == "flat"))
        ):
            raise ValueError("scheduled/indexed scoring requires mean/mean TP=1; asynchronous scheduling supports opportunity/flat only")
        if (
            isinstance(async_scoring_max_snapshot_age_ms, bool)
            or not isinstance(async_scoring_max_snapshot_age_ms, int)
            or async_scoring_max_snapshot_age_ms <= 0
        ):
            raise ValueError(
                "async_scoring_max_snapshot_age_ms must be a positive integer"
            )
        ema_rank_weight_floor = validate_ema_rank_weight_floor(
            ema_rank_weight_floor
        )
        parallel_config = vllm_config.parallel_config
        get_total_num_kv_heads = getattr(
            vllm_config.model_config, "get_total_num_kv_heads", None
        )
        total_num_kv_heads = (
            get_total_num_kv_heads() if get_total_num_kv_heads is not None else None
        )
        validate_parallel_scoring(
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            total_num_kv_heads=total_num_kv_heads,
            method=method,
            head_aggregation=head_aggregation,
            prefill_context_parallel_size=(
                getattr(parallel_config, "prefill_context_parallel_size", 1)
            ),
            decode_context_parallel_size=getattr(
                parallel_config, "decode_context_parallel_size", 1
            ),
            world_size=getattr(
                parallel_config, "world_size", parallel_config.tensor_parallel_size
            ),
            use_mla=bool(getattr(vllm_config.model_config, "use_mla", False)),
        )
        if scoring_mode == "asynchronous" and parallel_config.tensor_parallel_size != 1:
            raise ValueError("asynchronous scoring initially requires tensor_parallel_size=1")
        static_forward_context = vllm_config.compilation_config.static_forward_context
        query_capture_supported = bool(
            getattr(vllm_config, "use_v2_model_runner", False)
        )
        attention_layers = {
            layer_name: layer
            for layer_name, layer in static_forward_context.items()
            if hasattr(layer, "num_kv_heads") and hasattr(layer, "head_size")
        }
        probe_layer_name = select_probe_layer(attention_layers, selection=probe_layer)
        num_queries_per_kv = None
        if probe_layer_name is not None:
            probe_attention = attention_layers[probe_layer_name]
            num_attn_heads = vllm_config.model_config.get_num_attention_heads(
                parallel_config
            )
            if num_attn_heads % probe_attention.num_kv_heads:
                raise ValueError(
                    "attention heads must be divisible by KV heads for query capture"
                )
            num_queries_per_kv = num_attn_heads // probe_attention.num_kv_heads
            preflight_query_capture(vllm_config)
        elif preservation_events and query_capture_supported:
            # Instrumentation-only lifecycle screens still need the
            # model-runner execution boundary, but do not enable query scoring.
            preflight_query_capture(vllm_config)

        # CPUOffloadingWorker's own param is named block_size_factor or
        # blocks_per_chunk depending on the installed vLLM's version
        # (#48150 renamed it on some checkouts, not others still in use
        # across this project's machines) -- call via whichever name the
        # real base class actually declares. See _vllm_compat.py.
        if type(kv_svd_rank) is not int or not 0 <= kv_svd_rank <= 15:
            raise ValueError("kv_svd_rank must be an integer in 0..15")
        if kv_svd_rank:
            if blocks_per_chunk != 1 or vllm_config.model_config.dtype != torch.float16:
                raise ValueError("KV SVD requires FP16 and blocks_per_chunk=1")
            for layer in attention_layers.values():
                if layer.head_size != 128:
                    raise ValueError("KV SVD requires head_size=128")
            from semantic_offload.kv_svd import SVDTransport, SVDDirection
            transport = SVDTransport(kv_caches, num_cpu_blocks, kv_svd_rank, kv_svd_trace)
            self._store_handler = SVDDirection(transport, True)
            self._load_handler = SVDDirection(transport, False)
        else:
            init_cpu_offloading_worker_base(
                self,
                kv_caches=kv_caches,
                blocks_per_chunk=blocks_per_chunk,
                num_cpu_blocks=num_cpu_blocks,
            )
        self._method = method
        self._summary_fields = summary_fields
        self._head_aggregation = head_aggregation
        self._rank_ema_mode = rank_ema_mode
        self._scoring_mode = scoring_mode
        self._scoring_schedule = scoring_schedule
        self._candidate_search = ClusterSearch(search_config)
        self._score_transport_context = {}
        self._query_snapshots = {}
        self._query_snapshot_events = {}
        self._retired_async_request_ids = set()
        self._decision_async_signatures = {}
        self._query_snapshot_max_age_ns = query_snapshot_max_age_ms * 1_000_000
        self._query_snapshot_max_requests = query_snapshot_max_requests
        self._decision_score_sequence = 0
        self._profile_metadata_calls = 0
        self._ema_rank_weight_floor = ema_rank_weight_floor
        self._summary_offload_overlap = summary_offload_overlap
        self._preservation_events = preservation_events
        self._nvtx_ranges = nvtx_ranges
        self._attention_layers = attention_layers
        self._block_size = vllm_config.cache_config.block_size
        self._summary_stream = (
            torch.cuda.Stream()
            if probe_layer_name is not None and torch.cuda.is_available() else None
        )
        self._layout_checked: set[str] = set()
        # summaries[layer_name][block_id] = list[BlockSummary], one per KV head.
        # Keyed by physical GPU block ID, which is naturally bounded and
        # reused/overwritten as blocks are freed and reallocated -- no
        # separate eviction-cleanup hook needed to keep this bounded.
        self.summaries: dict[str, dict[int, list[_WorkerSummary]]] = {
            layer_name: {} for layer_name in self._attention_layers
        }

        # Step 1.3: durable, OffloadKey-keyed summaries (survive GPU block
        # reuse, unlike `self.summaries` above) + live query scoring.
        # Mid-model was used by the experiments that selected the current
        # scoring method. Keep the position configurable for controlled DGX
        # ablations and exact-name overrides.
        self._probe_layer_name = probe_layer_name
        # One BlockSummary per KV head (not averaged across heads -- see
        # issues log entry #9; entry #7's pooled-average version is
        # superseded).
        #
        # DEFENSIVE BACKSTOP, not the primary mechanism anymore (issues log
        # entries #62-64): receive_evicted_keys() now gets told precisely
        # which keys the real CachePolicy evicted, each step, and removes
        # exactly those -- the correct fix. This FIFO-by-insertion-order cap
        # was the original stopgap (bounds the unbounded growth but can
        # diverge from the manager's real eviction order, e.g. dropping a
        # block the real tier still holds); it's kept only to bound worst-
        # case growth from any removal path receive_evicted_keys doesn't yet
        # cover, such as a defensive backstop after a full cache reset.
        # Failed-store cleanup is handled at the transfer-result boundary.
        self._max_durable_summaries = max(num_cpu_blocks, 1)
        self.durable_summaries: dict[OffloadKey, list[_WorkerSummary]] = {}
        self._pending_job_keys: dict[int, list[OffloadKey]] = {}
        self._pending_job_blocks: dict[int, list[tuple[OffloadKey, list[int]]]] = {}
        self._transfer_directions: dict[int, bool] = {}
        self._transfer_started_at_ns: dict[int, int] = {}
        self._transfer_start_times: dict[int, float] = {}
        self._transfer_completion_times: dict[int, float] = {}
        self._transfer_destination_blocks: dict[int, set[int]] = {}
        self._failed_load_block_ids: set[int] = set()
        self._pending_transfer_events: dict[int, dict[str, object]] = {}
        self._pending_scores: dict[str, dict[str, list[tuple[OffloadKey, float]]]] = {}
        self._pending_score_head_counts: dict[str, dict[str, int]] = {}
        self._pending_async_score_observations: list[dict[str, object]] = []
        self._first_forward_times_ns: dict[str, int] = {}
        self._preempted_request_ids: set[str] = set()
        self._preempted_forward_times_ns: dict[str, int] = {}
        self._allocator_peak_stats: dict[str, int] = {}
        # Cache of durable_summaries stacked into batched tensors, kept in
        # sync incrementally (see _rebuild_stack_cache) rather than rebuilt
        # from scratch on every query capture -- query captures fire on
        # nearly every prefill/mixed step, while any single step only ever
        # inserts/evicts a handful of candidates, so a from-scratch rebuild
        # over the WHOLE resident pool on every dirty step was the dominant
        # cost in the entire hot path at real production scale (issues log
        # entries #53's follow-up, #72, #74).
        self._stack_cache_dirty = True
        self._stack_cache_keys: list[OffloadKey] = []
        self._stack_cache: dict[str, torch.Tensor] = {}
        self._stack_cache_index: dict[OffloadKey, int] = {}
        self._stack_pending_insert: set[OffloadKey] = set()
        self._stack_pending_remove: set[OffloadKey] = set()
        # Original stack-row indices ordered by OffloadKey bytes. Fast
        # ranking uses this as a tie-order view without mutating the legacy
        # cache-row order needed by non-finite fallback semantics.
        self._rank_tie_order: list[int] = []
        # Opt-in summary/offload overlap state. A store's source blocks remain
        # owned by the scheduler's TransferJobStatus until this worker emits
        # the completion. The summary event is an additional completion fence:
        # a source block cannot be reused until both the GPU->CPU copy and the
        # side-stream reduction have finished.
        self._summary_job_events: dict[int, torch.cuda.Event | None] = {}
        self._summary_job_publications: dict[
            int, dict[OffloadKey, SummaryPublicationId]
        ] = {}
        self._durable_summary_publications: dict[OffloadKey, SummaryPublicationId] = {}
        self._summary_job_generations: dict[int, int] = {}
        self._summary_publication_sequence = 0
        self._summary_generation = 0
        self._deferred_transfer_results: list[TransferResult] = []
        self._async_scoring: AsyncScoringController | None = None
        self._async_request_versions: dict[str, int] = {}
        self._async_submission_requests: dict[int, tuple[str, ...]] = {}
        self._async_scoring_stream = None
        self._async_scoring_max_snapshot_age_ns = (
            async_scoring_max_snapshot_age_ms * 1_000_000
        )
        if scoring_mode == "asynchronous":
            self._async_scoring_stream = (
                torch.cuda.Stream() if torch.cuda.is_available() else None
            )
            self._async_scoring = AsyncScoringController(
                self._score_async_payload,
                max_queue_items=async_scoring_max_queue_items,
                max_snapshot_bytes=async_scoring_max_snapshot_bytes,
                workers=1,  # A single owner for the single scoring stream.
            )
        # splice_id -> completion event. The scheduler keeps speculative
        # source blocks reserved until every worker reports this event done.
        self._splice_events: dict[int, torch.cuda.Event | None] = {}
        self._splice_event_evidence: dict[int, dict[str, object]] = {}
        self._completed_splice_evidence: dict[int, dict[str, object]] = {}
        self._query_capture_mode = None
        if self._probe_layer_name is not None or (
            self._preservation_events and query_capture_supported
        ):
            assert num_queries_per_kv is not None or self._preservation_events
            try:
                self._query_capture_mode = install_query_capture(
                    vllm_config,
                    self._probe_layer_name,
                    self._on_queries_captured
                    if self._probe_layer_name is not None
                    else None,
                    num_queries_per_kv=num_queries_per_kv or 1,
                    capture_stride=capture_stride,
                    on_forward=(
                        self._on_model_runner_forward
                        if self._preservation_events
                        else None
                    ),
                    emit_nvtx_ranges=self._nvtx_ranges,
                )
            except BaseException:
                async_scoring = getattr(self, "_async_scoring", None)
                if async_scoring is not None:
                    async_scoring.shutdown()
                    self._async_scoring = None
                super().shutdown()
                raise

    def shutdown(self) -> None:
        query_capture = self._query_capture_mode
        if query_capture is not None:
            # A failed close means a model execution still owns these resources.
            # Preserve them for a retry rather than detaching live readers.
            query_capture.close()
            self._query_capture_mode = None
        async_scoring = getattr(self, "_async_scoring", None)
        if async_scoring is not None:
            self._drain_async_scores(wait=True)
            async_scoring.shutdown()
            self._async_scoring = None
        summary_stream = getattr(self, "_summary_stream", None)
        if summary_stream is not None:
            summary_stream.synchronize()
        getattr(self, '_prefix_mean_centroid_cache', {}).clear()
        self._pending_prefix_score_observations = []
        super().shutdown()
        PROFILE.flush()

    def receive_job_keys(self, store_job_keys: dict[int, list[OffloadKey]]) -> None:
        """Called by SemanticOffloadingConnector before this step's
        submit_store() calls, with the OffloadKey(s) each pending store
        job_id represents (scheduler-side info the worker has no other way
        to see -- see connector.py and the issues log entry #6/7)."""
        self._pending_job_keys.update(store_job_keys)

    def receive_job_blocks(
        self,
        store_job_blocks: dict[int, list[tuple[OffloadKey, list[int]]]],
    ) -> None:
        """Receive an exact key-to-source-block layout for each store job."""
        self._pending_job_blocks.update(store_job_blocks)

    def reset_semantic_state(self) -> None:
        """Drop worker state whose identities belong to the old CPU cache."""
        async_scoring = getattr(self, "_async_scoring", None)
        if async_scoring is not None:
            async_scoring.reset()
        getattr(self, "_async_submission_requests", {}).clear()
        getattr(self, "_async_request_versions", {}).clear()
        getattr(self, "_query_snapshots", {}).clear()
        getattr(self, "_query_snapshot_events", {}).clear()
        getattr(self, '_prefix_mean_centroid_cache', {}).clear()
        self._pending_prefix_score_observations = []
        self._prefix_mean_scoring_stats = {}
        getattr(self, "_retired_async_request_ids", set()).clear()
        getattr(self, "_decision_async_signatures", {}).clear()
        if hasattr(self, "_candidate_search"):
            self._candidate_search.centroids = None
            self._candidate_search.groups = []
        summary_stream = getattr(self, "_summary_stream", None)
        if summary_stream is not None:
            # Reset is rare and destructive. Do not drop the last references
            # to tensors whose reductions are still running on the side stream.
            summary_stream.synchronize()
        # Keep stale job records long enough to consume their completion, but
        # make their generation explicit so a failed pre-reset job can never
        # remove a same-key summary published after this reset.
        self._summary_generation = getattr(self, "_summary_generation", 0) + 1
        for summaries in self.summaries.values():
            summaries.clear()
        self.durable_summaries.clear()
        getattr(self, "_durable_summary_publications", {}).clear()
        self._pending_job_keys.clear()
        self._pending_job_blocks.clear()
        self._pending_scores.clear()
        self._pending_score_head_counts.clear()
        getattr(self, "_pending_async_score_observations", []).clear()
        getattr(self, "_first_forward_times_ns", {}).clear()
        getattr(self, "_preempted_request_ids", set()).clear()
        getattr(self, "_preempted_forward_times_ns", {}).clear()
        getattr(self, "_allocator_peak_stats", {}).clear()
        getattr(self, "_transfer_directions", {}).clear()
        getattr(self, "_transfer_started_at_ns", {}).clear()
        getattr(self, "_transfer_start_times", {}).clear()
        getattr(self, "_transfer_completion_times", {}).clear()
        getattr(self, "_transfer_destination_blocks", {}).clear()
        getattr(self, "_failed_load_block_ids", set()).clear()
        getattr(self, "_pending_transfer_events", {}).clear()
        getattr(self, "_async_request_versions", {}).clear()
        getattr(self, "_async_submission_requests", {}).clear()
        self._stack_cache_dirty = True
        self._stack_cache_keys.clear()
        self._stack_cache.clear()
        self._stack_cache_index.clear()
        self._stack_pending_insert.clear()
        self._stack_pending_remove.clear()

    def _on_model_runner_forward(self, req_ids: list[str], timestamp_ns: int) -> None:
        """Record the worker-clock model-runner execution-entry boundary.

        The callback runs immediately before vLLM's ``execute_model`` body.
        It is not a device-kernel timestamp; CUDA launch/completion timing
        still requires a model-runner or profiler hook owned by vLLM.
        """
        for req_id in req_ids:
            prior = self._first_forward_times_ns.get(req_id)
            if prior is None or timestamp_ns < prior:
                self._first_forward_times_ns[req_id] = timestamp_ns
            preempted_request_ids = getattr(self, "_preempted_request_ids", set())
            if req_id in preempted_request_ids:
                preempted_forward_times_ns = getattr(
                    self, "_preempted_forward_times_ns", {}
                )
                prior = preempted_forward_times_ns.get(req_id)
                if prior is None or timestamp_ns < prior:
                    preempted_forward_times_ns[req_id] = timestamp_ns
                self._preempted_forward_times_ns = preempted_forward_times_ns
                preempted_request_ids.discard(req_id)
                self._preempted_request_ids = preempted_request_ids
        # Detailed allocator categories are audit telemetry. The native CUDA
        # peak counters remain available at export when this opt-in switch is off.
        if (os.environ.get("SEMANTIC_ALLOCATOR_AUDIT", "1") != "0"
                and torch.cuda.is_available()):
            stats = torch.cuda.memory_stats()
            for name in (
                "allocated_bytes.all.current",
                "reserved_bytes.all.current",
                "active_bytes.all.current",
                "inactive_split_bytes.all.current",
            ):
                value = stats.get(name)
                if value is not None:
                    self._allocator_peak_stats[name] = max(
                        self._allocator_peak_stats.get(name, 0), int(value)
                    )

    def pop_first_forward_times(self) -> dict[str, int]:
        values = getattr(self, "_first_forward_times_ns", {})
        self._first_forward_times_ns = {}
        return values

    def receive_preempted_request_ids(self, request_ids: Sequence[str]) -> None:
        """Arm post-preemption execution-entry attribution for this step."""
        pending = getattr(self, "_preempted_request_ids", None)
        if pending is None:
            pending = self._preempted_request_ids = set()
        pending.update(
            request_id for request_id in request_ids if isinstance(request_id, str)
        )

    def pop_preempted_forward_times(self) -> dict[str, int]:
        values = getattr(self, "_preempted_forward_times_ns", {})
        self._preempted_forward_times_ns = {}
        return values

    def _finish_summary_job(self, result: TransferResult) -> None:
        """Release a summary fence and clean up failed-store summaries."""
        summary_job_events = getattr(self, "_summary_job_events", {})
        summary_job_publications = getattr(self, "_summary_job_publications", {})
        durable_summary_publications = getattr(
            self, "_durable_summary_publications", {}
        )
        summary_job_generations = getattr(self, "_summary_job_generations", {})
        summary_job_events.pop(result.job_id, None)
        publications = summary_job_publications.pop(result.job_id, None)
        generation = summary_job_generations.pop(
            result.job_id, getattr(self, "_summary_generation", 0)
        )
        if (
            result.success
            or not publications
            or generation != getattr(self, "_summary_generation", 0)
        ):
            return
        removed = []
        for key, publication in publications.items():
            if durable_summary_publications.get(key) != publication:
                continue
            if self.durable_summaries.pop(key, None) is not None:
                durable_summary_publications.pop(key, None)
                removed.append(key)
        if removed:
            self._mark_removed_from_stack_cache(removed)

    def _register_summary_publications(
        self, job_id: int, keys: list[OffloadKey]
    ) -> None:
        """Assign unique publication identities to a store's summaries."""
        job_publications = getattr(self, "_summary_job_publications", None)
        if job_publications is None:
            job_publications = self._summary_job_publications = {}
        durable_publications = getattr(self, "_durable_summary_publications", None)
        if durable_publications is None:
            durable_publications = self._durable_summary_publications = {}
        sequence = getattr(self, "_summary_publication_sequence", 0)
        generation = getattr(self, "_summary_generation", 0)
        publications: dict[OffloadKey, SummaryPublicationId] = {}
        for key in keys:
            sequence += 1
            publication = (generation, job_id, sequence)
            publications[key] = publication
            durable_publications[key] = publication
        self._summary_publication_sequence = sequence
        job_publications[job_id] = publications

    def _rollback_summary_job(self, job_id: int) -> None:
        """Fence and retract a store whose base submission did not publish."""
        summary_job_events = getattr(self, "_summary_job_events", {})
        event = summary_job_events.get(job_id)
        if event is not None:
            # Do not retract Python-owned summaries until the side-stream
            # reduction that produced them has completed.
            event.synchronize()
        else:
            summary_stream = getattr(self, "_summary_stream", None)
            if summary_stream is not None:
                summary_stream.synchronize()
        self._finish_summary_job(TransferResult(job_id=job_id, success=False))

    def _summary_job_ready(self, job_id: int) -> bool:
        event = getattr(self, "_summary_job_events", {}).get(job_id)
        return event is None or event.query()

    @profiled('transfer.poll')
    def get_finished(self) -> list[TransferResult]:
        """Do not publish a store completion before its summary reduction.

        CPUOffloadingWorker's transfer handler removes a ready DMA result
        from its queue. Keep such a result locally while the summary event is
        pending, then publish it only after both consumers are complete. This
        preserves the scheduler's source-block ownership/refcount fence and
        also makes failed-store summary cleanup happen at the same lifecycle
        boundary as the transfer result.
        """
        ready: list[TransferResult] = []
        deferred: list[TransferResult] = []
        pending_results = getattr(self, "_deferred_transfer_results", [])
        for result in pending_results:
            if getattr(self, "_preservation_events", False):
                completion_times = getattr(self, "_transfer_completion_times", None)
                if completion_times is None:
                    completion_times = self._transfer_completion_times = {}
                completion_times.setdefault(result.job_id, time.perf_counter())
            if self._summary_job_ready(result.job_id):
                self._finish_summary_job(result)
                ready.append(result)
            else:
                deferred.append(result)
        self._deferred_transfer_results = deferred

        for result in super().get_finished():
            if getattr(self, "_preservation_events", False):
                completion_times = getattr(self, "_transfer_completion_times", None)
                if completion_times is None:
                    completion_times = self._transfer_completion_times = {}
                completion_times.setdefault(result.job_id, time.perf_counter())
            if not self._summary_job_ready(result.job_id):
                self._deferred_transfer_results.append(result)
                if _TIMING:
                    record_count("summary_offload_fence_deferred", 1)
                continue
            self._finish_summary_job(result)
            ready.append(result)
        transfer_events = getattr(self, "_pending_transfer_events", None)
        if not getattr(self, "_preservation_events", False):
            return ready
        if transfer_events is None:
            transfer_events = self._pending_transfer_events = {}
        directions = getattr(self, "_transfer_directions", {})
        for result in ready:
            is_load = directions.pop(result.job_id, None)
            if is_load is not None:
                cancelled = bool(getattr(result, "cancelled", False))
                success = bool(getattr(result, "success", False))
                if is_load and not success:
                    self._failed_load_block_ids.update(
                        self._transfer_destination_blocks.get(result.job_id, set())
                    )
                transfer_events[result.job_id] = {
                    "is_load": is_load,
                    "bytes": result.transfer_size,
                    "time_s": result.transfer_time,
                    "max_time_s": result.transfer_time,
                    "contributors": 1,
                    "success": success,
                    "status": (
                        "cancelled"
                        if cancelled
                        else "success"
                        if success
                        else "failure"
                    ),
                    "cancelled": cancelled,
                    "start_time_s": getattr(self, "_transfer_start_times", {}).pop(
                        result.job_id, None
                    ),
                    "completion_time_s": getattr(
                        self, "_transfer_completion_times", {}
                    ).pop(result.job_id, None),
                    "started_at_ns": getattr(self, "_transfer_started_at_ns", {}).pop(
                        result.job_id, None
                    ),
                    "completed_at_ns": time.monotonic_ns(),
                }
                self._transfer_destination_blocks.pop(result.job_id, None)
        return ready

    def _record_transfer_start(self, job_id: int) -> None:
        if not getattr(self, "_preservation_events", False):
            return
        starts = getattr(self, "_transfer_start_times", None)
        if starts is None:
            starts = self._transfer_start_times = {}
        starts.setdefault(job_id, time.perf_counter())

    def register_transfer_direction(
        self,
        job_id: int,
        *,
        is_load: bool,
        destination_block_ids=None,
    ) -> None:
        directions = getattr(self, "_transfer_directions", None)
        if directions is None:
            directions = self._transfer_directions = {}
        directions[job_id] = is_load
        if getattr(self, "_preservation_events", False):
            self._transfer_started_at_ns[job_id] = time.monotonic_ns()
            self._record_transfer_start(job_id)
        if is_load:
            raw_destination_block_ids = destination_block_ids
            self._transfer_destination_blocks[job_id] = {
                int(block_id)
                for block_id in (
                    ()
                    if raw_destination_block_ids is None
                    else raw_destination_block_ids
                )
                if int(block_id) != 0
            }

    def record_transfer_failure(
        self, job_id: int, *, is_load: bool, cancelled: bool = False
    ) -> None:
        events = getattr(self, "_pending_transfer_events", None)
        if events is None:
            events = self._pending_transfer_events = {}
        events[job_id] = {
            "is_load": is_load,
            "bytes": None,
            "time_s": None,
            "max_time_s": None,
            "contributors": 1,
            "success": False,
            "status": "cancelled" if cancelled else "failure",
            "cancelled": cancelled,
            "start_time_s": getattr(self, "_transfer_start_times", {}).pop(
                job_id, None
            ),
            "completion_time_s": (
                time.perf_counter()
                if getattr(self, "_preservation_events", False)
                else None
            ),
            "started_at_ns": getattr(self, "_transfer_started_at_ns", {}).pop(
                job_id, None
            ),
            "completed_at_ns": time.monotonic_ns(),
        }
        self._transfer_directions.pop(job_id, None)
        if is_load:
            self._failed_load_block_ids.update(
                self._transfer_destination_blocks.get(job_id, set())
            )
            self._transfer_destination_blocks.pop(job_id, None)

    def pop_failed_load_block_ids(self) -> set[int]:
        failed = self._failed_load_block_ids
        self._failed_load_block_ids = set()
        return failed

    def pop_transfer_events(self) -> dict[int, dict[str, object]]:
        events = getattr(self, "_pending_transfer_events", {})
        self._pending_transfer_events = {}
        return events

    def wait(self, job_ids: set[int]) -> None:
        """Wait for DMA and any summary fences before a source can be reused."""
        super().wait(job_ids)
        for job_id in job_ids:
            event = getattr(self, "_summary_job_events", {}).get(job_id)
            if event is not None:
                _t_fence = time.perf_counter() if _TIMING else 0.0
                event.synchronize()
                if _TIMING:
                    record_timing(
                        "summary_offload_fence_wait",
                        time.perf_counter() - _t_fence,
                    )

    def receive_evicted_keys(self, evicted_keys: list[OffloadKey]) -> None:
        """Called by SemanticOffloadingConnector with exactly the keys the
        manager's real CachePolicy evicted this step (issues log entries
        #62-64) -- the precise fix for durable_summaries' unbounded growth.
        Replaces the FIFO cap (`_prune_durable_summaries`) as the primary
        mechanism; that cap stays as a defensive backstop for any removal
        path this doesn't cover, such as a full cache reset."""
        if not evicted_keys:
            return
        removed = [
            k for k in evicted_keys if self.durable_summaries.pop(k, None) is not None
        ]
        for key in removed:
            getattr(self, "_durable_summary_publications", {}).pop(key, None)
        if removed:
            self._mark_removed_from_stack_cache(removed)
        debug_print(
            f"SEMANTIC_EVICT_DEBUG received={len(evicted_keys)} "
            f"removed={len(removed)} resident={len(self.durable_summaries)}"
        )

    def _mark_inserted_into_stack_cache(self, keys) -> None:
        """Record keys as needing a row in the stacked-tensor cache. A key
        that already has a synced row (an overwrite -- same OffloadKey
        re-stored with new content) is also queued for removal of its stale
        row, so the sync ends up with exactly one, up-to-date row per key."""
        for key in keys:
            if key in self._stack_cache_index:
                self._stack_pending_remove.add(key)
            self._stack_pending_insert.add(key)
        self._stack_cache_dirty = True

    def _mark_removed_from_stack_cache(self, keys) -> None:
        """Record keys as needing to be dropped from the stacked-tensor
        cache. A key never yet synced in (still only pending-insert) is
        simply un-queued -- no row was ever created for it."""
        for key in keys:
            self._stack_pending_insert.discard(key)
            if key in self._stack_cache_index:
                self._stack_pending_remove.add(key)
        self._stack_cache_dirty = True

    def pop_pending_scores(
        self,
    ) -> dict[str, dict[str, list[tuple[OffloadKey, float]]]]:
        scores, _, _ = self.pop_pending_score_metadata()
        return scores

    @profiled('metadata.pop_scores')
    def pop_pending_score_metadata(
        self,
    ) -> tuple[
        dict[str, dict[str, list[tuple[OffloadKey, float]]]],
        dict[str, dict[str, int]],
        str,
    ]:
        if getattr(self, "_scoring_mode", "synchronous") == "asynchronous":
            self._drain_async_scores()
        self._profile_metadata_calls = getattr(self, "_profile_metadata_calls", 0) + 1
        if PROFILE.directory and self._profile_metadata_calls % 128 == 0:
            PROFILE.flush()
        scores = self._pending_scores
        head_counts = getattr(self, "_pending_score_head_counts", {})
        self._pending_scores = {}
        self._pending_score_head_counts = {}
        return scores, head_counts, self._head_aggregation

    def pop_async_score_observations(self) -> list[dict[str, object]]:
        if getattr(self, "_scoring_mode", "synchronous") == "asynchronous":
            self._drain_async_scores()
        observations = list(
            getattr(self, "_pending_async_score_observations", [])
        )
        self._pending_async_score_observations = []
        return observations

    @profiled('scoring.stack_cache', cuda=True)
    def _rebuild_stack_cache(self) -> None:
        """Bring the stacked-tensor cache back in sync with
        durable_summaries incrementally: compact out pending removals with
        basic slices and one concatenation per field, then stack and append only
        the newly-inserted candidates -- instead of a from-scratch Python
        getattr+torch.stack rebuild over EVERY resident candidate on every
        dirty query capture. See the dirty-flag comment in __init__: on
        `chat`-style traffic the cache is dirtied on nearly every step while
        only a handful of candidates actually change per step, so the old
        full rebuild was by far the dominant cost in the whole hot path
        (40-100x every other bucket -- issues log entries #72/#74)."""
        fields = _METHOD_FIELDS[self._method]

        if self._stack_pending_remove:
            surviving_indices = [index for index, key in enumerate(self._stack_cache_keys)
                                 if key not in self._stack_pending_remove]
            surviving_keys = [self._stack_cache_keys[index] for index in surviving_indices]
            # CPU metadata determines contiguous runs. Basic slices avoid
            # implicit CPU-mask uploads and CUDA nonzero synchronization.
            runs = []
            for index in surviving_indices:
                if runs and runs[-1][1] == index:
                    runs[-1] = (runs[-1][0], index + 1)
                else:
                    runs.append((index, index + 1))
            for field in fields:
                existing = self._stack_cache[field]
                if len(surviving_keys) == len(self._stack_cache_keys):
                    continue
                if not runs:
                    self._stack_cache[field] = existing[:0]
                elif len(runs) == 1:
                    start, end = runs[0]
                    self._stack_cache[field] = existing[start:end]
                else:
                    self._stack_cache[field] = torch.cat(
                        [existing[start:end] for start, end in runs], dim=0
                    )
            self._stack_cache_keys = surviving_keys
            self._stack_cache_index = {k: i for i, k in enumerate(surviving_keys)}
            self._stack_pending_remove.clear()

        if self._stack_pending_insert:
            new_keys = [
                k for k in self._stack_pending_insert if k in self.durable_summaries
            ]
            self._stack_pending_insert.clear()
            if new_keys:
                new_summary_lists = [self.durable_summaries[k] for k in new_keys]
                num_heads = len(new_summary_lists[0])
                if not num_heads or any(len(summaries) != num_heads
                                        for summaries in new_summary_lists):
                    raise ValueError("stacked summaries must have equal positive head counts")
                for field in fields:
                    # [n_new, num_kv_heads, head_dim]
                    flat_stack = torch.stack([getattr(summary, field)
                                              for summaries in new_summary_lists
                                              for summary in summaries])
                    new_stack = flat_stack.reshape(len(new_keys), num_heads,
                                                   *flat_stack.shape[1:])
                    existing = self._stack_cache.get(field)
                    self._stack_cache[field] = (
                        torch.cat([existing, new_stack], dim=0)
                        if existing is not None and existing.numel()
                        else new_stack
                    )
                base = len(self._stack_cache_keys)
                for offset, key in enumerate(new_keys):
                    self._stack_cache_index[key] = base + offset
                self._stack_cache_keys.extend(new_keys)

        # The fast ranking path uses a stable descending tensor sort. Keep a
        # key-ordered index view rather than reordering cache rows: if scores
        # are non-finite, the fallback must see the exact legacy row order
        # before Python's `(-score, key)` sort is applied.
        if getattr(self, "_rank_ema_mode", "default") == "fast":
            self._rank_tie_order = sorted(
                range(len(self._stack_cache_keys)),
                key=self._stack_cache_keys.__getitem__,
            )
        else:
            self._rank_tie_order = []

        self._stack_cache_dirty = False

    def _on_query_captured(self, req_id: str, query_repr: torch.Tensor) -> None:
        """Single-request convenience wrapper over `_on_queries_captured`
        (tests and any external caller; the live capture path calls the
        batched form directly, once per step)."""
        self._on_queries_captured([req_id], query_repr.unsqueeze(0))

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    @torch.no_grad()
    def _score_async_payload(
        self, payload: _AsyncScorePayload
    ) -> tuple[
        dict[str, dict[str, list[tuple[OffloadKey, float]]]],
        dict[str, dict[str, int]],
    ]:
        fields = _METHOD_FIELDS[payload.method]
        # Retain enqueued output/cache buffers across an exception until the
        # outer scoring-stream fence completes; no extra tensor allocation.
        readback_owners = []

        def score_payload():
            with PROFILE.scope("async.stack", cuda=True):
                cache = {
                    field: torch.stack(
                        [
                            torch.stack(
                                [getattr(summary, field) for summary in summary_list]
                            )
                            for summary_list in payload.summaries
                        ]
                    )
                    for field in fields
                }
            with PROFILE.scope("async.math_enqueue", cuda=True):
                query = payload.queries.unsqueeze(1)
                if payload.method == "minmax":
                    per_head = score_minmax_batch(
                        query, cache["max"].unsqueeze(0), cache["min"].unsqueeze(0)
                    )
                elif payload.method == "mean":
                    per_head = score_mean_batch(query, cache["mean"].unsqueeze(0))
                else:
                    per_head = score_cuboid_mean_batch(
                        query, cache["mean"].unsqueeze(0), cache["mad"].unsqueeze(0)
                    )
                combined = (
                    per_head.mean(dim=-1)
                    if payload.head_aggregation == "mean"
                    else per_head.max(dim=-1).values
                )
            readback_owners.append((cache, per_head, combined))
            with PROFILE.scope("async.readback", cuda=False):
                if combined.is_cuda:
                    from semantic_offload.async_scoring import wait_for_owned_fence
                    completed = torch.cuda.Event()
                    readback_owners.append(completed)
                    completed.record(torch.cuda.current_stream(combined.device))
                    wait_for_owned_fence(completed)
                all_scores = combined.tolist()
            with PROFILE.scope("async.rank", cuda=False):
                ranked_by_request = {
                    req_id: sorted(
                        zip(payload.keys, scores), key=lambda item: (-item[1], item[0])
                    )
                    for req_id, scores in zip(payload.req_ids, all_scores)
                }
                head_counts = {
                    req_id: int(per_head.shape[-1]) for req_id in payload.req_ids
                }
            return (
                {payload.method: ranked_by_request},
                {payload.method: head_counts},
            )

        stream = getattr(self, "_async_scoring_stream", None)
        if stream is None:
            return score_payload()
        with torch.cuda.stream(stream):
            try:
                return score_payload()
            except BaseException:
                stream.synchronize()  # Failed Future alone is not a GPU reader fence.
                raise

    def _submit_async_scoring(self, req_ids, queries, *, candidate_keys=None,
                              feature_ns=None, query_fences=(), residency_versions=None, decision_leases=None):
        controller = self._async_scoring
        # Full item queues reject BEFORE scanning/retaining the bank.
        if not controller.can_accept(0):
            controller.record_pre_capture_drop(generation=self._summary_generation, snapshot_bytes=0)
            return None
        keys = tuple(key for key in (candidate_keys if candidate_keys is not None else self.durable_summaries)
                     if key in self.durable_summaries)
        if not keys:
            return None
        if (queries.ndim != 3 or queries.shape[0] != len(req_ids)
                or not queries.is_floating_point()):
            raise ValueError("async queries require real floating [requests, heads, dim]")
        fields = _METHOD_FIELDS[self._method]
        h, d = queries.shape[1:]
        if any(len(self.durable_summaries[k]) != h or any(
                getattr(head, f).shape != (d,) or not getattr(head, f).is_floating_point()
                for head in self.durable_summaries[k] for f in fields) for k in keys):
            raise ValueError("async summary fields must match query heads/dim and be real floating")
        # Project only required immutable tensor fields; never retain whole BlockSummary.
        from types import SimpleNamespace
        summaries = tuple(tuple(SimpleNamespace(**{f: getattr(summary, f) for f in fields})
                                  for summary in self.durable_summaries[key]) for key in keys)
        tensors = [getattr(summary, f) for row in summaries for summary in row for f in fields]
        seen = set(); retained = 0
        for tensor in tensors:
            storage = tensor.untyped_storage(); identity = (str(tensor.device), storage.data_ptr())
            if identity not in seen:
                seen.add(identity); retained += storage.nbytes()
        query_bytes = queries.numel() * 4
        # Snapshot copy is float32; torch promotion may retain wider summaries.
        dtype = torch.float32
        for tensor in tensors:
            dtype = torch.promote_types(dtype, tensor.dtype)
        element_bytes = torch.empty((), dtype=dtype).element_size()
        r, h, d = queries.shape
        k = len(keys)
        # Each stack may promote all rows/heads to the widest field dtype.
        field_bytes = len(fields)*k*h*d*element_bytes
        # mean: one broadcast product; minmax/cuboid: two products + max.
        # Cuboid additionally retains box_min and box_max [K,H,D].
        products = 1 if self._method == 'mean' else 3
        boxes = 2*k*h*d*element_bytes if self._method == 'cuboid_mean' else 0
        # Keep original Q backing as well as its owned float32 snapshot charged.
        query_backing = queries.untyped_storage().nbytes()
        snapshot_bytes = (retained + query_backing + query_bytes
            + 2*field_bytes + products*r*k*h*d*element_bytes + boxes
            + r*k*h*element_bytes + 3*r*k*(element_bytes+8))
        if not controller.can_accept(snapshot_bytes):
            controller.record_pre_capture_drop(generation=self._summary_generation, snapshot_bytes=snapshot_bytes)
            return None
        captured = time.monotonic_ns() if feature_ns is None else feature_ns
        max_age = self._async_scoring_max_snapshot_age_ns
        if getattr(self, '_scoring_schedule', 'continuous') == 'opportunity':
            max_age = min(max_age, self._query_snapshot_max_age_ns)
        context = getattr(self, '_score_transport_context', {})
        if residency_versions is None:
            residency_versions = context.get('residencies', {})
        query_snapshot = queries.detach().to(dtype=torch.float32, copy=True)
        fences = list(query_fences)
        if query_snapshot.is_cuda:
            ready = torch.cuda.Event(); ready.record(torch.cuda.current_stream(query_snapshot.device)); fences.append(ready)
            # Only selected publication production fences, never the unrelated summary-stream tail.
            for job in dict.fromkeys(self._durable_summary_publications[k][1] for k in keys):
                event = self._summary_job_events.get(job)
                if event is not None: fences.append(event)
        sequence = controller.submit(_AsyncScorePayload(
            req_ids=tuple(req_ids), queries=query_snapshot, keys=keys, summaries=summaries,
            method=self._method, head_aggregation=self._head_aggregation,
            request_versions=tuple((r,self._async_request_versions.get(r,0)) for r in req_ids),
            residency_versions=tuple((k,v) for k,v in (residency_versions or {}).items() if k in keys),
            decision_leases=tuple((r,tuple(ids)) for r,ids in (decision_leases or {}).items()),
            transport_epoch=context.get('epoch'),
            source_tokens=tuple((r, context.get('sources',{}).get(r)) for r in req_ids),
            expires_at_ns=captured+max_age),
            generation=self._summary_generation, snapshot_bytes=snapshot_bytes,
            key_versions={k:self._durable_summary_publications.get(k) for k in keys},
            observation_available_ns=captured, readiness_fences=fences)
        if sequence is not None:
            self._async_submission_requests[sequence] = tuple(req_ids)
            controller.record_event(event='snapshot_captured', observation_sequence=sequence,
                request_ids=list(req_ids), generation=self._summary_generation, key_count=len(keys),
                snapshot_bytes=snapshot_bytes, retained_unique_storage_bytes=retained,
                workspace_allowance_bytes=snapshot_bytes-retained-query_bytes,
                snapshot_captured_ns=captured, publication_ns=time.monotonic_ns())
        return sequence

    def _drain_async_scores(self, *, wait: bool = False) -> None:
        controller = getattr(self, "_async_scoring", None)
        if controller is None:
            return
        completions, failures = (
            controller.drain_all() if wait else controller.drain_ready()
        )
        current_generation = getattr(self, "_summary_generation", 0)
        current_publications = getattr(self, "_durable_summary_publications", {})
        for failure in failures:
            self._async_submission_requests.pop(
                int(failure["observation_sequence"]), None
            )
        for completion in completions:
            snapshot = completion.snapshot
            self._async_submission_requests.pop(snapshot.observation_sequence, None)
            if snapshot.generation != current_generation:
                controller.record_event(
                    event="publication_rejected",
                    observation_sequence=snapshot.observation_sequence,
                    reason="stale_generation",
                )
                continue
            publication_ns = time.monotonic_ns()
            max_age_ns = getattr(
                self, "_async_scoring_max_snapshot_age_ns", 1_000_000_000
            )
            expires_at_ns = min(snapshot.observation_available_ns + max_age_ns,
                snapshot.payload.expires_at_ns or snapshot.observation_available_ns + max_age_ns)
            if publication_ns > expires_at_ns:
                controller.record_event(
                    event="publication_rejected",
                    observation_sequence=snapshot.observation_sequence,
                    generation=snapshot.generation,
                    reason="snapshot_expired",
                    snapshot_captured_ns=snapshot.observation_available_ns,
                    observation_available_ns=completion.input_ready_ns,
                    publication_ns=publication_ns,
                    snapshot_age_ns=(
                        publication_ns - snapshot.observation_available_ns
                    ),
                    max_snapshot_age_ns=max_age_ns,
                    expires_at_ns=expires_at_ns,
                )
                continue
            valid_keys = {
                key
                for key, version in snapshot.key_versions
                if version is not None and current_publications.get(key) == version
            }
            scores, head_counts = completion.result
            request_versions = dict(completion.snapshot.payload.request_versions)
            observation_scores: dict[
                str, dict[str, list[tuple[OffloadKey, float]]]
            ] = {}
            observation_head_counts: dict[str, dict[str, int]] = {}
            published_request_ids: list[str] = []
            for method, requests in scores.items():
                for req_id, ranked in requests.items():
                    if self._async_request_versions.get(req_id, 0) != request_versions.get(
                        req_id, 0
                    ):
                        controller.record_event(
                            event="request_publication_rejected",
                            observation_sequence=snapshot.observation_sequence,
                            request_ids=[req_id],
                            reason="request_retired_or_aborted",
                        )
                        continue
                    if any(not math.isfinite(float(value)) for _, value in ranked):
                        controller.record_event(
                            event="request_publication_rejected",
                            observation_sequence=snapshot.observation_sequence,
                            request_ids=[req_id],
                            reason="nonfinite_score",
                        )
                        continue
                    filtered = [item for item in ranked if item[0] in valid_keys]
                    if not filtered:
                        continue
                    method_scores = observation_scores.setdefault(method, {})
                    method_heads = observation_head_counts.setdefault(method, {})
                    method_scores[req_id] = filtered
                    method_heads[req_id] = head_counts[method][req_id]
                    published_request_ids.append(req_id)
            if not published_request_ids:
                controller.record_event(
                    event="publication_rejected",
                    observation_sequence=snapshot.observation_sequence,
                    reason=(
                        "no_current_resident_version"
                        if not valid_keys
                        else "all_requests_retired_or_aborted"
                    ),
                )
            else:
                self._pending_async_score_observations.append(
                    {
                        "observation_sequence": snapshot.observation_sequence,
                        "async_contract": "bounded-v2",
                        "transport_epoch": snapshot.payload.transport_epoch,
                        "source_tokens": dict(snapshot.payload.source_tokens),
                        "request_ids": published_request_ids,
                        "scores": observation_scores,
                        "residency_versions": dict(snapshot.payload.residency_versions),
                        **({"decision_leases":dict(completion.snapshot.payload.decision_leases)} if completion.snapshot.payload.decision_leases else {}),
                        "head_counts": observation_head_counts,
                        "snapshot_captured_ns": snapshot.observation_available_ns,
                        "observation_available_ns": completion.input_ready_ns,
                        "expires_at_ns": expires_at_ns,
                    }
                )
                controller.record_event(
                    event="published",
                    observation_sequence=snapshot.observation_sequence,
                    request_ids=published_request_ids,
                    generation=snapshot.generation,
                    snapshot_captured_ns=snapshot.observation_available_ns,
                    observation_available_ns=completion.input_ready_ns,
                    publication_ns=publication_ns,
                    expires_at_ns=expires_at_ns,
                    max_snapshot_age_ns=max_age_ns,
                )
        pending_request_ids = {
            req_id
            for req_ids in self._async_submission_requests.values()
            for req_id in req_ids
        }
        retained_ids = pending_request_ids | set(getattr(self,'_query_snapshots',{}))
        retained_ids.update(r for obs in getattr(self,'_pending_async_score_observations',[]) for r in obs['request_ids'])
        getattr(self,'_retired_async_request_ids',set()).intersection_update(retained_ids)
        for req_id in list(self._async_request_versions):
            if req_id not in retained_ids:
                self._async_request_versions.pop(req_id, None)

    def retire_async_requests(self, request_ids: Sequence[str]) -> None:
        # Ordinary completion NEVER fences unrelated scoring. Final source queries
        # retain the same bounded feature lease; ready metadata polls publish later.
        if not request_ids:
            return
        # A terminal request can be cancelled before its first resumed forward.
        # Drop the pending marker in every scoring mode. Already observed
        # timestamps still belong to this step's metadata and are popped there.
        getattr(self, "_preempted_request_ids", set()).difference_update(request_ids)
        if not hasattr(self, '_retired_async_request_ids'):
            self._retired_async_request_ids = set()
        if getattr(self,'_scoring_mode','synchronous') != 'asynchronous':return
        live = set(getattr(self,'_query_snapshots',{}))
        live.update(r for ids in getattr(self,'_async_submission_requests',{}).values() for r in ids)
        live.update(r for obs in getattr(self,'_pending_async_score_observations',[]) for r in obs['request_ids'])
        self._retired_async_request_ids.intersection_update(live)
        self._retired_async_request_ids.update(set(request_ids).intersection(live))
        self._drain_async_scores()  # ready ordered prefix only

    def cancel_async_requests(self, request_ids):
        for req_id in request_ids:
            self._async_request_versions[req_id] = self._async_request_versions.get(req_id,0)+1
            getattr(self,'_query_snapshots',{}).pop(req_id,None)
            getattr(self,'_query_snapshot_events',{}).pop(req_id,None)
        cancelled=set(request_ids)
        observations=[]
        for obs in getattr(self,'_pending_async_score_observations',[]):
            scores={method:{r:values for r,values in reqs.items() if r not in cancelled}
                    for method,reqs in obs['scores'].items()}
            if any(reqs for reqs in scores.values()):
                obs=dict(obs,scores=scores,request_ids=[r for r in obs['request_ids'] if r not in cancelled])
                observations.append(obs)
        self._pending_async_score_observations=observations
        # In-flight owned readers remain charged until the controller completes.

    def pop_async_scoring_telemetry(self) -> list[dict[str, object]]:
        controller = getattr(self, "_async_scoring", None)
        return controller.pop_telemetry() if controller is not None else []

    @profiled("scoring.callback")
    def _on_queries_captured(self, req_ids: list[str], queries: torch.Tensor) -> None:
        PROFILE.count("scoring.capture_callbacks")
        PROFILE.count("scoring.captured_requests", len(req_ids))
        retired = getattr(self,'_retired_async_request_ids',set())
        for req_id in req_ids:
            if req_id in retired:
                self.cancel_async_requests([req_id]);retired.remove(req_id)
        schedule = getattr(self, "_scoring_schedule", "continuous")
        if schedule in ("opportunity", "capture_only"):
            # One bounded snapshot per request; no pool scan, score readback,
            # sorting or EMA updates during ordinary generation.
            with PROFILE.scope("scoring.query_snapshot", cuda=queries.is_cuda):
                now_ns = time.monotonic_ns()
                snapshots = self._query_snapshots
                for req_id, query in zip(req_ids, queries):
                    snapshots.pop(req_id, None)
                    snapshots[req_id] = (now_ns, query.detach().clone())
                if queries.is_cuda:
                    event = torch.cuda.Event(); event.record(torch.cuda.current_stream(queries.device))
                    events = self.__dict__.setdefault('_query_snapshot_events',{})
                    for req_id in req_ids: events[req_id] = event
                # Updates reinsert at the end, so expiry only walks expired
                # entries rather than the entire retained request pool.
                while snapshots and now_ns - next(iter(snapshots.values()))[0] > self._query_snapshot_max_age_ns:
                    del snapshots[next(iter(snapshots))]
                while len(snapshots) > self._query_snapshot_max_requests:
                    del snapshots[next(iter(snapshots))]
                if len(queries):
                    PROFILE.count("scoring.snapshot_bytes_observed",
                                  len(snapshots) * queries[0].numel() * queries.element_size())
                events = getattr(self,'_query_snapshot_events',{})
                for key in list(events):
                    if key not in snapshots: events.pop(key,None)
            return
        self._score_queries(req_ids, queries)

    def receive_scoring_context(self, context):
        """Scheduler identity changes invalidate before stride/capture, not after."""
        old = getattr(self, '_score_transport_context', {})
        sources = context.get('sources', {})
        changed = [r for r, token in old.get('sources', {}).items()
                   if old.get('epoch') != context.get('epoch') or sources.get(r) != token]
        if changed:
            self.cancel_async_requests(changed)
        self._score_transport_context = context

    def pop_prefix_score_observations(self):
        observations = getattr(self, '_pending_prefix_score_observations', [])
        self._pending_prefix_score_observations = []
        return observations

    @profiled('scoring.prefix_mean')
    def receive_prefix_score_requests(self, request):
        """Score exact prefixes against one common fresh, actively captured query.

        This transport contains scalars, never fabricated per-key ranks or EMA
        updates. Centroids average keys equally, matching the existing raw
        mean-key / mean-head metric up to FP32 reduction order.
        """
        context = getattr(self, '_score_transport_context', {})
        if not context.get('prefix_mean_prefetch_scoring') or not request:
            return
        if (getattr(self, '_method', None) != 'mean'
                or getattr(self, '_head_aggregation', 'mean') != 'mean'
                or getattr(self, '_scoring_mode', 'synchronous') != 'synchronous'):
            raise ValueError('prefix mean scoring requires synchronous mean/mean')
        candidates = request.get('candidates', {})
        if not candidates or len(candidates) > 16:
            PROFILE.count('scoring.prefix_request_over_bound')
            return
        now = time.monotonic_ns()
        snapshots = getattr(self, '_query_snapshots', {})
        selected = []
        for req_id in sorted(set(request.get('active_query_ids', ()))):
            snapshot = snapshots.get(req_id)
            token = context.get('sources', {}).get(req_id)
            if snapshot is not None and token is not None and 0 <= now - snapshot[0] <= self._query_snapshot_max_age_ns:
                selected.append((req_id, token, snapshot[0], snapshot[1]))
        if not selected:
            stats = self.__dict__.setdefault('_prefix_mean_scoring_stats', {})
            stats['missing_fresh_active_query'] = stats.get('missing_fresh_active_query', 0) + 1
            PROFILE.count('scoring.prefix_no_fresh_active_query')
            return
        # Existing query snapshots are owned clones. Work remains on the model
        # stream; no borrowed source allocation escapes this synchronous call.
        with torch.no_grad():
            for req_id, _, _, query in selected:
                event = getattr(self, '_query_snapshot_events', {}).get(req_id)
                if query.is_cuda and event is not None:
                    torch.cuda.current_stream(query.device).wait_event(event)
            query = torch.stack([x[3].detach().float() for x in selected]).mean(0)
            if query.ndim != 2:
                return
            if getattr(self, '_summary_stream', None) is not None:
                torch.cuda.current_stream(query.device).wait_stream(self._summary_stream)
            cache = self.__dict__.setdefault('_prefix_mean_centroid_cache', {})
            centroids, descriptors = [], []
            publications = getattr(self, '_durable_summary_publications', {})
            for event_id, descriptor in candidates.items():
                keys = tuple(descriptor.get('keys', ()))
                versions = tuple(publications.get(k) for k in keys)
                if not keys or any(v is None for v in versions) or any(k not in self.durable_summaries for k in keys):
                    continue
                signature = (getattr(self, '_summary_generation', 0), keys, versions)
                centroid = cache.get(signature)
                if centroid is None:
                    summaries = [self.durable_summaries[k] for k in keys]
                    h, d = query.shape
                    if len(keys) * h * d * 4 > 16 * 1024 * 1024 or any(
                            len(heads) != h or any(s.mean.shape != (d,) or s.mean.device != query.device for s in heads)
                            for heads in summaries):
                        PROFILE.count('scoring.prefix_summary_over_bound_or_mismatch')
                        continue
                    centroid = torch.stack([s.mean.detach() for heads in summaries for s in heads]).reshape(len(keys), h, d).float().mean(0)
                    # Cache only owned centroid storage, never per-head views.
                    if centroid.numel() * centroid.element_size() > 1024 * 1024:
                        continue
                    cache[signature] = centroid
                    while len(cache) > 16 or sum(t.untyped_storage().nbytes() for t in cache.values()) > 1024 * 1024:
                        cache.pop(next(iter(cache)))
                    stats = self.__dict__.setdefault('_prefix_mean_scoring_stats', {})
                    stats['centroid_builds'] = stats.get('centroid_builds', 0) + 1
                    PROFILE.count('scoring.prefix_centroid_builds')
                else:
                    stats = self.__dict__.setdefault('_prefix_mean_scoring_stats', {})
                    stats['centroid_hits'] = stats.get('centroid_hits', 0) + 1
                    PROFILE.count('scoring.prefix_centroid_hits')
                centroids.append(centroid)
                descriptors.append((event_id, dict(descriptor), versions))
            if not centroids:
                return
            values = (torch.stack(centroids) * query).sum(-1).mean(-1).tolist()
        self._prefix_score_sequence = getattr(self, '_prefix_score_sequence', 0) + 1
        captured = min(x[2] for x in selected)
        observation = {
            'contract': 'prefix-mean-v1', 'transport_epoch': context.get('epoch'),
            'observation_sequence': self._prefix_score_sequence,
            'query_sources': {r: token for r, token, _, _ in selected},
            'query_capture_ns': {r: stamp for r, _, stamp, _ in selected},
            'snapshot_captured_ns': captured, 'observation_available_ns': time.monotonic_ns(),
            'expires_at_ns': captured + self._query_snapshot_max_age_ns,
            'metric': 'raw_mean_key_dot_mean_head_common_active_query',
            'candidates': {eid: {'descriptor': desc, 'score': float(value),
                                  'summary_publications': versions}
                           for (eid, desc, versions), value in zip(descriptors, values)
                           if math.isfinite(value)},
        }
        # At most one bounded scalar batch is retained until metadata polling.
        self._pending_prefix_score_observations = [observation]
        stats = self.__dict__.setdefault('_prefix_mean_scoring_stats', {})
        stats['numerical_calls'] = stats.get('numerical_calls', 0) + 1
        stats['numerical_candidates'] = stats.get('numerical_candidates', 0) + len(observation['candidates'])
        PROFILE.count('scoring.prefix_numerical_calls')
        PROFILE.count('scoring.prefix_numerical_candidates', len(observation['candidates']))

    @profiled("scoring.decision_request")
    def receive_score_requests(self, requests: dict[str, list[OffloadKey]], residency_versions=None, lease_ids=None) -> None:
        if getattr(self, "_scoring_schedule", "continuous") != "opportunity":
            return
        for req_id, candidates in requests.items():
            snapshot = self._query_snapshots.get(req_id)
            now_ns = time.monotonic_ns()
            if snapshot is None or not 0 <= now_ns - snapshot[0] <= self._query_snapshot_max_age_ns:
                PROFILE.count("scoring.decision_snapshot_missing_or_stale")
                continue
            stamp, query = snapshot
            if getattr(self,'_scoring_mode','synchronous') == 'asynchronous':
                signatures = self.__dict__.setdefault('_decision_async_signatures',{})
                versions = (residency_versions or {}).get(req_id,{})
                signature = (stamp,tuple(candidates),tuple(versions.items()),tuple((lease_ids or {}).get(req_id,())),getattr(self,'_summary_generation',0))
                if signatures.get(req_id) == signature: continue
                event = getattr(self,'_query_snapshot_events',{}).get(req_id)
                sequence = self._submit_async_scoring([req_id],query.unsqueeze(0),candidate_keys=candidates,
                    feature_ns=stamp,query_fences=() if event is None else (event,),residency_versions=versions,
                    decision_leases={req_id:(lease_ids or {}).get(req_id,())} if lease_ids is not None else None)
                if sequence is not None: signatures[req_id] = signature
                # Bounded by the same current/retired snapshot lease map.
                for key in list(signatures):
                    if key not in self._query_snapshots: signatures.pop(key,None)
                continue
            # The existing observation transport preserves feature time,
            # availability, expiry and consumer checks. Re-scoring an old
            # query never renews its freshness lease.
            self._score_queries([req_id], query.unsqueeze(0), candidate_keys=candidates)
            ranked = self._pending_scores.get(self._method, {}).pop(req_id, None)
            heads = self._pending_score_head_counts.get(self._method, {}).pop(req_id, None)
            if not ranked:
                continue
            self._decision_score_sequence += 1
            self._pending_async_score_observations.append({
                "observation_sequence": self._decision_score_sequence,
                "request_ids": [req_id], "scores": {self._method: {req_id: ranked}},
                "head_counts": {self._method: {req_id: heads}},
                "snapshot_captured_ns": stamp,
                "observation_available_ns": time.monotonic_ns(),
                "expires_at_ns": stamp + self._query_snapshot_max_age_ns,
            })
            PROFILE.count("scoring.decision_scores_published")

    def _score_shortlists(self, req_ids, queries, keys, cache, candidate_keys):
        allowed = None if candidate_keys is None else set(candidate_keys)
        # An eligibility filter wins before index routing. Scanning Python
        # keys is cheap and does not perform vector comparisons.
        if allowed is not None:
            indices = [i for i, key in enumerate(keys) if key in allowed]
            rows = [indices for _ in req_ids]
            if (self._candidate_search.centroids is not None
                    and len(indices) >= self._candidate_search.config.min_pool):
                allowed_rows = set(indices)
                routed = self._candidate_search.candidates(queries)
                rows = [sorted(allowed_rows.intersection(found)) or indices for found in routed]
            PROFILE.count("search.eligibility_filtered_queries", len(req_ids))
        else:
            rows = self._candidate_search.candidates(queries)
        for req_id, query, selected in zip(req_ids, queries, rows):
            PROFILE.count("scoring.possible_block_comparisons", len(keys))
            PROFILE.count("scoring.exact_block_comparisons", len(selected))
            PROFILE.count("scoring.unobserved_blocks", len(keys) - len(selected))
            if not selected:
                continue
            with PROFILE.scope("search.gather", cuda=query.is_cuda):
                index = torch.tensor(selected, dtype=torch.long, device=query.device)
                means = cache["mean"].index_select(0, index)
            with PROFILE.scope("scoring.vector_math", cuda=query.is_cuda):
                # Exactly the existing mean-key, mean-head score for visited
                # rows. Never normalize queries/keys into cosine similarity.
                combined = score_mean_batch(query.unsqueeze(0), means).mean(dim=-1)
            with PROFILE.scope("scoring.readback"):
                values = combined.tolist()
            with PROFILE.scope("scoring.rank_metadata"):
                ranked = sorted(((keys[row], value) for row, value in zip(selected, values)),
                                key=lambda item: (-item[1], item[0]))
                self._pending_scores.setdefault(self._method, {})[req_id] = ranked
                self._pending_score_head_counts.setdefault(self._method, {})[req_id] = query.shape[0]

    @profiled("scoring.total")
    def _score_queries(self, req_ids: list[str], queries: torch.Tensor,
                       candidate_keys: list[OffloadKey] | None = None) -> None:
        if not self.durable_summaries or not req_ids:
            return
        if getattr(self, "_scoring_mode", "synchronous") == "asynchronous":
            self._submit_async_scoring(req_ids, queries)
            return
        _t_call = time.perf_counter() if _TIMING else 0.0
        if self._summary_stream is not None:
            # Durable tensors are produced on the summary stream. Scoring on
            # the model stream must not observe partially-written reductions.
            torch.cuda.current_stream().wait_stream(self._summary_stream)
        # 2026-08-01: test whether query_captured_total's per-call cost
        # growth (07-30/08-01 handoffs, rag@8.0) tracks batch size rather
        # than resident-pool size -- resident= stayed flat in the last
        # repro, so if batch size is flat too, neither is the mechanism and
        # the next lead is elsewhere (e.g. cache/stack_cache_dirty churn
        # rate itself). Recorded together so one grep correlates both
        # against the same call cadence as SEMANTIC_TIMING.
        if _TIMING:
            record_count("query_captured_batch_size", len(req_ids))
            record_count("query_captured_resident_pool", len(self.durable_summaries))
            record_count("query_captured_pending_scores", len(self._pending_scores))
            # 2026-08-02: allocator-fragmentation theory (08-02 handoff) --
            # snapshot GPU allocator counters on the same cadence to correlate
            # with query_captured_total's per-call cost climb.
            record_gpu_memory("query_captured")
            record_process_state("query_captured")
        # queries: [n_reqs, num_kv_heads, head_dim] -- one row per request
        # scheduled in this step. All concurrent requests are scored in ONE
        # batched pass with ONE GPU sync per step, instead of one pass-plus-
        # sync per request: per-call cost is flat (~1ms), but the capture
        # loop fires once per scheduled request per eligible step, so at
        # saturation the per-request loop's aggregate (~n_reqs x ~1ms/step,
        # n_reqs measured ~56 on the dev box) was the largest per-step cost
        # bucket -- see the cross-request batching entry in the issues log.
        # summaries are fp32 (Step 1.2 upcast).
        queries = queries.float()

        # Score and keep EVERY resident summary, not just a top-M slice. A
        # prior `_TOP_M=8` cap here meant most of a request's own blocks
        # never accumulated a relevance-EMA entry at all (fewer than 8 of
        # them would ever rank in some single query's own top-8, regardless
        # of how small the resident pool was -- confirmed unconditional, not
        # a crowding effect, empirically at pool sizes 21-645), which was
        # the dominant reason Step 1.5's prefetch splice could never find a
        # fully-scored request to exact-match against (issues log entries
        # #29-31).
        #
        # Scores only `self._method` (not all three) via one vectorized
        # batched pass over the whole candidate pool (index.py's
        # score_*_batch functions) -- one GPU sync total, not one per
        # candidate -- against a cache of the pool stacked into tensors,
        # rebuilt only when new candidates are inserted
        # (_rebuild_stack_cache), not on every query capture. Found and
        # fixed on a real B200 production-scale run (issues log entry #53):
        # the original per-candidate `.item()`-synchronizing design and the
        # per-query stack rebuild were both measured as "near-zero cost" on
        # the tiny dev model's small resident pool, but turned multi-second
        # per query capture at production scale (larger model, larger CPU
        # tier, thousands of resident candidates) -- see entry #53 for the
        # full investigation, including the third contributing bug (worker
        # was building summaries for every model layer, not just this one)
        # fixed in _build_summaries_body below.
        if self._stack_cache_dirty:
            _t_rebuild = time.perf_counter() if _TIMING else 0.0
            self._rebuild_stack_cache()
            search = getattr(self, "_candidate_search", None)
            if search is not None:
                search.build(self._stack_cache["mean"]) if self._method == "mean" else None
            if _TIMING:
                record_timing("stack_rebuild", time.perf_counter() - _t_rebuild)
        keys = self._stack_cache_keys
        cache = self._stack_cache
        search = getattr(self, "_candidate_search", None)
        if candidate_keys is not None or (search is not None and search.centroids is not None):
            self._score_shortlists(req_ids, queries, keys, cache, candidate_keys)
            return
        PROFILE.count("scoring.exact_block_comparisons", len(req_ids) * len(keys))

        # [n_reqs, 1, num_kv_heads, head_dim] broadcasts against the
        # [n_candidates, num_kv_heads, head_dim] stacks to score every
        # (request, candidate) pair in one op -- the score_*_batch functions
        # are broadcast-shape-agnostic, so no index.py change is needed.
        with PROFILE.scope('scoring.vector_math', cuda=queries.is_cuda):
            query = queries.unsqueeze(1)
            if self._method == "minmax":
                per_head = score_minmax_batch(
                    query, cache["max"].unsqueeze(0), cache["min"].unsqueeze(0)
                )
            elif self._method == "mean":
                per_head = score_mean_batch(query, cache["mean"].unsqueeze(0))
            else:  # cuboid_mean
                per_head = score_cuboid_mean_batch(
                    query, cache["mean"].unsqueeze(0), cache["mad"].unsqueeze(0)
                )
            # Combine KV heads before the one CPU/GPU synchronization. Mean is
            # the current evidence-backed default; max remains an audit arm.
            _t_sync = time.perf_counter() if _TIMING else 0.0
            # [n_reqs, n_candidates] -> list of per-request score lists.
            if self._head_aggregation == "mean":
                combined = per_head.mean(dim=-1)
            else:
                combined = per_head.max(dim=-1).values
        with PROFILE.scope('scoring.readback'):
            fast_rank_indices = None
            if getattr(self, "_rank_ema_mode", "default") == "fast":
                # Score outputs are finite for all supported index methods. Keep a
                # fail-closed fallback for malformed/custom tensors so the opt-in
                # path cannot change legacy NaN ordering semantics.
                if bool(torch.isfinite(combined).all()):
                    tie_order = getattr(self, "_rank_tie_order", [])
                    if len(tie_order) != len(keys):
                        tie_order = sorted(range(len(keys)), key=keys.__getitem__)
                    tie_order_tensor = torch.as_tensor(
                        tie_order, device=combined.device, dtype=torch.long
                    )
                    tie_ordered_scores = combined.index_select(1, tie_order_tensor)
                    tie_positions = torch.argsort(
                        tie_ordered_scores, dim=1, descending=True, stable=True
                    )
                    fast_rank_indices = tie_order_tensor[tie_positions]
                    all_scores = tie_ordered_scores.gather(1, tie_positions).tolist()
                    all_rank_indices = fast_rank_indices.tolist()
                else:
                    all_scores = combined.tolist()
                    all_rank_indices = None
            else:
                all_scores = combined.tolist()
                all_rank_indices = None
        if _TIMING:
            record_timing("query_captured_sync", time.perf_counter() - _t_sync)
        _t_rank = time.perf_counter() if _TIMING else 0.0
        with PROFILE.scope('scoring.rank_metadata'):
            method_scores = self._pending_scores.setdefault(self._method, {})
            head_counts = getattr(self, "_pending_score_head_counts", None)
            if head_counts is None:
                head_counts = {}
                self._pending_score_head_counts = head_counts
            method_head_counts = head_counts.setdefault(self._method, {})
            for row, (req_id, scores) in enumerate(zip(req_ids, all_scores)):
                if all_rank_indices is None:
                    ranked = sorted(zip(keys, scores), key=lambda kv: (-kv[1], kv[0]))
                else:
                    ranked = [
                        (keys[index], score)
                        for index, score in zip(all_rank_indices[row], scores)
                    ]
                method_scores[req_id] = ranked
                method_head_counts[req_id] = per_head.shape[-1]
                if _DEBUG:
                    debug_print(
                        f"SEMANTIC_STEP1_3_DEBUG req={req_id} method={self._method} "
                        f"n_summaries={len(self.durable_summaries)} "
                        f"ranked_keys={[k.hex()[:8] for k, _ in ranked]} "
                        f"scores={[round(s, 4) for _, s in ranked]}"
                    )
        if _TIMING:
            record_timing("query_rank_metadata", time.perf_counter() - _t_rank)
            record_timing("query_captured_total", time.perf_counter() - _t_call)

    def _check_layout(self, layer_name: str, layer, kv_cache: torch.Tensor) -> None:
        if layer_name in self._layout_checked:
            return
        self._layout_checked.add(layer_name)
        expected = layer.attn_backend.get_kv_cache_shape(
            kv_cache.shape[0], self._block_size, layer.num_kv_heads, layer.head_size
        )
        assert tuple(kv_cache.shape) == expected, (
            f"layer {layer_name}: kv_cache.shape={tuple(kv_cache.shape)} does not "
            f"match get_kv_cache_shape()={expected} -- this backend does not "
            "match the (num_blocks, num_kv_heads, block_size, 2*head_size) "
            "layout this module assumes; do not trust summaries built from it."
        )

    @profiled('summary.build_blocks')
    def _build_summaries_for_blocks(self, job_id: int, block_ids) -> None:
        """Overlaps the reduction work with inference via a side stream when
        CUDA is available; falls back to running it inline (still correct,
        just not overlapped) so this logic stays testable without a GPU.
        Durable re-keying (by OffloadKey) happens on the same stream, right
        after, so it observes the just-computed summaries in issue order --
        no separate cross-stream synchronization needed."""
        if len(block_ids) == 0:
            return
        if self._summary_stream is not None:
            current_stream = torch.cuda.current_stream()
            with torch.cuda.stream(self._summary_stream):
                self._summary_stream.wait_stream(current_stream)
                self._build_summaries_body(block_ids)
                self._durably_key_summaries(job_id, block_ids)
        else:
            self._build_summaries_body(block_ids)
            self._durably_key_summaries(job_id, block_ids)

    @profiled('summary.reductions', cuda=True)
    def _build_summaries_body(self, block_ids) -> None:
        # Real bug found on a B200 production-scale run (issues log entry
        # #53's follow-up): this used to loop over EVERY attention layer in
        # the model (self._attention_layers, ~28 for a 7B model), but
        # self.summaries is only ever READ for self._probe_layer_name
        # (_durably_key_summaries below, the only call site) -- every other
        # layer's summary was real, non-trivial tensor work (4 reductions
        # per KV head per block) computed and thrown away, unread, on
        # every single block stored. Restricting to just the probe layer
        # cuts this by roughly the model's layer count.
        if self._probe_layer_name is None:
            return
        layer = self._attention_layers.get(self._probe_layer_name)
        if layer is None:
            return
        kv_cache = getattr(layer, "kv_cache", None)
        if kv_cache is None or kv_cache.numel() == 0:
            return
        self._check_layout(self._probe_layer_name, layer, kv_cache)
        head_size = layer.head_size
        layer_summaries = self.summaries[self._probe_layer_name]
        block_ids = [int(block_id) for block_id in block_ids]
        if not block_ids:
            return
        if any(block_id < 0 or block_id >= kv_cache.shape[0] for block_id in block_ids):
            raise IndexError("summary source block is outside the KV cache")
        # Basic views + one stack avoid a host-to-device index upload. The
        # stack owns its storage before later GPU block reuse, and reductions
        # run for every block and head together rather than in Python loops.
        keys = torch.stack([
            kv_cache[block_id, ..., :head_size] for block_id in block_ids
        ]).float()
        for block_id, summaries in zip(block_ids, self._summarize_worker_key_batches(keys)):
            layer_summaries[block_id] = summaries

    def _summarize_worker_key_batches(self, keys: torch.Tensor) -> list[list[_WorkerSummary]]:
        if getattr(self, "_summary_fields", "all") == "required":
            if self._method != "mean" or self._scoring_schedule not in ("continuous", "opportunity"):
                raise ValueError("required summary fields require mean scoring with continuous or opportunity schedule")
            return self._summarize_mean_key_batches(keys)
        return self._summarize_key_batches(keys)

    @staticmethod
    def _summarize_mean_key_batches(keys: torch.Tensor) -> list[list[_MeanOnlySummary]]:
        """Same centroid reduction as the full builder, with owned mean storage."""
        if keys.ndim != 4 or any(size <= 0 for size in keys.shape[1:]):
            raise ValueError("mean summary keys require [groups, heads, tokens, dim] with positive inner dimensions")
        means = keys.mean(dim=2)
        summaries = [_MeanOnlySummary(mean)
                     for mean in means.flatten(0, 1).unbind(0)]
        heads = keys.shape[1]
        return [summaries[start : start + heads]
                for start in range(0, len(summaries), heads)]

    @staticmethod
    def _summarize_key_batches(keys: torch.Tensor) -> list[list[BlockSummary]]:
        """Summarize [groups, heads, tokens, dim] without crossing groups.

        Outputs own reduced storage independent of KV; slices share the small
        batch reduction buffers until the last summary from that batch retires.
        """
        means = keys.mean(dim=2)
        mins = keys.amin(dim=2)
        maxs = keys.amax(dim=2)
        mads = (keys - means.unsqueeze(2)).abs().mean(dim=2)
        # One unbind per field creates all row/head views in C++, avoiding
        # eight Python-dispatched select operations per summary. Regroup the
        # resulting Python objects without further tensor indexing.
        summaries = [BlockSummary(min=minimum, max=maximum, mean=mean, mad=mad)
                     for minimum, maximum, mean, mad in zip(
                         mins.flatten(0, 1).unbind(0),
                         maxs.flatten(0, 1).unbind(0),
                         means.flatten(0, 1).unbind(0),
                         mads.flatten(0, 1).unbind(0))]
        heads = keys.shape[1]
        return [summaries[start : start + heads]
                for start in range(0, len(summaries), heads)]

    @profiled('summary.publish')
    def _durably_key_summaries(self, job_id: int, block_ids) -> None:
        """Re-key this job's probe-layer summaries by OffloadKey so they
        outlive GPU block reuse. Job-level granularity: a store job's keys
        are attributed to all of that job's blocks (see connector.py's
        module docstring and the issues log for why, and the known blur this
        causes when one job covers many blocks)."""
        if self._probe_layer_name is None:
            return
        keys = self._pending_job_keys.pop(job_id, None)
        if not keys:
            return
        probe_summaries = self.summaries.get(self._probe_layer_name, {})
        # list[block] of list[BlockSummary] (one per KV head, unaveraged).
        block_summaries = [
            probe_summaries[int(b)] for b in block_ids if int(b) in probe_summaries
        ]
        if not block_summaries:
            return
        debug_print(
            f"SEMANTIC_STEP1_3_DEBUG store job={job_id} "
            f"keys={[k.hex()[:8] for k in keys]} block_ids={list(block_ids)}"
        )
        if len(block_summaries) != len(keys):
            debug_print(
                f"SEMANTIC_STORE_LAYOUT_SKIPPED job={job_id} "
                f"keys={len(keys)} blocks={len(block_summaries)}"
            )
            return
        for key, summary in zip(keys, block_summaries):
            self.durable_summaries[key] = summary
        self._register_summary_publications(job_id, list(keys))
        self._mark_inserted_into_stack_cache(keys)
        pruned = self._prune_durable_summaries()
        if pruned:
            self._mark_removed_from_stack_cache(pruned)

    @profiled('summary.build_job', cuda=True)
    def _build_durable_summaries_for_job(self, job_id: int) -> None:
        """Build summaries using an explicit key-to-GPU-block layout."""
        layout = self._pending_job_blocks.pop(job_id, None)
        if not layout or self._probe_layer_name is None:
            return
        layer = self._attention_layers.get(self._probe_layer_name)
        if layer is None:
            return
        kv_cache = getattr(layer, "kv_cache", None)
        if kv_cache is None or kv_cache.numel() == 0:
            return
        self._check_layout(self._probe_layer_name, layer, kv_cache)
        head_size = layer.head_size
        # Batch entries of the same block count so each reduction still sees
        # exactly that key's ordered concatenation of source-block tokens.
        # Grouping is CPU metadata only: no padding, head pooling, or borrowed
        # GPU indices; one gather owns all key data before reducing anything.
        groups: dict[int, list[tuple[int, Sequence[int]]]] = {}
        for position, (_, block_ids) in enumerate(layout):
            if len(block_ids):
                groups.setdefault(len(block_ids), []).append((position, block_ids))
        if not groups:
            return
        selected_block_ids = [int(block_id)
                              for entries in groups.values()
                              for _, block_ids in entries
                              for block_id in block_ids]
        # Unlike index_select, basic indexing accepts negative indices. Reject
        # them explicitly rather than silently attributing the final GPU block.
        if any(block_id < 0 or block_id >= kv_cache.shape[0]
               for block_id in selected_block_ids):
            raise IndexError("summary source block is outside the KV cache")
        selected = torch.stack([
            kv_cache[int(block_id), ..., :head_size]
            for block_id in selected_block_ids
        ]).float()
        summaries_by_position = {}
        offset = 0
        for count, entries in groups.items():
            num_blocks = count * len(entries)
            keys = selected[offset : offset + num_blocks]
            offset += num_blocks
            if count > 1:
                # [groups, blocks, heads, tokens, dim] ->
                # [groups, heads, blocks*tokens, dim], preserving token order.
                keys = keys.reshape(len(entries), count, *keys.shape[1:])
                keys = keys.permute(0, 2, 1, 3, 4).reshape(
                    len(entries), keys.shape[2], -1, head_size
                )
            for (position, _), summaries in zip(entries, self._summarize_worker_key_batches(keys)):
                summaries_by_position[position] = summaries
        inserted: list[OffloadKey] = []
        # Publish in original order, including last-write behavior for any
        # repeated key. Grouping by shape must not alter retention order.
        for position, (key, _) in enumerate(layout):
            if position not in summaries_by_position:
                continue
            self.durable_summaries[key] = summaries_by_position[position]
            inserted.append(key)
        if not inserted:
            return
        self._register_summary_publications(job_id, inserted)
        self._mark_inserted_into_stack_cache(inserted)
        pruned = self._prune_durable_summaries()
        if pruned:
            self._mark_removed_from_stack_cache(pruned)

    def _prune_durable_summaries(self) -> list[OffloadKey]:
        """Bound `durable_summaries` to (approximately) the real CPU tier's
        capacity -- see the `_max_durable_summaries` comment in __init__ for
        why this exists and its known imprecision. Evicts oldest-inserted
        first: plain dicts preserve insertion order, so this needs no extra
        bookkeeping, and insertion order is a reasonable proxy for "still
        resident" absent a real eviction signal. Returns the pruned keys so
        the caller can also drop their (possibly already-synced) rows from
        the stack cache -- this path bypasses receive_evicted_keys, so
        without this a pruned key's stale row would otherwise linger in the
        cache and keep being scored forever."""
        overflow = len(self.durable_summaries) - self._max_durable_summaries
        if overflow <= 0:
            return []
        pruned = list(itertools.islice(self.durable_summaries, overflow))
        for key in pruned:
            del self.durable_summaries[key]
            getattr(self, "_durable_summary_publications", {}).pop(key, None)
        return pruned

    @profiled('transfer.submit_store')
    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        self._record_transfer_start(job_id)
        summary_stream = getattr(self, "_summary_stream", None)
        # Instrumented ARC and CPU-only advice have no semantic probe. Their
        # stores still use the native copy fences, but need no summary stream
        # or event. Consume metadata here even if the base submission fails.
        if hasattr(self, "_probe_layer_name") and self._probe_layer_name is None:
            self._pending_job_blocks.pop(job_id, None)
            self._pending_job_keys.pop(job_id, None)
            return super().submit_store(job_id, src_spec, dst_spec)
        overlap = (
            bool(getattr(self, "_summary_offload_overlap", False))
            and summary_stream is not None
            and len(src_spec.block_ids) > 0
        )
        summary_event = None
        if overlap:
            current_stream = torch.cuda.current_stream()
            with torch.cuda.stream(summary_stream):
                summary_stream.wait_stream(current_stream)
                if job_id in self._pending_job_blocks:
                    self._build_durable_summaries_for_job(job_id)
                else:
                    # Legacy metadata remains exact only for one-block-per-key
                    # jobs. The caller already selected the side stream here.
                    self._build_summaries_body(src_spec.block_ids)
                    self._durably_key_summaries(job_id, src_spec.block_ids)
                if overlap:
                    summary_event = torch.cuda.Event()
                    summary_event.record(summary_stream)
        elif summary_stream is not None:
            # Preserve the serialized default path and its existing stream
            # ordering exactly. The worker-side mock/CPU tests also exercise
            # this branch without requiring a CUDA context.
            if job_id in self._pending_job_blocks:
                current_stream = torch.cuda.current_stream()
                with torch.cuda.stream(summary_stream):
                    summary_stream.wait_stream(current_stream)
                    self._build_durable_summaries_for_job(job_id)
            else:
                self._build_summaries_for_blocks(job_id, src_spec.block_ids)
            # The base GPU->CPU transfer stream waits on the summary stream
            # before its completion may release source block ownership.
            torch.cuda.current_stream().wait_stream(summary_stream)
        elif job_id in self._pending_job_blocks:
            self._build_durable_summaries_for_job(job_id)
        else:
            self._build_summaries_for_blocks(job_id, src_spec.block_ids)

        summary_job_generations = getattr(self, "_summary_job_generations", None)
        if summary_job_generations is None:
            summary_job_generations = self._summary_job_generations = {}
        summary_job_generations[job_id] = getattr(self, "_summary_generation", 0)

        if overlap:
            # The transfer itself may run concurrently with the reduction.
            # Its completion is withheld by get_finished()/wait() until this
            # event is complete, so the scheduler's refcount/flush fence still
            # protects source-block reuse for both consumers.
            summary_job_events = getattr(self, "_summary_job_events", None)
            if summary_job_events is None:
                summary_job_events = self._summary_job_events = {}
            summary_job_events[job_id] = summary_event
            if _TIMING:
                record_count("summary_offload_overlap_submitted", 1)
        try:
            success = super().submit_store(job_id, src_spec, dst_spec)
        except BaseException:
            # A raised base submission cannot leave a side-stream event or
            # publication ownership live: fence the reduction first, then
            # retract only the exact publication identities for this job.
            self._rollback_summary_job(job_id)
            raise
        if not success:
            self._rollback_summary_job(job_id)
        return success

    @profiled('transfer.submit_load')
    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        self._record_transfer_start(job_id)
        return super().submit_load(job_id, src_spec, dst_spec)

    @profiled('transfer.splice', cuda=True)
    def splice_gpu_blocks(
        self,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        *,
        splice_id: int | None = None,
    ) -> None:
        """Step 1.5: copy already-GPU-resident block content from
        `src_block_ids` to `dst_block_ids`, across every attention layer --
        moves a speculatively-prefetched block into a request's officially
        allocated one, avoiding a redundant CPU-tier reload for content
        that's already warm. Reuses the same `layer.kv_cache` tensor access
        validated in Step 1.2. Unlike vllm's own `copy_kv_blocks`
        (`CopyBlocksOp`), which is typed only for host<->device transfers,
        this is a same-tensor, same-device index copy -- ordinary torch
        indexing, no platform-specific op needed.

        Deliberately runs on the current (default) stream, not a side
        stream: this must complete, in program order, before the
        subsequent forward pass reads these blocks, and same-stream
        execution guarantees that ordering for free. Step 1.2's summary
        computation uses a side stream because its timing is not
        correctness-critical (a slightly-stale summary is harmless); a
        stale/incomplete splice would serve corrupted KV data, so ordinary
        in-order execution is the safer choice here, not an oversight.
        """
        if not src_block_ids or not dst_block_ids:
            return
        assert len(src_block_ids) == len(dst_block_ids)
        issued_at_ns = time.monotonic_ns()
        copied_on_cuda = False
        copied_layer_count = 0
        for layer in self._attention_layers.values():
            kv_cache = getattr(layer, "kv_cache", None)
            if kv_cache is None or kv_cache.numel() == 0:
                continue
            src_idx = torch.tensor(
                src_block_ids, device=kv_cache.device, dtype=torch.long
            )
            dst_idx = torch.tensor(
                dst_block_ids, device=kv_cache.device, dtype=torch.long
            )
            kv_cache.index_copy_(0, dst_idx, kv_cache.index_select(0, src_idx))
            copied_on_cuda = copied_on_cuda or kv_cache.is_cuda
            copied_layer_count += 1
        if splice_id is not None:
            events = getattr(self, "_splice_events", None)
            if events is None:
                events = self._splice_events = {}
            event = None
            if copied_on_cuda:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
            events[splice_id] = event
            evidence = getattr(self, "_splice_event_evidence", None)
            if evidence is None:
                evidence = self._splice_event_evidence = {}
            evidence[splice_id] = {
                "splice_id": splice_id,
                "issued_at_ns": issued_at_ns,
                "source_block_count": len(src_block_ids),
                "destination_block_count": len(dst_block_ids),
                "copied_layer_count": copied_layer_count,
                "copied_on_cuda": copied_on_cuda,
                "same_default_stream": True,
                "issued_before_base_bind_and_forward": True,
                "ordering_witness": (
                    "same_default_stream_copy_enqueued_before_base_bind_and_forward"
                ),
            }

    def pop_completed_splices(self) -> list[int]:
        """Return splice IDs whose source blocks are safe to release."""
        events = getattr(self, "_splice_events", {})
        completed = [
            splice_id
            for splice_id, event in events.items()
            if event is None or event.query()
        ]
        for splice_id in completed:
            del events[splice_id]
            evidence = getattr(self, "_splice_event_evidence", {}).pop(
                splice_id, {}
            )
            completed_evidence = getattr(
                self, "_completed_splice_evidence", None
            )
            if completed_evidence is None:
                completed_evidence = self._completed_splice_evidence = {}
            completed_evidence[splice_id] = {
                **evidence,
                "completion_observed_at_ns": time.monotonic_ns(),
                "completion_event_satisfied": True,
            }
        return completed

    def pop_completed_splice_evidence(self) -> dict[int, dict[str, object]]:
        evidence = dict(getattr(self, "_completed_splice_evidence", {}))
        getattr(self, "_completed_splice_evidence", {}).clear()
        return evidence
