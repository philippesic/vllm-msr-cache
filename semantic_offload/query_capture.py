# SPDX-License-Identifier: Apache-2.0
"""Live per-step query capture for Step 1.3.

Composes two independently-verified mechanisms (full investigation trail,
including two dead ends, is in .claude/docs/semantic-eviction-issues-log.md
entries #5-#6):

1. Execution ownership: patches `GPUModelRunner.execute_model` (the V2
   runner) to scope a fresh dispatch mode and per-execution context to one
   exact runner instance. Shutdown waits for active executions before
   restoring the class methods.
2. Row-boundary/req_id info: patches `GPUModelRunner.prepare_inputs` (the V2
   runner -- this project's target model defaults to V2, confirmed via
   `vllm_config.use_v2_model_runner`; a future session targeting the legacy
   runner needs a different patch point, see the issues log) at the class
   level. `prepare_inputs` is looked up via plain attribute access on every
   call (never a cached/bound reference), so this patch fires reliably on
   every step, including pure single-token decode steps, since it's
   orchestration code that itself feeds the (possibly compiled/graph-
   replayed) model forward rather than being part of what gets replayed.
3. Query data: the default `TorchDispatchMode` on
   `torch.ops.vllm.unified_attention_with_output`, or the opt-in
   `SEMANTIC_CAPTURE_BACKEND=attention` probe implementation hook. Both fire
   on Python-executed attention. This only fires on steps that include prefill or mixed
   prefill-decode batches -- vLLM's default `cudagraph_mode` fully captures
   pure decode-only batches with zero Python touchpoints during replay
   (verified empirically). Relevance updates are therefore opportunistic,
   not guaranteed every step; the manager's EMA is designed to tolerate that.

Zero vLLM source modifications. The attention backend patches only the probe's
implementation instance; it adds no interception to unrelated tensor operations.
"""

from semantic_offload.profiling import PROFILE, profiled
import os
import time
from collections.abc import Callable
from contextvars import ContextVar
from contextlib import nullcontext
from threading import Condition, RLock, get_ident

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils.torch_utils import _resolve_layer_name

from semantic_offload._debug import debug_print, ENABLED as _DEBUG


class _BatchLayout:
    __slots__ = ("boundaries", "num_tokens", "req_ids")

    def __init__(self, req_ids: list[str], boundaries: list[tuple[int, int]]):
        self.req_ids = req_ids
        self.boundaries = boundaries
        self.num_tokens = boundaries[-1][1] if boundaries else 0


_current_handle: ContextVar["QueryCaptureHandle | None"] = ContextVar(
    "semantic_query_capture_handle", default=None
)
_current_layout: ContextVar[_BatchLayout | None] = ContextVar(
    "semantic_query_capture_layout", default=None
)
_runner_patch: tuple[type, Callable, Callable, Callable, Callable] | None = None
_active_handle: "QueryCaptureHandle | None" = None
_lifecycle_lock = RLock()
_lifecycle_condition = Condition(_lifecycle_lock)


class QueryCaptureHandle:
    """Owns the single supported runner installation in this process."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        mode_factory: Callable[[], TorchDispatchMode],
        on_forward: Callable[[list[str], int], None] | None = None,
        on_complete: Callable[[object, object], None] | None = None,
        emit_nvtx_ranges: bool = False,
    ):
        self.vllm_config = vllm_config
        self._mode_factory = mode_factory
        self._runner = None
        self._active_executions = 0
        self._execution_threads: dict[int, int] = {}
        self._closing = False
        self._closed = False
        self._on_forward = on_forward
        self._on_complete = on_complete
        self._emit_nvtx_ranges = emit_nvtx_ranges
        self._capture_enabled = mode_factory is not None
        self._attention_patch = None

    def _begin_execution(self, runner) -> None:
        with _lifecycle_condition:
            if self._closing or self._closed:
                raise RuntimeError("query capture is closing")
            if self._runner is None:
                self._runner = runner
            elif self._runner is not runner:
                raise RuntimeError("query capture received an unexpected runner")
            self._active_executions += 1
            thread_id = get_ident()
            self._execution_threads[thread_id] = (
                self._execution_threads.get(thread_id, 0) + 1
            )

    def _end_execution(self) -> None:
        with _lifecycle_condition:
            self._active_executions -= 1
            thread_id = get_ident()
            remaining = self._execution_threads[thread_id] - 1
            if remaining:
                self._execution_threads[thread_id] = remaining
            else:
                del self._execution_threads[thread_id]
            _lifecycle_condition.notify_all()

    def close(self) -> None:
        global _active_handle

        with _lifecycle_condition:
            if self._closed:
                return
            if _active_handle is not self:
                raise RuntimeError("query capture handle no longer owns installation")
            if get_ident() in self._execution_threads:
                raise RuntimeError(
                    "cannot close query capture from an active execution"
                )
            self._closing = True
            while self._active_executions:
                _lifecycle_condition.wait()
            try:
                _unpatch_runner(self)
            finally:
                _active_handle = None
                self._runner = None
                self._closed = True
                _lifecycle_condition.notify_all()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _nvtx_execution_label(request_ids: list[str]) -> str:
    """Build a bounded NVTX label for one model-runner execution batch."""
    ids = [str(request_id) for request_id in request_ids]
    shown = ",".join(ids[:64])
    if len(ids) > 64:
        shown += ",..."
    return f"semantic_execute[n={len(ids)}]:{shown}"


def on_query_disabled(handle):
    return not handle._capture_enabled

def _patch_runner(handle: QueryCaptureHandle) -> None:
    global _runner_patch

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if _runner_patch is not None or hasattr(
        GPUModelRunner, "_semantic_query_capture_owner"
    ):
        raise RuntimeError("semantic query capture patch is already installed")
    original_prepare = GPUModelRunner.prepare_inputs
    original_execute = GPUModelRunner.execute_model

    @profiled('capture.prepare_layout')
    def prepare_wrapper(self_runner, scheduler_output, batch_desc):
        result = original_prepare(self_runner, scheduler_output, batch_desc)
        if on_query_disabled(handle) or _current_handle.get() is not handle or self_runner is not handle._runner:
            return result
        req_ids = list(result.req_ids)
        token_counts = [int(c) for c in result.num_scheduled_tokens]
        boundaries = []
        start = 0
        for count in token_counts:
            boundaries.append((start, start + count))
            start += count
        _current_layout.set(_BatchLayout(req_ids=req_ids, boundaries=boundaries))
        return result

    @profiled('model.execute_host', cuda=True)
    def execute_wrapper(self_runner, *args, **kwargs):
        if getattr(self_runner, "vllm_config", None) is not handle.vllm_config:
            return original_execute(self_runner, *args, **kwargs)
        handle._begin_execution(self_runner)
        handle_token = _current_handle.set(handle)
        layout_token = _current_layout.set(None)
        nvtx_range_open = False
        try:
            scheduler_output = args[0] if args else kwargs.get("scheduler_output")
            if (
                handle._emit_nvtx_ranges
                and torch.cuda.is_available()
                and scheduler_output is not None
                and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0
                and not kwargs.get("dummy_run", False)
            ):
                request_ids = list(
                    getattr(scheduler_output, "num_scheduled_tokens", {})
                )
                torch.cuda.nvtx.range_push(_nvtx_execution_label(request_ids))
                nvtx_range_open = True
            if (
                handle._on_forward is not None
                and scheduler_output is not None
                and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0
                and not kwargs.get("dummy_run", False)
            ):
                handle._on_forward(
                    list(getattr(scheduler_output, "num_scheduled_tokens", {})),
                    time.monotonic_ns(),
                )
            with (handle._mode_factory() if handle._mode_factory is not None else nullcontext()):
                result = original_execute(self_runner, *args, **kwargs)
            if handle._on_complete is not None:
                handle._on_complete(self_runner, result)
            return result
        finally:
            if nvtx_range_open:
                torch.cuda.nvtx.range_pop()
            _current_layout.reset(layout_token)
            _current_handle.reset(handle_token)
            handle._end_execution()

    GPUModelRunner.prepare_inputs = prepare_wrapper
    GPUModelRunner.execute_model = execute_wrapper
    GPUModelRunner._semantic_query_capture_owner = handle
    _runner_patch = (
        GPUModelRunner,
        original_prepare,
        prepare_wrapper,
        original_execute,
        execute_wrapper,
    )


def _unpatch_runner(handle: QueryCaptureHandle) -> None:
    global _runner_patch

    conflicts = []
    if handle._attention_patch is not None:
        impl, original, wrapper, had_instance_attr = handle._attention_patch
        if impl.forward is wrapper:
            if had_instance_attr:
                impl.forward = original
            else:
                del impl.forward
        else:
            conflicts.append("attention implementation forward")
        handle._attention_patch = None
    patch = _runner_patch
    if patch is None:
        if conflicts:
            raise RuntimeError("query capture methods changed: " + ", ".join(conflicts))
        return
    runner_cls, original_prepare, prepare_wrapper, original_execute, execute_wrapper = (
        patch
    )
    if runner_cls.prepare_inputs is prepare_wrapper:
        runner_cls.prepare_inputs = original_prepare
    else:
        conflicts.append("prepare_inputs")
    if runner_cls.execute_model is execute_wrapper:
        runner_cls.execute_model = original_execute
    else:
        conflicts.append("execute_model")
    if getattr(runner_cls, "_semantic_query_capture_owner", None) is handle:
        delattr(runner_cls, "_semantic_query_capture_owner")
    _runner_patch = None
    if conflicts:
        raise RuntimeError(
            "GPUModelRunner methods changed while semantic capture owned them: "
            + ", ".join(conflicts)
        )


def _should_sample_step(step_index: int, stride: int) -> bool:
    """True on every `stride`-th eligible query-capture step (stride<=1:
    every step, the historical default). Counted once per real query-
    capture-eligible dispatch event (one per step), not per request within
    it, so concurrent requests sharing a step are throttled together as a
    unit rather than independently -- a coarser cadence relies on the
    manager's own EMA staleness-tolerance (Step 1.4) to carry relevance
    signal forward across skipped steps, not on every step being scored.
    See semantic-eviction-plan.md's TTFT-tax follow-up investigation for
    why this knob exists (leads #1/#3): stack_rebuild/update_relevance's
    per-call cost is bounded but non-trivial (issues log entries #62-65),
    and their aggregate cost scales with how often query-capture fires, not
    just candidate-pool size."""
    return stride <= 1 or step_index % stride == 0


def _validate_cudagraph_mode(vllm_config: VllmConfig) -> None:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation_config, "cudagraph_mode", None)
    mixed_mode = getattr(mode, "mixed_mode", None)
    if callable(mixed_mode):
        mode = mixed_mode()
    if getattr(mode, "name", str(mode)) == "FULL":
        raise ValueError(
            "semantic query capture requires prefill and mixed batches outside "
            "FULL CUDA graphs; use NONE, PIECEWISE, FULL_DECODE_ONLY, or "
            "FULL_AND_PIECEWISE"
        )


def preflight_install(vllm_config: VllmConfig) -> None:
    """Validate capture compatibility without changing process state."""
    if not vllm_config.use_v2_model_runner:
        raise ValueError("semantic query capture requires the V2 GPUModelRunner")
    _validate_cudagraph_mode(vllm_config)

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    with _lifecycle_condition:
        if (
            _active_handle is not None
            or _runner_patch is not None
            or hasattr(GPUModelRunner, "_semantic_query_capture_owner")
        ):
            raise RuntimeError("query capture is already installed in this process")


def _select_last_query_rows(query: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Copy selected rows without constructing a CPU-to-device index tensor.

    Basic slices retain the original device and concatenate into owned storage,
    including the one-row case. The caller supplies nonempty, valid row indices.
    """
    return torch.cat([query[index : index + 1] for index in indices], dim=0)


def install(
    vllm_config: VllmConfig,
    probe_layer_name: str | None,
    on_query: Callable[[list[str], torch.Tensor], None] | None,
    num_queries_per_kv: int = 1,
    capture_stride: int = 1,
    on_forward: Callable[[list[str], int], None] | None = None,
    on_complete: Callable[[object, object], None] | None = None,
    emit_nvtx_ranges: bool = False,
    capture_backend: str | None = None,
    capture_steps: str | None = None,
) -> QueryCaptureHandle:
    """Install both patches. `on_query(req_ids, query_reprs)` fires ONCE per
    step whenever live query data is captured for the probe layer, covering
    every request scheduled in that step -- `query_reprs` is a
    `[len(req_ids), num_kv_heads, head_dim]` tensor: each request's last
    real token's query, grouped by which KV head each query
    head attends against (GQA group, contiguous per
    triton_unified_attention.py's `query_offset_1 = kv_head_idx *
    num_queries_per_kv + ...` convention -- verified against this backend's
    kernel, not assumed) and averaged only within the group, not across all
    heads. See issues log entry #8 for why last-token (vs. whole-step mean)
    was adopted, and entry #9 for why per-KV-head (vs. fully-pooled) was
    adopted on top of that. Returns an owning handle; caller must keep it alive
    and call `close()` during worker shutdown.

    Only one runner is supported per process. A concurrent installation or a
    second runner using the owning config fails before publishing capture state.

    `capture_stride`: only every `capture_stride`-th eligible step actually
    fires `on_query` (default 1: unchanged, every step) -- see
    `_should_sample_step`.

    `capture_backend`: "dispatch" (default) or "attention", alternatively selected
    by SEMANTIC_CAPTURE_BACKEND. The latter hooks the configured probe's backend
    implementation, called dynamically inside vLLM's opaque attention custom op.
    Full graph replay still has no Python callback. It requires the probe's
    implementation to expose the installed vLLM `forward(layer, query, ...)`
    contract; no fallback silently changes the requested backend.

    `capture_steps`: "all" (default) or "multi_token", alternatively selected
    by SEMANTIC_CAPTURE_STEPS. Multi-token capture keeps a batch only if some
    request has more than one scheduled token, then captures every nonempty
    request in that batch, including decode requests. It skips pure single-token
    batches, including single-token prefills. This reduces semantic observations
    and may leave relevance stale; it does not identify actual prefill status.
    Skipped batches still advance the eligible-step stride counter.
    """
    if capture_stride < 1:
        raise ValueError("capture_stride must be >= 1")
    if on_query is not None and num_queries_per_kv < 1:
        raise ValueError("num_queries_per_kv must be >= 1")
    global _active_handle
    capture_backend = capture_backend or os.environ.get(
        "SEMANTIC_CAPTURE_BACKEND", "dispatch"
    )
    if capture_backend not in ("dispatch", "attention"):
        raise ValueError("capture_backend must be dispatch or attention")
    if capture_steps is None:
        capture_steps = os.environ.get("SEMANTIC_CAPTURE_STEPS", "all")
    if capture_steps not in ("all", "multi_token"):
        raise ValueError("capture_steps must be all or multi_token")

    state: dict = {"step_index": 0}

    @profiled('capture.extract', cuda=True)
    def capture_query(layout: _BatchLayout, query: torch.Tensor) -> None:
        step_index = state["step_index"]
        state["step_index"] = step_index + 1
        if not _should_sample_step(step_index, capture_stride):
            return
        if capture_steps == "multi_token" and not any(
            end - start > 1 for start, end in layout.boundaries
        ):
            return
        owner = getattr(on_query, '__self__', None)
        if (getattr(owner, '_scoring_schedule', 'continuous') == 'continuous'
                and hasattr(owner, 'durable_summaries') and not owner.durable_summaries):
            return  # Preserve eligible-step stride, but continuous empty banks cannot score.
        req_ids = []
        last_indices = []
        for req_id, (start, end) in zip(layout.req_ids, layout.boundaries):
            if end > start:
                req_ids.append(req_id)
                last_indices.append(end - 1)
        if not req_ids:
            return
        with PROFILE.scope("capture.group_query", cuda=query.is_cuda):
            with PROFILE.scope("capture.query_select", cuda=query.is_cuda):
                last_q = _select_last_query_rows(query, last_indices)
            n_reqs, num_query_heads, head_dim = last_q.shape
            if num_query_heads % num_queries_per_kv:
                raise ValueError("query heads must be divisible by num_queries_per_kv")
            num_kv_heads = num_query_heads // num_queries_per_kv
            with PROFILE.scope("capture.head_group_reduce", cuda=query.is_cuda):
                grouped = last_q.view(
                    n_reqs, num_kv_heads, num_queries_per_kv, head_dim
                ).mean(dim=2)
        if on_query is not None:
            on_query(req_ids, grouped)

    attention_ops = {}

    def capture_prepared_query(query):
        layout = _current_layout.get()
        # Consume before any callback, including callbacks which reenter.
        _current_layout.set(None)
        try:
            if (
                layout is not None
                and query is not None
                and query.shape[0] >= layout.num_tokens
            ):
                capture_query(layout, query)
        finally:
            _current_layout.set(None)

    class ProbeMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            sample = False
            if PROFILE.directory:
                PROFILE.count("capture.dispatch_calls")
                sequence = getattr(self, "_profile_dispatch_sequence", 0) + 1
                self._profile_dispatch_sequence = sequence
                sample = sequence % 64 == 1
            if sample:
                profile_wall, profile_cpu = time.perf_counter_ns(), time.thread_time_ns()
            kwargs = kwargs or {}
            match = attention_ops.get(func)
            if match is None:
                match = "unified_attention_with_output" in str(func)
                attention_ops[func] = match
            if match:
                query = args[0] if len(args) > 0 else kwargs.get("query")
                layer_name_arg = args[4] if len(args) > 4 else kwargs.get("layer_name")
                resolved_name = (
                    _resolve_layer_name(layer_name_arg)
                    if layer_name_arg is not None
                    else None
                )
                if resolved_name == probe_layer_name:
                    layout = _current_layout.get()
                    # Consume before extraction/callback: reentry must not
                    # capture the same prepared batch a second time.
                    _current_layout.set(None)
                    try:
                        if _DEBUG:
                            debug_print(
                                "SEMANTIC_QUERY_CAPTURE_DEBUG "
                                "query_shape0="
                                f"{query.shape[0] if query is not None else None} "
                                "layout_num_tokens="
                                f"{layout.num_tokens if layout else None} "
                                f"layout_req_ids={layout.req_ids if layout else None}"
                            )
                        # CUDA-graph replay can pad the query tensor, but real
                        # tokens occupy its prefix. Consume each prepare-inputs
                        # layout once at the configured probe layer.
                        if (
                            layout is not None
                            and query is not None
                            and query.shape[0] >= layout.num_tokens
                        ):
                            capture_query(layout, query)
                    finally:
                        _current_layout.set(None)
            if sample:
                PROFILE.observe("capture.dispatch_intercept_sample.host_wall", time.perf_counter_ns() - profile_wall)
                PROFILE.observe("capture.dispatch_intercept_sample.thread_cpu", time.thread_time_ns() - profile_cpu)
            return func(*args, **kwargs)

    with _lifecycle_condition:
        preflight_install(vllm_config)
        handle = QueryCaptureHandle(
            vllm_config,
            ProbeMode if on_query is not None else None,
            on_forward=on_forward,
            on_complete=on_complete,
            emit_nvtx_ranges=emit_nvtx_ranges,
        )
        try:
            if capture_backend == "attention" and on_query is not None:
                context = getattr(vllm_config.compilation_config,
                                  "static_forward_context", {})
                attention = context.get(probe_layer_name)
                impl = getattr(attention, "impl", None)
                if impl is None or not callable(getattr(impl, "forward", None)):
                    raise ValueError("attention capture requires a probe implementation")
                original = impl.forward
                had_instance_attr = "forward" in getattr(impl, "__dict__", {})

                def attention_wrapper(layer, query, *args, **kwargs):
                    if layer is attention and _current_handle.get() is handle:
                        capture_prepared_query(query)
                    return original(layer, query, *args, **kwargs)

                impl.forward = attention_wrapper
                handle._attention_patch = (
                    impl, original, attention_wrapper, had_instance_attr
                )
                handle._mode_factory = None
            _patch_runner(handle)
        except BaseException:
            _unpatch_runner(handle)
            raise
        _active_handle = handle
        return handle
