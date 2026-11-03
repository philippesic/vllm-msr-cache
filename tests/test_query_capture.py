# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for query-capture throttling and hook lifecycle ownership."""

from threading import Event, Thread
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from semantic_offload import query_capture
from semantic_offload.query_capture import (
    _nvtx_execution_label,
    _should_sample_step,
    install,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("indices", [[1], [1, 2, 5], [0, 1, 2]])
def test_query_row_selection_matches_indexing_without_index_upload(
    monkeypatch, dtype, strided, indices
):
    query = torch.arange(8 * 8 * 4, dtype=torch.float32).reshape(8, 8, 4).to(dtype)
    if strided:
        query = query.transpose(1, 2)
    expected = query[torch.tensor(indices)].clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("row selection must not construct an index tensor")

    monkeypatch.setattr(torch, "tensor", forbidden)
    selected = query_capture._select_last_query_rows(query, indices)
    assert torch.equal(selected, expected)
    assert selected.dtype == query.dtype and selected.device == query.device
    # Ownership matters even when only one row is selected: model buffers may
    # be overwritten after the capture callback returns.
    selected.zero_()
    assert torch.equal(query[indices[0]], expected[0])


@pytest.mark.parametrize("group_size", [1, 4])
def test_capture_preserves_mixed_rows_gqa_padding_and_consumes_before_reentry(
    monkeypatch, group_size
):
    config = SimpleNamespace(use_v2_model_runner=True)
    callbacks = []

    class FakeAttention:
        def __str__(self):
            return "vllm.unified_attention_with_output.default"

        def __call__(self, *args, **kwargs):
            return "attention-result"

    query = torch.arange(8 * 8 * 4, dtype=torch.float32).reshape(8, 8, 4)
    # Rows6..7 are graph padding; a zero-token request owns no query row.
    query[6:] = -1000
    expected = query[[1, 2, 5]].reshape(3, 8 // group_size, group_size, 4).mean(2)
    args = (query, None, None, None, "layer")
    mode = None

    def callback(req_ids, grouped):
        assert query_capture._current_layout.get() is None
        callbacks.append((req_ids, grouped.clone()))
        assert mode.__torch_dispatch__(FakeAttention(), (), args) == "attention-result"

    handle = install(config, "layer", callback, num_queries_per_kv=group_size)
    try:
        mode = handle._mode_factory()
        query_capture._current_layout.set(query_capture._BatchLayout(
            ["prepared-b", "empty", "prepared-a", "prepared-c"],
            [(0, 2), (2, 2), (2, 3), (3, 6)],
        ))
        assert mode.__torch_dispatch__(FakeAttention(), (), args) == "attention-result"
        assert len(callbacks) == 1
        assert callbacks[0][0] == ["prepared-b", "prepared-a", "prepared-c"]
        assert torch.equal(callbacks[0][1], expected)
        query_capture._current_layout.set(query_capture._BatchLayout(["empty"], [(0, 0)]))
        mode.__torch_dispatch__(FakeAttention(), (), args)
        assert len(callbacks) == 1
    finally:
        handle.close()


def test_nvtx_execution_label_is_bounded_and_reports_batch_size():
    label = _nvtx_execution_label([f"req-{index}" for index in range(66)])

    assert label.startswith("semantic_execute[n=66]:req-0,req-1")
    assert label.endswith("req-63,...")
    assert "req-64" not in label
    assert "req-65" not in label


def test_nvtx_execution_label_handles_empty_batch():
    assert _nvtx_execution_label([]) == "semantic_execute[n=0]:"


def _worker_config(mixed_cudagraph_mode: str, *, use_v2_model_runner: bool = True):
    class CUDAGraphMode:
        def mixed_mode(self):
            return SimpleNamespace(name=mixed_cudagraph_mode)

    attention = SimpleNamespace(num_kv_heads=1, head_size=1)
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode(),
            static_forward_context={"layer": attention},
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            world_size=1,
        ),
        model_config=SimpleNamespace(
            get_total_num_kv_heads=lambda: 1,
            get_num_attention_heads=lambda parallel_config: 1,
            use_mla=False,
        ),
        cache_config=SimpleNamespace(block_size=16),
    )


def test_stride_one_samples_every_step():
    """Default behavior (capture_stride=1) must be unchanged from before
    this knob existed: every eligible step fires."""
    assert all(_should_sample_step(i, 1) for i in range(10))


def test_stride_zero_or_negative_treated_as_no_throttle():
    """Defensive: a misconfigured stride (<=1) must not silently disable
    capture entirely -- falls back to "every step", the safe default."""
    assert all(_should_sample_step(i, 0) for i in range(5))
    assert all(_should_sample_step(i, -3) for i in range(5))


def test_stride_n_samples_every_nth_step_only():
    stride = 4
    sampled = [i for i in range(20) if _should_sample_step(i, stride)]
    assert sampled == [0, 4, 8, 12, 16]


def test_install_close_and_reinstall_own_prepare_inputs_patch(monkeypatch):
    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        if scheduler_output == "raise":
            raise RuntimeError("prepare failed")
        return SimpleNamespace(req_ids=["req"], num_scheduled_tokens=[2])

    def prior_execute_model(self_runner, scheduler_output=None):
        return GPUModelRunner.prepare_inputs(self_runner, scheduler_output, None)

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", prior_execute_model)
    config = SimpleNamespace(use_v2_model_runner=True)

    first = install(config, "layer", lambda req_ids, query: None)
    assert GPUModelRunner.prepare_inputs is not prior_prepare_inputs
    assert GPUModelRunner.execute_model is not prior_execute_model

    with pytest.raises(RuntimeError, match="already installed"):
        install(config, "layer", lambda req_ids, query: None)

    owner = SimpleNamespace(vllm_config=config)
    result = GPUModelRunner.execute_model(owner, None)
    assert result.req_ids == ["req"]
    assert query_capture._current_layout.get() is None

    with pytest.raises(RuntimeError, match="unexpected runner"):
        GPUModelRunner.execute_model(SimpleNamespace(vllm_config=config), None)

    other = SimpleNamespace(vllm_config=SimpleNamespace())
    assert GPUModelRunner.execute_model(other, None).req_ids == ["req"]
    assert query_capture._current_layout.get() is None

    with pytest.raises(RuntimeError, match="prepare failed"):
        GPUModelRunner.execute_model(owner, "raise")
    assert query_capture._current_layout.get() is None
    assert query_capture._current_handle.get() is None

    first.close()
    assert GPUModelRunner.prepare_inputs is prior_prepare_inputs
    first.close()

    second = install(config, "other-layer", lambda req_ids, query: None)
    second.close()
    assert GPUModelRunner.prepare_inputs is prior_prepare_inputs


def test_forward_callback_binds_scheduled_requests_and_skips_empty_runs(monkeypatch):
    calls = []

    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    def prior_execute_model(self_runner, scheduler_output=None, **kwargs):
        calls.append("execute")

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", prior_execute_model)
    config = SimpleNamespace(use_v2_model_runner=True)
    forwards = []
    handle = install(
        config,
        None,
        None,
        on_forward=lambda req_ids, timestamp_ns: forwards.append(
            (req_ids, timestamp_ns)
        ),
    )
    owner = SimpleNamespace(vllm_config=config)
    try:
        GPUModelRunner.execute_model(
            owner,
            SimpleNamespace(
                total_num_scheduled_tokens=3,
                num_scheduled_tokens={"req-a": 2, "req-b": 1},
            ),
        )
        GPUModelRunner.execute_model(
            owner,
            SimpleNamespace(total_num_scheduled_tokens=0, num_scheduled_tokens={}),
        )
        GPUModelRunner.execute_model(
            owner,
            SimpleNamespace(
                total_num_scheduled_tokens=3,
                num_scheduled_tokens={"req-c": 3},
            ),
            dummy_run=True,
        )
    finally:
        handle.close()

    assert calls == ["execute", "execute", "execute"]
    assert [item[0] for item in forwards] == [["req-a", "req-b"]]
    assert isinstance(forwards[0][1], int)


def test_complete_callback_receives_successful_output_and_skips_failure(monkeypatch):
    output = object()

    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    def prior_execute_model(self_runner, scheduler_output=None, **kwargs):
        if scheduler_output == "raise":
            raise RuntimeError("model failed")
        return output

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", prior_execute_model)
    config = SimpleNamespace(use_v2_model_runner=True)
    completed = []
    handle = install(
        config,
        None,
        None,
        on_complete=lambda runner, result: completed.append((runner, result)),
    )
    owner = SimpleNamespace(vllm_config=config)
    try:
        assert GPUModelRunner.execute_model(owner, None) is output
        with pytest.raises(RuntimeError, match="model failed"):
            GPUModelRunner.execute_model(owner, "raise")
    finally:
        handle.close()

    assert completed == [(owner, output)]


def test_worker_shutdown_closes_query_capture_before_base_shutdown(monkeypatch):
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

    from semantic_offload.worker import SemanticOffloadingWorker

    calls = []

    class Handle:
        def close(self):
            calls.append("capture")

    worker = object.__new__(SemanticOffloadingWorker)
    worker._query_capture_mode = Handle()
    monkeypatch.setattr(
        CPUOffloadingWorker, "shutdown", lambda self: calls.append("base")
    )

    SemanticOffloadingWorker.shutdown(worker)

    assert calls == ["capture", "base"]
    assert worker._query_capture_mode is None


def test_worker_shutdown_preserves_resources_when_capture_close_fails(monkeypatch):
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

    from semantic_offload.worker import SemanticOffloadingWorker

    calls = []

    class Handle:
        def close(self):
            calls.append("capture")
            raise RuntimeError("capture still active")

    handle = Handle()
    worker = object.__new__(SemanticOffloadingWorker)
    worker._query_capture_mode = handle
    monkeypatch.setattr(
        CPUOffloadingWorker, "shutdown", lambda self: calls.append("base")
    )

    with pytest.raises(RuntimeError, match="still active"):
        SemanticOffloadingWorker.shutdown(worker)

    assert calls == ["capture"]
    assert worker._query_capture_mode is handle


def test_worker_rejects_full_cudagraph_before_base_allocation(monkeypatch):
    from semantic_offload import worker as worker_module

    base_allocations = []
    monkeypatch.setattr(
        worker_module,
        "init_cpu_offloading_worker_base",
        lambda *args, **kwargs: base_allocations.append(kwargs),
    )

    with pytest.raises(ValueError, match="outside FULL CUDA graphs"):
        worker_module.SemanticOffloadingWorker(
            kv_caches=None,
            blocks_per_chunk=1,
            num_cpu_blocks=1,
            vllm_config=_worker_config("FULL"),
        )

    assert base_allocations == []


def test_instrumentation_only_worker_skips_query_capture_on_v1(monkeypatch):
    from semantic_offload import worker as worker_module

    base_allocations = []
    monkeypatch.setattr(
        worker_module,
        "init_cpu_offloading_worker_base",
        lambda *args, **kwargs: base_allocations.append(kwargs),
    )
    monkeypatch.setattr(
        worker_module,
        "preflight_query_capture",
        lambda *_args, **_kwargs: pytest.fail("V1 must not preflight query capture"),
    )
    monkeypatch.setattr(
        worker_module,
        "install_query_capture",
        lambda *_args, **_kwargs: pytest.fail("V1 must not install query capture"),
    )

    worker = worker_module.SemanticOffloadingWorker(
        kv_caches=None,
        blocks_per_chunk=1,
        num_cpu_blocks=1,
        vllm_config=_worker_config("PIECEWISE", use_v2_model_runner=False),
        probe_layer=None,
        preservation_events=True,
    )

    assert worker._query_capture_mode is None
    assert base_allocations


def test_worker_rejects_duplicate_owner_before_base_allocation(monkeypatch):
    from semantic_offload import worker as worker_module

    owner_config = SimpleNamespace(use_v2_model_runner=True)
    owner = install(owner_config, "layer", lambda req_ids, query: None)
    base_allocations = []
    monkeypatch.setattr(
        worker_module,
        "init_cpu_offloading_worker_base",
        lambda *args, **kwargs: base_allocations.append(kwargs),
    )
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            worker_module.SemanticOffloadingWorker(
                kv_caches=None,
                blocks_per_chunk=1,
                num_cpu_blocks=1,
                vllm_config=_worker_config("PIECEWISE"),
            )
    finally:
        owner.close()

    assert base_allocations == []


def test_callback_exception_consumes_layout_and_cannot_replay(monkeypatch):
    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=["req"], num_scheduled_tokens=[2])

    class FakeAttention:
        def __str__(self):
            return "vllm.unified_attention_with_output.default"

        def __call__(self, *args, **kwargs):
            return "attention-result"

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    config = SimpleNamespace(use_v2_model_runner=True)
    callbacks = []

    def fail_callback(req_ids, query):
        callbacks.append(req_ids)
        raise RuntimeError("callback failed")

    handle = install(config, "layer", fail_callback)
    try:
        query_capture._current_layout.set(query_capture._BatchLayout(["req"], [(0, 2)]))
        args = (torch.ones(2, 2, 1), None, None, None, "layer")
        mode = handle._mode_factory()

        other_args = (torch.ones(2, 2, 1), None, None, None, "other-layer")
        assert (
            mode.__torch_dispatch__(FakeAttention(), (), other_args)
            == "attention-result"
        )
        assert query_capture._current_layout.get() is not None

        with pytest.raises(RuntimeError, match="callback failed"):
            mode.__torch_dispatch__(FakeAttention(), (), args)

        assert query_capture._current_layout.get() is None
        assert mode.__torch_dispatch__(FakeAttention(), (), args) == "attention-result"
        assert callbacks == [["req"]]
    finally:
        handle.close()


def test_mode_entry_failure_clears_execution_context(monkeypatch):
    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    def prior_execute_model(self_runner):
        return None

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", prior_execute_model)
    original_enter = TorchDispatchMode.__enter__

    def fail_enter(self):
        raise RuntimeError("mode enter failed")

    monkeypatch.setattr(TorchDispatchMode, "__enter__", fail_enter)
    config = SimpleNamespace(use_v2_model_runner=True)
    handle = install(config, "layer", lambda req_ids, query: None)
    with pytest.raises(RuntimeError, match="mode enter failed"):
        GPUModelRunner.execute_model(SimpleNamespace(vllm_config=config))

    assert query_capture._current_layout.get() is None
    assert query_capture._current_handle.get() is None
    handle.close()

    monkeypatch.setattr(TorchDispatchMode, "__enter__", original_enter)
    handle = install(config, "layer", lambda req_ids, query: None)
    handle.close()


def test_full_cudagraph_mode_is_rejected_before_patching(monkeypatch):
    class FullMode:
        name = "FULL"

        def mixed_mode(self):
            return self

    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    config = SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(cudagraph_mode=FullMode()),
    )

    with pytest.raises(ValueError, match="outside FULL CUDA graphs"):
        install(config, "layer", lambda req_ids, query: None)

    assert GPUModelRunner.prepare_inputs is prior_prepare_inputs


def test_sequential_installations_do_not_reuse_callbacks(monkeypatch):
    class FakeAttention:
        def __str__(self):
            return "vllm.unified_attention_with_output.default"

        def __call__(self, *args, **kwargs):
            return None

    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    config = SimpleNamespace(use_v2_model_runner=True)
    callbacks = []
    args = (torch.ones(1, 1, 1), None, None, None, "layer")

    first = install(
        config, "layer", lambda req_ids, query: callbacks.append(("first", req_ids))
    )
    query_capture._current_layout.set(query_capture._BatchLayout(["a"], [(0, 1)]))
    first._mode_factory().__torch_dispatch__(FakeAttention(), (), args)
    first.close()

    second = install(
        config, "layer", lambda req_ids, query: callbacks.append(("second", req_ids))
    )
    query_capture._current_layout.set(query_capture._BatchLayout(["b"], [(0, 1)]))
    second._mode_factory().__torch_dispatch__(FakeAttention(), (), args)
    second.close()

    assert callbacks == [("first", ["a"]), ("second", ["b"])]


def test_external_patch_conflict_is_not_overwritten(monkeypatch):
    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    def external_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=["external"], num_scheduled_tokens=[1])

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    config = SimpleNamespace(use_v2_model_runner=True)
    handle = install(config, "layer", lambda req_ids, query: None)
    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", external_prepare_inputs)
    with pytest.raises(RuntimeError, match="methods changed"):
        handle.close()

    assert GPUModelRunner.prepare_inputs is external_prepare_inputs
    replacement = install(config, "layer", lambda req_ids, query: None)
    replacement.close()
    assert GPUModelRunner.prepare_inputs is external_prepare_inputs


def test_close_waits_for_execution_on_another_thread(monkeypatch):
    execution_started = Event()
    release_execution = Event()
    errors = []

    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    def prior_execute_model(self_runner):
        execution_started.set()
        if not release_execution.wait(timeout=2):
            raise RuntimeError("test execution was not released")

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", prior_execute_model)
    config = SimpleNamespace(use_v2_model_runner=True)
    handle = install(config, "layer", lambda req_ids, query: None)
    owner = SimpleNamespace(vllm_config=config)

    def execute():
        try:
            GPUModelRunner.execute_model(owner)
        except BaseException as exc:  # noqa: BLE001 - surface failures from the thread
            errors.append(exc)

    def close():
        try:
            handle.close()
        except BaseException as exc:  # noqa: BLE001 - surface failures from the thread
            errors.append(exc)

    execute_thread = Thread(target=execute)
    execute_thread.start()
    assert execution_started.wait(timeout=2)
    close_thread = Thread(target=close)
    close_thread.start()
    with query_capture._lifecycle_condition:
        assert handle._closing
        assert handle._active_executions == 1
    assert close_thread.is_alive()

    release_execution.set()
    execute_thread.join(timeout=2)
    close_thread.join(timeout=2)

    assert not execute_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []
    assert GPUModelRunner.prepare_inputs is prior_prepare_inputs
    assert GPUModelRunner.execute_model is prior_execute_model


def test_close_from_active_execution_fails_without_deadlock(monkeypatch):
    handle = None

    def prior_prepare_inputs(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=[], num_scheduled_tokens=[])

    def prior_execute_model(self_runner):
        handle.close()

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prior_prepare_inputs)
    monkeypatch.setattr(GPUModelRunner, "execute_model", prior_execute_model)
    config = SimpleNamespace(use_v2_model_runner=True)
    handle = install(config, "layer", lambda req_ids, query: None)

    with pytest.raises(RuntimeError, match="active execution"):
        GPUModelRunner.execute_model(SimpleNamespace(vllm_config=config))

    assert handle._active_executions == 0
    handle.close()
    assert GPUModelRunner.execute_model is prior_execute_model


@pytest.mark.parametrize("group_size", [1, 4])
def test_attention_backend_captures_owned_probe_once_without_dispatch(monkeypatch, group_size):
    calls = []

    class Impl:
        def forward(self, layer, query, *args, **kwargs):
            calls.append((layer, query))
            return "attention-result"

    impl = Impl()
    layer = SimpleNamespace(impl=impl)
    other = SimpleNamespace(impl=impl)
    original = impl.forward
    config = SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}),
    )
    query = torch.arange(8 * 8 * 4, dtype=torch.float32).reshape(8, 8, 4)
    expected = query[[1, 2, 5]].reshape(3, 8 // group_size, group_size, 4).mean(2)
    callbacks = []

    def callback(ids, grouped):
        callbacks.append((ids, grouped.clone()))
        assert query_capture._current_layout.get() is None
        impl.forward(layer, query)  # Reentry cannot reuse a consumed layout.

    def prepare(self_runner, scheduler_output, batch_desc):
        return SimpleNamespace(req_ids=["a", "empty", "b", "c"],
                               num_scheduled_tokens=[2, 0, 1, 3])

    def execute(self_runner):
        GPUModelRunner.prepare_inputs(self_runner, None, None)
        impl.forward(other, query)
        assert query_capture._current_layout.get() is not None
        impl.forward(layer, query)
        impl.forward(layer, query)
        return "execute-result"

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prepare)
    monkeypatch.setattr(GPUModelRunner, "execute_model", execute)
    handle = install(config, "layer", callback, num_queries_per_kv=group_size,
                     capture_backend="attention")
    try:
        assert handle._mode_factory is None
        assert not query_capture.on_query_disabled(handle)
        # Calls outside the owning model execution do not consume layout.
        query_capture._current_layout.set(query_capture._BatchLayout(["outside"], [(0, 1)]))
        impl.forward(layer, query)
        assert query_capture._current_layout.get() is not None
        query_capture._current_layout.set(None)
        assert GPUModelRunner.execute_model(SimpleNamespace(vllm_config=config)) == "execute-result"
        assert len(callbacks) == 1
        assert callbacks[0][0] == ["a", "b", "c"]
        assert torch.equal(callbacks[0][1], expected)
    finally:
        handle.close()
    assert "forward" not in vars(impl)
    assert impl.forward == original


def test_attention_backend_stride_and_callback_failure_cleanup(monkeypatch):
    class Impl:
        def forward(self, layer, query):
            return query

    impl = Impl()
    layer = SimpleNamespace(impl=impl)
    config = SimpleNamespace(use_v2_model_runner=True,
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}))
    callbacks = []

    def callback(ids, query):
        callbacks.append(ids)
        raise RuntimeError("callback failed")

    def execute(runner):
        GPUModelRunner.prepare_inputs(runner, None, None)
        return impl.forward(layer, torch.ones(2, 2, 1))

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", lambda *args:
                        SimpleNamespace(req_ids=["req"], num_scheduled_tokens=[2]))
    monkeypatch.setattr(GPUModelRunner, "execute_model", execute)
    handle = install(config, "layer", callback, capture_stride=2, capture_backend="attention")
    runner = SimpleNamespace(vllm_config=config)
    try:
        with pytest.raises(RuntimeError, match="callback failed"):
            GPUModelRunner.execute_model(runner)
        assert query_capture._current_handle.get() is None
        assert query_capture._current_layout.get() is None
        GPUModelRunner.execute_model(runner)  # stride-skipped, no callback
        assert callbacks == [["req"]]
    finally:
        handle.close()


def test_attention_backend_does_not_overwrite_external_patch():
    impl = SimpleNamespace(forward=lambda *args: None)
    original = impl.forward
    config = SimpleNamespace(use_v2_model_runner=True,
        compilation_config=SimpleNamespace(static_forward_context={"layer": SimpleNamespace(impl=impl)}))
    handle = install(config, "layer", lambda *args: None, capture_backend="attention")
    external = lambda *args: "external"
    impl.forward = external
    with pytest.raises(RuntimeError, match="methods changed"):
        handle.close()
    assert impl.forward is external
    # Conflicting close still releases runner hooks and ownership.
    impl.forward = original
    replacement = install(config, "layer", lambda *args: None, capture_backend="attention")
    replacement.close()
    assert impl.forward is original


def test_attention_backend_missing_probe_rolls_back_without_runner_patch():
    config = SimpleNamespace(use_v2_model_runner=True,
        compilation_config=SimpleNamespace(static_forward_context={}))
    original = GPUModelRunner.execute_model
    with pytest.raises(ValueError, match="probe implementation"):
        install(config, "layer", lambda *args: None, capture_backend="attention")
    assert GPUModelRunner.execute_model is original
    assert query_capture._active_handle is None


def test_attention_backend_preserves_outer_dispatch_mode(monkeypatch):
    operations = []

    class OuterMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operations.append(str(func))
            return func(*args, **(kwargs or {}))

    class Impl:
        def forward(self, layer, query):
            return query + 1

    impl = Impl()
    layer = SimpleNamespace(impl=impl)
    config = SimpleNamespace(use_v2_model_runner=True,
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}))
    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", lambda *args:
                        SimpleNamespace(req_ids=["req"], num_scheduled_tokens=[2]))

    def execute(runner):
        GPUModelRunner.prepare_inputs(runner, None, None)
        return impl.forward(layer, torch.ones(2, 2, 1))

    monkeypatch.setattr(GPUModelRunner, "execute_model", execute)
    callbacks = []
    handle = install(config, "layer", lambda ids, query: callbacks.append(ids),
                     capture_backend="attention")
    try:
        with OuterMode():
            result = GPUModelRunner.execute_model(SimpleNamespace(vllm_config=config))
        assert torch.equal(result, torch.full((2, 2, 1), 2.0))
        assert callbacks == [["req"]]
        assert any("aten.add" in op for op in operations)
        assert any("aten.mean" in op for op in operations)
    finally:
        handle.close()


@pytest.mark.parametrize("backend", ["dispatch", "attention"])
@pytest.mark.parametrize("capture_steps,expected", [
    ("all", [["decode"], ["single-prefill"], ["decode", "prefill"]]),
    ("multi_token", [["decode", "prefill"]]),
])
def test_capture_steps_keeps_all_mixed_rows_and_skips_only_single_token_batches(
    monkeypatch, backend, capture_steps, expected
):
    class Impl:
        def forward(self, layer, query):
            return query

    class FakeAttention:
        def __str__(self):
            return "vllm.unified_attention_with_output.default"

        def __call__(self, *args):
            return None

    impl = Impl()
    layer = SimpleNamespace(impl=impl)
    config = SimpleNamespace(use_v2_model_runner=True,
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}))
    callbacks = []
    handle = None
    batches = [(["decode"], [1]), (["single-prefill"], [1]),
               (["empty"], [0]), (["decode", "empty", "prefill"], [1, 0, 3])]

    def prepare(runner, index, batch_desc):
        ids, counts = batches[index]
        return SimpleNamespace(req_ids=ids, num_scheduled_tokens=counts)

    query = torch.arange(4 * 4 * 2, dtype=torch.float32).reshape(4, 4, 2)

    def execute(runner, index):
        GPUModelRunner.prepare_inputs(runner, index, None)
        if backend == "attention":
            impl.forward(layer, query)
        else:
            handle._mode_factory().__torch_dispatch__(
                FakeAttention(), (), (query, None, None, None, "layer"))

    monkeypatch.setattr(GPUModelRunner, "prepare_inputs", prepare)
    monkeypatch.setattr(GPUModelRunner, "execute_model", execute)
    handle = install(config, "layer", lambda ids, grouped: callbacks.append((ids, grouped.clone())),
                     capture_backend=backend, capture_steps=capture_steps)
    try:
        runner = SimpleNamespace(vllm_config=config)
        for index in range(len(batches)):
            GPUModelRunner.execute_model(runner, index)
        assert [ids for ids, grouped in callbacks] == expected
        assert torch.equal(callbacks[-1][1], query[[0, 3]])
    finally:
        handle.close()


def test_multi_token_skips_still_advance_stride(monkeypatch):
    # Index0 is filtered, index1 is stride-skipped, index2 is captured.
    class FakeAttention:
        def __str__(self):
            return "vllm.unified_attention_with_output.default"

        def __call__(self, *args):
            return None

    callbacks = []
    config = SimpleNamespace(use_v2_model_runner=True)
    handle = install(config, "layer", lambda ids, grouped: callbacks.append(ids),
                     capture_stride=2, capture_steps="multi_token")
    try:
        mode = handle._mode_factory()
        for count in [1, 3, 3]:
            query_capture._current_layout.set(
                query_capture._BatchLayout(["req"], [(0, count)]))
            mode.__torch_dispatch__(FakeAttention(), (),
                (torch.ones(3, 2, 1), None, None, None, "layer"))
        assert callbacks == [["req"]]
    finally:
        handle.close()


@pytest.mark.parametrize("invalid", ["prefill", "", "ALL"])
def test_invalid_capture_steps_rejected_before_patch(monkeypatch, invalid):
    original = GPUModelRunner.execute_model
    monkeypatch.setenv("SEMANTIC_CAPTURE_STEPS", invalid)
    with pytest.raises(ValueError, match="capture_steps must be all or multi_token"):
        install(SimpleNamespace(use_v2_model_runner=True), "layer", lambda *args: None)
    assert GPUModelRunner.execute_model is original
    assert query_capture._active_handle is None


def test_capture_steps_explicit_parameter_overrides_environment(monkeypatch):
    monkeypatch.setenv("SEMANTIC_CAPTURE_STEPS", "invalid")
    handle = install(SimpleNamespace(use_v2_model_runner=True), "layer", lambda *args: None,
                     capture_steps="all")
    handle.close()
