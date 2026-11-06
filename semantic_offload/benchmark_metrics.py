"""Opt-in scalar activity receipts through vLLM's native offloading metrics."""

from vllm.v1.kv_offload.base import OffloadingCounterMetadata

RELEVANCE_BATCHES = "vllm:semantic_relevance_update_batches"


def definitions(extra_config):
    enabled = extra_config.get("benchmark_telemetry", False)
    if type(enabled) is not bool:
        raise TypeError("benchmark_telemetry must be a JSON boolean")
    if not enabled:
        return {}
    return {
        RELEVANCE_BATCHES: OffloadingCounterMetadata(
            documentation="Scheduler updates containing scored relevance actually applied to the manager. Not scoring quality, comparisons, or useful cache hits."
        )
    }


def record_relevance_batch(scheduler, amount=1):
    if not getattr(scheduler, "_benchmark_telemetry", False):
        return
    scheduler._connector_stats.increase_counter(RELEVANCE_BATCHES, amount)



def allocation_receipt(worker, *, num_cpu_blocks, kv_bytes_per_chunk,
                       requested_cpu_bytes, blocks_per_chunk, model_config):
    """Observe allocated backing without allocating, transferring or changing it."""
    from functools import partial

    dtype = getattr(model_config, "dtype", None)
    result = dict(schema="semantic-offload-allocation-v1", available=False,
        num_cpu_blocks=num_cpu_blocks, kv_bytes_per_chunk=kv_bytes_per_chunk,
        requested_cpu_bytes=requested_cpu_bytes, blocks_per_chunk=blocks_per_chunk,
        model_dtype=str(dtype) if dtype is not None else None,
        model_quantization=getattr(model_config, "quantization", None),
        cpu_buffers=[])
    try:
        store, load = worker._store_handler, worker._load_handler
        cpu, returning = store.dst_tensors, load.src_tensors
        if not isinstance(cpu, (list, tuple)) or not isinstance(returning, (list, tuple)):
            raise TypeError("Unsupported native offloading buffer containers")
        def selected_callable(handler):
            function = getattr(handler, "_swap_blocks_batch", None)
            while isinstance(function, partial):
                function = function.func
            if function is None:
                return None
            return (getattr(function, "__module__", type(function).__module__)
                    + "." + getattr(function, "__qualname__", type(function).__qualname__))
        for index, tensor in enumerate(cpu):
            other = returning[index] if index < len(returning) else None
            shared = (len(cpu) == len(returning) and other is not None
                and tensor.device == other.device and tensor.dtype == other.dtype
                and tuple(tensor.shape) == tuple(other.shape)
                and tensor.data_ptr() == other.data_ptr())
            result["cpu_buffers"].append(dict(device=str(tensor.device),
                dtype=str(tensor.dtype), shape=list(tensor.shape),
                nbytes=tensor.numel()*tensor.element_size(),
                pinned=bool(tensor.is_pinned()), load_uses_same_storage=bool(shared)))
        result.update(available=bool(cpu), selected_store_callable=selected_callable(store),
                      selected_load_callable=selected_callable(load))
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        result.update(available=False, error=f"{type(exc).__name__}: {exc}")
    return result
