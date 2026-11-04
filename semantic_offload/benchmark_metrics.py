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
