from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from wait_hint_offload.vllm_adapter.connector import WaitHintScheduler


def native_fixture(
    enabled: bool = True, mechanism: str = "recency"
) -> tuple[CPUOffloadingSpec, Any, KVCacheConfig]:
    cache = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[KVCacheTensor(size=1024, shared_by=["layer"])],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=4, num_kv_heads=1, head_size=4, dtype=torch.float16
                ),
            )
        ],
    )
    raw = {
        "namespace": {
            "model_revision": "fixture-model",
            "tokenizer_revision": "fixture-tokenizer",
            "adapter_namespace": "none",
            "layout_namespace": "fp16-dense-tp1",
        },
        "cpu_bytes_to_use": 1024,
        "advice_enabled": enabled,
        "advice_mechanism": mechanism,
    }
    config = OffloadingConfig(
        groups=(OffloadingGroupConfig(4, ("layer",)),),
        worker_kv_bytes_per_block=64,
        enable_kv_cache_events=False,
        extra_config=raw,
        engine_id="fixture-engine",
        model=OffloadingModelConfig("fixture-model", "float16"),
        cache=OffloadingCacheConfig(4, 1),
        parallel=OffloadingParallelConfig(0, 1, 1, 1, 1, 1, 0, True),
    )
    spec = CPUOffloadingSpec(config)
    spec.eviction_policy = "arc"
    spec.offload_prompt_only = False
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=1),
        speculative_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )
    return spec, vllm_config, cache


def scheduler_fixture(
    enabled: bool = True, mechanism: str = "recency"
) -> WaitHintScheduler:
    spec, config, cache = native_fixture(enabled, mechanism)
    return WaitHintScheduler(spec, config, cache)
