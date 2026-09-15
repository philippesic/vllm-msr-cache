# SPDX-License-Identifier: Apache-2.0
"""Step 1.1 acceptance checks.

1. The spec loads via vLLM's dynamic spec_module_path path (the mechanism
   external packages use, per vllm/v1/kv_offload/factory.py) and constructs
   end-to-end with a real VllmConfig/KVCacheConfig.
2. SemanticOffloadingManager's observable behavior (prepare_store / touch /
   evict / events) is identical to CPUOffloadingManager(cache_policy="lru")
   given the same operation sequence -- this is Step 1.1's "behavior
   identical to Step 0.1" requirement, checked at the manager level since
   that's the cheapest level that actually catches a divergence (see
   AGENTS.md test-design guidance: unit over integration over e2e).
3. index.py's scoring interface (stubbed now, wired in at Step 1.3) dispatches
   all three methods and rejects an unknown one.

Full end-to-end server verification (CPU-cache hit via /metrics with our spec
selected, mirroring Step 0.1's demonstration) is run manually against real
GPU hardware -- see .claude/docs/semantic-eviction-plan.md Step 1.1 and the
project memory for that run's result.
"""

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.v1.kv_offload.base import OffloadingSpec, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

from semantic_offload.index import BlockSummary, build_summary, score
from semantic_offload.manager import (
    PreservationOffloadingManager,
    SemanticOffloadingManager,
)
from semantic_offload.spec import SemanticOffloadingSpec

# ---------------------------------------------------------------------------
# 1. Spec loads via the real factory dynamic-import path
# ---------------------------------------------------------------------------


def _make_vllm_config(
    cpu_bytes_to_use: int = 65536, extra_config_overrides: dict | None = None
):
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        KVTransferConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=10000,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    extra_config = {
        "spec_name": "SemanticOffloadingSpec",
        "spec_module_path": "semantic_offload.spec",
        "cpu_bytes_to_use": cpu_bytes_to_use,
    }
    extra_config.update(extra_config_overrides or {})
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def _make_kv_cache_config():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    num_blocks = 16
    num_kv_heads = 1
    head_size = 1
    dtype = torch.float32
    page_size = 2 * num_kv_heads * head_size * torch.finfo(dtype).bits // 8
    kv_tensor = KVCacheTensor(
        size=num_blocks * page_size, shared_by=["layer"], block_stride=0
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[kv_tensor],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            )
        ],
    )


def test_spec_loads_via_spec_module_path():
    """SemanticOffloadingSpec is not pre-registered; it must resolve via the
    same spec_module_path dynamic-import path documented for external
    packages in vllm/v1/kv_offload/factory.py.

    get_spec_cls() takes the raw kv_connector_extra_config mapping directly
    (not the whole VllmConfig) as of vLLM #48150 ("Define clean backend
    configuration boundary") -- mirrors the real call site's own signature,
    same as create_spec below."""
    config = _make_vllm_config()
    extra_config = config.kv_transfer_config.kv_connector_extra_config
    spec_cls = OffloadingSpecFactory.get_spec_cls(extra_config)
    assert spec_cls is SemanticOffloadingSpec
    assert issubclass(spec_cls, OffloadingSpec)


def test_spec_constructs_end_to_end_and_serves_manager():
    """Full factory -> spec construction, then get_manager() returns a live
    SemanticOffloadingManager -- the "spec loads and serves a request" check,
    at the manager-construction granularity."""
    config = _make_vllm_config()
    kv_cache_config = _make_kv_cache_config()
    offloading_config = build_offloading_config(config, kv_cache_config)
    spec = OffloadingSpecFactory.create_spec(offloading_config)
    assert isinstance(spec, SemanticOffloadingSpec)
    assert spec.num_blocks > 0

    manager = spec.get_manager()
    assert isinstance(manager, SemanticOffloadingManager)
    # get_manager() must be idempotent, like the base CPUOffloadingSpec.
    assert spec.get_manager() is manager


def test_spec_extra_config_method_selects_real_scoring_method():
    """issues log entry #34: `method` must be selectable via
    kv_connector_extra_config (the same real launch-config path every other
    knob here uses), not just at the Python-constructor level -- this is
    what Step 1.6's benchmark grid actually launches servers with."""
    config = _make_vllm_config(extra_config_overrides={"method": "cuboid_mean"})
    kv_cache_config = _make_kv_cache_config()
    offloading_config = build_offloading_config(config, kv_cache_config)
    spec = OffloadingSpecFactory.create_spec(offloading_config)

    manager = spec.get_manager()

    assert manager._policy._method == "cuboid_mean"


def test_spec_extra_config_selects_policy_alpha_without_changing_default():
    configured = _make_vllm_config(extra_config_overrides={"alpha": 0.6})
    configured_spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(configured, _make_kv_cache_config())
    )
    default = _make_vllm_config()
    default_spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(default, _make_kv_cache_config())
    )

    assert configured_spec.get_manager()._policy._alpha == 0.6
    assert default_spec.get_manager()._policy._alpha == 0.5


def test_decision_feature_trace_is_opt_in_and_provenance_bound():
    default = _make_vllm_config()
    default_spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(default, _make_kv_cache_config())
    )
    default_manager = default_spec.get_manager()
    assert default_manager._decision_feature_trace_enabled is False
    assert default_manager._latest_relevance is None
    assert default_manager._policy._decision_feature_sink is None

    configured = _make_vllm_config(
        extra_config_overrides={
            "preservation_events": True,
            "decision_feature_trace": True,
        }
    )
    configured_spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(configured, _make_kv_cache_config())
    )
    configured_manager = configured_spec.get_manager()
    assert configured_manager._decision_feature_trace_enabled is True
    assert configured_manager._latest_relevance == {}
    assert configured_manager._policy._decision_feature_sink is not None

    unbound = _make_vllm_config(extra_config_overrides={"decision_feature_trace": True})
    unbound_spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(unbound, _make_kv_cache_config())
    )
    with pytest.raises(ValueError, match="requires preservation_events"):
        unbound_spec.get_manager()


def test_spec_extra_config_selects_opt_in_prefix_and_rejects_future_feedback():
    configured = _make_vllm_config(
        extra_config_overrides={
            "prefix_aware": True,
            "future_reuse_feedback": True,
        }
    )
    spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(configured, _make_kv_cache_config())
    )

    with pytest.raises(ValueError, match="offline-only"):
        spec.get_manager()

    configured = _make_vllm_config(extra_config_overrides={"prefix_aware": True})
    spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(configured, _make_kv_cache_config())
    )
    assert spec.get_manager()._policy._prefix_aware is True
    default = OffloadingSpecFactory.create_spec(
        build_offloading_config(_make_vllm_config(), _make_kv_cache_config())
    )
    assert default.get_manager()._policy._prefix_aware is False


@pytest.mark.parametrize(
    ("alpha", "error"),
    [
        (True, TypeError),
        ("0.6", TypeError),
        (None, TypeError),
        (-0.1, ValueError),
        (1.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_spec_rejects_invalid_policy_alpha(alpha, error):
    config = _make_vllm_config(extra_config_overrides={"alpha": alpha})
    spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(config, _make_kv_cache_config())
    )

    with pytest.raises(error, match="alpha"):
        spec.get_manager()


@pytest.mark.parametrize(
    ("floor", "error"),
    [
        (True, TypeError),
        ("0.03", TypeError),
        (-0.01, ValueError),
        (0.31, ValueError),
        (float("nan"), ValueError),
    ],
)
def test_spec_rejects_invalid_ema_rank_weight_floor(floor, error):
    config = _make_vllm_config(
        extra_config_overrides={"ema_rank_weight_floor": floor}
    )
    spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(config, _make_kv_cache_config())
    )

    with pytest.raises(error, match="ema_rank_weight_floor"):
        spec.get_manager()


def test_spec_admission_requires_events_and_propagates_namespace():
    unbound = _make_vllm_config(
        extra_config_overrides={"admission_mode": "historical_frequency"}
    )
    unbound_spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(unbound, _make_kv_cache_config())
    )
    with pytest.raises(ValueError, match="requires preservation_events"):
        unbound_spec.get_manager()

    namespace = {
        "model_revision": "model-rev",
        "tokenizer_revision": "tokenizer-rev",
        "adapter_namespace": "none",
        "layout_namespace": "tp1-bf16",
    }
    configured = _make_vllm_config(
        extra_config_overrides={
            "preservation_events": True,
            "admission_mode": "historical_frequency_cost_per_byte",
            "admission_namespace": namespace,
        }
    )
    spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(configured, _make_kv_cache_config())
    )
    manager = spec.get_manager()
    assert manager._admission_mode == "historical_frequency_cost_per_byte"
    assert manager._admission_namespace == namespace


@pytest.mark.parametrize(
    "overrides, expected_type",
    [
        ({}, SemanticOffloadingManager),
        (
            {"instrumentation_only": True, "eviction_policy": "arc"},
            PreservationOffloadingManager,
        ),
        (
            {"instrumentation_only": True, "eviction_policy": "lru"},
            PreservationOffloadingManager,
        ),
    ],
)
def test_spec_propagates_exact_chain_admission_bound_to_all_policies(
    overrides, expected_type
):
    namespace = {
        "model_revision": "model-rev",
        "tokenizer_revision": "tokenizer-rev",
        "adapter_namespace": "none",
        "layout_namespace": "tp1-bf16",
    }
    configured = _make_vllm_config(
        extra_config_overrides={
            **overrides,
            "store_threshold": 2,
            "admission_namespace": namespace,
            "chain_admission_max_prefix_blocks": 7,
        }
    )
    spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(configured, _make_kv_cache_config())
    )

    manager = spec.get_manager()

    assert isinstance(manager, expected_type)
    assert manager.store_threshold == 2
    assert manager._chain_admission_max_prefix_blocks == 7
    assert manager._admission_namespace == namespace


def test_spec_exact_chain_admission_default_is_disabled_and_validation_is_strict():
    default_config = _make_vllm_config()
    default_spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(default_config, _make_kv_cache_config())
    )
    assert default_spec.get_manager()._chain_admission_max_prefix_blocks == 0

    invalid = _make_vllm_config(
        extra_config_overrides={"chain_admission_max_prefix_blocks": True}
    )
    invalid_spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(invalid, _make_kv_cache_config())
    )
    with pytest.raises(TypeError, match="chain_admission_max_prefix_blocks"):
        invalid_spec.get_manager()


# ---------------------------------------------------------------------------
# 2. Behavior identical to plain LRU (Step 1.1's pass-through requirement)
# ---------------------------------------------------------------------------

_EMPTY_REQ_CTX = ReqContext(req_id="")


def to_keys(int_hashes: list[int]):
    return [make_offload_key(str(i).encode(), 0) for i in int_hashes]


def _run_scenario(manager) -> list:
    """A store/touch/evict/store scenario exercising every CachePolicy method."""
    manager.prepare_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX)
    manager.complete_store(to_keys([1, 2, 3]), _EMPTY_REQ_CTX)
    manager.touch(to_keys([1]), _EMPTY_REQ_CTX)  # block 1 now most-recently-used
    # Only 1 free block (capacity 4, 3 used) -- forces eviction of the LRU
    # candidate among {2, 3} (1 was just touched, so it's protected by recency).
    result = manager.prepare_store(to_keys([4, 5]), _EMPTY_REQ_CTX)
    return result.evicted_keys if result is not None else None


def test_semantic_manager_matches_lru_manager_behavior():
    """SemanticOffloadingManager (SemanticPolicy, delegating to LRU) must
    produce identical eviction decisions to CPUOffloadingManager(cache_policy=
    "lru") given the same operations -- the literal "behavior identical to
    Step 0.1" acceptance check."""
    lru_manager = CPUOffloadingManager(num_blocks=4, cache_policy="lru")
    semantic_manager = SemanticOffloadingManager(num_blocks=4)

    lru_evicted = _run_scenario(lru_manager)
    semantic_evicted = _run_scenario(semantic_manager)

    assert semantic_evicted == lru_evicted
    assert semantic_evicted is not None


def test_prepare_store_evictions_are_drainable():
    """Regression for issues log entries #62-64: prepare_store() must
    surface the base CPUOffloadingManager's own PrepareStoreOutput.
    evicted_keys via drain_evicted_keys(), so the worker can learn exactly
    which blocks were evicted instead of approximating with a FIFO cap.
    drain_evicted_keys() must also clear its buffer once read."""
    manager = SemanticOffloadingManager(num_blocks=4)
    result = _run_scenario(manager)

    assert result  # the scenario forces a real eviction
    assert manager.drain_evicted_keys() == result
    assert manager.drain_evicted_keys() == []  # drained, not re-readable


def test_lifecycle_release_removes_only_idle_cpu_blocks():
    manager = SemanticOffloadingManager(num_blocks=3)
    keys = to_keys([1, 2])
    manager.prepare_store(keys, _EMPTY_REQ_CTX)
    manager.complete_store(keys, _EMPTY_REQ_CTX)
    manager.prepare_load([keys[0]], _EMPTY_REQ_CTX)

    released, skipped = manager.release_keys(keys, _EMPTY_REQ_CTX)

    assert released == [keys[1]]
    assert skipped == [keys[0]]
    assert manager.lookup(keys[1], _EMPTY_REQ_CTX).name == "MISS"
    assert manager.lookup(keys[0], _EMPTY_REQ_CTX).name == "HIT"
    assert manager.drain_evicted_keys() == [keys[1]]

    manager.complete_load([keys[0]], _EMPTY_REQ_CTX)
    released, skipped = manager.release_keys([keys[0]], _EMPTY_REQ_CTX)
    assert released == [keys[0]]
    assert skipped == []


@pytest.mark.parametrize(
    "manager",
    [
        pytest.param(SemanticOffloadingManager(num_blocks=2), id="semantic"),
        pytest.param(
            PreservationOffloadingManager(num_blocks=2, cache_policy="arc"),
            id="arc",
        ),
        pytest.param(
            PreservationOffloadingManager(num_blocks=2, cache_policy="lru"),
            id="lru",
        ),
    ],
)
def test_bound_staging_source_is_protected_equivalently_until_release(manager):
    target, pressure, replacement = to_keys([60, 61, 62])
    first = manager.prepare_store([target, pressure], _EMPTY_REQ_CTX)
    assert first is not None
    manager.complete_store([target, pressure], _EMPTY_REQ_CTX)

    protection = manager.protect_staging_keys("prefetch-binding-1", [target])
    assert protection["outcome"] == "accepted"
    assert manager.staging_protection_stats()["active_bindings"] == 1

    pressured = manager.prepare_store([replacement], _EMPTY_REQ_CTX)
    assert pressured is not None
    assert target not in pressured.evicted_keys
    assert manager.lookup(target, _EMPTY_REQ_CTX).name == "HIT"

    released = manager.release_staging_keys("prefetch-binding-1")
    assert released["outcome"] == "accepted"
    assert manager.staging_protection_stats()["active_bindings"] == 0
    records = manager.drain_staging_protection_records()
    assert [record["action"] for record in records] == ["protect", "release"]
    assert all(record["critical_path_wall_time_ns"] >= 0 for record in records)
    assert all(record["cpu_time_ns"] >= 0 for record in records)
    assert all(record["synchronization_wait_ns"] == 0 for record in records)


def test_staging_protection_is_atomic_idempotent_and_reference_counted():
    manager = SemanticOffloadingManager(num_blocks=2)
    first, second, missing = to_keys([70, 71, 72])
    output = manager.prepare_store([first, second], _EMPTY_REQ_CTX)
    assert output is not None
    manager.complete_store([first, second], _EMPTY_REQ_CTX)

    rejected = manager.protect_staging_keys("bad-binding", [first, missing])
    assert rejected["outcome"] == "rejected"
    assert manager._policy.get(first).ref_cnt == 0
    accepted = manager.protect_staging_keys("shared-a", [first])
    duplicate = manager.protect_staging_keys("shared-a", [first])
    manager.protect_staging_keys("shared-b", [first])
    assert accepted["outcome"] == "accepted"
    assert duplicate["outcome"] == "noop"
    assert manager._policy.get(first).ref_cnt == 2

    manager.release_staging_keys("shared-a")
    assert manager._policy.get(first).ref_cnt == 1
    assert manager.lookup(first, _EMPTY_REQ_CTX).name == "HIT"
    manager.release_staging_keys("shared-b")
    assert manager._policy.get(first).ref_cnt == 0
    assert manager.release_staging_keys("shared-b")["outcome"] == "noop"


def test_lifecycle_owner_release_is_shared_generation_safe_and_ledgered():
    manager = SemanticOffloadingManager(num_blocks=5)
    keys = to_keys([10, 11, 12, 13])
    manager.prepare_store(keys, _EMPTY_REQ_CTX)
    manager.complete_store(keys, _EMPTY_REQ_CTX)

    manager.register_lifecycle_owner("dead", 1, keys[:3], state="future")
    manager.register_lifecycle_owner("active", 1, [keys[0]], state="active")
    manager.register_lifecycle_owner("future", 1, [keys[0]], state="future")
    manager.set_lifecycle_owner_state("dead", 1, "dead")

    result = manager.release_lifecycle_owner("dead", 1, keys[:3], _EMPTY_REQ_CTX)

    assert result.outcome == "accepted"
    assert result.released_keys == tuple(keys[1:3])
    assert result.affected_owner_ids == ("active", "future")
    assert manager.lookup(keys[0], _EMPTY_REQ_CTX).name == "HIT"
    assert manager.lookup(keys[1], _EMPTY_REQ_CTX).name == "MISS"
    assert manager.lookup(keys[2], _EMPTY_REQ_CTX).name == "MISS"

    duplicate = manager.release_lifecycle_owner("dead", 1, keys[:3], _EMPTY_REQ_CTX)
    wrong_generation = manager.release_lifecycle_owner(
        "dead", 2, keys[:3], _EMPTY_REQ_CTX
    )
    assert duplicate.outcome == "noop"
    assert duplicate.reason == "already_released"
    assert wrong_generation.outcome == "rejected"
    assert wrong_generation.reason == "wrong_generation"

    records = manager.drain_lifecycle_release_ledger()
    assert [record["outcome"] for record in records] == [
        "accepted",
        "noop",
        "rejected",
    ]
    assert records[0]["affected_owner_ids"] == ["active", "future"]


def test_lifecycle_owner_release_rejects_active_reference_without_mutation():
    manager = SemanticOffloadingManager(num_blocks=2)
    keys = to_keys([20, 21])
    manager.prepare_store(keys, _EMPTY_REQ_CTX)
    manager.complete_store(keys, _EMPTY_REQ_CTX)
    manager.register_lifecycle_owner("active", 4, [keys[0]], state="active")
    manager.set_lifecycle_owner_state("active", 4, "dead")
    manager.prepare_load([keys[0]], _EMPTY_REQ_CTX)

    result = manager.release_lifecycle_owner("active", 4, [keys[0]], _EMPTY_REQ_CTX)

    assert result.outcome == "rejected"
    assert result.reason == "active_reference"
    assert result.protected_keys == (keys[0],)
    assert manager.lookup(keys[0], _EMPTY_REQ_CTX).name == "HIT"

    manager.complete_load([keys[0]], _EMPTY_REQ_CTX)
    retry = manager.release_lifecycle_owner("active", 4, [keys[0]], _EMPTY_REQ_CTX)
    assert retry.outcome == "accepted"
    assert manager.lookup(keys[0], _EMPTY_REQ_CTX).name == "MISS"


def test_lifecycle_owner_release_is_atomic_when_one_key_is_active():
    manager = SemanticOffloadingManager(num_blocks=3)
    keys = to_keys([30, 31])
    manager.prepare_store(keys, _EMPTY_REQ_CTX)
    manager.complete_store(keys, _EMPTY_REQ_CTX)
    manager.register_lifecycle_owner("dead", 1, keys, state="dead")
    manager.prepare_load([keys[1]], _EMPTY_REQ_CTX)

    result = manager.release_lifecycle_owner("dead", 1, keys, _EMPTY_REQ_CTX)

    assert result.outcome == "rejected"
    assert result.protected_keys == (keys[1],)
    assert manager.lookup(keys[0], _EMPTY_REQ_CTX).name == "HIT"
    assert manager.lookup(keys[1], _EMPTY_REQ_CTX).name == "HIT"

    manager.complete_load([keys[1]], _EMPTY_REQ_CTX)
    retry = manager.release_lifecycle_owner("dead", 1, keys, _EMPTY_REQ_CTX)
    assert retry.outcome == "accepted"
    assert manager.lookup(keys[0], _EMPTY_REQ_CTX).name == "MISS"
    assert manager.lookup(keys[1], _EMPTY_REQ_CTX).name == "MISS"


def test_semantic_manager_method_is_selectable():
    """issues log entry #34: `method` used to be hardcoded to
    manager.py's _DEFAULT_METHOD with no way to select a different scoring
    method at construction time -- Step 1.6's benchmark grid needs
    {minmax, mean, cuboid_mean} as separate, runnable configs, so this must
    actually thread through to the real SemanticPolicy instance."""
    default_manager = SemanticOffloadingManager(num_blocks=4)
    mean_manager = SemanticOffloadingManager(num_blocks=4, method="mean")
    cuboid_manager = SemanticOffloadingManager(num_blocks=4, method="cuboid_mean")

    assert default_manager._policy._method == "mean"
    assert mean_manager._policy._method == "mean"
    assert cuboid_manager._policy._method == "cuboid_mean"


# ---------------------------------------------------------------------------
# 3. index.py's scoring interface stub
# ---------------------------------------------------------------------------


def test_build_summary_shapes():
    keys = torch.randn(16, 8)  # [block_tokens, head_dim]
    summary = build_summary(keys)
    assert isinstance(summary, BlockSummary)
    for field in (summary.min, summary.max, summary.mean, summary.mad):
        assert field.shape == (8,)


def test_score_dispatches_all_three_methods():
    keys = torch.randn(16, 8)
    query = torch.randn(8)
    summary = build_summary(keys)
    for method in ("minmax", "mean", "cuboid_mean"):
        value = score(method, query, summary)
        assert isinstance(value, float)


def test_score_rejects_unknown_method():
    keys = torch.randn(16, 8)
    query = torch.randn(8)
    summary = build_summary(keys)
    try:
        score("not_a_method", query, summary)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "not_a_method" in str(e)
