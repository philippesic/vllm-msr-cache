# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.v1.kv_offload.base import ReqContext, make_offload_key

from benchmarks.preservation import (
    REQUEST_EVENT_ADMISSION_DECISION_REASON,
    validate_admission_decision_records,
)
from semantic_offload.manager import (
    PreservationOffloadingManager,
    SemanticOffloadingManager,
)


def key(value: int, group: int = 0):
    return make_offload_key(value.to_bytes(8, "big"), group)


CHAIN_NAMESPACE = {
    "model_revision": "model-rev",
    "tokenizer_revision": "tokenizer-rev",
    "adapter_namespace": "none",
    "layout_namespace": "tp1-bf16-block16",
}


def chain_context(
    request_id: str,
    *,
    session_id: str = "session-a",
    generation: int = 0,
    digest: str = "a" * 64,
):
    return ReqContext(
        req_id=request_id,
        kv_transfer_params={
            "run_id": "run-1",
            "session_id": session_id,
            "generation": generation,
            "key_chain_digest": digest,
        },
    )


def managers(**kwargs):
    return [
        PreservationOffloadingManager(
            num_blocks=1, cache_policy="lru", **kwargs
        ),
        PreservationOffloadingManager(
            num_blocks=1, cache_policy="arc", **kwargs
        ),
        SemanticOffloadingManager(num_blocks=1, **kwargs),
    ]


def store_ready(manager, cache_key, request_id):
    context = ReqContext(req_id=request_id)
    output = manager.prepare_store([cache_key], context)
    assert output is not None
    manager.complete_store(output.keys_to_store, context)
    return output


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_native_policy_shared_prefetch_scorer_tracks_scores_without_policy_swap(
    policy,
):
    manager = PreservationOffloadingManager(
        num_blocks=2,
        cache_policy=policy,
        shared_prefetch_scorer=True,
        method="mean",
    )
    first, second = key(100), key(101)
    store_ready(manager, first, "first")
    store_ready(manager, second, "second")
    policy_type = type(manager._policy)

    manager.update_relevance(
        {"mean": {"probe": [(second, 2.0), (first, 1.0)]}}
    )

    assert manager.top_relevant_keys([first, second], 1) == [second]
    assert type(manager._policy) is policy_type
    assert manager._policy.get(first) is not None
    assert manager._policy.get(second) is not None


def test_native_policy_shared_prefetch_scorer_default_is_inert():
    manager = PreservationOffloadingManager(num_blocks=1, cache_policy="arc")
    resident = key(102)
    store_ready(manager, resident, "resident")

    manager.update_relevance({"mean": {"probe": [(resident, 1.0)]}})

    assert manager.relevance_ema == {}
    assert manager.top_relevant_keys([resident], 1) == []


def test_native_policy_shared_prefetch_scores_expire_causally():
    manager = PreservationOffloadingManager(
        num_blocks=1,
        cache_policy="arc",
        shared_prefetch_scorer=True,
    )
    resident = key(103)
    store_ready(manager, resident, "resident")
    manager.update_relevance({"mean": {"probe": [(resident, 1.0)]}})
    updated_ns = manager._relevance_updated_ns["mean"][resident]

    assert manager.top_relevant_keys(
        [resident], 1, max_age_ns=10, now_ns=updated_ns + 10
    ) == [resident]
    assert manager.top_relevant_keys(
        [resident], 1, max_age_ns=10, now_ns=updated_ns + 11
    ) == []


@pytest.mark.parametrize(
    "manager", managers(admission_mode="historical_frequency_cost_per_byte")
)
def test_equal_frequency_cost_rejects_without_evicting(manager):
    resident, candidate = key(1), key(2)
    store_ready(manager, resident, "resident-store")
    manager.lookup(resident, ReqContext(req_id="resident-read"))
    manager.lookup(candidate, ReqContext(req_id="candidate-read"))

    output = manager.prepare_store([candidate], ReqContext(req_id="candidate-store"))

    assert output is not None
    assert output.keys_to_store == []
    assert output.evicted_keys == []
    assert manager._policy.get(resident) is not None
    records = manager.drain_admission_records()
    decision = records[-1]
    assert decision["action"] == "reject"
    assert decision["reason"] == "candidate_utility_not_greater"
    assert decision["candidate_features_available"] is True
    assert decision["victim_features_available"] is True
    assert decision["candidate_utility"] == decision["victim_utility"] == 1.0
    assert "future" not in " ".join(decision).lower()
    assert decision["decision_timestamp_ns"] > 0
    assert manager.stores_skipped_in_current_batch == 0
    assert manager.drain_admission_accounting() == {
        "admitted": 1,
        "rejected": 1,
        "compulsory_fallbacks": 0,
    }


@pytest.mark.parametrize(
    "manager", managers(admission_mode="historical_frequency_cost_per_byte")
)
def test_more_frequent_candidate_admits_and_evicts_exact_victim(manager):
    resident, candidate = key(10), key(11)
    store_ready(manager, resident, "resident-store")
    manager.lookup(resident, ReqContext(req_id="resident-read"))
    manager.lookup(candidate, ReqContext(req_id="candidate-read-1"))
    manager.lookup(candidate, ReqContext(req_id="candidate-read-2"))

    output = manager.prepare_store([candidate], ReqContext(req_id="candidate-store"))

    assert output is not None
    assert output.keys_to_store == [candidate]
    assert output.evicted_keys == [resident]
    decision = manager.drain_admission_records()[-1]
    assert decision["victim_key"] == resident.hex()
    assert decision["candidate_frequency"] == 2
    assert decision["victim_frequency"] == 1
    assert decision["reason"] == "candidate_utility_greater"


def test_configured_cost_per_byte_changes_decision_with_provenance():
    manager = PreservationOffloadingManager(
        num_blocks=1,
        cache_policy="lru",
        admission_mode="historical_frequency_cost_per_byte",
        admission_cost_per_byte={"1": 2.0},
        admission_default_cost_per_byte=1.0,
    )
    resident, candidate = key(20, 0), key(21, 1)
    store_ready(manager, resident, "resident-store")
    manager.lookup(resident, ReqContext(req_id="resident-read"))
    manager.lookup(candidate, ReqContext(req_id="candidate-read"))

    output = manager.prepare_store([candidate], ReqContext(req_id="candidate-store"))

    assert output is not None and output.evicted_keys == [resident]
    decision = manager.drain_admission_records()[-1]
    assert decision["candidate_cost_per_byte"] == 2.0
    assert decision["candidate_cost_origin"] == "configured_group"
    assert decision["victim_cost_per_byte"] == 1.0
    assert decision["victim_cost_origin"] == "configured_default"


def test_historical_frequency_is_unit_cost_control():
    manager = PreservationOffloadingManager(
        num_blocks=1,
        cache_policy="lru",
        admission_mode="historical_frequency",
        admission_cost_per_byte={"1": 100.0},
    )
    resident, candidate = key(22, 0), key(23, 1)
    store_ready(manager, resident, "resident-store")
    manager.lookup(resident, ReqContext(req_id="resident-read"))
    manager.lookup(candidate, ReqContext(req_id="candidate-read"))

    output = manager.prepare_store([candidate], ReqContext(req_id="candidate-store"))

    assert output is not None and output.evicted_keys == []
    decision = manager.drain_admission_records()[-1]
    assert decision["action"] == "reject"
    assert decision["candidate_cost_per_byte"] == 1.0
    assert decision["candidate_cost_origin"] == "unit_cost_frequency_control"


def test_request_historical_cost_sample_is_causal_and_persists_with_resident():
    manager = SemanticOffloadingManager(
        num_blocks=1, admission_mode="historical_frequency_cost_per_byte"
    )
    candidate = key(24)
    context = ReqContext(
        req_id="candidate",
        kv_transfer_params={
            "admission_cost_per_byte": 3.5,
            "admission_cost_provenance": "screen_baseline_profile_v1",
            "admission_cost_observation_available_ns": 1,
            "admission_cost_input_publication_ns": 2,
        },
    )
    manager.lookup(candidate, context)
    output = manager.prepare_store([candidate], context)

    assert output is not None
    decision = manager.drain_admission_records()[-1]
    assert decision["candidate_cost_per_byte"] == 3.5
    assert decision["candidate_cost_origin"] == (
        "historical_request_sample:screen_baseline_profile_v1"
    )
    assert manager._admission_resident_cost_per_byte[candidate][:2] == (
        3.5,
        "historical_request_sample:screen_baseline_profile_v1",
    )


def test_future_timestamped_cost_sample_is_not_used():
    manager = SemanticOffloadingManager(
        num_blocks=1, admission_mode="historical_frequency_cost_per_byte"
    )
    resident, candidate = key(26), key(27)
    store_ready(manager, resident, "resident-store")
    manager.lookup(resident, ReqContext(req_id="resident-read"))
    context = ReqContext(
        req_id="candidate",
        kv_transfer_params={
            "admission_cost_per_byte": 100.0,
            "admission_cost_provenance": "not_yet_available",
            "admission_cost_observation_available_ns": 2**63 - 1,
            "admission_cost_input_publication_ns": 2**63 - 1,
        },
    )
    manager.lookup(candidate, context)

    output = manager.prepare_store([candidate], context)

    assert output is not None and output.evicted_keys == [resident]
    decision = manager.drain_admission_records()[-1]
    assert decision["candidate_cost_per_byte"] is None
    assert decision["candidate_cost_origin"] == "unavailable_noncausal_cost_sample"
    assert decision["reason"] == "compulsory_fallback_unavailable_features"
    assert decision["causal_binding_complete"] is False


@pytest.mark.parametrize(
    "manager",
    [
        PreservationOffloadingManager(
            num_blocks=1,
            cache_policy="lru",
            store_threshold=2,
            max_tracker_size=1,
            admission_mode="historical_frequency",
        ),
        PreservationOffloadingManager(
            num_blocks=1,
            cache_policy="arc",
            store_threshold=2,
            max_tracker_size=1,
            admission_mode="historical_frequency",
        ),
        SemanticOffloadingManager(
            num_blocks=1,
            store_threshold=2,
            max_tracker_size=1,
            admission_mode="historical_frequency",
        ),
    ],
)
def test_preview_protection_matches_native_threshold_filtered_keys(manager):
    resident, candidate = key(28), key(29)
    context = ReqContext(req_id="resident")
    manager.lookup(resident, context)
    manager.lookup(resident, context)
    store_ready(manager, resident, "resident-store")
    candidate_context = ReqContext(req_id="candidate")
    manager.lookup(candidate, candidate_context)
    manager.lookup(candidate, candidate_context)

    output = manager.prepare_store(
        [resident, candidate], ReqContext(req_id="combined-store")
    )

    assert output is not None and output.evicted_keys == [resident]
    decision = manager.drain_admission_records()[-1]
    assert decision["victim_key"] == resident.hex()
    assert decision["reason"] == "compulsory_fallback_unavailable_features"


def test_admission_record_binds_causal_namespace_and_monotonic_use():
    namespace = {
        "model_revision": "model-rev",
        "tokenizer_revision": "tokenizer-rev",
        "adapter_namespace": "none",
        "layout_namespace": "tp1-bf16",
    }
    manager = SemanticOffloadingManager(
        num_blocks=1,
        admission_mode="historical_frequency_cost_per_byte",
        admission_namespace=namespace,
    )
    candidate = key(25)
    params = {
        "run_id": "run-1",
        "session_id": "session-1",
        "generation": 2,
        "scheduler_step": 7,
        "key_chain_digest": "a" * 64,
    }
    context = ReqContext(req_id="candidate", kv_transfer_params=params)
    manager.lookup(candidate, context)
    manager.prepare_store([candidate], context)

    decision = manager.drain_admission_records()[-1]
    assert decision["causal_binding_complete"] is True
    assert decision["engine_run_id"] == "run-1"
    assert decision["model_revision"] == "model-rev"
    assert decision["generation"] == 2
    assert decision["scheduler_step"] == 7
    assert (
        decision["observation_available_ns"]
        <= decision["input_publication_ns"]
        <= decision["consumer_decision_ns"]
        <= decision["consumer_use_ns"]
    )
    assert (
        validate_admission_decision_records(
            [decision], request_id="candidate", run_id="run-1"
        )
        is None
    )
    decision["future_reuse"] = True
    assert (
        validate_admission_decision_records(
            [decision], request_id="candidate", run_id="run-1"
        )
        == REQUEST_EVENT_ADMISSION_DECISION_REASON
    )


def test_missing_frequency_falls_back_to_compulsory_admission():
    manager = SemanticOffloadingManager(
        num_blocks=1, admission_mode="historical_frequency_cost_per_byte"
    )
    resident, candidate = key(30), key(31)
    store_ready(manager, resident, "resident-store")
    manager.lookup(resident, ReqContext(req_id="resident-read"))

    output = manager.prepare_store([candidate], ReqContext(req_id="candidate-store"))

    assert output is not None and output.evicted_keys == [resident]
    decision = manager.drain_admission_records()[-1]
    assert decision["action"] == "admit"
    assert decision["reason"] == "compulsory_fallback_unavailable_features"
    assert decision["candidate_features_available"] is False


def test_native_store_threshold_is_applied_before_admission():
    manager = PreservationOffloadingManager(
        num_blocks=1,
        cache_policy="lru",
        store_threshold=2,
        admission_mode="historical_frequency_cost_per_byte",
    )
    candidate = key(40)
    manager.lookup(candidate, ReqContext(req_id="candidate-read"))

    output = manager.prepare_store([candidate], ReqContext(req_id="candidate-store"))

    assert output is not None and output.keys_to_store == []
    assert manager.drain_admission_records() == []
    assert manager.stores_skipped_in_current_batch == 1


def test_failed_store_and_reset_clear_admission_metadata():
    manager = SemanticOffloadingManager(
        num_blocks=1, admission_mode="historical_frequency_cost_per_byte"
    )
    candidate = key(50)
    manager.lookup(candidate, ReqContext(req_id="candidate-read"))
    output = manager.prepare_store([candidate], ReqContext(req_id="candidate-store"))
    assert output is not None
    assert candidate in manager._admission_resident_cost_per_byte

    manager.complete_store(
        output.keys_to_store, ReqContext(req_id="candidate-store"), success=False
    )
    assert candidate not in manager._admission_resident_cost_per_byte
    manager.reset_cache()
    assert manager._admission_frequencies == {}
    assert manager.drain_admission_records() == []
    assert manager.drain_admission_accounting() == {
        "admitted": 0,
        "rejected": 0,
        "compulsory_fallbacks": 0,
    }


def test_admission_behavior_digest_binds_threshold_and_tracker_capacity():
    base = SemanticOffloadingManager(
        num_blocks=1,
        admission_mode="historical_frequency",
        store_threshold=0,
        max_tracker_size=8,
    )
    threshold = SemanticOffloadingManager(
        num_blocks=1,
        admission_mode="historical_frequency",
        store_threshold=2,
        max_tracker_size=8,
    )
    tracker = SemanticOffloadingManager(
        num_blocks=1,
        admission_mode="historical_frequency",
        store_threshold=0,
        max_tracker_size=9,
    )

    assert len({
        base._admission_config_hash,
        threshold._admission_config_hash,
        tracker._admission_config_hash,
    }) == 3


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"admission_mode": "oracle"}, ValueError),
        ({"admission_mode": True}, TypeError),
        ({"admission_cost_per_byte": []}, TypeError),
        ({"admission_cost_per_byte": {"-1": 1}}, ValueError),
        ({"admission_cost_per_byte": {"0": float("nan")}}, ValueError),
        ({"admission_default_cost_per_byte": -1}, ValueError),
    ],
)
def test_invalid_admission_configuration_fails_closed(kwargs, expected):
    with pytest.raises(expected):
        SemanticOffloadingManager(num_blocks=1, **kwargs)


def test_disabled_admission_preserves_base_behavior_and_emits_no_records():
    manager = SemanticOffloadingManager(num_blocks=1)
    first, second = key(60), key(61)
    store_ready(manager, first, "first")
    output = manager.prepare_store([second], ReqContext(req_id="second"))
    assert output is not None and output.evicted_keys == [first]
    assert manager.drain_admission_records() == []


@pytest.mark.parametrize(
    "manager",
    [
        PreservationOffloadingManager(
            num_blocks=4,
            cache_policy="lru",
            store_threshold=2,
            admission_namespace=CHAIN_NAMESPACE,
        ),
        PreservationOffloadingManager(
            num_blocks=4,
            cache_policy="arc",
            store_threshold=2,
            admission_namespace=CHAIN_NAMESPACE,
        ),
        SemanticOffloadingManager(
            num_blocks=4,
            store_threshold=2,
            admission_namespace=CHAIN_NAMESPACE,
        ),
    ],
)
def test_exact_chain_admission_default_preserves_native_first_miss_frontier(manager):
    chain = [key(70), key(71), key(72)]
    first = chain_context("first")
    manager.lookup(chain[0], first)
    output = manager.prepare_store(chain, first)
    assert output is not None and output.keys_to_store == []

    second = chain_context("second")
    manager.lookup(chain[0], second)
    output = manager.prepare_store(chain, second)

    assert output is not None and output.keys_to_store == [chain[0]]
    assert manager.drain_chain_admission_records() == []


@pytest.mark.parametrize(
    "manager",
    [
        PreservationOffloadingManager(
            num_blocks=4,
            cache_policy="lru",
            store_threshold=2,
            admission_namespace=CHAIN_NAMESPACE,
            chain_admission_max_prefix_blocks=3,
        ),
        PreservationOffloadingManager(
            num_blocks=4,
            cache_policy="arc",
            store_threshold=2,
            admission_namespace=CHAIN_NAMESPACE,
            chain_admission_max_prefix_blocks=3,
        ),
        SemanticOffloadingManager(
            num_blocks=4,
            store_threshold=2,
            admission_namespace=CHAIN_NAMESPACE,
            chain_admission_max_prefix_blocks=3,
        ),
    ],
)
def test_second_distinct_exact_chain_access_admits_bounded_contiguous_prefix(manager):
    chain = [key(80), key(81), key(82), key(83)]
    first = chain_context("first")
    manager.lookup(chain[0], first)
    output = manager.prepare_store(chain, first)
    assert output is not None and output.keys_to_store == []

    second = chain_context("second")
    manager.lookup(chain[0], second)
    output = manager.prepare_store(chain, second)
    assert output is not None and output.keys_to_store == chain[:3]
    manager.complete_store(output.keys_to_store, second)

    records = manager.drain_chain_admission_records()
    assert [record["action"] for record in records] == [
        "observe",
        "admit",
        "store_complete",
    ]
    assert records[1]["reason"] == "bounded_exact_chain_prefix_admitted"
    assert records[-1]["reason"] == "exact_chain_store_completed"
    assert [item["position"] for item in records[1]["admitted_keys"]] == [0, 1, 2]
    assert [item["key"] for item in records[1]["admitted_keys"]] == [
        item.hex() for item in chain[:3]
    ]
    accounting = manager.drain_chain_admission_accounting()
    assert accounting["observations"] == 2
    assert accounting["matched_observations"] == 1
    assert accounting["qualified_chains"] == 1
    assert accounting["qualified_keys"] == 3
    assert accounting["admitted_keys"] == 3
    assert accounting["stored_keys"] == 3
    assert accounting["overhead_cpu_time_ns"] >= 0
    assert accounting["overhead_wall_time_ns"] >= 0


def test_exact_chain_admission_does_not_change_threshold_one_boundary():
    manager = SemanticOffloadingManager(
        num_blocks=3,
        store_threshold=1,
        admission_namespace=CHAIN_NAMESPACE,
        chain_admission_max_prefix_blocks=3,
    )
    chain = [key(90), key(91), key(92)]

    output = manager.prepare_store(chain, chain_context("first"))

    assert output is not None and output.keys_to_store == chain
    assert manager.drain_chain_admission_records() == []


@pytest.mark.parametrize(
    "changed",
    [
        {"session_id": "session-b"},
        {"generation": 1},
        {"digest": "b" * 64},
    ],
)
def test_exact_chain_identity_mismatch_cannot_share_accesses(changed):
    manager = SemanticOffloadingManager(
        num_blocks=3,
        store_threshold=2,
        admission_namespace=CHAIN_NAMESPACE,
        chain_admission_max_prefix_blocks=3,
    )
    chain = [key(100), key(101), key(102)]
    first = manager.prepare_store(chain, chain_context("first"))
    assert first is not None and first.keys_to_store == []

    second = manager.prepare_store(chain, chain_context("second", **changed))

    assert second is not None and second.keys_to_store == []
    records = manager.drain_chain_admission_records()
    assert [record["reason"] for record in records] == [
        "first_exact_chain_access",
        "first_exact_chain_access",
    ]


def test_exact_chain_namespace_and_position_mismatch_fail_closed():
    manager = SemanticOffloadingManager(
        num_blocks=3,
        store_threshold=2,
        admission_namespace=CHAIN_NAMESPACE,
        chain_admission_max_prefix_blocks=3,
    )
    chain = [key(110), key(111), key(112)]
    first = manager.prepare_store(chain, chain_context("first"))
    assert first is not None and first.keys_to_store == []

    manager._admission_namespace = {**CHAIN_NAMESPACE, "layout_namespace": "tp2"}
    different_namespace = manager.prepare_store(chain, chain_context("second"))
    assert different_namespace is not None
    assert different_namespace.keys_to_store == []

    manager._admission_namespace = dict(CHAIN_NAMESPACE)
    reordered = manager.prepare_store(
        [chain[0], chain[2], chain[1]], chain_context("third")
    )
    assert reordered is not None and reordered.keys_to_store == []
    records = manager.drain_chain_admission_records()
    assert records[-1]["reason"] == "exact_chain_prefix_mismatch"
    assert manager.drain_chain_admission_accounting()["prefix_rejections"] == 1


def test_duplicate_request_does_not_satisfy_second_chain_access():
    manager = SemanticOffloadingManager(
        num_blocks=2,
        store_threshold=2,
        admission_namespace=CHAIN_NAMESPACE,
        chain_admission_max_prefix_blocks=2,
    )
    chain = [key(120), key(121)]
    context = chain_context("same-request")
    first = manager.prepare_store(chain, context)
    second = manager.prepare_store(chain, context)

    assert first is not None and first.keys_to_store == []
    assert second is not None and second.keys_to_store == []
    assert manager.drain_chain_admission_records()[-1]["reason"] == (
        "duplicate_request_access"
    )


def test_failed_chain_store_and_reset_clear_bounded_metadata():
    manager = SemanticOffloadingManager(
        num_blocks=2,
        store_threshold=2,
        max_tracker_size=2,
        admission_namespace=CHAIN_NAMESPACE,
        chain_admission_max_prefix_blocks=2,
    )
    chain = [key(130), key(131)]
    manager.prepare_store(chain, chain_context("first"))
    second_context = chain_context("second")
    output = manager.prepare_store(chain, second_context)
    assert output is not None and output.keys_to_store == chain
    manager.complete_store(output.keys_to_store, second_context, success=False)

    accounting = manager.drain_chain_admission_accounting()
    assert accounting["failed_store_keys"] == 2
    assert manager._chain_admission_pending_stores == {}
    manager.reset_cache()
    assert manager._chain_admission_observations == {}
    assert manager.drain_chain_admission_records() == []
    assert not any(manager.drain_chain_admission_accounting().values())


def test_exact_chain_observation_tracker_evicts_oldest_identity_without_credit():
    manager = SemanticOffloadingManager(
        num_blocks=1,
        store_threshold=2,
        max_tracker_size=2,
        admission_namespace=CHAIN_NAMESPACE,
        chain_admission_max_prefix_blocks=1,
    )
    candidate = [key(140)]
    manager.prepare_store(
        candidate, chain_context("a-first", session_id="session-a")
    )
    manager.prepare_store(
        candidate, chain_context("b-first", session_id="session-b")
    )
    manager.prepare_store(
        candidate, chain_context("c-first", session_id="session-c")
    )

    revisited = manager.prepare_store(
        candidate, chain_context("a-second", session_id="session-a")
    )

    assert revisited is not None and revisited.keys_to_store == []
    records = manager.drain_chain_admission_records()
    assert records[-1]["reason"] == "first_exact_chain_access"
    accounting = manager.drain_chain_admission_accounting()
    assert accounting["tracker_evictions"] == 2
    assert accounting["qualified_chains"] == 0


@pytest.mark.parametrize(
    "value, expected",
    [(True, TypeError), (-1, ValueError), (1.5, TypeError)],
)
def test_invalid_exact_chain_admission_bound_fails_closed(value, expected):
    with pytest.raises(expected):
        SemanticOffloadingManager(
            num_blocks=1,
            chain_admission_max_prefix_blocks=value,
        )


def test_exact_chain_admission_bound_cannot_exceed_native_tracker_capacity():
    with pytest.raises(ValueError, match="cannot exceed max_tracker_size"):
        SemanticOffloadingManager(
            num_blocks=1,
            max_tracker_size=1,
            chain_admission_max_prefix_blocks=2,
        )
