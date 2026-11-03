# SPDX-License-Identifier: Apache-2.0
"""CPU qualification for mean-only continuous and opportunity payloads."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_offload.base import make_offload_key

from semantic_offload.index import BlockSummary
from semantic_offload.scoring_search import ClusterSearch, SearchConfig
from semantic_offload.worker import SemanticOffloadingWorker, _MeanOnlySummary
from tests.test_step_1_2_summaries import _make_layer, _make_worker


def _key(value):
    return make_offload_key(str(value).encode(), 0)


def _worker(layer, fields="required", schedule="opportunity"):
    worker = _make_worker({"probe": layer}, 16)
    worker._summary_fields = fields
    worker._method = "mean"
    worker._scoring_schedule = schedule
    worker._scoring_mode = "synchronous"
    worker._head_aggregation = "mean"
    worker._candidate_search = ClusterSearch(SearchConfig())
    worker._pending_score_head_counts = {}
    worker._query_snapshots = {}
    worker._query_snapshot_events = {}
    worker._query_snapshot_max_age_ns = 1_000_000_000
    worker._query_snapshot_max_requests = 16
    worker._pending_async_score_observations = []
    worker._decision_score_sequence = 0
    return worker


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mean_only_exact_key_mapping_ownership_and_duplicate_last_write(dtype):
    layer = _make_layer(9, 3, 16, 8, seed=924)
    layer.kv_cache = layer.kv_cache.to(dtype)
    worker = _worker(layer)
    full = _worker(layer, "all")
    layout = [(_key(1), [7, 2]), (_key(2), [0]), (_key(3), [5, 5, 1]),
              (_key(4), []), (_key(1), [8])]
    worker.receive_job_blocks({12: layout})
    full.receive_job_blocks({12: layout})
    worker._build_durable_summaries_for_job(12)
    full._build_durable_summaries_for_job(12)
    assert list(worker.durable_summaries) == [_key(1), _key(2), _key(3)]
    layer.kv_cache.fill_(999)
    for key, rows in worker.durable_summaries.items():
        for reduced, original in zip(rows, full.durable_summaries[key]):
            assert isinstance(reduced, _MeanOnlySummary)
            assert isinstance(original, BlockSummary)
            assert reduced.mean.dtype == torch.float32
            assert torch.equal(reduced.mean, original.mean)
            assert all(not hasattr(reduced, field) for field in ("min", "max", "mad"))
    # Group/head identities must be distinct, even with repeated source blocks.
    assert not torch.equal(worker.durable_summaries[_key(1)][0].mean,
                           worker.durable_summaries[_key(2)][0].mean)
    assert not torch.equal(worker.durable_summaries[_key(3)][0].mean,
                           worker.durable_summaries[_key(3)][1].mean)


def test_mean_only_legacy_block_mapping_matches_full_builder():
    layer = _make_layer(8, 2, 16, 8, seed=925)
    worker, full = _worker(layer), _worker(layer, "all")
    ids = [5, 0, 7, 5]
    worker._build_summaries_body(ids)
    full._build_summaries_body(ids)
    layer.kv_cache.zero_()
    assert list(worker.summaries["probe"]) == [5, 0, 7]
    for block in (5, 0, 7):
        for reduced, original in zip(worker.summaries["probe"][block], full.summaries["probe"][block]):
            assert torch.equal(reduced.mean, original.mean)


def test_mean_only_opportunity_scoring_and_cache_mutations_equal_full_fields():
    layer = _make_layer(8, 2, 16, 8, seed=926)
    workers = [_worker(layer), _worker(layer, "all")]
    queries = torch.randn(1, 2, 8, generator=torch.Generator().manual_seed(13))
    expected = None
    for worker in workers:
        worker.receive_job_blocks({1: [(_key(1), [3]), (_key(2), [0, 1]), (_key(3), [4])]})
        worker._build_durable_summaries_for_job(1)
        worker._rebuild_stack_cache()
        assert set(worker._stack_cache) == {"mean"}
        worker.receive_evicted_keys([_key(2)])
        worker.receive_job_blocks({2: [(_key(1), [7]), (_key(4), [5])]})
        worker._build_durable_summaries_for_job(2)
        worker._on_queries_captured(["req"], queries)
        stamp = worker._query_snapshots["req"][0]
        worker.receive_score_requests({"req": [_key(1), _key(3), _key(4)]})
        observations = worker.pop_async_score_observations()
        assert len(observations) == 1
        assert observations[0]["snapshot_captured_ns"] == stamp
        assert observations[0]["expires_at_ns"] == stamp + worker._query_snapshot_max_age_ns
        scores = observations[0]["scores"]
        if expected is None:
            expected = scores
        else:
            assert scores == expected
        assert set(worker._stack_cache_keys) == {_key(1), _key(3), _key(4)}


def test_mean_only_empty_layout_does_not_publish():
    layer = _make_layer(2, 1, 16, 8, seed=927)
    worker = _worker(layer)
    worker.receive_job_blocks({1: [(_key(1), [])]})
    with patch("torch.stack", side_effect=AssertionError("must not gather")):
        worker._build_durable_summaries_for_job(1)
        worker._build_summaries_body([])
    assert not worker.durable_summaries and not worker.summaries["probe"]
    assert SemanticOffloadingWorker._summarize_mean_key_batches(torch.empty(0, 2, 16, 8)) == []


@pytest.mark.parametrize("block", [-1, 3])
def test_mean_only_invalid_block_layout_never_publishes_other_keys(block):
    layer = _make_layer(3, 1, 16, 8, seed=928)
    worker = _worker(layer)
    worker.receive_job_blocks({1: [(_key(1), [0]), (_key(2), [block])]})
    with pytest.raises(IndexError, match="outside the KV cache"):
        worker._build_durable_summaries_for_job(1)
    assert not worker.durable_summaries
    with pytest.raises(IndexError, match="outside the KV cache"):
        worker._build_summaries_body([block])
    assert not worker.summaries["probe"]


def test_mean_only_backend_shape_mismatch_fails_closed():
    layer = _make_layer(3, 1, 16, 8, seed=929)
    layer.attn_backend.get_kv_cache_shape = lambda *args: (1, 2, 3)
    worker = _worker(layer)
    worker.receive_job_blocks({1: [(_key(1), [0])]})
    with pytest.raises(AssertionError, match="does not match"):
        worker._build_durable_summaries_for_job(1)
    assert not worker.durable_summaries


@pytest.mark.parametrize("shape", [(2, 16, 8), (2, 0, 16, 8), (2, 3, 0, 8), (2, 3, 16, 0)])
def test_mean_only_rejects_invalid_reduction_shape(shape):
    with pytest.raises(ValueError, match="positive inner dimensions"):
        SemanticOffloadingWorker._summarize_mean_key_batches(torch.empty(shape))


def test_mean_only_reductions_do_not_evaluate_unused_fields():
    keys = torch.randn(3, 2, 16, 8)
    with patch.object(torch.Tensor, "amin", side_effect=AssertionError("unused min")), \
         patch.object(torch.Tensor, "amax", side_effect=AssertionError("unused max")), \
         patch.object(torch.Tensor, "abs", side_effect=AssertionError("unused MAD")):
        groups = SemanticOffloadingWorker._summarize_mean_key_batches(keys)
    assert len(groups) == 3 and len(groups[0]) == 2
    assert torch.equal(groups[1][1].mean, keys[1, 1].mean(0))


@pytest.mark.parametrize("method,schedule", [("minmax", "opportunity"), ("cuboid_mean", "opportunity"),
    ("minmax", "continuous"), ("cuboid_mean", "continuous"), ("mean", "capture_only")])
def test_required_fields_reject_unsupported_modes_before_allocation(monkeypatch, method, schedule):
    monkeypatch.setenv("SEMANTIC_SUMMARY_FIELDS", "required")
    monkeypatch.setattr("semantic_offload.worker.init_cpu_offloading_worker_base",
                        lambda *args, **kwargs: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="require mean scoring"):
        SemanticOffloadingWorker(None, 1, 1, None, method=method, scoring_schedule=schedule)


@pytest.mark.parametrize("setting", ["", "mean", "REQUIRED"])
def test_required_fields_reject_invalid_environment(monkeypatch, setting):
    monkeypatch.setenv("SEMANTIC_SUMMARY_FIELDS", setting)
    with pytest.raises(ValueError, match="must be all or required"):
        SemanticOffloadingWorker(None, 1, 1, None)


@pytest.mark.parametrize("schedule", ["continuous", "opportunity"])
def test_required_fields_constructor_acceptance_and_default_full_payload(monkeypatch, schedule):
    monkeypatch.setattr("semantic_offload.worker.init_cpu_offloading_worker_base", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = SimpleNamespace(
        use_v2_model_runner=False,
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, world_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
        model_config=SimpleNamespace(use_mla=False, get_total_num_kv_heads=lambda: 1),
        cache_config=SimpleNamespace(block_size=16),
    )
    monkeypatch.setenv("SEMANTIC_SUMMARY_FIELDS", "required")
    worker = SemanticOffloadingWorker(None, 1, 1, config, scoring_schedule=schedule, probe_layer=None)
    assert worker._summary_fields == "required"
    monkeypatch.delenv("SEMANTIC_SUMMARY_FIELDS")
    original = SemanticOffloadingWorker(None, 1, 1, config, probe_layer=None)
    assert original._summary_fields == "all"


def test_mean_only_rollback_waits_for_fence_and_preserves_newer_publication():
    layer = _make_layer(3, 1, 16, 8, seed=930)
    worker = _worker(layer)
    worker._summary_job_events = {}
    worker.receive_job_blocks({1: [(_key(1), [0])]})
    worker._build_durable_summaries_for_job(1)
    worker.receive_job_blocks({2: [(_key(1), [1])]})
    worker._build_durable_summaries_for_job(2)
    current = worker.durable_summaries[_key(1)]
    synchronized = []
    worker._summary_job_events[1] = SimpleNamespace(synchronize=lambda: synchronized.append(1))
    worker._rollback_summary_job(1)
    assert synchronized == [1]
    assert worker.durable_summaries[_key(1)] is current
    worker._summary_job_events[2] = SimpleNamespace(synchronize=lambda: synchronized.append(2))
    worker._rollback_summary_job(2)
    assert synchronized == [1, 2]
    assert _key(1) not in worker.durable_summaries


@pytest.mark.parametrize("schedule", ["continuous", "opportunity"])
def test_mean_only_async_projection_and_scores_equal_full_payload(schedule):
    from tests.test_async_scoring import _worker as make_async_worker, _wait_async_observations
    expected = None
    for required in (False, True):
        worker = make_async_worker("asynchronous")
        worker._scoring_schedule = schedule
        worker._query_snapshot_max_age_ns = 1_000_000_000
        if required:
            worker._summary_fields = "required"
            worker.durable_summaries = {
                key: [_MeanOnlySummary(row.mean) for row in rows]
                for key, rows in worker.durable_summaries.items()
            }
        try:
            worker._score_queries(["req"], torch.ones(1, 1, 4))
            pending = next(iter(worker._async_scoring._pending.values()))
            projected = pending.snapshot.payload.summaries
            assert all(hasattr(row, "mean") and not hasattr(row, "min")
                       for rows in projected for row in rows)
            observations = _wait_async_observations(worker)
            assert len(observations) == 1
            scores = observations[0]["scores"]
            if expected is None:
                expected = scores
            else:
                assert scores == expected
        finally:
            worker._async_scoring.shutdown()


@pytest.mark.parametrize("request_count", [1, 3])
def test_continuous_mean_only_online_scores_and_transport_equal_full_payload(request_count):
    from tests.test_opportunity_metadata import _connector
    layer = _make_layer(8, 2, 16, 8, seed=414)
    queries = torch.randn(request_count, 2, 8, generator=torch.Generator().manual_seed(15))
    req_ids = [f"req-{index}" for index in range(request_count)]
    expected = None
    for fields in ("all", "required"):
        worker = _worker(layer, fields, schedule="continuous")
        worker.receive_job_blocks({1: [(_key(1), [3]), (_key(2), [0, 1]), (_key(3), [4])]})
        worker._build_durable_summaries_for_job(1)
        worker._on_queries_captured(req_ids, queries)
        assert set(worker._pending_scores["mean"]) == set(req_ids)
        assert worker._pending_score_head_counts["mean"] == dict.fromkeys(req_ids, 2)
        assert not worker._query_snapshots
        assert not worker._pending_async_score_observations
        metadata = _connector(worker).build_connector_worker_meta()
        assert metadata.complete_async_score_observations() == []
        actual = (metadata.complete_scores(), metadata.complete_relevance_updates(),
                  metadata.complete_latest_relevance(), metadata.score_head_counts,
                  metadata.score_reduction)
        if request_count == 1:
            assert actual[0] and not actual[1]
        else:
            assert actual[1] and not actual[0]
        if expected is None:
            expected = actual
        else:
            assert actual == expected
        # A later capture remains live; metadata extraction must not turn the
        # reduced payload into a one-shot or deferred-only scorer.
        worker._on_queries_captured(req_ids, -queries)
        assert set(worker._pending_scores["mean"]) == set(req_ids)
