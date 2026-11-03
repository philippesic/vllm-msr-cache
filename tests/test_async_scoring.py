# SPDX-License-Identifier: Apache-2.0

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.outputs import KVConnectorOutput

from semantic_offload.async_scoring import AsyncScoringController
from semantic_offload.connector import (
    SemanticOffloadingConnector,
    SemanticWorkerMetadata,
)
from semantic_offload.index import BlockSummary
from semantic_offload.manager import SemanticOffloadingManager
from semantic_offload.worker import SemanticOffloadingWorker


class Fence:
    def __init__(self):
        self.event = threading.Event()

    def synchronize(self):
        self.event.wait(timeout=2)


def _summary(value: float) -> BlockSummary:
    tensor = torch.full((4,), value)
    return BlockSummary(tensor, tensor, tensor, torch.zeros(4))


def _key(value: int):
    return make_offload_key(str(value).encode(), 0)


def _worker(mode: str) -> SemanticOffloadingWorker:
    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    keys = (_key(1), _key(2))
    worker._method = "mean"
    worker._head_aggregation = "mean"
    worker._rank_ema_mode = "default"
    worker._scoring_mode = mode
    worker._summary_generation = 0
    worker._summary_stream = None
    worker._async_scoring_stream = None
    worker._async_request_versions = {}
    worker._async_submission_requests = {}
    worker._pending_async_score_observations = []
    worker._async_scoring_max_snapshot_age_ns = 1_000_000_000
    worker.summaries = {}
    worker._pending_job_keys = {}
    worker._pending_job_blocks = {}
    worker.durable_summaries = {keys[0]: [_summary(2.0)], keys[1]: [_summary(-1.0)]}
    worker._durable_summary_publications = {
        keys[0]: (0, 0, 1),
        keys[1]: (0, 0, 2),
    }
    worker._pending_scores = {}
    worker._pending_score_head_counts = {}
    worker._stack_cache_dirty = True
    worker._stack_cache_keys = []
    worker._stack_cache = {}
    worker._stack_cache_index = {}
    worker._stack_pending_insert = set(keys)
    worker._stack_pending_remove = set()
    worker._async_scoring = (
        AsyncScoringController(
            worker._score_async_payload,
            max_queue_items=2,
            max_snapshot_bytes=4096,
        )
        if mode == "asynchronous"
        else None
    )
    return worker


def _wait_scores(worker: SemanticOffloadingWorker):
    if worker._scoring_mode == "asynchronous":
        observations = _wait_async_observations(worker)
        return observations[0]["scores"] if observations else {}
    for _ in range(1000):
        scores = worker.pop_pending_scores()
        if scores:
            return scores
        time.sleep(0.001)
    return {}


def _wait_async_observations(worker: SemanticOffloadingWorker):
    for _ in range(1000):
        observations = worker.pop_async_score_observations()
        if observations:
            return observations
        if worker._async_scoring.stats()["pending_items"] == 0:
            return []
        time.sleep(0.001)
    return []


def test_completed_later_observation_waits_for_ordered_publication():
    first_fence = Fence()
    controller = AsyncScoringController(
        lambda value: value * 2,
        max_queue_items=2,
        max_snapshot_bytes=16,
        workers=2,
    )
    controller.submit(
        1,
        generation=0,
        snapshot_bytes=4,
        key_versions={},
        observation_available_ns=1,
        readiness_fences=[first_fence],
    )
    controller.submit(
        2,
        generation=0,
        snapshot_bytes=4,
        key_versions={},
        observation_available_ns=2,
    )
    assert controller.drain_ready() == ([], [])

    first_fence.event.set()
    for _ in range(1000):
        completions, failures = controller.drain_ready()
        if completions:
            break
        time.sleep(0.001)
    assert failures == []
    assert [item.snapshot.observation_sequence for item in completions] == [0, 1]
    assert [item.result for item in completions] == [2, 4]
    controller.shutdown()


def test_drop_newest_enforces_item_and_byte_bounds():
    fence = Fence()
    controller = AsyncScoringController(
        lambda value: value,
        max_queue_items=1,
        max_snapshot_bytes=4,
    )
    assert controller.submit(
        "kept",
        generation=0,
        snapshot_bytes=4,
        key_versions={},
        observation_available_ns=1,
        readiness_fences=[fence],
    ) == 0
    assert controller.submit(
        "dropped",
        generation=0,
        snapshot_bytes=1,
        key_versions={},
        observation_available_ns=2,
    ) is None
    assert controller.stats()["pending_snapshot_bytes"] == 4
    assert any(
        record["event"] == "submission_dropped"
        for record in controller.pop_telemetry()
    )
    fence.event.set()
    controller.shutdown()


def test_reset_waits_for_fence_then_discards_owned_snapshot():
    fence = Fence()
    controller = AsyncScoringController(
        lambda value: value,
        max_queue_items=1,
        max_snapshot_bytes=4,
    )
    controller.submit(
        "value",
        generation=0,
        snapshot_bytes=4,
        key_versions={"key": "version"},
        observation_available_ns=1,
        readiness_fences=[fence],
    )
    reset_done = threading.Event()

    def reset():
        controller.reset()
        reset_done.set()

    thread = threading.Thread(target=reset)
    thread.start()
    assert not reset_done.wait(timeout=0.01)
    fence.event.set()
    thread.join(timeout=2)
    assert reset_done.is_set()
    assert controller.stats()["pending_items"] == 0
    controller.shutdown()


def test_reset_rejects_new_submissions_until_owned_readers_finish():
    fence = Fence()
    controller = AsyncScoringController(
        lambda value: value,
        max_queue_items=2,
        max_snapshot_bytes=8,
    )
    controller.submit(
        "owned",
        generation=0,
        snapshot_bytes=4,
        key_versions={},
        observation_available_ns=1,
        readiness_fences=[fence],
    )
    reset_thread = threading.Thread(target=controller.reset)
    reset_thread.start()
    for _ in range(1000):
        if controller.stats()["resetting"]:
            break
        time.sleep(0.001)
    assert controller.submit(
        "late",
        generation=0,
        snapshot_bytes=1,
        key_versions={},
        observation_available_ns=2,
    ) is None
    assert any(
        event.get("reason") == "controller_resetting"
        for event in controller.pop_telemetry()
    )
    fence.event.set()
    reset_thread.join(timeout=2)
    assert not reset_thread.is_alive()
    assert controller.stats()["pending_items"] == 0
    controller.shutdown()


def test_exception_is_reported_and_does_not_block_later_publication():
    def score(value):
        if value == "bad":
            raise RuntimeError("boom")
        return value

    controller = AsyncScoringController(
        score, max_queue_items=2, max_snapshot_bytes=8, workers=1
    )
    controller.submit(
        "bad",
        generation=0,
        snapshot_bytes=1,
        key_versions={},
        observation_available_ns=1,
    )
    controller.submit(
        "good",
        generation=0,
        snapshot_bytes=1,
        key_versions={},
        observation_available_ns=2,
    )
    all_completions = []
    all_failures = []
    for _ in range(1000):
        completions, failures = controller.drain_ready()
        all_completions.extend(completions)
        all_failures.extend(failures)
        if completions:
            break
        time.sleep(0.001)
    assert [completion.result for completion in all_completions] == ["good"]
    assert all_failures[0]["exception_type"] == "RuntimeError"
    controller.shutdown()


def test_shutdown_rejects_new_work_and_telemetry_is_bounded():
    controller = AsyncScoringController(
        lambda value: value,
        max_queue_items=1,
        max_snapshot_bytes=1,
        telemetry_limit=3,
    )
    controller.shutdown()
    for value in range(10):
        assert controller.submit(
            value,
            generation=0,
            snapshot_bytes=1,
            key_versions={},
            observation_available_ns=value,
        ) is None
    assert controller.stats()["telemetry_items"] == 3


def test_drain_all_waits_for_final_owned_snapshot():
    controller = AsyncScoringController(
        lambda value: value + 1,
        max_queue_items=2,
        max_snapshot_bytes=1024,
        workers=1,
    )
    sequence = controller.submit(
        41,
        generation=0,
        snapshot_bytes=16,
        key_versions={},
        observation_available_ns=time.monotonic_ns(),
    )
    assert sequence == 0
    completions, failures = controller.drain_all()
    assert failures == []
    assert [item.result for item in completions] == [42]
    assert controller.stats()["pending_items"] == 0
    assert controller.stats()["pending_snapshot_bytes"] == 0
    controller.shutdown()


def test_worker_async_snapshot_is_query_immutable_and_matches_sync_formula():
    async_worker = _worker("asynchronous")
    sync_worker = _worker("synchronous")
    query = torch.ones(1, 4)

    async_worker._on_query_captured("req", query)
    query.fill_(-100.0)
    sync_worker._on_query_captured("req", torch.ones(1, 4))

    async_scores = _wait_scores(async_worker)
    sync_scores = sync_worker.pop_pending_scores()
    assert async_scores == sync_scores
    assert async_scores["mean"]["req"][0][0] == _key(1)
    publication = next(
        event
        for event in async_worker.pop_async_scoring_telemetry()
        if event.get("event") == "published"
    )
    assert publication["observation_available_ns"] <= publication["publication_ns"]
    async_worker._async_scoring.shutdown()


def test_worker_async_publication_rejects_evicted_reinserted_version():
    worker = _worker("asynchronous")
    key = _key(1)
    worker._on_query_captured("req", torch.ones(1, 4))
    worker.durable_summaries[key] = [_summary(100.0)]
    worker._durable_summary_publications[key] = (1, 0, 99)

    scores = _wait_scores(worker)

    assert key not in dict(scores["mean"]["req"])
    assert _key(2) in dict(scores["mean"]["req"])
    worker._async_scoring.shutdown()


def test_worker_async_publication_rejects_stale_generation():
    worker = _worker("asynchronous")
    worker._on_query_captured("req", torch.ones(1, 4))
    worker._summary_generation = 1

    assert _wait_scores(worker) == {}
    assert any(
        event.get("reason") == "stale_generation"
        for event in worker.pop_async_scoring_telemetry()
    )
    worker._async_scoring.shutdown()


def test_worker_retirement_preserves_eventual_final_snapshot_without_global_drain():
    worker = _worker("asynchronous")
    worker._on_query_captured("req", torch.ones(1, 4))
    worker.retire_async_requests(["req"])

    observations = _wait_async_observations(worker)
    assert observations[0]["request_ids"] == ["req"]
    assert worker._async_scoring.stats()["pending_items"] == 0
    assert worker._async_scoring.stats()["pending_snapshot_bytes"] == 0
    assert worker._async_request_versions == {}
    assert any(
        event.get("event") == "published"
        for event in worker.pop_async_scoring_telemetry()
    )
    worker._async_scoring.shutdown()


def test_worker_reset_clears_async_request_and_submission_state():
    worker = _worker("asynchronous")
    worker._async_request_versions["req"] = 1
    worker._async_submission_requests[9] = ("req",)

    worker.reset_semantic_state()

    assert worker._async_request_versions == {}
    assert worker._async_submission_requests == {}
    assert worker._pending_async_score_observations == []
    worker._async_scoring.shutdown()


def test_worker_expires_delayed_snapshot_without_releasing_reader_early():
    worker = _worker("asynchronous")
    worker._async_scoring.shutdown()
    gate = threading.Event()
    scorer = worker._score_async_payload

    def delayed(payload):
        gate.wait(timeout=2)
        return scorer(payload)

    worker._async_scoring = AsyncScoringController(
        delayed, max_queue_items=2, max_snapshot_bytes=4096
    )
    worker._async_scoring_max_snapshot_age_ns = 1
    worker._on_query_captured("req", torch.ones(1, 4))
    assert worker._async_scoring.stats()["pending_items"] == 1
    time.sleep(0.001)
    gate.set()

    assert _wait_async_observations(worker) == []
    assert worker._async_scoring.stats()["pending_items"] == 0
    assert any(
        event.get("reason") == "snapshot_expired"
        for event in worker.pop_async_scoring_telemetry()
    )
    worker._async_scoring.shutdown()


def test_connector_applies_same_request_observations_in_order(monkeypatch, capsys):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    key = _key(1)
    observations = [
        {
            "observation_sequence": 0,
            "request_ids": ["req"],
            "scores": {"mean": {"req": [(key, 1.0)]}},
            "observation_available_ns": 1,
            "expires_at_ns": time.monotonic_ns() + 1_000_000_000,
        },
        {
            "observation_sequence": 1,
            "request_ids": ["req"],
            "scores": {"mean": {"req": [(key, 0.0)]}},
            "observation_available_ns": 2,
            "expires_at_ns": time.monotonic_ns() + 1_000_000_000,
        },
    ]
    expected = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    expected.relevance_ema = {}
    for observation in observations:
        expected.update_relevance(observation["scores"])
    actual = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    actual.relevance_ema = {}
    connector = SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
    connector.connector_scheduler = SimpleNamespace(
        manager=actual,
        _preservation_requests={},
        _preservation_finished={},
    )
    monkeypatch.setattr(
        OffloadingConnector, "update_connector_output", lambda self, output: None
    )
    metadata = SemanticWorkerMetadata(
        async_score_observations=observations,
        score_group_size=1,
        score_contributors=1,
    )

    connector.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=metadata)
    )

    assert actual.relevance_ema["mean"] == pytest.approx(
        expected.relevance_ema["mean"]
    )
    assert actual.relevance_ema["mean"][key] == pytest.approx(0.7)
    output = capsys.readouterr().out
    assert output.count('"event":"consumed"') == 2


def test_worker_preserves_two_ready_observations_for_same_request():
    worker = _worker("asynchronous")
    worker._on_query_captured("req", torch.ones(1, 4))
    worker._on_query_captured("req", -torch.ones(1, 4))
    for pending in list(worker._async_scoring._pending.values()):
        pending.future.result(timeout=2)

    observations = worker.pop_async_score_observations()

    assert [item["observation_sequence"] for item in observations] == [0, 1]
    assert len(observations) == 2
    assert observations[0]["scores"] != observations[1]["scores"]
    published = [
        event
        for event in worker.pop_async_scoring_telemetry()
        if event.get("event") == "published"
    ]
    assert [event["observation_sequence"] for event in published] == [0, 1]
    worker._async_scoring.shutdown()


def test_connector_rejects_observation_expired_before_consumption(
    monkeypatch, capsys
):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    manager = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    manager.relevance_ema = {}
    connector = SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
    connector.connector_scheduler = SimpleNamespace(
        manager=manager,
        _preservation_requests={},
        _preservation_finished={},
    )
    monkeypatch.setattr(
        OffloadingConnector, "update_connector_output", lambda self, output: None
    )
    metadata = SemanticWorkerMetadata(
        async_score_observations=[
            {
                "observation_sequence": 0,
                "request_ids": ["req"],
                "scores": {"mean": {"req": [(_key(1), 1.0)]}},
                "observation_available_ns": 0,
                "expires_at_ns": 0,
            }
        ],
        score_group_size=1,
        score_contributors=1,
    )

    connector.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=metadata)
    )

    assert manager.relevance_ema == {}
    assert "snapshot_expired_at_consumer" in capsys.readouterr().out


def test_worker_drops_before_snapshot_capture_when_byte_bound_is_full():
    worker = _worker("asynchronous")
    worker._async_scoring.shutdown()
    worker._async_scoring = AsyncScoringController(
        worker._score_async_payload,
        max_queue_items=1,
        max_snapshot_bytes=1,
    )

    worker._on_query_captured("req", torch.ones(1, 4))

    assert worker.pop_pending_scores() == {}
    assert worker._async_scoring.stats()["pending_items"] == 0
    assert any(
        event.get("reason") == "drop_newest_before_snapshot_capture"
        for event in worker.pop_async_scoring_telemetry()
    )
    worker._async_scoring.shutdown()


def test_spec_async_controls_are_opt_in_bounded_and_require_events():
    from semantic_offload import spec as spec_module

    captured = {}

    class FakeWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    spec = spec_module.SemanticOffloadingSpec.__new__(
        spec_module.SemanticOffloadingSpec
    )
    spec.extra_config = {
        "scoring_mode": "asynchronous",
        "preservation_events": True,
        "async_scoring_max_queue_items": 3,
        "async_scoring_max_snapshot_bytes": 4096,
        "async_scoring_workers": 2,
        "async_scoring_max_snapshot_age_ms": 250,
    }
    spec.tokens_per_block = [16]
    spec.num_blocks = 4
    spec.vllm_config = object()
    with (
        patch.object(spec_module, "SemanticOffloadingWorker", FakeWorker),
        patch.object(spec_module, "spec_blocks_per_chunk", return_value=1),
    ):
        spec.create_worker(None)
    assert captured["scoring_mode"] == "asynchronous"
    assert captured["async_scoring_max_queue_items"] == 3
    assert captured["async_scoring_max_snapshot_bytes"] == 4096
    assert captured["async_scoring_max_snapshot_age_ms"] == 250

    spec.extra_config = {"scoring_mode": "asynchronous"}
    with pytest.raises(ValueError, match="requires preservation_events"):
        spec.create_worker(None)


def test_spec_async_limit_rejects_bool_and_nonpositive_values():
    from semantic_offload import spec as spec_module

    for value in (True, 0, -1, 1.5):
        spec = spec_module.SemanticOffloadingSpec.__new__(
            spec_module.SemanticOffloadingSpec
        )
        spec.extra_config = {"async_scoring_max_queue_items": value}
        spec.num_blocks = 4
        spec.vllm_config = object()
        with pytest.raises(ValueError, match="positive JSON integer"):
            spec.create_worker(None)


def test_spec_async_snapshot_age_rejects_bool_and_nonpositive_values():
    from semantic_offload import spec as spec_module

    for value in (True, 0, -1, 1.5):
        spec = spec_module.SemanticOffloadingSpec.__new__(
            spec_module.SemanticOffloadingSpec
        )
        spec.extra_config = {"async_scoring_max_snapshot_age_ms": value}
        spec.num_blocks = 4
        spec.vllm_config = object()
        with pytest.raises(ValueError, match="positive JSON integer"):
            spec.create_worker(None)


def test_async_metadata_events_aggregate_only_with_matching_status():
    status = {
        "schema": "semantic-async-scoring-status-v1",
        "pending_items": 0,
        "pending_snapshot_bytes": 0,
    }
    left = SemanticWorkerMetadata(
        async_scoring_events=[{"event": "submitted", "observation_sequence": 0}],
        async_scoring_status=status,
    )
    right = SemanticWorkerMetadata(
        async_scoring_events=[{"event": "published", "observation_sequence": 0}],
        async_scoring_status=status,
    )

    merged = left.aggregate(right)

    assert [event["event"] for event in merged.async_scoring_events] == [
        "submitted",
        "published",
    ]
    conflicting = SemanticWorkerMetadata(
        async_scoring_status={**status, "pending_items": 1}
    )
    with pytest.raises(ValueError, match="async scoring statuses"):
        left.aggregate(conflicting)
