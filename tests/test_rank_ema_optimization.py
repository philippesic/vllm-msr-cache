# SPDX-License-Identifier: Apache-2.0
"""Differential coverage for the opt-in rank/EMA hot path."""

from copy import deepcopy
from math import isnan

import pytest
import torch
from vllm.v1.kv_offload.base import make_offload_key

from semantic_offload.connector import SemanticWorkerMetadata
from semantic_offload.index import BlockSummary
from semantic_offload.manager import (
    SemanticOffloadingManager,
    compose_relevance_updates,
)
from semantic_offload.worker import (
    SemanticOffloadingWorker,
    validate_rank_ema_mode,
)


def _key(value: int):
    return make_offload_key(str(value).encode(), 0)


def _summary(value: float, dim: int = 4) -> BlockSummary:
    vector = torch.full((dim,), value)
    return BlockSummary(min=vector, max=vector, mean=vector, mad=torch.zeros(dim))


def _make_worker(mode: str) -> SemanticOffloadingWorker:
    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    worker._probe_layer_name = "layer0"
    worker._summary_stream = None
    worker.summaries = {"layer0": {}}
    worker._pending_job_keys = {}
    worker._pending_job_blocks = {}
    worker.durable_summaries = {}
    worker._max_durable_summaries = 10_000
    worker._pending_scores = {}
    worker._pending_score_head_counts = {}
    worker._method = "mean"
    worker._head_aggregation = "mean"
    worker._rank_ema_mode = mode
    worker._stack_cache_dirty = True
    worker._stack_cache_keys = []
    worker._stack_cache = {}
    worker._stack_cache_index = {}
    worker._stack_pending_insert = set()
    worker._stack_pending_remove = set()
    return worker


def _seed_worker(worker, values: dict) -> None:
    worker.durable_summaries.update(
        {key: [_summary(value)] for key, value in values.items()}
    )
    worker._mark_inserted_into_stack_cache(values)


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_fast_worker_ranking_matches_default_for_ties_and_batches(batch_size):
    # Deliberately insert keys out of byte order. The legacy path's tie rule is
    # key bytes, not insertion order, and the fast path must retain it exactly.
    values = {
        _key(10): 2.0,
        _key(2): 2.0,
        _key(1): 2.0,
        _key(30): -1.0,
        _key(4): 1.0,
    }
    query = torch.stack([torch.ones(1, 4) * (index + 1) for index in range(batch_size)])
    baseline = _make_worker("default")
    fast = _make_worker("fast")
    _seed_worker(baseline, values)
    _seed_worker(fast, values)

    baseline._on_queries_captured([f"req-{i}" for i in range(batch_size)], query)
    fast._on_queries_captured([f"req-{i}" for i in range(batch_size)], query)

    assert fast.pop_pending_score_metadata() == baseline.pop_pending_score_metadata()


def test_fast_nonfinite_fallback_preserves_legacy_cache_row_order():
    # Python's ordering of NaN keys is input-order-sensitive. The fast path
    # must therefore retain the original cache rows for its non-finite
    # fallback, even though finite scores use a key-ordered tensor view.
    values = {_key(2): float("nan"), _key(1): 1.0, _key(3): 2.0}
    query = torch.ones(1, 1, 4)
    baseline = _make_worker("default")
    fast = _make_worker("fast")
    _seed_worker(baseline, values)
    _seed_worker(fast, values)

    baseline._on_queries_captured(["req"], query)
    fast._on_queries_captured(["req"], query)
    baseline_ranked = baseline.pop_pending_scores()["mean"]["req"]
    fast_ranked = fast.pop_pending_scores()["mean"]["req"]

    assert [key for key, _ in fast_ranked] == [key for key, _ in baseline_ranked]
    for (_, baseline_score), (_, fast_score) in zip(baseline_ranked, fast_ranked):
        if isnan(baseline_score):
            assert isnan(fast_score)
        else:
            assert fast_score == baseline_score


def test_fast_ema_composition_matches_default_with_seeded_state_and_first_observations():
    keys = [_key(10), _key(2), _key(1), _key(30)]
    scores = {
        "mean": {
            "req-a": [(keys[0], 0.5), (keys[1], 0.5), (keys[2], -1.0)],
            "req-b": [(keys[3], 2.0), (keys[0], 1.0)],
            "req-c": [(keys[2], 3.0), (keys[1], 0.0), (keys[3], -2.0)],
        }
    }
    initial = {"mean": {keys[0]: 0.125, keys[3]: -0.75}}
    baseline = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    baseline.relevance_ema = deepcopy(initial)
    fast = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    fast.relevance_ema = deepcopy(initial)

    baseline.update_relevance(scores)
    fast.apply_relevance_updates(compose_relevance_updates(scores, "fast"))

    assert fast.relevance_ema == baseline.relevance_ema
    assert fast.ranked_keys("mean") == baseline.ranked_keys("mean")


def _tp_metadata(
    mode: str, values: list[tuple[object, float]]
) -> SemanticWorkerMetadata:
    return SemanticWorkerMetadata(
        pending_scores={"mean": {"req": values}},
        score_head_counts={"mean": {"req": 1}},
        score_worker_counts={"mean": {"req": 1}},
        score_reduction="mean",
        score_group_size=2,
        score_contributors=1,
        rank_ema_mode=mode,
    )


def test_fast_rank_ema_mode_preserves_supported_tp_aggregation_and_global_ema():
    keys = [_key(10), _key(2), _key(1)]
    left = [(keys[1], 2.0), (keys[0], 1.0), (keys[2], 0.0)]
    right = [(keys[2], 2.0), (keys[0], 3.0), (keys[1], 0.0)]
    default = _tp_metadata("default", left).aggregate(_tp_metadata("default", right))
    fast = _tp_metadata("fast", left).aggregate(_tp_metadata("fast", right))

    assert fast.rank_ema_mode == "fast"
    assert fast.complete_scores() == default.complete_scores()

    initial = {"mean": {keys[0]: 0.25}}
    default_manager = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    default_manager.relevance_ema = deepcopy(initial)
    fast_manager = SemanticOffloadingManager.__new__(SemanticOffloadingManager)
    fast_manager.relevance_ema = deepcopy(initial)
    default_scores = default.complete_scores()
    fast_scores = fast.complete_scores()
    default_manager.update_relevance(default_scores)
    fast_manager.apply_relevance_updates(
        compose_relevance_updates(fast_scores, fast.rank_ema_mode)
    )

    assert fast_manager.relevance_ema == default_manager.relevance_ema
    assert fast_manager.ranked_keys("mean") == default_manager.ranked_keys("mean")


def test_rank_ema_mode_validation_is_opt_in_and_explicit():
    validate_rank_ema_mode("default")
    validate_rank_ema_mode("fast")
    with pytest.raises(ValueError, match="rank_ema_mode"):
        validate_rank_ema_mode("unknown")


def test_metadata_rejects_mixed_rank_ema_modes():
    with pytest.raises(ValueError, match="rank_ema_mode"):
        _tp_metadata("default", [(_key(1), 1.0)]).aggregate(
            _tp_metadata("fast", [(_key(1), 1.0)])
        )


def test_metadata_rejects_mixed_negative_evidence_floors():
    left = _tp_metadata("default", [(_key(1), 1.0)])
    right = _tp_metadata("default", [(_key(1), 1.0)])
    left.ema_rank_weight_floor = 0.01
    right.ema_rank_weight_floor = 0.03

    with pytest.raises(ValueError, match="ema_rank_weight_floor"):
        left.aggregate(right)
