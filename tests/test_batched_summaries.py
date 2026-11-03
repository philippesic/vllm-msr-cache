# SPDX-License-Identifier: Apache-2.0
"""Batched summary equivalence and key/storage ownership checks."""
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_offload.base import make_offload_key

from semantic_offload.index import build_summary
from tests.test_step_1_2_summaries import _make_layer, _make_worker


def _key(value):
    return make_offload_key(str(value).encode(), 0)


def _reference(layer, block_ids):
    keys = layer.kv_cache[block_ids, ..., :layer.head_size].float()
    keys = keys.permute(1, 0, 2, 3).reshape(layer.num_kv_heads, -1, layer.head_size)
    return [build_summary(head) for head in keys]


def _assert_summary_equal(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        for name in ("min", "max", "mean", "mad"):
            value = getattr(a, name)
            assert value.dtype == torch.float32
            torch.testing.assert_close(value, getattr(b, name), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_variable_layout_lengths_heads_order_and_owned_storage(dtype):
    layer = _make_layer(12, 3, 16, 32, seed=923)
    layer.kv_cache = layer.kv_cache.to(dtype)
    worker = _make_worker({"probe": layer}, block_size=16)
    layout = [(_key(1), [7, 2]), (_key(2), [0]), (_key(3), [5, 5, 1]),
              (_key(4), []), (_key(5), [4]), (_key(6), [9, 3])]
    expected = {key: _reference(layer, ids) for key, ids in layout if ids}
    worker.receive_job_blocks({19: layout})
    with patch("torch.as_tensor", side_effect=AssertionError("index upload forbidden")), \
         patch("torch.stack", wraps=torch.stack) as stack:
        worker._build_durable_summaries_for_job(19)
    assert stack.call_count == 1
    assert list(worker.durable_summaries) == list(expected)
    assert 19 not in worker._pending_job_blocks
    layer.kv_cache.fill_(999)  # Later GPU block reuse cannot alter summaries.
    for key, values in expected.items():
        _assert_summary_equal(worker.durable_summaries[key], values)
    # Deliberately wrong block attribution is observably different.
    assert not torch.equal(expected[_key(1)][0].mean, expected[_key(2)][0].mean)
    assert not torch.equal(expected[_key(1)][0].mean, expected[_key(1)][1].mean)


def test_duplicate_key_last_write_and_original_insertion_order():
    layer = _make_layer(7, 2, 16, 8, seed=4)
    worker = _make_worker({"probe": layer}, block_size=16)
    layout = [(_key(1), [0]), (_key(2), [2, 3]), (_key(1), [5, 6])]
    worker._pending_job_blocks[11] = layout
    worker._build_durable_summaries_for_job(11)
    assert list(worker.durable_summaries) == [_key(1), _key(2)]
    _assert_summary_equal(worker.durable_summaries[_key(1)], _reference(layer, [5, 6]))
    _assert_summary_equal(worker.durable_summaries[_key(2)], _reference(layer, [2, 3]))


def test_legacy_blocks_are_batched_without_cross_block_reduction():
    layer = _make_layer(8, 2, 16, 32, seed=123)
    worker = _make_worker({"probe": layer}, block_size=16)
    ids = [5, 0, 7, 5]
    expected = {block: _reference(layer, [block]) for block in ids}
    with patch("torch.stack", wraps=torch.stack) as stack:
        worker._build_summaries_body(ids)
    assert stack.call_count == 1
    layer.kv_cache.zero_()
    assert list(worker.summaries["probe"]) == [5, 0, 7]
    for block, values in expected.items():
        _assert_summary_equal(worker.summaries["probe"][block], values)


def test_empty_layout_does_not_gather_or_publish():
    layer = _make_layer(2, 1, 16, 8, seed=3)
    worker = _make_worker({"probe": layer}, block_size=16)
    worker._pending_job_blocks[7] = [(_key(1), [])]
    with patch("torch.stack", side_effect=AssertionError("must not gather")):
        worker._build_durable_summaries_for_job(7)
        worker._build_summaries_body([])
    assert worker.durable_summaries == {}
    assert worker.summaries["probe"] == {}


def test_pruning_uses_original_layout_order_and_preserves_remaining_values():
    layer = _make_layer(5, 1, 16, 8, seed=32)
    worker = _make_worker({"probe": layer}, block_size=16)
    worker._max_durable_summaries = 2
    worker._pending_job_blocks[12] = [(_key(1), [0]), (_key(2), [1, 2]), (_key(3), [3])]
    worker._build_durable_summaries_for_job(12)
    assert list(worker.durable_summaries) == [_key(2), _key(3)]
    _assert_summary_equal(worker.durable_summaries[_key(2)], _reference(layer, [1, 2]))
    _assert_summary_equal(worker.durable_summaries[_key(3)], _reference(layer, [3]))


@pytest.mark.parametrize("block_id", [-1, 3])
def test_invalid_block_id_cannot_alias_another_key(block_id):
    layer = _make_layer(3, 1, 16, 8, seed=22)
    worker = _make_worker({"probe": layer}, block_size=16)
    worker._pending_job_blocks[17] = [(_key(1), [0]), (_key(2), [block_id])]
    with pytest.raises(IndexError, match="outside the KV cache"):
        worker._build_durable_summaries_for_job(17)
    assert worker.durable_summaries == {}
    with pytest.raises(IndexError, match="outside the KV cache"):
        worker._build_summaries_body([block_id])
    assert worker.summaries["probe"] == {}


def test_summary_view_creation_does_not_dispatch_per_head_selects():
    from torch.utils._python_dispatch import TorchDispatchMode
    from semantic_offload.worker import SemanticOffloadingWorker

    class CountSelects(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.selects = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.selects += func is torch.ops.aten.select.int
            return func(*args, **(kwargs or {}))

    keys = torch.randn(17, 3, 16, 8, generator=torch.Generator().manual_seed(1103))
    with CountSelects() as mode:
        summaries = SemanticOffloadingWorker._summarize_key_batches(keys)
    assert mode.selects == 0
    assert len(summaries) == 17
    for row, group in enumerate(summaries):
        _assert_summary_equal(group, [build_summary(head) for head in keys[row]])


def _cache_worker(method):
    from semantic_offload.worker import SemanticOffloadingWorker
    worker = _make_worker({}, block_size=16)
    worker._method = method
    keys = torch.randn(7, 3, 16, 8, generator=torch.Generator().manual_seed(915))
    summaries = SemanticOffloadingWorker._summarize_key_batches(keys)
    worker.durable_summaries = {_key(index): summary for index, summary in enumerate(summaries)}
    worker._mark_inserted_into_stack_cache(worker.durable_summaries)
    return worker


def _assert_stack_matches_live_summaries(worker):
    assert len(worker._stack_cache_keys) == len(worker._stack_cache_index)
    for field, tensor in worker._stack_cache.items():
        for index, key in enumerate(worker._stack_cache_keys):
            expected = torch.stack([getattr(summary, field)
                                    for summary in worker.durable_summaries[key]])
            assert worker._stack_cache_index[key] == index
            assert torch.equal(tensor[index], expected)


@pytest.mark.parametrize("method,fields", [("mean", 1), ("minmax", 2), ("cuboid_mean", 2)])
def test_stack_cache_uses_one_stack_per_field_and_preserves_heads(method, fields):
    worker = _cache_worker(method)
    with patch("torch.stack", wraps=torch.stack) as stack:
        worker._rebuild_stack_cache()
    assert stack.call_count == fields
    _assert_stack_matches_live_summaries(worker)


@pytest.mark.parametrize("remove_positions", [[0], [2, 3], [1, 3, 5], list(range(7)), []])
def test_stack_cache_slice_compaction_keeps_order_and_excludes_removed_values(remove_positions):
    worker = _cache_worker("cuboid_mean")
    worker._rebuild_stack_cache()
    original_order = list(worker._stack_cache_keys)
    removed = [original_order[position] for position in remove_positions]
    worker.receive_evicted_keys(removed)
    with patch("torch.ones", side_effect=AssertionError("no CPU tensor mask")):
        worker._rebuild_stack_cache()
    assert worker._stack_cache_keys == [key for key in original_order if key not in removed]
    _assert_stack_matches_live_summaries(worker)
    for tensor in worker._stack_cache.values():
        assert tensor.shape == (7 - len(removed), 3, 8)


def test_stack_cache_overwrite_moves_one_updated_row_after_survivors():
    worker = _cache_worker("mean")
    worker._rebuild_stack_cache()
    original_order = list(worker._stack_cache_keys)
    overwritten = original_order[2]
    worker.durable_summaries[overwritten] = worker._summarize_key_batches(
        torch.full((1, 3, 16, 8), 456.0)
    )[0]
    worker._mark_inserted_into_stack_cache([overwritten])
    worker._rebuild_stack_cache()
    assert worker._stack_cache_keys == [key for key in original_order if key != overwritten] + [overwritten]
    _assert_stack_matches_live_summaries(worker)
    assert worker._stack_cache["mean"][-1].eq(456.0).all()


def test_stack_cache_empty_survivors_then_insert_cannot_resurrect_stale_rows():
    worker = _cache_worker("mean")
    worker._rebuild_stack_cache()
    worker.receive_evicted_keys(list(worker._stack_cache_keys))
    worker._rebuild_stack_cache()
    new_key = _key(90)
    worker.durable_summaries[new_key] = worker._summarize_key_batches(
        torch.full((1, 3, 16, 8), -123.0)
    )[0]
    worker._mark_inserted_into_stack_cache([new_key])
    worker._rebuild_stack_cache()
    assert worker._stack_cache_keys == [new_key]
    assert worker._stack_cache["mean"].shape == (1, 3, 8)
    _assert_stack_matches_live_summaries(worker)


def test_stack_cache_no_survivor_change_reuses_existing_tensor():
    worker = _cache_worker("mean")
    worker._rebuild_stack_cache()
    previous = worker._stack_cache["mean"]
    worker._stack_pending_remove.add(_key(999))
    worker._rebuild_stack_cache()
    assert worker._stack_cache["mean"] is previous
    _assert_stack_matches_live_summaries(worker)
