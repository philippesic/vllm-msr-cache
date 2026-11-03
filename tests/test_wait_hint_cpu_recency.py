"""Causal wait hints may refresh CPU ARC without reserving or copying KV."""
from unittest.mock import patch
REASON = "cpu_prefix_recency_refreshed"

import pytest

from semantic_offload.manager import PreservationOffloadingManager
from vllm.v1.kv_offload.base import get_offload_block_hash
from tests.test_step_1_5_prefetch import (
    _StubBlockPool, _insert_resident, _make_connector_scheduler,
    _register_wait_prefix, _wait_hint, to_key,
)


@pytest.fixture
def retention():
    manager = PreservationOffloadingManager(num_blocks=8, cache_policy="arc")
    keys = [to_key(i) for i in range(4)]
    for i, key in enumerate(keys):
        _insert_resident(manager, key, i)
    pool = _StubBlockPool(20, free_blocks=5)
    scheduler = _make_connector_scheduler(manager, pool)
    scheduler._wait_hint_cpu_retention_only = True
    prefix_id, boundary = _register_wait_prefix(scheduler, keys)
    hint = _wait_hint(scheduler, prefix_id, boundary)
    return scheduler, manager, pool, keys, hint


def test_cpu_retention_refreshes_arc_without_ownership_or_gpu_work(retention):
    scheduler, manager, pool, keys, hint = retention
    original = {k: (manager._policy.get(k).block_id,
                    manager._policy.get(k).ref_cnt) for k in keys}
    free_before = pool.get_num_free_blocks()
    result = scheduler.publish_wait_hint(hint)
    assert result['reason'] == REASON
    assert result['ready_cpu_prefix_keys'] == len(keys)
    assert result['reserved_gpu_blocks'] == result['speculative_transfer_bytes'] == 0
    assert set(manager._policy.t1) == set(keys) and not manager._policy.t2
    assert {k: (manager._policy.get(k).block_id,
                manager._policy.get(k).ref_cnt) for k in keys} == original
    assert manager._num_evictable_cache_blocks == len(keys)
    assert pool.get_num_free_blocks() == free_before
    assert scheduler._prefetch_reserved_blocks == 0
    assert not scheduler._wait_prefetch and not scheduler._wait_prefetch_jobs
    assert not scheduler._jobs and not manager._staging_protection_keys
    # Ordinary ARC eviction remains possible: hints never pin these blocks.
    assert len(manager._policy.evict(len(keys), set())) == len(keys)


@pytest.mark.parametrize('bad', ['namespace', 'generation', 'source',
                                'future_clock', 'expired', 'future_field'])
def test_invalid_or_stale_hint_cannot_promote_cpu_blocks(retention, bad):
    scheduler, manager, pool, keys, hint = retention
    if bad == 'namespace':
        hint['namespace'] = dict(hint['namespace'], model_revision='other')
    elif bad == 'generation':
        scheduler._observed_prefix_generations[('run', 'cell', 'session')] = 1
    elif bad == 'source':
        scheduler._observed_prefixes.clear()
    elif bad == 'future_clock':
        hint['input_publication_ns'] += 10_000_000_000
    elif bad == 'future_field':
        hint['actual_resume_ns'] = hint['input_publication_ns'] + 1
    clock = hint['observation_available_ns'] + (3_000_000_000 if bad == 'expired' else 1)
    with patch('semantic_offload.connector.time.monotonic_ns', return_value=clock):
        result = scheduler.publish_wait_hint(hint)
    assert result['reason'] != REASON
    assert not manager._policy.t2
    assert set(manager._policy.t1) == set(keys)
    assert not scheduler._wait_prefetch and not scheduler._jobs


@pytest.mark.parametrize('gap', ['absent', 'pending'])
def test_cpu_prefix_stops_at_first_missing_or_pending_block(retention, gap):
    scheduler, manager, pool, keys, hint = retention
    if gap == 'absent':
        manager._policy.remove(keys[1])
    else:
        manager._policy.get(keys[1]).ref_cnt = -1
    result = scheduler.publish_wait_hint(hint)
    assert result['ready_cpu_prefix_keys'] == 1
    assert not manager._policy.t2 and list(manager._policy.t1)[-1] == keys[0]
    assert keys[2] in manager._policy.t1 and keys[3] in manager._policy.t1


def test_replayed_event_does_not_promote_again(retention):
    scheduler, manager, pool, keys, hint = retention
    first = scheduler.publish_wait_hint(hint)
    with patch.object(manager, 'refresh_wait_hint_cpu_recency', side_effect=AssertionError('duplicate refresh')):
        second = scheduler.publish_wait_hint(hint)
    assert first['reason'] == second['reason'] == REASON
    assert second['idempotent_replay'] is True


def test_ready_active_cpu_load_keeps_its_reference(retention):
    scheduler, manager, pool, keys, hint = retention
    block = manager._policy.get(keys[0])
    block.ref_cnt = 1
    manager._policy.mark_non_evictable(keys[0])
    manager._num_evictable_cache_blocks -= 1
    result = scheduler.publish_wait_hint(hint)
    assert result['ready_cpu_prefix_keys'] == len(keys)
    assert block.ref_cnt == 1
    assert manager._num_evictable_cache_blocks == len(keys) - 1
    assert not manager._staging_protection_keys


def test_advice_preserves_both_queues_ghost_history_and_adaptive_target(retention):
    scheduler, manager, pool, keys, hint = retention
    policy = manager._policy
    policy.t2[keys[1]] = policy.t1.pop(keys[1])
    ghost1, ghost2 = to_key(81), to_key(82)
    policy.b1[ghost1] = None
    policy.b2[ghost2] = None
    policy.target_t1_size = 2.5
    assert manager.refresh_wait_hint_cpu_recency([*keys, ghost1, ghost2]) == 4
    assert list(policy.t1) == [keys[3], keys[2], keys[0]]
    assert list(policy.t2) == [keys[1]]
    assert list(policy.b1) == [ghost1] and list(policy.b2) == [ghost2]
    assert policy.target_t1_size == 2.5


def test_unsupported_native_lru_does_not_receive_speculative_hits():
    manager = PreservationOffloadingManager(num_blocks=8, cache_policy="lru")
    keys = [to_key(i) for i in range(4)]
    for i, key in enumerate(keys):
        _insert_resident(manager, key, i)
    before = [(key, manager._policy.get(key).ref_cnt) for key in keys]
    before_order = list(manager._policy.evictable_blocks)
    assert manager.refresh_wait_hint_cpu_recency(keys) == 0
    assert list(manager._policy.evictable_blocks) == before_order
    assert [(key, manager._policy.get(key).ref_cnt) for key in keys] == before


@pytest.fixture
def construct_retention_scheduler(monkeypatch):
    from types import SimpleNamespace as NS
    from tests.test_step_1_5_prefetch import (
        GroupOffloadConfig, SchedulerOffloadConfig,
        SemanticOffloadingConnectorScheduler,
    )
    config = SchedulerOffloadConfig(
        kv_group_configs=(GroupOffloadConfig(0, 16, 16, 1, None, None),),
        blocks_per_chunk=1, num_workers=1, offload_prompt_only=False,
    )
    selected_config = [config]
    def base(scheduler, *args):
        scheduler.config = selected_config[0]
        scheduler.manager = object()
    monkeypatch.setattr('semantic_offload.connector.construct_scheduler_base', base)
    def construct(extra, *, offload_config=None, cache=None):
        selected_config[0] = offload_config or config
        spec = NS(extra_config=extra, replicated_layout=False,
                  cpu_page_size_per_worker=128,
                  kv_bytes_per_chunk=128 * selected_config[0].num_workers)
        return SemanticOffloadingConnectorScheduler(
            spec, NS(model_config=NS(model='test-model'),
                     cache_config=NS(block_size=16)), cache or NS())
    return construct


@pytest.mark.parametrize('invalid', [0, 1, None, 'true', []])
def test_retention_option_requires_a_json_boolean(construct_retention_scheduler, invalid):
    with pytest.raises(TypeError, match='wait_hint_cpu_retention_only must be a JSON boolean'):
        construct_retention_scheduler({'wait_hint_cpu_retention_only': invalid})


@pytest.mark.parametrize('extra,enabled', [
    ({}, False),
    ({'wait_hint_cpu_retention_only': False}, False),
    ({'wait_hint_cpu_retention_only': True,
      'prefetch_trigger': 'known_prefix_wait_hint',
      'preservation_events': True}, True),
])
def test_retention_constructor_defaults_off_and_accepts_explicit_mode(
    construct_retention_scheduler, extra, enabled,
):
    assert construct_retention_scheduler(extra)._wait_hint_cpu_retention_only is enabled


def test_retention_option_requires_causal_wait_hint_trigger(construct_retention_scheduler):
    with pytest.raises(ValueError, match='requires known-prefix wait hints'):
        construct_retention_scheduler({'wait_hint_cpu_retention_only': True})


@pytest.mark.parametrize('retain_only', [False, True])
def test_global_disable_prevents_any_cpu_advice(retention, retain_only):
    scheduler, manager, pool, keys, hint = retention
    scheduler._wait_hint_cpu_retention_only = retain_only
    scheduler._prefetch_budget_fraction = 0
    before = list(manager._policy.t1)
    with patch.object(manager, 'refresh_wait_hint_cpu_recency',
                      side_effect=AssertionError('advice despite global disable')):
        result = scheduler.publish_wait_hint(hint)
    assert result['reason'] != REASON
    assert list(manager._policy.t1) == before and not manager._policy.t2
    assert not scheduler._wait_prefetch and not scheduler._jobs


def _enable_gpu_bridge(retention, gpu_prefix_length):
    scheduler, manager, pool, keys, hint = retention
    scheduler._wait_hint_cpu_retention_gpu_aware = True
    pool._cached_block_hashes = {
        get_offload_block_hash(key): index + 1
        for index, key in enumerate(keys[:gpu_prefix_length])
    }
    return scheduler, manager, pool, keys, hint


@pytest.mark.parametrize('active_suffix_load', [False, True])
def test_gpu_prefix_bridges_missing_cpu_roots_without_ownership_changes(
    retention, active_suffix_load,
):
    scheduler, manager, pool, keys, hint = _enable_gpu_bridge(retention, 2)
    manager._policy.remove(keys[0])
    manager._num_evictable_cache_blocks -= 1
    if active_suffix_load:
        manager._policy.get(keys[2]).ref_cnt = 1
        manager._policy.mark_non_evictable(keys[2])
        manager._num_evictable_cache_blocks -= 1
    before = {key: (manager._policy.get(key).block_id,
                    manager._policy.get(key).ref_cnt) for key in keys[1:]}
    free_before = pool.get_num_free_blocks()
    result = scheduler.publish_wait_hint(hint)
    assert result['cpu_only_prefix_keys'] == 0
    assert result['local_gpu_prefix_keys'] == 2
    assert result['additional_ready_cpu_suffix_keys'] == 2
    assert result['ready_cpu_prefix_keys'] == 2
    assert list(manager._policy.t1)[-2:] == [keys[3], keys[2]]
    assert {key: (manager._policy.get(key).block_id,
                  manager._policy.get(key).ref_cnt) for key in keys[1:]} == before
    assert manager._num_evictable_cache_blocks == 3 - active_suffix_load
    assert pool.get_num_free_blocks() == free_before
    assert not scheduler._wait_prefetch and not scheduler._wait_prefetch_jobs
    assert not scheduler._jobs and not manager._staging_protection_keys
    assert result['reserved_gpu_blocks'] == result['speculative_transfer_bytes'] == 0
    # A hint still provides no immunity from normal native ARC eviction.
    assert [key for key, _ in manager._policy.evict(1, set())] == [keys[1]]


def test_gpu_bridge_preserves_original_roots_and_single_refresh_order(retention):
    scheduler, manager, pool, keys, hint = _enable_gpu_bridge(retention, 3)
    manager._policy.remove(keys[1])
    manager._num_evictable_cache_blocks -= 1
    with patch.object(manager, 'refresh_wait_hint_cpu_recency',
                      wraps=manager.refresh_wait_hint_cpu_recency) as refresh:
        result = scheduler.publish_wait_hint(hint)
    refresh.assert_called_once_with([keys[0], keys[3]])
    assert result['cpu_only_prefix_keys'] == 1
    assert result['additional_ready_cpu_suffix_keys'] == 1
    assert list(manager._policy.t1)[-2:] == [keys[3], keys[0]]


@pytest.mark.parametrize('gap', ['absent', 'pending'])
def test_gpu_bridge_cannot_jump_a_cpu_suffix_gap(retention, gap):
    scheduler, manager, pool, keys, hint = _enable_gpu_bridge(retention, 2)
    manager._policy.remove(keys[1])
    if gap == 'absent':
        manager._policy.remove(keys[2])
    else:
        manager._policy.get(keys[2]).ref_cnt = -1
    result = scheduler.publish_wait_hint(hint)
    assert result['cpu_only_prefix_keys'] == 1
    assert result['additional_ready_cpu_suffix_keys'] == 0
    assert result['ready_cpu_prefix_keys'] == 1
    assert list(manager._policy.t1)[-1] == keys[0]


def test_cpu_hole_not_covered_by_gpu_prefix_still_blocks_bridge(retention):
    scheduler, manager, pool, keys, hint = _enable_gpu_bridge(retention, 1)
    manager._policy.remove(keys[1])
    result = scheduler.publish_wait_hint(hint)
    assert result['additional_ready_cpu_suffix_keys'] == 0
    assert result['ready_cpu_prefix_keys'] == 1


def test_gpu_bridge_falls_back_when_native_index_unavailable(retention):
    scheduler, manager, pool, keys, hint = _enable_gpu_bridge(retention, 2)
    manager._policy.remove(keys[1])
    pool.get_cached_block = None
    result = scheduler.publish_wait_hint(hint)
    assert result['local_gpu_coverage_available'] is False
    assert result['additional_ready_cpu_suffix_keys'] == 0
    assert result['ready_cpu_prefix_keys'] == 1


def test_complete_cpu_prefix_avoids_gpu_index_lookup(retention):
    scheduler, manager, pool, keys, hint = _enable_gpu_bridge(retention, 4)
    with patch.object(pool, 'get_cached_block', side_effect=AssertionError('scan')):
        result = scheduler.publish_wait_hint(hint)
    assert result['ready_cpu_prefix_keys'] == len(keys)
    assert result['gpu_prefix_lookup_performed'] is False
    assert result['local_gpu_coverage_available'] is None
    assert result['additional_ready_cpu_suffix_keys'] == 0


def test_gpu_bridge_replay_does_not_rescan_or_refresh(retention):
    scheduler, manager, pool, keys, hint = _enable_gpu_bridge(retention, 2)
    manager._policy.remove(keys[0])
    first = scheduler.publish_wait_hint(hint)
    with patch.object(pool, 'get_cached_block', side_effect=AssertionError('rescan')):
        with patch.object(manager, 'refresh_wait_hint_cpu_recency',
                          side_effect=AssertionError('refresh')):
            second = scheduler.publish_wait_hint(hint)
    assert second['idempotent_replay'] is True
    assert second['additional_ready_cpu_suffix_keys'] == first['additional_ready_cpu_suffix_keys']


def test_gpu_bridge_reports_actual_refresh_count(retention):
    scheduler, manager, pool, keys, hint = _enable_gpu_bridge(retention, 2)
    manager._policy.remove(keys[0])
    with patch.object(manager, 'refresh_wait_hint_cpu_recency', return_value=0):
        result = scheduler.publish_wait_hint(hint)
    assert result['additional_ready_cpu_suffix_keys'] == 2
    assert result['refreshed_cpu_key_count'] == 0
    assert result['ready_cpu_prefix_keys'] == 0


@pytest.mark.parametrize('invalid', [0, 1, None, 'true', []])
def test_gpu_aware_retention_requires_json_boolean(construct_retention_scheduler, invalid):
    with pytest.raises(TypeError, match='wait_hint_cpu_retention_gpu_aware'):
        construct_retention_scheduler({'wait_hint_cpu_retention_gpu_aware': invalid})


def test_gpu_aware_retention_requires_retention_only(construct_retention_scheduler):
    with pytest.raises(ValueError, match='requires CPU retention-only'):
        construct_retention_scheduler({'wait_hint_cpu_retention_gpu_aware': True})


def test_gpu_aware_retention_is_opt_in(construct_retention_scheduler):
    from types import SimpleNamespace as NS
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    assert not construct_retention_scheduler({})._wait_hint_cpu_retention_gpu_aware
    cache = NS(kv_cache_groups=[NS(kv_cache_spec=FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=4, dtype=torch.float16,
    ))])
    scheduler = construct_retention_scheduler({
        'wait_hint_cpu_retention_gpu_aware': True,
        'wait_hint_cpu_retention_only': True,
        'prefetch_trigger': 'known_prefix_wait_hint',
        'preservation_events': True,
        'wait_hint_native_adoption': True,
    }, cache=cache)
    assert scheduler._wait_hint_cpu_retention_gpu_aware is True


@pytest.mark.parametrize('unsupported', ['multi_block', 'multi_group', 'multi_worker'])
def test_gpu_bridge_rejects_unqualified_layouts(
    construct_retention_scheduler, unsupported,
):
    from types import SimpleNamespace as NS
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from tests.test_step_1_5_prefetch import GroupOffloadConfig, SchedulerOffloadConfig

    group = GroupOffloadConfig(0, 16, 16, 1, None, None)
    config = SchedulerOffloadConfig((group,), 1, 1, False)
    if unsupported == 'multi_block':
        config = config._replace(
            blocks_per_chunk=2,
            kv_group_configs=(group._replace(tokens_per_chunk=32, hashes_per_chunk=2),),
        )
    elif unsupported == 'multi_group':
        config = config._replace(kv_group_configs=(group, group._replace(group_idx=1)))
    else:
        config = config._replace(num_workers=2)
    cache = NS(kv_cache_groups=[NS(kv_cache_spec=FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=4, dtype=torch.float16,
    ))])
    with pytest.raises(ValueError, match='dense TP1 single-block'):
        construct_retention_scheduler({
            'wait_hint_cpu_retention_gpu_aware': True,
            'wait_hint_cpu_retention_only': True,
            'prefetch_trigger': 'known_prefix_wait_hint',
            'preservation_events': True,
            'wait_hint_native_adoption': True,
        }, offload_config=config, cache=cache)


def test_fully_gpu_resident_source_preserves_only_existing_cpu_roots(retention):
    scheduler, manager, pool, keys, hint = _enable_gpu_bridge(retention, 4)
    manager._policy.remove(keys[1])
    with patch.object(manager, 'refresh_wait_hint_cpu_recency',
                      wraps=manager.refresh_wait_hint_cpu_recency) as refresh:
        result = scheduler.publish_wait_hint(hint)
    refresh.assert_called_once_with([keys[0]])
    assert result['additional_ready_cpu_suffix_keys'] == 0
    assert result['refreshed_cpu_key_count'] == 1


@pytest.mark.parametrize('bad', ['namespace', 'generation', 'source',
                                'future_clock', 'expired', 'global_disable'])
def test_gpu_bridge_is_behind_causal_and_global_guards(retention, bad):
    scheduler, manager, pool, keys, hint = _enable_gpu_bridge(retention, 2)
    manager._policy.remove(keys[0])
    before = list(manager._policy.t1)
    if bad == 'namespace':
        hint['namespace'] = dict(hint['namespace'], model_revision='other')
    elif bad == 'generation':
        scheduler._observed_prefix_generations[('run', 'cell', 'session')] = 1
    elif bad == 'source':
        scheduler._observed_prefixes.clear()
    elif bad == 'future_clock':
        hint['input_publication_ns'] += 10_000_000_000
    elif bad == 'global_disable':
        scheduler._prefetch_budget_fraction = 0
    clock = hint['observation_available_ns'] + (3_000_000_000 if bad == 'expired' else 1)
    with patch('semantic_offload.connector.time.monotonic_ns', return_value=clock):
        with patch.object(pool, 'get_cached_block', side_effect=AssertionError('scan')):
            with patch.object(manager, 'refresh_wait_hint_cpu_recency',
                              side_effect=AssertionError('refresh')):
                result = scheduler.publish_wait_hint(hint)
    assert result['reason'] != REASON
    assert list(manager._policy.t1) == before
