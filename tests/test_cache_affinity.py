# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the opt-in cache-affinity scheduler queue."""

from types import SimpleNamespace

import pytest

from semantic_offload.cache_affinity import (
    CacheAffinityConfig,
    CacheAffinityRequestQueue,
    CacheAffinityScheduler,
    _config_from_vllm_config,
    _local_cached_blocks,
)


def request(name: str):
    return SimpleNamespace(request_id=name)


def make_queue(scores, **kwargs):
    probes = []

    def probe(req):
        probes.append(req.request_id)
        return scores.get(req.request_id, 0)

    clock = kwargs.pop("clock", lambda: 0.0)
    return (
        CacheAffinityRequestQueue(probe, clock=clock, **kwargs),
        probes,
    )


def test_cache_warm_requests_move_only_within_bounded_window():
    queue, _ = make_queue({"cold": 0, "warm": 8}, window_size=2)
    cold, warm, tail = request("cold"), request("warm"), request("tail")
    queue.add_request(cold)
    queue.add_request(warm)
    queue.add_request(tail)

    assert [req.request_id for req in queue] == ["warm", "cold", "tail"]


def test_preempted_sticky_front_is_never_reordered():
    queue, _ = make_queue({"new": 20, "preempted": 0})
    preempted, new = request("preempted"), request("new")
    queue.add_request(new)
    queue.prepend_request(preempted)

    assert queue.peek_request() is preempted
    assert queue.pop_request() is preempted
    assert queue.pop_request() is new


def test_bulk_abort_is_idempotent_across_waiting_queues():
    queue, _ = make_queue({})
    other_queue, _ = make_queue({})
    sticky, sortable, other, survivor = [
        request(name) for name in ("sticky", "sortable", "other", "survivor")
    ]
    queue.prepend_request(sticky)
    queue.add_request(sortable)
    queue.add_request(survivor)
    other_queue.add_request(other)
    aborts = [request("absent"), sticky, sortable, sticky, other]
    survivor_clock = queue._enqueued_at["survivor"]
    queue.remove_requests(iter(aborts))
    other_queue.remove_requests(iter(aborts))
    assert list(queue) == [survivor]
    assert not other_queue
    assert queue._enqueued_at == {"survivor": survivor_clock}
    assert queue._overtakes == {"survivor": 0}
    queue.remove_requests(aborts)
    queue.remove_requests([])
    later = request("later")
    queue.add_request(later)
    assert queue.pop_request() is survivor
    assert queue.pop_request() is later
    assert not queue._enqueued_at and not queue._overtakes


def test_single_remove_missing_request_remains_strict():
    queue, _ = make_queue({})
    with pytest.raises(ValueError):
        queue.remove_request(request("absent"))


def test_aging_overrides_cache_affinity():
    now = [0.0]
    queue, _ = make_queue(
        {"cold": 0, "warm": 20},
        max_wait_s=0.2,
        clock=lambda: now[0],
    )
    cold, warm = request("cold"), request("warm")
    queue.add_request(cold)
    queue.add_request(warm)
    now[0] = 0.3

    assert queue.peek_request() is cold


def test_overtake_cap_prevents_repeated_bypass():
    scores = {"cold": 0}
    queue, _ = make_queue(
        scores,
        max_wait_s=0.2,
        max_overtakes=8,
    )
    cold = request("cold")
    queue.add_request(cold)
    for index in range(8):
        warm = request(f"warm-{index}")
        scores[warm.request_id] = 20
        queue.add_request(warm)
        queue.reorder(force=True)
        if index < 7:
            assert queue.pop_request() is warm
    assert queue.peek_request() is cold


def test_priority_mode_is_not_represented_by_this_fcfs_queue():
    # The scheduler plugin leaves vLLM's PriorityRequestQueue untouched.  This
    # test documents that this queue has no priority reinterpretation.
    queue, _ = make_queue({"a": 100, "b": 0})
    queue.add_request(request("a"))
    queue.add_request(request("b"))
    assert [req.request_id for req in queue] == ["a", "b"]


def test_probe_failure_falls_back_to_fcfs():
    def failing_probe(_):
        raise RuntimeError("probe unavailable")

    queue = CacheAffinityRequestQueue(failing_probe)
    first, second = request("first"), request("second")
    queue.add_request(first)
    queue.add_request(second)

    assert queue.pop_request() is first
    assert queue.pop_request() is second


def test_probe_is_bounded_to_window():
    queue, probes = make_queue(
        {"a": 1, "b": 2, "c": 3, "d": 4},
        window_size=2,
    )
    for name in ("a", "b", "c", "d"):
        queue.add_request(request(name))

    queue.reorder()
    assert probes == ["a", "b"]


def test_forced_reorder_reprobes_after_cache_state_changes():
    scores = {"cold": 0, "warm": 8}
    queue, probes = make_queue(scores)
    cold, warm = request("cold"), request("warm")
    queue.add_request(cold)
    queue.add_request(warm)
    queue.reorder(force=True)
    scores["cold"] = 9
    queue.reorder(force=True)

    assert [req.request_id for req in queue] == ["cold", "warm"]
    assert probes == ["cold", "warm", "warm", "cold"]


def test_reorder_applies_dynamic_scheduler_window():
    queue, probes = make_queue({"a": 1, "b": 2, "c": 3, "d": 4})
    for name in ("a", "b", "c", "d"):
        queue.add_request(request(name))

    queue.reorder(force=True, window_size=2)
    assert probes == ["a", "b"]


def test_one_cached_block_is_affinity_eligible():
    queue, _ = make_queue({"one": 1, "none": 0})
    one, none = request("one"), request("none")
    queue.add_request(one)
    queue.add_request(none)

    assert queue.peek_request() is one


def test_local_probe_uses_only_the_side_effect_free_coordinator():
    calls = []

    class Coordinator:
        def find_longest_cache_hit(self, block_hashes, max_length):
            calls.append((block_hashes, max_length))
            return (object(), 32, 0)

    class Manager:
        coordinator = Coordinator()
        enable_caching = True

        def prefix_cache_lookup_enabled(self, _request):
            return True

        def get_computed_blocks(self, _request):
            raise AssertionError("the side-effectful manager lookup was used")

    fake_scheduler = SimpleNamespace(
        kv_cache_manager=Manager(),
        block_size=16,
    )
    fake_request = SimpleNamespace(
        num_tokens=48,
        block_hashes=["h0", "h1"],
        skip_reading_prefix_cache=False,
    )

    assert _local_cached_blocks(fake_scheduler, fake_request) == 2
    assert calls == [(["h0", "h1"], 47)]


def test_plugin_is_loadable_as_a_vllm_scheduler_subclass():
    from vllm.v1.core.sched.scheduler import Scheduler

    assert issubclass(CacheAffinityScheduler, Scheduler)


def test_scheduler_caps_window_at_four_times_max_num_seqs(monkeypatch):
    from vllm.v1.core.sched.request_queue import SchedulingPolicy
    from vllm.v1.core.sched.scheduler import Scheduler

    calls = []
    fake_scheduler = CacheAffinityScheduler.__new__(CacheAffinityScheduler)
    fake_scheduler.policy = SchedulingPolicy.FCFS
    fake_scheduler._cache_affinity_enabled = True
    fake_scheduler.scheduler_config = SimpleNamespace(max_num_seqs=1)
    fake_scheduler.waiting = SimpleNamespace(
        reorder=lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(Scheduler, "schedule", lambda self, **kwargs: "ok")

    assert CacheAffinityScheduler.schedule(fake_scheduler) == "ok"
    assert calls == [{"force": True, "window_size": 4}]


def test_disabled_scheduler_does_not_call_affinity_queue(monkeypatch):
    from vllm.v1.core.sched.request_queue import SchedulingPolicy
    from vllm.v1.core.sched.scheduler import Scheduler

    calls = []
    fake_scheduler = CacheAffinityScheduler.__new__(CacheAffinityScheduler)
    fake_scheduler.policy = SchedulingPolicy.FCFS
    fake_scheduler._cache_affinity_enabled = False
    fake_scheduler.waiting = SimpleNamespace()
    monkeypatch.setattr(
        Scheduler, "schedule", lambda self, **kwargs: calls.append(kwargs)
    )

    CacheAffinityScheduler.schedule(fake_scheduler)
    assert calls == [{"throttle_prefills": False}]


def test_malformed_additional_config_disables_plugin():
    assert CacheAffinityConfig.from_mapping({"enabled": "yes"}).enabled is False
    assert CacheAffinityConfig.from_mapping({"unknown": True}).enabled is False
    assert CacheAffinityConfig.from_mapping([]).enabled is False
    assert (
        CacheAffinityConfig.from_mapping({"max_wait_s": float("nan")}).enabled is False
    )
    assert CacheAffinityConfig.from_mapping({"window_size": True}).enabled is False
    assert CacheAffinityConfig.from_mapping({"window_size": 64}).enabled is False
    assert CacheAffinityConfig.from_mapping({"max_wait_s": 0.3}).enabled is False
    assert CacheAffinityConfig.from_mapping({"max_overtakes": 9}).enabled is False


def test_explicit_vllm_namespace_is_read_and_malformed_values_disable():
    enabled = SimpleNamespace(
        additional_config={
            "semantic_offload.cache_affinity": {
                "enabled": True,
                "window_size": 4,
            }
        }
    )
    assert _config_from_vllm_config(enabled, CacheAffinityConfig()).window_size == 4
    assert not _config_from_vllm_config(
        SimpleNamespace(additional_config={"semantic_offload.cache_affinity": None}),
        CacheAffinityConfig(),
    ).enabled


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_invalid_window(bad):
    with pytest.raises(ValueError, match="window_size"):
        CacheAffinityRequestQueue(lambda _: 0, window_size=bad)
