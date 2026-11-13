from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus

from wait_hint_offload.observed import ObservedGroup
from wait_hint_offload.vllm_adapter.arc import (
    ARCHolds,
    ARCRecency,
    GroupARCRecency,
    HoldTarget,
    Owner,
)


def test_actual_arc_prefix_order_references_and_pending_boundary() -> None:
    policy = ARCCachePolicy(8)
    keys = tuple(bytes([i]) for i in range(1, 5))
    for i, raw in enumerate(keys):
        block = BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(OffloadKey(raw), block)
    policy.t1[OffloadKey(keys[1])].ref_cnt = 1
    policy.t1[OffloadKey(keys[2])].ref_cnt = -1
    policy.b1[OffloadKey(b"ghost")] = None
    before = [(key, block.block_id, block.ref_cnt) for key, block in policy.t1.items()]
    recency = ARCRecency(policy)
    assert recency.refresh_prefix(keys) == 2
    assert list(policy.t1) == [keys[2], keys[3], keys[1], keys[0]]
    assert {
        (key, block.block_id, block.ref_cnt) for key, block in policy.t1.items()
    } == set(before)
    assert not policy.t2 and list(policy.b1) == [b"ghost"]
    assert policy.target_t1_size == 0
    victims = policy.evict(1, set())
    assert victims is not None and [key for key, _ in victims] == [keys[3]]


def test_pinned_arc_fallback_limitation_is_characterized() -> None:
    policy = ARCCachePolicy(4)
    block = BlockStatus(1)
    block.ref_cnt = 0
    policy.insert(OffloadKey(b"a"), block)
    policy.target_t1_size = 2
    assert policy.evict(1, set()) is None
    assert list(policy.t1) == [b"a"]


def test_group_advice_requires_aligned_ready_window_and_checkpoint() -> None:
    policy = ARCCachePolicy(32)
    full = tuple(bytes([i]) for i in range(1, 5))
    sliding = tuple(bytes([i]) for i in range(11, 15))
    recurrent = tuple(bytes([i]) for i in range(21, 23))
    for i, key in enumerate(full + sliding + recurrent):
        block = BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(OffloadKey(key), block)
    groups = (
        ObservedGroup(4, None, full),
        ObservedGroup(4, 2, sliding),
        ObservedGroup(8, 1, recurrent),
    )
    recency = GroupARCRecency(policy, groups)
    assert set(recency.ready_prefix(full)) == set(full + sliding[2:] + recurrent[1:])
    policy.t1[OffloadKey(recurrent[-1])].ref_cnt = -1
    assert set(recency.ready_prefix(full)) == set(
        full[:2] + sliding[:2] + recurrent[:1]
    )
    policy.t1[OffloadKey(sliding[1])].ref_cnt = -1
    before = list(policy.t1)
    assert recency.refresh_prefix(full) == 0
    assert list(policy.t1) == before
    policy.t1[OffloadKey(recurrent[-1])].ref_cnt = 0
    assert recency.refresh_prefix(full) == 7
    assert set(policy.t1) == set(before)
    assert not policy.t2


def ready_policy(count: int, capacity: int = 8) -> tuple[ARCCachePolicy, list[bytes]]:
    policy = ARCCachePolicy(capacity)
    keys = [bytes([i]) for i in range(1, count + 1)]
    for i, raw in enumerate(keys):
        block = BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(OffloadKey(raw), block)
    return policy, keys


def test_promotion_moves_only_the_ready_prefix_to_the_frequent_queue() -> None:
    policy, keys = ready_policy(4)
    policy.t1[OffloadKey(keys[2])].ref_cnt = -1
    assert ARCRecency(policy).promote_prefix(tuple(keys)) == 2
    assert list(policy.t2) == [keys[1], keys[0]]
    assert list(policy.t1) == [keys[2], keys[3]]
    assert policy.target_t1_size == 0


def test_holds_are_bounded_expire_and_yield_rather_than_fail_a_store() -> None:
    policy, keys = ready_policy(4, capacity=4)
    now = [0]
    owners: dict[Owner, object] = {}
    holds = ARCHolds(policy, 0.5, lambda: now[0], owners.get)
    first, second = ("scope", "a"), ("scope", "b")
    owners[first] = owners[second] = token = object()
    held = [OffloadKey(keys[0]), OffloadKey(keys[1])]
    assert holds.hold(first, held, 10, token) == 2
    assert holds.hold(second, [OffloadKey(keys[2])], 10, token) == 0
    victims = policy.evict(1, set())
    assert victims is not None and [key for key, _ in victims] == [keys[2]]
    victims = policy.evict(2, set())
    assert victims is not None and {key for key, _ in victims} == {keys[0], keys[1]}
    assert holds.yielded == 1
    policy, keys = ready_policy(2)
    holds = ARCHolds(policy, 1.0, lambda: now[0], owners.get)
    assert holds.hold(first, [OffloadKey(keys[0])], 10, token) == 1
    owners[first] = object()
    assert holds.active() == set()
    owners[first] = token
    assert holds.hold(first, [OffloadKey(keys[0])], 10, token) == 1
    now[0] = 10
    assert holds.active() == set()
    now[0] = 0
    assert holds.hold(first, [OffloadKey(keys[0])], 10, token) == 1
    holds.release(first)
    assert holds.active() == set()
    assert holds.hold(first, [OffloadKey(keys[0])], 10, token) == 1
    holds.clear()
    assert holds.active() == set()


def test_hold_target_promotes_then_holds_the_ready_prefix() -> None:
    policy, keys = ready_policy(3)
    policy.t1[OffloadKey(keys[2])].ref_cnt = -1
    owner: Owner = ("scope", "session")
    token = object()
    holds = ARCHolds(policy, 0.5, lambda: 0, {owner: token}.get)
    target = HoldTarget(ARCRecency(policy), holds, owner, 5, token)
    assert target.refresh_prefix(tuple(keys)) == 2
    assert target.held == 2
    assert list(policy.t2) == [keys[1], keys[0]]
    assert holds.active() == {keys[0], keys[1]}


def test_unreachable_keys_are_window_keys_before_the_ready_window() -> None:
    policy = ARCCachePolicy(32)
    full = tuple(bytes([i]) for i in range(1, 5))
    sliding = tuple(bytes([i]) for i in range(11, 15))
    recurrent = tuple(bytes([i]) for i in range(21, 23))
    for i, key in enumerate(full + sliding + recurrent):
        block = BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(OffloadKey(key), block)
    groups = (
        ObservedGroup(4, None, full),
        ObservedGroup(4, 2, sliding),
        ObservedGroup(8, 1, recurrent),
    )
    recency = GroupARCRecency(policy, groups)
    assert recency.unreachable(full) == [sliding[0], sliding[1], recurrent[0]]
    policy.t1[OffloadKey(recurrent[-1])].ref_cnt = -1
    assert recency.unreachable(full) == []
    policy.t1[OffloadKey(full[0])].ref_cnt = -1
    assert recency.unreachable(full) == []
    assert ARCRecency(policy).unreachable(full) == []


def test_unreachable_keys_are_evicted_first_and_atomically() -> None:
    policy, keys = ready_policy(6, capacity=6)
    ARCRecency(policy).promote_prefix(tuple(keys[4:]))
    owners: dict[Owner, object] = {}
    holds = ARCHolds(policy, 0.5, lambda: 0, owners.get)
    dead = [OffloadKey(keys[5]), OffloadKey(keys[3]), OffloadKey(b"gone")]
    assert holds.mark_dead(dead) == 3
    policy.t1[OffloadKey(keys[3])].ref_cnt = 1
    victims = policy.evict(2, set())
    assert victims is not None and [key for key, _ in victims] == [keys[5], keys[0]]
    assert keys[5] in policy.b2 and keys[0] in policy.b1
    assert list(holds.dead) == [keys[3]] and holds.dead_evicted == 1
    policy.t1[OffloadKey(keys[3])].ref_cnt = 0
    before = (list(policy.t1), list(policy.t2), list(holds.dead))
    assert policy.evict(6, set()) is None
    assert (list(policy.t1), list(policy.t2), list(holds.dead)) == before
    victims = policy.evict(1, {OffloadKey(keys[3])})
    assert victims is not None and [key for key, _ in victims] == [keys[1]]
    holds.clear()
    assert not holds.dead


def test_partial_yield_releases_the_oldest_hold_first() -> None:
    policy, keys = ready_policy(4, capacity=4)
    first, second = ("scope", "a"), ("scope", "b")
    token = object()
    holds = ARCHolds(policy, 1.0, lambda: 0, {first: token, second: token}.get, True)
    assert holds.hold(first, [OffloadKey(keys[0]), OffloadKey(keys[1])], 5, token)
    assert holds.hold(second, [OffloadKey(keys[2])], 9, token)
    victims = policy.evict(2, set())
    assert victims is not None and [key for key, _ in victims] == [keys[0], keys[1]]
    assert holds.yielded == 1 and set(holds.holds) == {second}


def test_reachable_hold_target_marks_unreachable_keys() -> None:
    policy = ARCCachePolicy(16)
    full = (b"f1", b"f2")
    sliding = (b"s1", b"s2")
    for i, key in enumerate(full + sliding):
        block = BlockStatus(i)
        block.ref_cnt = 0
        policy.insert(OffloadKey(key), block)
    groups = (ObservedGroup(4, None, full), ObservedGroup(4, 1, sliding))
    owner: Owner = ("scope", "session")
    token = object()
    holds = ARCHolds(policy, 0.5, lambda: 0, {owner: token}.get, True)
    recency = GroupARCRecency(policy, groups)
    target = HoldTarget(recency, holds, owner, 5, token, evict_unreachable=True)
    assert target.refresh_prefix(full) == 3
    assert target.held == 3 and target.unreachable == 1
    assert list(holds.dead) == [b"s1"]
    assert holds.active() == {b"f1", b"f2", b"s2"}
