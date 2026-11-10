from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus

from wait_hint_offload.vllm_adapter.arc import ARCRecency


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
