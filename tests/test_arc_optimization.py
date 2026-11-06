"""Differential checks against native ARC; advice cannot alter ownership."""
from types import SimpleNamespace
import random

import pytest

from semantic_offload.arc_optimization import ARCAdvice, AdviceOptions, LinearARCCachePolicy
from tests.test_wait_hint_cpu_recency import retention
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus


def state(p):
    return (tuple((name, tuple((k, b.block_id, b.ref_cnt) for k, b in getattr(p, name).items()))
                  for name in ('t1', 't2')), tuple(p.b1), tuple(p.b2), p.target_t1_size)


def policy(seed, cls=ARCCachePolicy):
    rng = random.Random(seed)
    p = cls(64)
    p.target_t1_size = rng.uniform(0, 64)
    for key in range(64):
        b = BlockStatus(key)
        b.ref_cnt = rng.choice([-1, 0, 0, 0, 1, 2])
        (p.t1 if rng.random() < .5 else p.t2)[key] = b
    p.b1.update((i, None) for i in range(64, 100))
    p.b2.update((i, None) for i in range(100, 140))
    return p


def test_linear_eviction_matches_native_atomic_state():
    for seed in range(1000):
        a, b = policy(seed), policy(seed, LinearARCCachePolicy)
        rng = random.Random(seed + 999)
        for _ in range(5):
            protected = {rng.randrange(64) for _ in range(rng.randrange(15))}
            n = rng.randrange(25)
            before = state(a)
            x, y = a.evict(n, protected), b.evict(n, protected)
            assert (None if x is None else [k for k, _ in x]) == (None if y is None else [k for k, _ in y])
            assert state(a) == state(b)
            if x is None:
                assert state(a) == before


@pytest.mark.parametrize('options', [
    {}, {'queue':'recent'}, {'queue':'frequent'}, {'boost':.25}, {'boost':.5},
    {'fraction':.25}, {'max_keys':8}, {'order':'suffix'}, {'cooldown_ms':2000},
    {'min_fill':1}, {'session_budget':16},
])
def test_advice_only_reorders_ready_members(options):
    p = policy(42)
    before = state(p)
    members = {name: set(getattr(p, name)) for name in ('t1', 't2')}
    refs = {k:(b.block_id,b.ref_cnt) for q in (p.t1,p.t2) for k,b in q.items()}
    advice = ARCAdvice(options)
    count = advice.refresh(SimpleNamespace(_policy=p), list(range(80)), ('session',0), 1000000)
    assert count >= 0
    assert {name:set(getattr(p,name)) for name in members} == members
    assert {k:(b.block_id,b.ref_cnt) for q in (p.t1,p.t2) for k,b in q.items()} == refs
    assert (tuple(p.b1),tuple(p.b2),p.target_t1_size) == before[1:]


def test_default_advice_matches_existing_refresh():
    for seed in range(100):
        p, q = policy(seed), policy(seed)
        keys = list(range(40))
        for k in reversed(keys):
            block = p.get(k)
            if block is not None and block.is_ready:
                (p.t1 if k in p.t1 else p.t2).move_to_end(k)
        ARCAdvice({}).refresh(SimpleNamespace(_policy=q),keys,('s',0),0)
        assert state(p) == state(q)


def test_budget_cooldown_and_state_bounds():
    p = LinearARCCachePolicy(8)
    for i in range(8):
        b = BlockStatus(i); b.ref_cnt = 0; p.t1[i] = b
    m = SimpleNamespace(_policy=p)
    a = ARCAdvice({'session_budget':4, 'cooldown_ms':1000, 'max_state':4})
    assert a.refresh(m,list(range(8)),('s',0),0) == 4
    assert a.refresh(m,list(range(8)),('s',0),2_000_000_000) == 0
    assert a.refresh(m,list(range(8)),('s',1),500_000_000) == 0
    assert a.refresh(m,list(range(8)),('s',1),2_000_000_000) == 4
    for i in range(20):
        a.refresh(m,list(range(8)),('s',i),i*2_000_000_000)
    assert len(a.sessions) <= 4 and len(a.key_times) <= 4


@pytest.mark.parametrize('raw', [None, [], {'typo':1}, {'boost':float('nan')},
    {'fraction':2}, {'max_keys':True}, {'max_state':0}, {'queue':'all'}, {'cooldown_ms':-1}])
def test_invalid_options_rejected(raw):
    with pytest.raises((TypeError,ValueError)):
        AdviceOptions.parse(raw)


def test_manager_selects_linear_policy_and_rejects_other_policy():
    from semantic_offload.manager import PreservationOffloadingManager
    m = PreservationOffloadingManager(num_blocks=8,cache_policy='arc',arc_linear_eviction=True)
    assert isinstance(m._policy,LinearARCCachePolicy)
    with pytest.raises(ValueError):
        PreservationOffloadingManager(num_blocks=8,cache_policy='lru',arc_linear_eviction=True)
    with pytest.raises(TypeError):
        PreservationOffloadingManager(num_blocks=8,cache_policy='arc',arc_linear_eviction='true')


def test_reset_clears_optional_advice_state():
    a=ARCAdvice({'cooldown_ms':1000,'session_budget':10})
    a.key_times['old']=0;a.sessions['old']=10;a.last={'applied':2}
    a.clear()
    assert not a.key_times and not a.sessions and not a.last


def test_wait_scope_budget_uses_future_owner_identity(retention):
    from tests.test_step_1_5_prefetch import _wait_hint
    scheduler,manager,pool,keys,hint=retention
    scheduler._wait_hint_arc_advice=ARCAdvice({'session_budget':2,'budget_scope':'wait'})
    assert scheduler.publish_wait_hint(hint)['ready_cpu_prefix_keys']==2
    again={**hint,'event_id':'b'*64}
    assert scheduler.publish_wait_hint(again)['ready_cpu_prefix_keys']==0
    following={**hint,'event_id':'c'*64,'owner_id':'next-future-owner'}
    assert scheduler.publish_wait_hint(following)['ready_cpu_prefix_keys']==2
    assert set(manager._policy.t1)==set(keys) and not manager._policy.t2


@pytest.mark.parametrize('retain,declared,grace,elapsed,expected',[
    (True,750,0,1_000_000_000,False),
    (True,750,750,1_000_000_000,True),
    (True,750,750,1_500_000_000,False),
    (True,0,750,1,False),
    (False,750,750,1_000_000_000,False),
])
def test_nomination_grace_is_bounded_cpu_only_advice(retention,retain,declared,grace,elapsed,expected):
    from unittest.mock import patch
    scheduler,manager,pool,keys,hint=retention
    scheduler._wait_hint_cpu_retention_only=retain
    scheduler._wait_hint_cpu_nomination_grace_ms=grace
    hint['declared_wait_ms']=declared
    before=pool.get_num_free_blocks()
    with patch('semantic_offload.connector.time.monotonic_ns',return_value=hint['input_publication_ns']+elapsed):
        result=scheduler.publish_wait_hint(hint)
    assert (result['reason']=='cpu_prefix_recency_refreshed')==expected
    if expected:
        assert result['cpu_nomination_grace_ms']==grace
        assert result['ready_cpu_prefix_keys']==len(keys)
        assert result['reserved_gpu_blocks']==result['speculative_transfer_bytes']==0
    assert pool.get_num_free_blocks()==before
    assert not scheduler._wait_prefetch and not scheduler._jobs


from tests.test_wait_hint_cpu_recency import construct_retention_scheduler

@pytest.mark.parametrize('grace',[-1,3001,True,1.5,'750'])
def test_nomination_grace_rejects_invalid_values(construct_retention_scheduler,grace):
    with pytest.raises(ValueError,match='integer in 0..3000'):
        construct_retention_scheduler({'wait_hint_cpu_nomination_grace_ms':grace})


def test_nomination_grace_requires_cpu_only_native_arc(construct_retention_scheduler):
    with pytest.raises(ValueError,match='requires native ARC CPU retention-only'):
        construct_retention_scheduler({'wait_hint_cpu_nomination_grace_ms':750})
    extra=dict(wait_hint_cpu_nomination_grace_ms=750,wait_hint_cpu_retention_only=True,
               prefetch_trigger='known_prefix_wait_hint',eviction_policy='arc',instrumentation_only=True,
               preservation_events=True)
    assert construct_retention_scheduler(extra)._wait_hint_cpu_nomination_grace_ms==750
