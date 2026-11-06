"""Behavioral CPU controls for the isolated predicted-reuse experiment."""
import pytest
from tests.test_wait_hint_cpu_recency import (
 retention, test_invalid_or_stale_hint_cannot_promote_cpu_blocks,
 test_global_disable_prevents_any_cpu_advice,
 test_replayed_event_does_not_promote_again,
 test_unsupported_native_lru_does_not_receive_speculative_hits,
 test_ready_active_cpu_load_keeps_its_reference,
)
from tests.test_step_1_5_prefetch import _insert_resident,to_key

def test_ready_advice_promotes_without_refs_gpu_or_ghost_learning(retention):
 s,m,p,keys,hint=retention;q=m._policy
 q.t2[keys[1]]=q.t1.pop(keys[1]);q.b1[to_key(80)]=None;q.b2[to_key(81)]=None;q.target_t1_size=2.5
 before={k:(q.get(k).block_id,q.get(k).ref_cnt) for k in keys};free=p.get_num_free_blocks();evictable=m._num_evictable_cache_blocks
 ack=s.publish_wait_hint(hint)
 assert ack['cpu_advice_queue_policy']=='experimental_ready_resident_T2_promotion' and ack['ready_cpu_prefix_keys']==4
 assert ack['cpu_advice_promoted_t1_keys']==3 and ack['cpu_advice_refreshed_t2_keys']==1
 assert not q.t1 and list(q.t2)==list(reversed(keys))
 assert list(q.b1)==[to_key(80)] and list(q.b2)==[to_key(81)] and q.target_t1_size==2.5
 assert {k:(q.get(k).block_id,q.get(k).ref_cnt) for k in keys}==before
 assert p.get_num_free_blocks()==free and m._num_evictable_cache_blocks==evictable
 assert not s._jobs and not s._wait_prefetch_jobs and not s._wait_prefetch and not m._staging_protection_keys
 assert ack['reserved_gpu_blocks']==ack['speculative_transfer_bytes']==0
 assert len(q.evict(4,set()))==4

@pytest.mark.parametrize('gap',['missing','pending'])
def test_promotion_only_contiguous_ready_prefix(retention,gap):
 s,m,p,keys,hint=retention;q=m._policy
 if gap=='missing':q.remove(keys[1])
 else:q.get(keys[1]).ref_cnt=-1
 ack=s.publish_wait_hint(hint)
 assert ack['ready_cpu_prefix_keys']==1 and list(q.t2)==[keys[0]]
 assert keys[2] in q.t1 and keys[3] in q.t1

def test_mixed_queue_eviction_competition_is_changed_explicitly(retention):
 s,m,p,keys,hint=retention;q=m._policy
 old=[to_key(4),to_key(5)]
 for i,k in enumerate(old):
  _insert_resident(m,k,4+i);q.t2[k]=q.t1.pop(k)
 s.publish_wait_hint(hint)
 # No pinning: the unchanged native ARC picks older T2 entries first.
 assert [k for k,b in q.evict(2,set())]==old
 assert all(q.get(k) is not None for k in keys)
 assert len(q.t1)+len(q.t2)<=q.cache_capacity

def test_repeat_ready_advice_bounded_no_ghost_adaptation(retention):
 s,m,p,keys,hint=retention;q=m._policy;q.b1[to_key(81)]=None;q.b2[to_key(82)]=None;q.target_t1_size=1.5
 for _ in range(1000):assert m.refresh_wait_hint_cpu_recency([*keys,to_key(81),to_key(82),to_key(900)])==4
 assert not q.t1 and list(q.t2)==list(reversed(keys))
 assert list(q.b1)==[to_key(81)] and list(q.b2)==[to_key(82)] and q.target_t1_size==1.5
 assert all(q.get(k).ref_cnt==0 for k in keys)

def test_promotion_works_with_native_audit_off_and_bounded_state():
 from tests.test_preservation_audit_mode import native,source,_wait_hint
 scheduler,manager=native();req,prefix,keys=source(scheduler)
 ctx=scheduler._req_status[req.request_id].req_context
 prepared=manager.prepare_store(keys,ctx);manager.complete_store(prepared.keys_to_store,ctx)
 scheduler._register_observed_prefix(req)
 pool=scheduler.gpu_block_pool;free=pool.get_num_free_blocks();refs=[b.ref_cnt for b in pool.blocks]
 result=scheduler.publish_wait_hint(_wait_hint(scheduler,prefix,32,session_id="one"))
 assert result['ready_cpu_prefix_keys']==2 and result['cpu_advice_queue_policy']=='experimental_ready_resident_T2_promotion'
 assert not manager._policy.t1 and set(manager._policy.t2)==set(keys)
 assert [manager._policy.get(k).ref_cnt for k in keys]==[0,0]
 assert not scheduler._preservation_requests and not scheduler._wait_prefetch and not scheduler._jobs
 assert pool.get_num_free_blocks()==free and [b.ref_cnt for b in pool.blocks]==refs
 assert len(scheduler._observed_prefix_generations)==len(scheduler._observed_prefixes)==1

def test_per_call_opportunity_counts_no_manager_history(retention):
 s,m,p,keys,hint=retention;fields=set(vars(m))
 n,stats=m.refresh_wait_hint_cpu_recency(keys,return_stats=True)
 assert n==4 and stats=={'cpu_advice_promoted_t1_keys':4,'cpu_advice_refreshed_t2_keys':0}
 n,stats=m.refresh_wait_hint_cpu_recency(keys,return_stats=True)
 assert n==4 and stats=={'cpu_advice_promoted_t1_keys':0,'cpu_advice_refreshed_t2_keys':4}
 assert set(vars(m))==fields

def test_per_call_zero_and_pending_counts(retention):
 s,m,p,keys,hint=retention;m._policy.get(keys[0]).ref_cnt=-1
 n,stats=m.refresh_wait_hint_cpu_recency([keys[0],to_key(900)],return_stats=True)
 assert n==0 and stats=={'cpu_advice_promoted_t1_keys':0,'cpu_advice_refreshed_t2_keys':0}
 assert keys[0] in m._policy.t1 and not m._policy.t2

def test_per_call_lru_counts_zero():
 from semantic_offload.manager import PreservationOffloadingManager
 m=PreservationOffloadingManager(num_blocks=8,cache_policy='lru')
 _insert_resident(m,to_key(1),0)
 assert m.refresh_wait_hint_cpu_recency([to_key(1)],return_stats=True)==(0,{'cpu_advice_promoted_t1_keys':0,'cpu_advice_refreshed_t2_keys':0})

def test_replayed_ack_carries_original_stats_without_new_action(retention):
 from unittest.mock import patch
 s,m,p,keys,hint=retention;first=s.publish_wait_hint(hint)
 with patch.object(m,'refresh_wait_hint_cpu_recency',side_effect=AssertionError('duplicate promotion')):second=s.publish_wait_hint(hint)
 assert second['idempotent_replay'] is True
 assert first['cpu_advice_promoted_t1_keys']==second['cpu_advice_promoted_t1_keys']==4
 assert first['cpu_advice_refreshed_t2_keys']==second['cpu_advice_refreshed_t2_keys']==0
