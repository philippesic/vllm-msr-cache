from unittest.mock import patch
import pytest
from semantic_offload.gpu_cache_recency import refresh_gpu_prefix_recency
from test_gpu_coverage_nov3 import cached,snapshot
from tests.test_step_1_5_prefetch import to_key
from tests.test_wait_hint_cpu_recency import retention
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_offload.base import get_offload_block_hash

def stable(pool):return [(b.block_id,b.ref_cnt,b.block_hash,b.block_hash_num_tokens,b.is_null) for b in pool.blocks]

def test_free_exact_prefix_moves_to_tail_without_ownership_changes():
 p=BlockPool(12,True,16);keys,b=cached(p);p.free_blocks(b)
 others=p.get_new_blocks(2);p.free_blocks(others)
 before=stable(p);free=p.get_num_free_blocks();initial=snapshot(p)['free']
 with patch.object(p,'get_new_blocks',side_effect=AssertionError('allocation')),patch.object(p,'touch',side_effect=AssertionError('reference acquisition')):
  r=refresh_gpu_prefix_recency(p,keys)
 assert stable(p)==before and p.get_num_free_blocks()==free
 assert r['gpu_recency_refreshed_free_blocks']==2 and r['gpu_recency_eligible_free_blocks']==2
 assert r['gpu_recency_queue_mutation_performed'] and r['gpu_observation_free_queue_validated']
 after=snapshot(p)['free'];assert set(after)==set(initial) and len(after)==len(set(after))==free
 assert after[-2:]==[b[1].block_id,b[0].block_id]
 # Real native allocation still works and consumes all other blocks first.
 acquired=p.get_new_blocks(free-2);assert all(x not in b for x in acquired)
 acquired+=p.get_new_blocks(2);assert p.get_num_free_blocks()==0
 p.free_blocks(acquired);assert p.get_num_free_blocks()==free

def test_active_blocks_never_move_or_change_references():
 p=BlockPool(8,True,16);keys,b=cached(p);p.free_blocks([b[1]])
 before=stable(p);r=refresh_gpu_prefix_recency(p,keys)
 assert stable(p)==before and b[0].ref_cnt==1 and r['gpu_recency_refreshed_free_blocks']==1
 assert b[0].prev_free_block is b[0].next_free_block is None

def test_duplicate_cached_ids_move_once():
 p=BlockPool(8,True,16);keys,b=cached(p,1);p.free_blocks(b)
 before=stable(p);r=refresh_gpu_prefix_recency(p,[keys[0],keys[0]])
 assert stable(p)==before and r['gpu_recency_refreshed_free_blocks']==1
 assert len(snapshot(p)['free'])==len(set(snapshot(p)['free']))

def test_recycled_hash_never_refreshes_stale_identity():
 p=BlockPool(4,True,16);keys,b=cached(p);p.free_blocks(b);new=p.get_new_blocks(3)
 before=snapshot(p);r=refresh_gpu_prefix_recency(p,keys)
 assert snapshot(p)==before and r['gpu_recency_refreshed_free_blocks']==0 and not r['gpu_recency_queue_mutation_performed']
 p.free_blocks(new)

def test_invalid_membership_fails_before_any_mutation():
 p=BlockPool(8,True,16);keys,b=cached(p);p.free_blocks(b);p.free_block_queue.num_free_blocks+=1
 try:
  before=snapshot(p);r=refresh_gpu_prefix_recency(p,keys);assert snapshot(p)==before
  assert not r['gpu_observation_available'] and r['gpu_recency_eligible_free_blocks'] is None and r['gpu_recency_refreshed_free_blocks']==0
 finally:p.free_block_queue.num_free_blocks-=1

def test_repeated_refresh_keeps_queue_integrity_and_state_bounded():
 p=BlockPool(12,True,16);keys,b=cached(p);p.free_blocks(b);before=stable(p);fields=set(vars(p))
 for _ in range(1000):
  r=refresh_gpu_prefix_recency(p,keys)
  assert r['gpu_recency_refreshed_free_blocks']==2 and len(snapshot(p)['free'])==p.get_num_free_blocks()
 assert stable(p)==before and set(vars(p))==fields

def native_hint():
 from tests.test_preservation_audit_mode import native,source,_wait_hint
 s,m=native(wait_hint_native_adoption=True,wait_hint_gpu_cache_recency=True)
 req,prefix,keys=source(s);ctx=s._req_status[req.request_id].req_context
 prepared=m.prepare_store(keys,ctx);m.complete_store(prepared.keys_to_store,ctx);s._register_observed_prefix(req)
 blocks=s.gpu_block_pool.get_new_blocks(2);s.gpu_block_pool.cache_full_blocks(req,blocks,0,2,16,0);s.gpu_block_pool.free_blocks(blocks)
 return s,m,keys,_wait_hint(s,prefix,32,session_id='one')

def test_native_audit_off_hint_changes_order_only_and_replay_is_noop():
 s,m,keys,hint=native_hint();p=s.gpu_block_pool;before=stable(p);free=p.get_num_free_blocks()
 r=s.publish_wait_hint(hint);assert r['gpu_recency_refreshed_free_blocks']==2 and r['ready_cpu_prefix_keys']==2
 assert stable(p)==before and p.get_num_free_blocks()==free and not s._jobs and not s._wait_prefetch
 assert not s._preservation_requests and len(s._observed_prefix_generations)==len(s._observed_prefixes)==1
 after=snapshot(p)
 with patch('semantic_offload.gpu_cache_recency.refresh_gpu_prefix_recency',side_effect=AssertionError('replayed mutation')):second=s.publish_wait_hint(hint)
 assert second['idempotent_replay'] and snapshot(p)==after

@pytest.mark.parametrize('bad',['namespace','generation','source','future_clock','expired','global_disable'])
def test_causal_guards_precede_gpu_mutation(retention,bad):
 s,m,p,keys,hint=retention;s._wait_hint_gpu_cache_recency=True
 if bad=='namespace':hint['namespace']=dict(hint['namespace'],model_revision='other')
 elif bad=='generation':s._observed_prefix_generations[('run','cell','session')]=1
 elif bad=='source':s._observed_prefixes.clear()
 elif bad=='future_clock':hint['input_publication_ns']+=10_000_000_000
 elif bad=='global_disable':s._prefetch_budget_fraction=0
 clock=hint['observation_available_ns']+(3_000_000_000 if bad=='expired' else 1)
 with patch('semantic_offload.connector.time.monotonic_ns',return_value=clock),patch('semantic_offload.gpu_cache_recency.refresh_gpu_prefix_recency',side_effect=AssertionError('invalid mutation')):r=s.publish_wait_hint(hint)
 assert r['reason']!='cpu_prefix_recency_refreshed'

@pytest.mark.parametrize('flag',[0,1,None,'true'])
def test_gpu_recency_flag_requires_boolean(flag):
 from tests.test_preservation_audit_mode import native
 with pytest.raises(TypeError,match='must be a JSON boolean'):native(wait_hint_gpu_cache_recency=flag)

def test_gpu_recency_requires_qualified_adoption_layout():
 from tests.test_preservation_audit_mode import native
 with pytest.raises(ValueError,match='dense TP1'):native(wait_hint_gpu_cache_recency=True)

def test_last_preflight_lookup_failure_is_atomic_no_partial_moves():
 p=BlockPool(8,True,16);keys,b=cached(p);p.free_blocks(b);before=snapshot(p)
 real=p.get_cached_block;calls=0
 def lookup(*args,**kw):
  nonlocal calls
  calls+=1
  return None if calls==4 else real(*args,**kw)
 with patch.object(p,'get_cached_block',side_effect=lookup),patch.object(p.free_block_queue,'remove',side_effect=AssertionError('partial move')),patch.object(p.free_block_queue,'append',side_effect=AssertionError('partial move')):
  r=refresh_gpu_prefix_recency(p,keys)
 assert calls==4 and snapshot(p)==before and r['gpu_recency_refreshed_free_blocks']==0
 assert r['gpu_recency_unavailable_reason']=='cached_index_changed'

def test_missing_middle_keeps_unreachable_cached_suffix_unmoved():
 p=BlockPool(10,True,16);keys,b=cached(p,3);p.free_blocks(b)
 assert p._maybe_evict_cached_block(b[1])
 others=p.get_new_blocks(2);p.free_blocks(others);before=snapshot(p)['free']
 real_remove=p.free_block_queue.remove
 with patch.object(p.free_block_queue,'remove',wraps=real_remove) as remove:
  r=refresh_gpu_prefix_recency(p,keys)
 assert r['gpu_observation_cached_prefix_keys']==r['gpu_recency_refreshed_free_blocks']==1
 assert remove.call_count==1 and remove.call_args.args[0] is b[0]
 assert snapshot(p)['free']==[i for i in before if i!=b[0].block_id]+[b[0].block_id]
 assert p.get_cached_block(get_offload_block_hash(keys[2]),[0])[0] is b[2]

def test_all_active_prefix_has_zero_queue_mutation():
 p=BlockPool(8,True,16);keys,b=cached(p);before=snapshot(p)
 with patch.object(p.free_block_queue,'remove',side_effect=AssertionError('active move')),patch.object(p.free_block_queue,'append',side_effect=AssertionError('active move')):
  r=refresh_gpu_prefix_recency(p,keys)
 assert snapshot(p)==before and r['gpu_recency_eligible_free_blocks']==r['gpu_recency_refreshed_free_blocks']==0
 assert r['gpu_observation_cached_active_prefix_keys']==2 and not r['gpu_recency_queue_mutation_performed']
