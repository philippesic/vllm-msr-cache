from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from semantic_offload.gpu_prefix_coverage import describe_gpu_prefix
from tests.test_step_1_5_prefetch import to_key
from tests.test_wait_hint_cpu_recency import retention
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_offload.base import get_offload_block_hash

def cached(pool,n=2):
 keys=[to_key(i+1) for i in range(n)];blocks=pool.get_new_blocks(n)
 pool.cache_full_blocks(NS(block_hashes=[get_offload_block_hash(k) for k in keys]),blocks,0,n,16,0)
 return keys,blocks

def snapshot(pool):
 return {'free':[b.block_id for b in pool.free_block_queue.get_all_free_blocks()],'blocks':[(b.block_id,b.ref_cnt,b.block_hash,b.block_hash_num_tokens,id(b.prev_free_block),id(b.next_free_block)) for b in pool.blocks]}

def test_unsupported_is_unavailable_not_zero():
 r=describe_gpu_prefix(object(),[to_key(1)])
 assert r['gpu_observation_available'] is False and r['gpu_observation_cached_prefix_keys'] is None

def test_exact_free_active_coverage_readonly():
 p=BlockPool(8,True,16);keys,b=cached(p);p.free_blocks([b[0]])
 before=snapshot(p);r=describe_gpu_prefix(p,keys);assert snapshot(p)==before
 assert r['gpu_observation_available'] and r['gpu_observation_free_queue_validated']
 assert r['gpu_observation_full_prefix_cached'] and r['gpu_observation_cached_prefix_keys']==2
 assert r['gpu_observation_cached_free_prefix_keys']==r['gpu_observation_cached_active_prefix_keys']==1

def test_contiguous_prefix_stops_at_missing_root():
 p=BlockPool(8,True,16);keys,b=cached(p);p.free_blocks(b);before=snapshot(p)
 r=describe_gpu_prefix(p,[to_key(900),*keys]);assert snapshot(p)==before
 assert r['gpu_observation_available'] and r['gpu_observation_cached_prefix_keys']==0 and not r['gpu_observation_full_prefix_cached']

def test_native_recycling_removes_old_prefix():
 p=BlockPool(4,True,16);keys,b=cached(p);p.free_blocks(b)
 new=p.get_new_blocks(3);before=snapshot(p);r=describe_gpu_prefix(p,keys)
 assert snapshot(p)==before and r['gpu_observation_cached_prefix_keys']==0
 p.free_blocks(new)

def test_invalid_queue_count_unavailable_not_zero():
 p=BlockPool(8,True,16);keys,b=cached(p);p.free_blocks(b);p.free_block_queue.num_free_blocks+=1
 try:
  before=snapshot(p);r=describe_gpu_prefix(p,keys);assert snapshot(p)==before
  assert not r['gpu_observation_available'] and r['gpu_observation_cached_prefix_keys'] is None
 finally:p.free_block_queue.num_free_blocks-=1

def test_foreign_cached_block_identity_rejected():
 p=BlockPool(8,True,16);keys,b=cached(p);other=BlockPool(8,True,16);_,foreign=cached(other);before=snapshot(p)
 with patch.object(p,'get_cached_block',return_value=[foreign[0]]):r=describe_gpu_prefix(p,keys)
 assert snapshot(p)==before and not r['gpu_observation_available']

@pytest.mark.parametrize('bad',['namespace','generation','source','future_clock','expired','global_disable'])
def test_causal_guards_precede_observation(retention,bad):
 s,m,p,keys,hint=retention;s._wait_hint_gpu_prefix_observation=True
 if bad=='namespace':hint['namespace']=dict(hint['namespace'],model_revision='other')
 elif bad=='generation':s._observed_prefix_generations[('run','cell','session')]=1
 elif bad=='source':s._observed_prefixes.clear()
 elif bad=='future_clock':hint['input_publication_ns']+=10_000_000_000
 elif bad=='global_disable':s._prefetch_budget_fraction=0
 clock=hint['observation_available_ns']+(3_000_000_000 if bad=='expired' else 1)
 with patch('semantic_offload.connector.time.monotonic_ns',return_value=clock),patch('semantic_offload.gpu_prefix_coverage.describe_gpu_prefix',side_effect=AssertionError('invalid observation')):r=s.publish_wait_hint(hint)
 assert r['reason']!='cpu_prefix_recency_refreshed'

def test_full_cpu_ready_still_observes_gpu_without_mutation():
 from tests.test_preservation_audit_mode import native,source,_wait_hint
 s,m=native(wait_hint_native_adoption=True,wait_hint_gpu_prefix_observation=True)
 req,prefix,keys=source(s);ctx=s._req_status[req.request_id].req_context
 prepared=m.prepare_store(keys,ctx);m.complete_store(prepared.keys_to_store,ctx);s._register_observed_prefix(req)
 blocks=s.gpu_block_pool.get_new_blocks(2);s.gpu_block_pool.cache_full_blocks(req,blocks,0,2,16,0);s.gpu_block_pool.free_blocks(blocks)
 before=snapshot(s.gpu_block_pool);r=s.publish_wait_hint(_wait_hint(s,prefix,32,session_id='one'))
 assert snapshot(s.gpu_block_pool)==before and r['ready_cpu_prefix_keys']==2
 assert r['gpu_observation_full_prefix_cached'] and r['gpu_observation_cached_free_prefix_keys']==2
 assert not s._jobs and not s._wait_prefetch and not s._preservation_requests

@pytest.mark.parametrize('flag',[0,1,None,'true'])
def test_observation_flag_requires_boolean(flag):
 from tests.test_preservation_audit_mode import native
 with pytest.raises(TypeError,match='must be a JSON boolean'):native(wait_hint_gpu_prefix_observation=flag)

def test_observation_requires_qualified_adoption_layout():
 from tests.test_preservation_audit_mode import native
 with pytest.raises(ValueError,match='dense TP1'):native(wait_hint_gpu_prefix_observation=True)
