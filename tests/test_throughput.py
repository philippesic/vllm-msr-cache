"""Native allocator cache preservation; CUDA forbidden by CPU driver.

BlockPool owns prefix hashes, free ordering, and refs. Failure guarded: treating
cached reclaimable blocks as scratch erases the first hash of a reusable chain.
Use the real allocator and scheduler capacity method; no model is needed.
"""
from types import SimpleNamespace as N
import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from semantic_offload.connector import SemanticOffloadingConnectorScheduler

def pool_and_chain():
 p=BlockPool(8,True,16);blocks=p.get_new_blocks(7);hashes=[(i+1).to_bytes(32,'big') for i in range(4)]
 for block,h in zip(blocks,hashes):p._insert_block_hash(make_block_hash_with_group_id(h,0),block,16)
 p.free_blocks(blocks[:4]);return p,blocks,hashes

def scheduler(pool,enabled=True):
 # Capacity is a pure allocator view; actual constructor integration/semantic
 # scarce-budget selection remains in retained real-native scheduler suite.
 s=SemanticOffloadingConnectorScheduler.__new__(SemanticOffloadingConnectorScheduler)
 s.gpu_block_pool=pool;s._prefetch_preserve_gpu_prefix_cache=enabled;s._prefetch_budget_fraction=1.;s._prefetch_headroom_blocks=0
 return s

def test_native_scratch_can_destroy_first_reusable_prefix_hash():
 p,blocks,hashes=pool_and_chain();assert all(p.get_cached_block(h,[0]) for h in hashes)
 victim=p.get_new_blocks(1);assert victim[0] is blocks[0]
 assert p.get_cached_block(hashes[0],[0]) is None
 assert all(p.get_cached_block(h,[0]) for h in hashes[1:])
 p.free_blocks(victim)

def test_safe_capacity_defers_without_touching_cached_chain_or_refs():
 p,blocks,hashes=pool_and_chain();order=list(p.free_block_queue.get_all_free_blocks());s=scheduler(p)
 assert s._prefetch_free_capacity()==0
 assert p.free_block_queue.get_all_free_blocks()==order
 assert all(p.get_cached_block(h,[0]) for h in hashes)
 assert p.get_num_free_blocks()==4 and all(b.ref_cnt==0 for b in blocks[:4])

def test_safe_head_reserves_only_unhashed_blocks_through_native_allocator():
 p,blocks,hashes=pool_and_chain();p.free_blocks(blocks[4:6]);s=scheduler(p)
 assert s._prefetch_free_capacity()==2
 reserved=p.get_new_blocks(s._prefetch_free_capacity());assert [b.block_id for b in reserved]==[b.block_id for b in blocks[4:6]]
 assert all(b.ref_cnt==1 for b in reserved) and all(p.get_cached_block(h,[0]) for h in hashes)
 p.free_blocks(reserved);assert s._prefetch_free_capacity()==2
 assert p.get_num_free_blocks()==6

def test_disabled_option_keeps_legacy_capacity_and_unsupported_pool_fails_closed():
 p,_,_=pool_and_chain();assert scheduler(p,False)._prefetch_free_capacity()==4
 stub=N(num_gpu_blocks=8,get_num_free_blocks=lambda:5)
 assert scheduler(stub,True)._prefetch_free_capacity()==0
 assert scheduler(stub,False)._prefetch_free_capacity()==5

def test_alias_hash_also_prevents_scratch_reservation():
 p=BlockPool(3,True,16);head=p.free_block_queue.get_all_free_blocks()[0]
 p.cached_block_hashes_by_block[head.block_id]={make_block_hash_with_group_id(bytes(32),0)}
 assert scheduler(p)._prefetch_free_capacity()==0

def test_safe_budget_enters_actual_competition_and_consumes_real_native_scores():
 # Actual constructors/native metadata branch from the preserved round2 suite.
 # Only its CUDA transfer-handler constructor is stubbed; CPU query tensors
 # substitute for the model probe. No rating/lease insertion bypasses selection.
 import torch
 from tests.test_round2 import native,source,_wait_hint
 from vllm.v1.core.sched.output import SchedulerOutput
 from vllm.v1.request import RequestStatus
 from vllm.v1.outputs import KVConnectorOutput
 c,s,w,m,a,b=native();s._prefetch_preserve_gpu_prefix_cache=True
 s._prefetch_budget_fraction=.1;s._prefetch_headroom_blocks=0
 p=s.gpu_block_pool;all_blocks=p.get_new_blocks(319)
 cached=all_blocks[-16:];scratch=all_blocks[-32:-16]
 cache_hashes=[(i+900).to_bytes(32,'big') for i in range(16)]
 for block,h in zip(cached,cache_hashes):p._insert_block_hash(make_block_hash_with_group_id(h,0),block,1)
 p.free_blocks(cached);p.free_blocks(scratch)
 assert p.get_num_free_blocks()==32 and s._prefetch_free_capacity()==16
 try:
  requests=[(source(s,label,keys),label) for keys,label in [(a,'a'),(b,'b')]]
  c.bind_connector_metadata(s.build_connector_meta(SchedulerOutput.make_empty()))
  w._on_queries_captured(['warm-a','warm-b'],torch.ones(2,1,4))
  for (req,prefix),label in requests:
   req.status=RequestStatus.FINISHED_STOPPED;s.request_finished(req)
   assert s.publish_wait_hint(_wait_hint(s,prefix,16,session_id='session-'+label,event_id=label*64))['status']=='accepted'
  first=s.build_connector_meta(SchedulerOutput.make_empty())
  assert first.scoring_requests=={'warm-a':a,'warm-b':b}
  assert not first.prefetch_load_jobs and all(x.status=='pending' for x in s._wait_prefetch.values())
  c.bind_connector_metadata(first);w._drain_async_scores(wait=True)
  meta=c.build_connector_worker_meta();assert len(meta.async_score_observations)==2
  c.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))
  assert m.relevance_ema['mean'][a[0]]>m.relevance_ema['mean'][b[0]]
  s.build_connector_meta(SchedulerOutput.make_empty())
  flight=[state for state in s._wait_prefetch.values() if state.status=='in_flight']
  assert len(flight)==1 and flight[0].source.source_request_id=='warm-a'
  assert list(flight[0].keys)==a
  assert set(flight[0].gpu_block_ids)=={block.block_id for block in scratch}
  assert all(p.get_cached_block(h,[0]) for h in cache_hashes)
  assert s._prefetch_free_capacity()==0 and p.get_num_free_blocks()==16
  assert 'arc' in type(m._policy).__name__.lower()
 finally:w._async_scoring.shutdown()

def test_allocator_ledger_reports_actual_victims_without_changing_native_result():
 import sys
 from pathlib import Path
 from unittest.mock import patch
 from tests import allocation_ledger as ledger
 import vllm.v1.core.block_pool as module
 p1,b1,h1=pool_and_chain();plain=p1.get_new_blocks(1)
 expected=([x.block_id for x in plain],p1.get_num_free_blocks(),
           [(x.block_id,x.ref_cnt,x.block_hash) for x in p1.blocks],
           [p1.get_cached_block(h,[0]) is not None for h in h1])
 p2,b2,h2=pool_and_chain();ledger.COUNTS.clear();ledger.ROWS.clear();ledger.PATCHED.discard(module.__name__)
 try:
  with patch.object(BlockPool,'get_new_blocks',BlockPool.get_new_blocks),patch.object(BlockPool,'_maybe_evict_cached_block',BlockPool._maybe_evict_cached_block):
   ledger.patch(module);token=ledger.CONTEXT.set(('staging_wait','source'))
   try:observed=p2.get_new_blocks(1)
   finally:ledger.CONTEXT.reset(token)
   actual=([x.block_id for x in observed],p2.get_num_free_blocks(),
           [(x.block_id,x.ref_cnt,x.block_hash) for x in p2.blocks],
           [p2.get_cached_block(h,[0]) is not None for h in h2])
   assert actual==expected
   row=ledger.ROWS[-1];assert row['context']==('staging_wait','source')
   assert row['victims']==[(b2[0].block_id,(make_block_hash_with_group_id(h2[0],0),))]
   assert ledger.COUNTS['victim_blocks:staging_wait']==1
   before=p2.get_num_free_blocks()
   with pytest.raises(ValueError):p2.get_new_blocks(before+1)
   assert p2.get_num_free_blocks()==before and ledger.COUNTS['allocation_errors']==1
 finally:ledger.PATCHED.discard(module.__name__)
