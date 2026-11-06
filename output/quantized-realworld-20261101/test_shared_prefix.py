"""Private shared-prefix correctness, using real native allocator and CPU tensors.
CUDA transfer execution is qualified separately, never inferred from these tests.
"""
import importlib.util,sys,time
from pathlib import Path
import pytest,torch
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
spec=importlib.util.spec_from_file_location('shared_prefix_base_fixtures',ROOT/'tests/test_step_1_5_prefetch.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from semantic_offload.connector import SemanticTransferCompletion,SemanticWorkerMetadata

def fixture(refs=0,complete=True):
 f=base._native_adoption_fixture(torch.float16);old=f.state
 f.sched._retire_wait_prefetch(old,'cancel',now_ns=time.monotonic_ns());base._native_adoption_complete(f)
 assert f.cache.block_pool.get_num_free_blocks()==11
 pool=f.cache.block_pool;local=pool.get_new_blocks(1)[0]
 f.backing[local.block_id].copy_(f.dram[0]);pool.cache_full_blocks(f.warm,[local],0,1,16,0)
 if refs==0:pool.free_blocks([local])
 elif refs>1:pool.touch([local]*(refs-1))
 f.local=local;f.local_original_refs=refs;f.sched._wait_hint_shared_local_prefix=True;f.sched._prefetch_require_full_prefix=True
 f.hint=base._wait_hint(f.sched,f.hint['observed_prefix_id'],32,event_id='b'*64)
 result=f.sched.publish_wait_hint(f.hint);assert result['status']=='accepted',result
 f.sched._retry_wait_prefetches();f.state=f.sched._wait_prefetch[f.hint['event_id']]
 assert f.state.status=='in_flight';assert f.state.publication_local_gpu_block_ids==(local.block_id,)
 assert f.state.keys==f.state.source.keys[1:];assert f.sched._prefetch_reserved_blocks==1
 job=f.sched._current_batch_prefetch_load_jobs[f.state.job_id]
 assert list(job.dst_spec.block_indices)==[1]
 f.sched._current_batch_prefetch_load_jobs.clear()
 if complete:finish(f)
 return f

def finish(f,success=True):
 state=f.state
 for key,bid in zip(state.keys,state.gpu_block_ids):f.backing[bid].copy_(f.dram[state.source.keys.index(key)])
 now=time.monotonic_ns();f.sched._preservation_job_events[state.job_id]=SemanticTransferCompletion(is_load=True,bytes=len(state.keys)*128,time_s=.001,contributors=1,success=success,started_at_ns=now-1_000_000,completed_at_ns=now)
 f.sched._complete_wait_prefetch_jobs(SemanticWorkerMetadata(completed_jobs={state.job_id:1} if success else {},failed_jobs={} if success else {state.job_id:1}))

@pytest.mark.parametrize('refs',[0,1,3])
def test_exact_shared_prefix_adoption_keeps_other_owners_and_only_counts_suffix(refs):
 f=fixture(refs);pool=f.cache.block_pool;suffix=list(f.state.gpu_block_ids);before=f.local.ref_cnt
 req=base._native_adoption_bind(f)
 assert f.state.status=='published';assert f.local.ref_cnt==before
 assert all(pool.blocks[i].ref_cnt==0 for i in suffix)
 assert f.sched._prefetch_reserved_blocks==0
 pub=f.sched._published_wait_prefixes[req.request_id]
 assert pub['block_ids']==[f.local.block_id]+suffix
 assert pub['prefetched_block_count']==1 and pub['local_prefix_count']==1
 blocks=base._native_adoption_allocate(f,req)
 assert f.local.ref_cnt==before+1
 assert torch.equal(f.backing[[b.block_id for b in blocks.blocks[0][:2]]],f.dram)
 f.cache.free(req);assert f.local.ref_cnt==before

@pytest.mark.parametrize('invalid',['evicted','wrong_tokens','wrong_hash','missing_fence','suffix_already_resident','different_local_physical'])
def test_rejected_shared_bind_preserves_local_owners_and_never_splices(invalid):
 f=fixture();pool=f.cache.block_pool;req=base._native_adoption_request('resume')
 if invalid=='evicted':pool.evict_blocks({f.local.block_id})
 elif invalid=='wrong_tokens':req=base._native_adoption_request('resume',[90]*35)
 elif invalid=='wrong_hash':req.block_hashes[1]=b'wrong-suffix'
 elif invalid=='missing_fence':f.state.transfer_record['completed_at_ns']=None
 elif invalid=='suffix_already_resident':
  extra=pool.get_new_blocks(1);pool.cache_full_blocks(f.warm,[f.local,*extra],1,2,16,0);pool.free_blocks(extra)
 elif invalid=='different_local_physical':
  pool.evict_blocks({f.local.block_id});extra=pool.get_new_blocks(1);pool.cache_full_blocks(f.warm,extra,0,1,16,0);pool.free_blocks(extra)
 oldrefs=f.local.ref_cnt;ids=f.state.gpu_block_ids;base._native_adoption_bind(f,req)
 assert req.request_id not in f.sched._published_wait_prefixes
 assert req.request_id not in f.sched._prefetched
 assert f.local.ref_cnt==oldrefs
 assert all(pool.blocks[i].ref_cnt==0 for i in ids)
 assert f.sched._prefetch_reserved_blocks==0

@pytest.mark.parametrize('finish_success',[True,False])
def test_prefix_eviction_while_transfer_pending_never_recycles_before_fence(finish_success):
 f=fixture(complete=False);pool=f.cache.block_pool;ids=f.state.gpu_block_ids
 pool.evict_blocks({f.local.block_id});req=base._native_adoption_bind(f)
 assert req.request_id not in f.sched._prefetched
 assert all(pool.blocks[i].ref_cnt==1 for i in ids)
 assert f.state.status=='retired_pending_fence'
 finish(f,success=finish_success)
 assert all(pool.blocks[i].ref_cnt==0 for i in ids)
 assert f.sched._prefetch_reserved_blocks==0
 assert not f.sched._published_wait_prefixes

@pytest.mark.parametrize('invalid',['duplicate','foreign','shared_suffix_ref','wrong_group','wrong_size','skip','no_fence','noncontiguous','wrong_prefix_order'])
def test_suffix_pool_rejects_atomically_without_touching_prefix(invalid):
 f=fixture();pool=f.cache.block_pool;local=[f.local];suffix=[pool.blocks[i] for i in f.state.gpu_block_ids];req=base._native_adoption_request('resume');args=dict(block_size=16,kv_cache_group_id=0,completion_verified=True)
 if invalid=='duplicate':suffix=[f.local]
 elif invalid=='foreign':
  from vllm.v1.core.block_pool import BlockPool
  suffix=BlockPool(12,True,16).get_new_blocks(1)
 elif invalid=='shared_suffix_ref':pool.touch(suffix)
 elif invalid=='wrong_group':args['kv_cache_group_id']=1
 elif invalid=='wrong_size':args['block_size']=32
 elif invalid=='skip':req.skip_reading_prefix_cache=True
 elif invalid=='no_fence':args['completion_verified']=False
 elif invalid=='noncontiguous':local=[]
 elif invalid=='wrong_prefix_order':req.block_hashes[0],req.block_hashes[1]=req.block_hashes[1],req.block_hashes[0]
 before=(pool.get_num_free_blocks(),f.local.ref_cnt,f.local.block_hash)
 assert pool.publish_completed_suffix(req,local,suffix,**args) is False
 assert before==(pool.get_num_free_blocks(),f.local.ref_cnt,f.local.block_hash)


def test_suffix_publication_exception_rolls_back_only_suffix(monkeypatch):
 f=fixture(refs=2);pool=f.cache.block_pool;local=f.local;suffix=[pool.blocks[i] for i in f.state.gpu_block_ids];original=pool.cache_full_blocks
 def fail(*a,**kw):original(*a,**kw);raise RuntimeError('injected failure')
 monkeypatch.setattr(pool,'cache_full_blocks',fail);before=(local.ref_cnt,local.block_hash,pool.get_num_free_blocks())
 with pytest.raises(RuntimeError,match='injected failure'):pool.publish_completed_suffix(f.warm,[local],suffix,block_size=16,kv_cache_group_id=0,completion_verified=True)
 assert (local.ref_cnt,local.block_hash,pool.get_num_free_blocks())==before
 assert all(b.ref_cnt==1 and b.block_hash is None for b in suffix)
 assert pool.get_cached_block(f.warm.block_hashes[0],[0])[0] is local

@pytest.mark.parametrize('invalid',['no_native','unsupported_dtype','tp','factor','not_full','nonbool'])
def test_shared_feature_constructor_fails_closed(invalid,monkeypatch):
 from types import SimpleNamespace
 from semantic_offload.connector import SemanticOffloadingConnectorScheduler
 from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import SchedulerOffloadConfig,GroupOffloadConfig
 from vllm.v1.kv_cache_interface import KVCacheConfig,KVCacheGroupSpec,FullAttentionSpec
 native=SchedulerOffloadConfig(kv_group_configs=(GroupOffloadConfig(0,16,16,1,None,None),),blocks_per_chunk=2 if invalid=='factor' else 1,num_workers=2 if invalid=='tp' else 1,offload_prompt_only=False)
 def initialize(scheduler,*a):scheduler.config=native;scheduler.manager=object()
 monkeypatch.setattr('semantic_offload.connector.construct_scheduler_base',initialize)
 opts={'wait_hint_native_adoption':invalid!='no_native','prefetch_require_full_prefix':invalid!='not_full','wait_hint_shared_local_prefix':'yes' if invalid=='nonbool' else True}
 spec=SimpleNamespace(extra_config=opts,replicated_layout=False,cpu_page_size_per_worker=128,kv_bytes_per_chunk=128*native.num_workers)
 cfg=SimpleNamespace(model_config=SimpleNamespace(model='test-model'),cache_config=SimpleNamespace(block_size=16))
 cache=KVCacheConfig(12,[],[KVCacheGroupSpec(['layer'],FullAttentionSpec(block_size=16,num_kv_heads=1,head_size=4,dtype=torch.float32 if invalid=='unsupported_dtype' else torch.float16))])
 with pytest.raises((ValueError,TypeError),match='wait_hint_shared_local_prefix'):SemanticOffloadingConnectorScheduler(spec,cfg,cache)


def test_prefix_eviction_during_reservation_unwinds_without_source_load(monkeypatch):
 f=fixture();s=f.sched;pool=f.cache.block_pool
 s._retire_wait_prefetch(f.state,'cancel',now_ns=time.monotonic_ns())
 f.hint=base._wait_hint(s,f.hint['observed_prefix_id'],32,event_id='c'*64)
 assert s.publish_wait_hint(f.hint)['status']=='accepted'
 state=s._wait_prefetch[f.hint['event_id']];allocated=[];original=pool.get_new_blocks
 def allocate(n):
  result=original(n);allocated.extend(b.block_id for b in result);pool.evict_blocks({f.local.block_id});return result
 monkeypatch.setattr(pool,'get_new_blocks',allocate)
 old_refs=[s.manager._policy.get(k).ref_cnt for k in state.source.keys]
 s._retry_wait_prefetches()
 assert state.status=='pending' and state.job_id is None
 assert state.selection_evidence['status']=='local_prefix_changed_during_reservation'
 assert s._prefetch_reserved_blocks==0 and not s._current_batch_prefetch_load_jobs
 assert all(pool.blocks[i].ref_cnt==0 for i in allocated)
 assert set(allocated).isdisjoint(s._current_batch_allocated_block_ids)
 assert old_refs==[s.manager._policy.get(k).ref_cnt for k in state.source.keys]
