import ast,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
H=Path(__file__).resolve().parent
@pytest.fixture
def scheduler():
 tree=ast.parse((H/'sources/demand-capacity/semantic_offload/connector.py').read_text());names={'_native_prefix_free_block_ids','_reclaim_wait_prefetch_for_demand'}
 body=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
 for n in body:n.decorator_list=[]
 events=[];ns={'config_blocks_per_chunk':lambda c:c.chunk,'group_config_tokens_per_block':lambda g:16,'cdiv':lambda a,b:(a+b-1)//b,'record_wait_hint_event':events.append,'time':NS(monotonic_ns=lambda:999)}
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+body,type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),'actual-demand-capacity','exec'),ns)
 blocks={b'a':NS(block_id=2,ref_cnt=0,is_null=False),b'b':NS(block_id=3,ref_cnt=1,is_null=False),b'c':NS(block_id=4,ref_cnt=0,is_null=False)}
 pool=NS(enable_caching=True,get_cached_block=lambda h,g:[blocks[h]] if h in blocks else None,get_num_free_blocks=lambda:6)
 st=NS(bound_request_id=None,status='ready',gpu_block_ids=(10,11),lease=NS(hint=NS(input_publication_ns=1,event_id='e')))
 x=NS(config=NS(chunk=1,kv_group_configs=[NS(group_idx=0)]),gpu_block_pool=pool,_staging_mode='owned_reclaimable',_wait_prefetch={'e':st},_prefetched={},_jobs={},_prefetch_bytes_per_key=16,freed=[])
 def free(state,reason):x.freed.append(reason);x._wait_prefetch.pop('e')
 x._free_wait_prefetch=free
 for name in names:setattr(x,name,types.MethodType(ns[name],x))
 req=NS(request_id='actual-demand',block_hashes=[b'a',b'b',b'c'],num_prompt_tokens=128)
 return x,req,blocks,events
def test_counts_cached_free_pages_but_shared_pages_cost_no_new_slot(scheduler):
 s,r,b,e=scheduler;before=[x.ref_cnt for x in b.values()]
 assert s._native_prefix_free_block_ids(r,48)=={2,4}
 assert [x.ref_cnt for x in b.values()]==before
def test_reclaims_when_remainder_alone_fits_but_native_admission_cannot(scheduler):
 s,r,b,e=scheduler
 # 80 new tokens =>5 blocks; two cached zero-ref pages also need ownership.
 # Six free pages cannot serve seven acquisitions. Old remainder-only math
 # wrongly skipped reclaim here.
 s._reclaim_wait_prefetch_for_demand(r,48,protected_event_id=None)
 assert s.freed==['demand_reclaim']
 assert e[-1]['demand_blocks']==7 and e[-1]['native_cached_free_prefix_blocks']==2
 assert e[-1]['shortfall_blocks']==1 and e[-1]['reclaimed_ready_blocks']==2
def test_no_reclaim_when_real_capacity_suffices(scheduler):
 s,r,b,e=scheduler;s.gpu_block_pool.get_num_free_blocks=lambda:7
 s._reclaim_wait_prefetch_for_demand(r,48,protected_event_id=None);assert not s.freed and not e
@pytest.mark.parametrize('mode',['protected','bound','inflight'])
def test_never_reclaims_request_owner_or_unfinished_transfer(scheduler,mode):
 s,r,b,e=scheduler
 if mode=='bound':s._wait_prefetch['e'].bound_request_id='actual-request'
 if mode=='inflight':s._wait_prefetch['e'].status='in_flight'
 s._reclaim_wait_prefetch_for_demand(r,48,protected_event_id='e' if mode=='protected' else None)
 assert not s.freed
 if mode=='inflight':assert e[-1]['in_flight_unreclaimable_blocks']==2 and e[-1]['demand_wait_required']
@pytest.mark.parametrize('bad',['multi','chunk','disabled','unaligned','missing_lookup','missing_key','positive','duplicate','null'])
def test_layout_and_cache_controls(scheduler,bad):
 s,r,b,e=scheduler;expected={2,4}
 if bad=='multi':s.config.kv_group_configs.append(NS(group_idx=1));expected=set()
 if bad=='chunk':s.config.chunk=2;expected=set()
 if bad=='disabled':s.gpu_block_pool.enable_caching=False;expected=set()
 if bad=='missing_lookup':s.gpu_block_pool.get_cached_block=None;expected=set()
 if bad=='missing_key':b.pop(b'c');expected={2}
 if bad=='positive':b[b'a'].ref_cnt=2;expected={4}
 if bad=='duplicate':b[b'c']=b[b'a'];expected={2}
 if bad=='null':b[b'a'].is_null=True;expected={4}
 assert s._native_prefix_free_block_ids(r,47 if bad=='unaligned' else 48)==(set() if bad=='unaligned' else expected)
