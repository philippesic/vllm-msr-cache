import ast,time,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from test_partial import case,digest
from test_tailfirst import Pool,load as old_pool_methods
H=Path(__file__).resolve().parent
def load(path,names,ns):
 tree=ast.parse((H/path).read_text());body=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(body)==len(names)
 for n in body:n.decorator_list=[]
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+body,type_ignores=[])
 exec(compile(ast.fix_missing_locations(mod),'actual-evictable-warm','exec'),ns)
 return ns
def native():return load('runtime-evictable-warm/vllm/v1/core/block_pool.py',{'publish_completed_observed_keys'},{'get_group_id':lambda k:int.from_bytes(k[-4:],'big'),'get_block_hash':lambda k:k[:-4]})
@pytest.fixture
def pool():
 p=Pool();p.enable_kv_cache_events=False;p.semantic_allocation_epochs=[7]*8
 def insert(key,b,num_tokens):b.block_hash=key;b.block_hash_num_tokens=num_tokens;p.cached[key[:-4]]=b
 p._insert_block_hash=insert
 p.publish=types.MethodType(native()['publish_completed_observed_keys'],p)
 return p,[bytes([v])*32+b'\0'*4 for v in [1,2]]
def test_native_publication_preserves_bytes_epochs_and_releases_all_reservations(pool):
 p,keys=pool;data=[b.data for b in p.pages];epochs=p.semantic_allocation_epochs[:]
 assert p.publish(keys,p.pages[:2],start_index=3,block_size=16,completion_verified=True)
 assert [b.block_hash_num_tokens for b in p.pages[:2]]==[64,80]
 assert [b.ref_cnt for b in p.pages[:2]]==[0,0] and [b.block_id for b in p.queue]==[2,1]
 assert [b.data for b in p.pages]==data and epochs==p.semantic_allocation_epochs
@pytest.mark.parametrize('bad',['unfenced','events','size','negative_start','bool_start','duplicate_key','duplicate_page','foreign_group','null','borrowed','already_hashed','indexed_alias','duplicate_cached','wrong_physical'])
def test_native_rejection_leaves_all_ownership_and_visibility_unchanged(pool,bad):
 p,keys=pool;bs=p.pages[:2];args=dict(start_index=3,block_size=16,completion_verified=True)
 if bad=='unfenced':args['completion_verified']=False
 if bad=='events':p.enable_kv_cache_events=True
 if bad=='size':args['block_size']=32
 if bad=='negative_start':args['start_index']=-1
 if bad=='bool_start':args['start_index']=True
 if bad=='duplicate_key':keys[1]=keys[0]
 if bad=='duplicate_page':bs=[bs[0],bs[0]]
 if bad=='foreign_group':keys[1]=keys[1][:-4]+b'\0\0\0\1'
 if bad=='null':bs[1].is_null=True
 if bad=='borrowed':bs[1].ref_cnt=2
 if bad=='already_hashed':bs[1].block_hash=b'existing'
 if bad=='indexed_alias':p.cached_block_hashes_by_block[bs[1].block_id]={b'alias'}
 if bad=='duplicate_cached':p.cached[keys[1][:-4]]=p.pages[2]
 if bad=='wrong_physical':p.blocks=list(p.blocks);p.blocks[bs[1].block_id]=NS()
 before=[(b.ref_cnt,b.block_hash) for b in p.pages];cache=dict(p.cached)
 assert not p.publish(keys,bs,**args)
 assert before==[(b.ref_cnt,b.block_hash) for b in p.pages] and p.cached==cache and not p.queue
def scheduler_methods():
 events=[];ns={'time':time,'group_config_tokens_per_block':lambda _:16,'group_config_tokens_per_chunk':lambda _:16,'config_blocks_per_chunk':lambda _:1,'make_offload_key':lambda h,g:h,'_observed_prefix_digest':digest,'record_wait_hint_event':events.append,'get_offload_block_hash':lambda k:k,'get_offload_group_idx':lambda k:0}
 names={'_warm_completed_wait_prefetch','_warmed_wait_prefix_valid','_try_publish_wait_prefix','_record_native_prefix_adoption'}
 return load('sources/evictable-warm/semantic_offload/connector.py',names,ns),events
def warm_case():
 ns,events=scheduler_methods();s,st,req,p=case();s._wait_hint_cold_run_publication=True;s._wait_hint_evictable_warm=True
 st.destination_released=False;st.warmed_allocation_epochs=();st.lease.hint.run_id='r';st.lease.hint.cell_id='c';st.lease.hint.session_id='s';st.lease.hint.generation=1
 s._observed_prefix_generations={('r','c','s'):1};st.bound_request_id=None
 for name in ['_warmed_wait_prefix_valid','_warm_completed_wait_prefetch']:setattr(s,name,types.MethodType(ns[name],s))
 # Isolate caller tests from the independently tested native hash publisher.
 def publish(keys,blocks,**kw):
  for i,(k,b) in enumerate(zip(keys,blocks)):b.block_hash=k;b.block_hash_num_tokens=(kw['start_index']+i+1)*16;p.cached[k]=b
  p.free_blocks(reversed(blocks));return True
 p.publish_completed_observed_keys=publish
 p.publish_completed_prefix=old_pool_methods('runtime-cold-run')['publish_completed_prefix'].__get__(p)
 return s,st,req,p,ns,events
def test_completed_copy_releases_capacity_before_return_and_adopts_same_pages_once():
 s,st,r,p,ns,events=warm_case()
 assert s._warm_completed_wait_prefetch(st)
 assert st.destination_released and s._prefetch_reserved_blocks==0 and all(b.ref_cnt==0 for b in p.pages[:2])
 assert not s._warm_completed_wait_prefetch(st)
 st.bound_request_id=r.request_id
 assert ns['_try_publish_wait_prefix'](s,st)
 assert s._prefetch_reserved_blocks==0 and all(b.ref_cnt==0 for b in p.pages[:2])
 for b in p.pages[:2]:b.ref_cnt=1
 ns['_record_native_prefix_adoption'](s,r,NS(blocks=[p.pages[:2]]),16)
 assert events[-1]['event']=='native_prefix_adopted'
@pytest.mark.parametrize('bad',['disabled','bound','pending','source_not_fenced','contributors','released','failed','cancelled','missing_fence','future_fence','generation','expired','wrong_source'])
def test_scheduler_does_not_publish_unqualified_transfer(bad):
 # Parameters deliberately exercise actual production guards, not an alternate
 # fake implementation of the scheduler qualification policy.
 s,st,r,p,ns,events=warm_case()
 if bad=='disabled':s._wait_hint_evictable_warm=False
 if bad=='bound':st.bound_request_id='request'
 if bad=='pending':st.status='in_flight'
 if bad=='source_not_fenced':st.source_released=False
 if bad=='contributors':st.pending_contributors=1
 if bad=='released':st.destination_released=True
 if bad=='failed':st.transfer_record['success']=False
 if bad=='cancelled':st.transfer_record['cancelled']=True
 if bad=='missing_fence':st.transfer_record['completed_at_ns']=None
 if bad=='future_fence':st.transfer_record['completed_at_ns']=time.monotonic_ns()+10**12
 if bad=='generation':s._observed_prefix_generations[('r','c','s')]=2
 if bad=='expired':st.lease.expires_at_ns=1
 if bad=='wrong_source':st.source.keys=(b'wrong',b'b',b'c')
 assert not s._warm_completed_wait_prefetch(st) and not p.queue and s._prefetch_reserved_blocks==2
@pytest.mark.parametrize('bad',['epoch','hash','physical','missing'])
def test_evicted_or_overwritten_warm_pages_cannot_be_credited(bad):
 s,st,r,p,ns,events=warm_case();assert s._warm_completed_wait_prefetch(st)
 if bad=='epoch':p.semantic_allocation_epochs[1]+=1
 if bad=='hash':p.pages[0].block_hash=b'wrong'
 if bad=='physical':p.cached[b'a']=p.pages[2]
 if bad=='missing':p.cached.pop(b'a')
 assert not s._warmed_wait_prefix_valid(st)
