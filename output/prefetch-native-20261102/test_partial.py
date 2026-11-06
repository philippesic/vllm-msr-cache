"""Execute actual isolated scheduler methods with adversarial ownership controls."""
import ast,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from test_tailfirst import Pool,load as pool_methods
H=Path(__file__).resolve().parent

def digest(tokens,namespace):return hashlib.sha256(json.dumps([tokens,namespace]).encode()).hexdigest()
def methods():
 nodes=ast.parse((H/'sources/partial-native/semantic_offload/connector.py').read_text())
 names={'_try_publish_wait_prefix','_record_native_prefix_adoption'}
 body=[n for n in ast.walk(nodes) if isinstance(n,ast.FunctionDef) and n.name in names]
 for n in body:n.decorator_list=[]
 tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+body,type_ignores=[])
 events=[];ns={'time':time,'group_config_tokens_per_block':lambda _:16,'group_config_tokens_per_chunk':lambda _:16,'config_blocks_per_chunk':lambda _:1,'make_offload_key':lambda h,g:h,'_observed_prefix_digest':digest,'record_wait_hint_event':events.append}
 exec(compile(ast.fix_missing_locations(tree),'<actual-partial-connector>','exec'),ns)
 return ns,events

def case(partial=True):
 pool=Pool();pool.semantic_allocation_epochs=[1]*8
 pool.publish_completed_prefix=pool_methods('runtime-partial-native')['publish_completed_prefix'].__get__(pool)
 req=NS(request_id='request',num_computed_tokens=0,prompt_token_ids=[1]*49,num_tokens=49,block_hashes=[b'a',b'b',b'c'],skip_reading_prefix_cache=False)
 state=NS(publication_local_gpu_prefix_count=0,publication_local_gpu_block_ids=(),bound_request_id='request',source_released=True,pending_contributors=0,status='ready',keys=(b'a',b'b'),gpu_block_ids=(1,2),source=NS(keys=(b'a',b'b',b'c')),transfer_record={'success':True,'cancelled':False,'contributors':1,'expected_contributors':1,'completed_at_ns':time.monotonic_ns()},lease=NS(hint=NS(event_id='event',prefix_boundary_tokens=48,observed_prefix_id=digest([1]*48,{})),state='ready',expires_at_ns=time.monotonic_ns()+10**12))
 scheduler=NS(_wait_hint_native_adoption=True,_wait_hint_native_adoption_layout_supported=True,_wait_hint_shared_local_prefix=True,_wait_hint_partial_native_publication=partial,_wait_hint_namespace={},config=NS(num_workers=1,kv_group_configs=[NS(group_idx=0)]),gpu_block_pool=pool,_req_status={'request':NS(req=req,num_locally_computed_tokens=32)},_prefetch_reserved_blocks=2,_wait_prefetch={'event':state},_wait_hint_request_bindings={},_published_wait_prefixes={},_release_wait_prefetch_protection=lambda _:None,_wait_state_identity=lambda _: {},_preservation_requests={})
 return scheduler,state,req,pool

def test_partial_publication_requires_no_full_source_change():
 ns,events=methods();s,st,req,p=case()
 assert ns['_try_publish_wait_prefix'](s,st)
 assert [b.block_id for b in p.queue]==[2,1]
 assert p.pages[2].block_hash is None and p.pages[2].ref_cnt==1
 assert s._prefetch_reserved_blocks==0 and st.destination_released
 assert s._published_wait_prefixes['request']['block_ids']==[1,2]
 assert [b.data for b in p.pages]==['owned-KV-1','owned-KV-2','owned-KV-3']

@pytest.mark.parametrize('bad',['disabled','unordered','wrong_source','wrong_digest','no_fence','oversized'])
def test_publication_wrong_controls_rejected_without_ownership_change(bad):
 ns,_=methods();s,st,req,p=case()
 if bad=='disabled':s._wait_hint_partial_native_publication=False
 elif bad=='unordered':st.keys=(b'b',b'a')
 elif bad=='wrong_source':st.source.keys=(b'a',b'b',b'd')
 elif bad=='wrong_digest':st.lease.hint.observed_prefix_id='wrong'
 elif bad=='no_fence':st.transfer_record['success']=False
 elif bad=='oversized':st.keys=(b'a',b'b',b'c',b'd');st.gpu_block_ids=(1,2,3,4)
 assert not ns['_try_publish_wait_prefix'](s,st)
 assert not p.queue and all(b.ref_cnt==1 for b in p.pages)

@pytest.mark.parametrize('bad',[None,'epoch','hash','physical','unowned','not_native','external_destination'])
def test_consumption_witness_counts_only_unmodified_native_hit_pages(bad):
 ns,events=methods();s,st,req,p=case();assert ns['_try_publish_wait_prefix'](s,st)
 for b in p.pages[:2]:b.ref_cnt=1
 if bad=='epoch':p.semantic_allocation_epochs[1]+=1
 elif bad=='hash':p.pages[0].block_hash=b'wrong'
 elif bad=='physical':p.pages[0],p.pages[1]=p.pages[1],p.pages[0]
 elif bad=='unowned':p.pages[0].ref_cnt=0
 elif bad=='not_native':s._req_status['request'].num_locally_computed_tokens=0
 elif bad=='external_destination':s._req_status['request'].num_locally_computed_tokens=16
 ns['_record_native_prefix_adoption'](s,req,NS(blocks=[p.pages[:2]]),16)
 w=[e for e in events if e['event']=='native_prefetch_consumption_witness'][0]
 assert w['confirmed_prefetched_blocks']==(2 if bad is None else 1 if bad=='external_destination' else 0)

def test_allocation_and_reset_epoch_invalidation_are_in_actual_runtime():
 nodes=ast.parse((H/'runtime-partial-native/vllm/v1/core/block_pool.py').read_text())
 names={n.name:ast.unparse(n) for n in ast.walk(nodes) if isinstance(n,ast.FunctionDef)}
 for name in ['get_new_blocks','_remove_cached_block_hashes','reset_prefix_cache']:
  assert 'self.semantic_allocation_epochs[block.block_id] += 1' in names[name]
 assert 'semantic_allocation_epochs' not in names['touch']
