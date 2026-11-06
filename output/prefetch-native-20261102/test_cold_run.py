import ast,time,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from test_partial import case,digest
from test_tailfirst import load as pool_methods
H=Path(__file__).resolve().parent
def methods():
 tree=ast.parse((H/'sources/cold-run/semantic_offload/connector.py').read_text())
 names={'_wait_staging_candidates','_wait_required_staging_keys','_wait_local_prefix_valid','_wait_complete_staging_available','_try_publish_wait_prefix','_record_native_prefix_adoption'}
 body=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
 for n in body:n.decorator_list=[]
 events=[]
 def local(pool,keys):
  ids=[]
  for key in keys:
   found=pool.get_cached_block(key,[0])
   if not found:break
   ids.append(found[0].block_id)
  return len(ids),ids,True
 ns={'time':time,'group_config_tokens_per_block':lambda _:16,'group_config_tokens_per_chunk':lambda _:16,'config_blocks_per_chunk':lambda _:1,'make_offload_key':lambda h,g:h,'_observed_prefix_digest':digest,'record_wait_hint_event':events.append,'get_offload_block_hash':lambda k:k,'get_offload_group_idx':lambda k:0,'_local_gpu_prefix':local,'_any_local_gpu_key':lambda p,keys:any(p.get_cached_block(k,[0]) for k in keys)}
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+body,type_ignores=[])
 exec(compile(ast.fix_missing_locations(mod),'actual-cold-run','exec'),ns);return ns,events
def bind(s,ns):
 for k,v in ns.items():
  if k in {'_wait_staging_candidates','_wait_required_staging_keys','_wait_local_prefix_valid','_wait_complete_staging_available'}:setattr(s,k,types.MethodType(v,s))
 return s
def test_cold_run_stops_at_cached_island_and_copies_no_resident_keys():
 ns,_=methods();s=bind(NS(_wait_hint_cold_run_publication=True,gpu_block_pool=NS(get_cached_block=lambda k,_:[NS(block_id=9)] if k in {b'a',b'd',b'f'} else None)),ns)
 assert s._wait_staging_candidates(tuple(b'abcdef'[i:i+1] for i in range(6)),1)==(b'b',b'c')
 assert s._wait_staging_candidates((b'b',b'c'),0)==(b'b',b'c')
 s._wait_hint_cold_run_publication=False;assert s._wait_staging_candidates((b'a',b'b',b'd'),1)==(b'b',b'd')
 s._wait_hint_cold_run_publication=True;s.gpu_block_pool=NS();assert s._wait_staging_candidates((b'b',),0)==()
@pytest.mark.parametrize('changed',['none','head','island_gone','cold_became_hot','candidate_reordered'])
def test_revalidates_exact_head_and_cold_run_before_any_publication(changed):
 ns,_=methods();cache={b'a':3,b'd':6};s=bind(NS(_wait_hint_cold_run_publication=True,_wait_hint_shared_local_prefix=True,gpu_block_pool=NS(get_cached_block=lambda k,_:[NS(block_id=cache[k])] if k in cache else None)),ns)
 st=NS(source=NS(keys=(b'a',b'b',b'c',b'd',b'e')),publication_local_gpu_prefix_count=1,publication_local_gpu_block_ids=(3,),publication_candidate_keys=(b'b',b'c'))
 if changed=='head':cache[b'a']=4
 if changed=='island_gone':cache.pop(b'd')
 if changed=='cold_became_hot':cache[b'b']=5
 if changed=='candidate_reordered':st.publication_candidate_keys=(b'c',b'b')
 assert s._wait_local_prefix_valid(st) is (changed=='none')
 assert s._wait_complete_staging_available(st.source.keys,st.publication_candidate_keys,st.publication_candidate_keys,1,(3,)) is (changed=='none')
@pytest.mark.parametrize('bad',[None,'disabled','unordered','wrong_full_source','wrong_full_digest','unfenced','oversized'])
def test_partial_run_publication_still_validates_entire_observed_source(bad):
 ns,_=methods();s,st,req,p=case();s._wait_hint_cold_run_publication=True
 p.publish_completed_prefix=pool_methods('runtime-cold-run')['publish_completed_prefix'].__get__(p)
 if bad=='disabled':s._wait_hint_cold_run_publication=False
 if bad=='unordered':st.keys=(b'b',b'a')
 if bad=='wrong_full_source':st.source.keys=(b'a',b'b',b'd')
 if bad=='wrong_full_digest':st.lease.hint.observed_prefix_id='wrong'
 if bad=='unfenced':st.transfer_record['success']=False
 if bad=='oversized':st.keys=(b'a',b'b',b'c',b'd');st.gpu_block_ids=(1,2,3,4)
 assert ns['_try_publish_wait_prefix'](s,st) is (bad is None)
 if bad is not None:assert all(b.ref_cnt==1 for b in p.pages) and s._prefetch_reserved_blocks==2
@pytest.mark.parametrize('bad',[None,'epoch','hash','physical','unowned','missing_native_hit','external_only','short_native_hit'])
def test_adoption_requires_native_positions_epoch_and_positive_owner_even_with_later_demand_load(bad):
 ns,events=methods();s,st,req,p=case();s._wait_hint_cold_run_publication=True
 p.publish_completed_prefix=pool_methods('runtime-cold-run')['publish_completed_prefix'].__get__(p)
 assert ns['_try_publish_wait_prefix'](s,st)
 for b in p.pages[:2]:b.ref_cnt=1
 if bad=='epoch':p.semantic_allocation_epochs[1]+=1
 if bad=='hash':p.pages[0].block_hash=b'wrong'
 if bad=='physical':p.pages[0],p.pages[1]=p.pages[1],p.pages[0]
 if bad=='unowned':p.pages[0].ref_cnt=0
 if bad=='missing_native_hit':s._req_status['request'].num_locally_computed_tokens=None
 if bad=='external_only':s._req_status['request'].num_locally_computed_tokens=0
 if bad=='short_native_hit':s._req_status['request'].num_locally_computed_tokens=16
 ns['_record_native_prefix_adoption'](s,req,NS(blocks=[p.pages[:2]]),16)
 assert events[-1]['event']==('native_prefix_adopted' if bad is None else 'native_prefix_missed')
def test_epochs_cover_all_overwrites_but_not_readonly_touch():
 tree=ast.parse((H/'runtime-cold-run/vllm/v1/core/block_pool.py').read_text());nodes={n.name:ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
 for name in ['get_new_blocks','_remove_cached_block_hashes','reset_prefix_cache']:assert 'self.semantic_allocation_epochs[block.block_id] += 1' in nodes[name]
 assert 'semantic_allocation_epochs' not in nodes['touch']
