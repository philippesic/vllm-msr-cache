import ast,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
H=Path(__file__).resolve().parent
@pytest.fixture
def state():
 tree=ast.parse((H/'sources/residency-refresh/semantic_offload/connector.py').read_text())
 fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_refresh_wait_hint_gpu_residency')
 ns={'config_blocks_per_chunk':lambda c:c.chunk,'get_offload_group_idx':lambda k:k[1],'get_offload_block_hash':lambda k:k[0]}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual-residency-refresh','exec'),ns)
 keys=[(bytes([i]),0) for i in range(1,5)]
 blocks=[NS(block_id=i,block_hash=k,ref_cnt=0,is_null=False) for i,k in enumerate(keys)]
 blocks[2].ref_cnt=2
 index={k[0]:b for k,b in zip(keys,blocks)}
 queue=[0,1,3];touched=[]
 def touch(bs):
  for b in bs:
   assert b.ref_cnt==0;queue.remove(b.block_id);b.ref_cnt+=1;touched.append(b.block_id)
 def free(bs):
  for b in bs:b.ref_cnt-=1;queue.append(b.block_id)
 pool=NS(enable_caching=True,get_cached_block=lambda h,g:[index[h]] if h in index else None,blocks=blocks,touch=touch,free_blocks=free)
 s=NS(config=NS(chunk=1,kv_group_configs=[NS(group_idx=0)]),gpu_block_pool=pool)
 s.refresh=types.MethodType(ns[fn.name],s)
 return s,NS(keys=tuple(keys)),blocks,index,queue,touched
def test_refresh_preserves_refs_hashes_and_free_capacity_with_prefix_priority(state):
 s,source,bs,idx,q,t=state;before=[(b.ref_cnt,b.block_hash) for b in bs]
 assert s.refresh(source)==3
 assert q==[3,1,0] and t==[0,1,3]
 assert before==[(b.ref_cnt,b.block_hash) for b in bs]
def test_cached_islands_are_refreshed_without_fabricating_missing_pages(state):
 s,source,bs,idx,q,t=state;idx.pop(source.keys[1][0])
 assert s.refresh(source)==2 and q==[1,3,0]
@pytest.mark.parametrize('bad',['disabled','multi','chunk','missing_lookup','hash','negative_ref','null','wrong_physical','group'])
def test_validation_rejects_before_mutating_any_page(state,bad):
 s,source,bs,idx,q,t=state;before=list(q)
 if bad=='disabled':s.gpu_block_pool.enable_caching=False
 if bad=='multi':s.config.kv_group_configs.append(NS(group_idx=1))
 if bad=='chunk':s.config.chunk=2
 if bad=='missing_lookup':s.gpu_block_pool.get_cached_block=None
 if bad=='hash':bs[-1].block_hash=b'wrong'
 if bad=='negative_ref':bs[-1].ref_cnt=-1
 if bad=='null':bs[-1].is_null=True
 if bad=='wrong_physical':s.gpu_block_pool.blocks=list(bs);s.gpu_block_pool.blocks[-1]=NS()
 if bad=='group':source.keys=source.keys+((b'other',1),)
 assert s.refresh(source)==0 and q==before and not t
def test_duplicate_key_changes_recency_once(state):
 s,source,bs,idx,q,t=state;source.keys=source.keys+source.keys
 assert s.refresh(source)==3 and len(t)==3
def test_hint_identity_and_clock_checks_precede_residency_refresh():
 tree=ast.parse((H/'sources/residency-refresh/semantic_offload/connector.py').read_text())
 fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='publish_wait_hint')
 text=ast.unparse(fn)
 at=text.index('self._refresh_wait_hint_gpu_residency(source)')
 for guard in ['namespace_mismatch','observed_prefix_unavailable','superseded_generation','causal_clock_mismatch','nomination_deadline_elapsed']:
  assert text.index(guard)<at
