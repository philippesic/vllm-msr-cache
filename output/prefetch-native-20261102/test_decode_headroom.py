import ast,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
H=Path(__file__).resolve().parent
def case():
 tree=ast.parse((H/'sources/decode-headroom/semantic_offload/connector.py').read_text())
 names={'_prefetch_decode_headroom_blocks','_prefetch_free_capacity'}
 body=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
 ns={'group_config_tokens_per_block':lambda g:16,'cdiv':lambda a,b:(a+b-1)//b}
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'actual-decode-headroom','exec'),ns)
 r=NS(request_id='r',status=NS(name='RUNNING'),num_prompt_tokens=128,num_computed_tokens=144,sampling_params=NS(max_tokens=96))
 p=NS(num_gpu_blocks=416,get_num_free_blocks=lambda:100)
 s=NS(_prefetch_reserve_decode_budget=True,config=NS(kv_group_configs=[NS()]),_req_status={'r':NS(req=r)},gpu_block_pool=p,_prefetch_preserve_gpu_prefix_cache=False)
 for name in names:setattr(s,name,types.MethodType(ns[name],s))
 return s,r
def test_reserves_only_remaining_declared_growth_without_mutating_request():
 s,r=case();before=repr(r)
 assert s._prefetch_decode_headroom_blocks()==5
 assert s._prefetch_free_capacity()==95 and repr(r)==before
@pytest.mark.parametrize('status',['WAITING','FINISHED_STOPPED','FINISHED_ABORTED','PREEMPTED'])
def test_unadmitted_or_finished_requests_do_not_reserve_decode_space(status):
 s,r=case();r.status.name=status
 assert s._prefetch_decode_headroom_blocks()==0 and s._prefetch_free_capacity()==100
def test_remote_admitted_requests_and_duplicate_index_entries():
 s,r=case();r.status.name='WAITING_FOR_REMOTE_KVS';s._req_status['alias']=NS(req=r)
 assert s._prefetch_decode_headroom_blocks()==5
def test_opt_in_is_default_off_and_suppresses_nothing_when_disabled():
 s,r=case();s._prefetch_reserve_decode_budget=False
 assert s._prefetch_free_capacity()==100
@pytest.mark.parametrize('bad',[None,True,0,-1])
def test_unknown_or_invalid_decode_budget_is_not_borrowed(bad):
 s,r=case();r.sampling_params.max_tokens=bad
 assert s._prefetch_free_capacity()==0
def test_completed_budget_needs_no_extra_pages():
 s,r=case();r.num_computed_tokens=224
 assert s._prefetch_free_capacity()==100
