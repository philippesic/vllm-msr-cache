"""Actual opt-in alias method: no output/future access, no uncomputed KV aliases."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
H=Path(__file__).resolve().parent

@pytest.fixture
def case():
 t=ast.parse((H/'sources/full-input-alias/semantic_offload/connector.py').read_text())
 m=next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name=='_register_completed_input')
 ns={'Request':object,'_ObservedPrefix':NS,'group_config_tokens_per_chunk':lambda g:16,'make_offload_key':lambda h,g:h,'_observed_prefix_digest':lambda tokens,namespace:tuple(tokens),'time':NS(monotonic_ns=lambda:101)}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[m],type_ignores=[])),'actual-alias','exec'),ns)
 class Request:
  @property
  def all_token_ids(self):raise AssertionError('alias must never read output or future tokens')
 r=Request();r.request_id='request';r.status=NS(name='FINISHED_STOPPED');r.num_computed_tokens=48;r.num_prompt_tokens=35;r.prompt_token_ids=list(range(35));r.block_hashes=['a','b','output-hash']
 status=NS(group_states=[NS(offload_keys=['a','b','output-hash'])],update_offload_keys=lambda:None)
 saved=[];w=NS(_req_status={'request':status},config=NS(kv_group_configs=[None]),_calc_num_offloadable_tokens=lambda status,n:n,_wait_hint_namespace={},_save_observed_prefix=lambda record,**kw:saved.append((record,kw)))
 declared=NS(run_id='run',cell_id='cell',session_id='session',generation=2,boundary_tokens=16,keys=('a',))
 return ns['_register_completed_input'],w,r,declared,saved

def test_new_input_alias_extends_prior_declared_source_without_output(case):
 fn,w,r,d,saved=case;fn(w,r,d);assert len(saved)==1
 x,kw=saved[0];assert x.boundary_tokens==32 and x.keys==('a','b')
 assert x.observed_prefix_id==tuple(range(32)) and x.generation==2 and x.source_request_id=='request'
 assert kw['provenance']=='completed_full_input_prefix'

@pytest.mark.parametrize('computed',[None,True,-1,31,34])
def test_uncomputed_input_cannot_be_registered(case,computed):
 fn,w,r,d,saved=case;r.num_computed_tokens=computed;fn(w,r,d);assert not saved

@pytest.mark.parametrize('status',['RUNNING','WAITING','FINISHED_ABORTED','FINISHED_ERROR'])
def test_noncompleted_request_cannot_create_alias(case,status):
 fn,w,r,d,saved=case;r.status.name=status;fn(w,r,d);assert not saved

@pytest.mark.parametrize('keys',[[],['a'],['a','wrong'],['a','a']])
def test_wrong_missing_or_duplicate_physical_keys_rejected(case,keys):
 fn,w,r,d,saved=case;w._req_status['request'].group_states[0].offload_keys=keys;fn(w,r,d);assert not saved

def test_offloadable_boundary_must_cover_alias(case):
 fn,w,r,d,saved=case;w._calc_num_offloadable_tokens=lambda status,n:31;fn(w,r,d);assert not saved

def test_missing_request_ownership_rejected(case):
 fn,w,r,d,saved=case;w._req_status.clear();fn(w,r,d);assert not saved
