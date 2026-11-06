import ast,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
H=Path(__file__).resolve().parent
@pytest.fixture
def case():
 tree=ast.parse((H/'sources/demand-fallback/semantic_offload/connector.py').read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_bind_wait_hint_request');fn.decorator_list=[]
 events=[];ns={'time':NS(monotonic_ns=lambda:100),'record_wait_hint_event':events.append}
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),'actual-demand-fallback','exec'),ns)
 identity=dict(owner_id='owner',run_id='run',cell_id='cell',session_id='session',generation=0,observed_prefix_id='digest',prefix_boundary_tokens=32,event_id='e')
 hint=NS(**identity);lease=NS(hint=hint,active_at=lambda now:True,resume=lambda:None)
 st=NS(lease=lease,status='pending',job_id=None,gpu_block_ids=(),pending_contributors=0,publication_local_gpu_prefix_count=2,bound_request_id=None)
 s=NS(_wait_prefetch={'e':st},_observed_prefix_generations={('run','cell','session'):0},_wait_hint_shared_local_prefix=True,_wait_hint_request_bindings={},_preservation_requests={},_wait_state_identity=lambda _: {},_wait_local_prefix_valid=lambda _:True,freed=[],retired=[],attached=[])
 def free(state,reason):s.freed.append(reason);s._wait_prefetch.pop('e')
 s._free_wait_prefetch=free;s._retire_wait_prefetch=lambda *a,**kw:s.retired.append(a[1]);s._attach_ready_wait_prefetch=lambda st:s.attached.append(st)
 params={k:v for k,v in identity.items() if k not in {'owner_id','event_id','prefix_boundary_tokens'}};params.update(external_request_id='owner',observed_prefix_boundary_tokens=32,wait_hint_event_id='e')
 r=NS(request_id='actual',kv_transfer_params=params)
 return s,st,r,ns['_bind_wait_hint_request'],events
def test_no_copy_nomination_falls_back_without_binding_or_waiting(case):
 s,st,r,fn,events=case;fn(s,r)
 assert s.freed==['demand_arrived_before_dispatch'] and not s._wait_hint_request_bindings and not s.attached
 assert st.bound_request_id is None and events[-1]['reason']=='undispatched_nomination_demand_fallback'
@pytest.mark.parametrize('field',['external_request_id','run_id','cell_id','session_id','generation','observed_prefix_id','observed_prefix_boundary_tokens'])
def test_wrong_identity_cannot_cancel_another_owners_nomination(case,field):
 s,st,r,fn,events=case;r.kv_transfer_params[field]='wrong';fn(s,r)
 assert not s.freed and not s._wait_hint_request_bindings and st.bound_request_id is None
 assert events[-1]['reason']=='resume_identity_mismatch'
@pytest.mark.parametrize('guard',['generation','expired','changed_prefix'])
def test_existing_generation_lease_and_local_prefix_controls_remain(case,guard):
 s,st,r,fn,events=case
 if guard=='generation':s._observed_prefix_generations[('run','cell','session')]=1
 if guard=='expired':st.lease.active_at=lambda _:False
 if guard=='changed_prefix':s._wait_local_prefix_valid=lambda _:False
 fn(s,r);assert not s.freed and not s._wait_hint_request_bindings
@pytest.mark.parametrize('status',['ready','in_flight'])
def test_started_or_completed_copy_keeps_native_binding_and_fences(case,status):
 s,st,r,fn,events=case;st.status=status;st.job_id=9;st.gpu_block_ids=(4,5);st.pending_contributors=int(status=='in_flight')
 fn(s,r);assert not s.freed and s._wait_hint_request_bindings=={'actual':'e'}
 assert st.bound_request_id=='actual' and bool(s.attached)==(status=='ready')
@pytest.mark.parametrize('field,value',[('job_id',9),('gpu_block_ids',(4,)),('pending_contributors',1)])
def test_inconsistent_pending_state_does_not_bypass_transfer_ownership(case,field,value):
 s,st,r,fn,events=case;setattr(st,field,value);fn(s,r);assert not s.freed
 assert s._wait_hint_request_bindings=={'actual':'e'}
