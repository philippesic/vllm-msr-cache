import ast,copy,json,time
from pathlib import Path
from types import SimpleNamespace as N
import pytest
import demand_guard_observer as probe
from offline import analyze
H=Path(__file__).resolve().parent;OLD=H.parent/'vram-frontier-20261102'

def method(path):
 tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SemanticOffloadingConnectorScheduler');return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_num_new_matched_tokens')
def compile_class(path,name):
 class Base:
  def get_num_new_matched_tokens(self,request,num):self.base_calls+=1;return self.result
 node=ast.ClassDef(name=name,bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[copy.deepcopy(method(path))],decorator_list=[])
 env={'Base':Base,'Request':N,'time':time,'_demand_guard_probe':probe.observe};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),env);return env[name]
Old=compile_class(OLD/'sources/shared-lookup/semantic_offload/connector.py','Old');New=compile_class(H/'sources/shared-lookup/semantic_offload/connector.py','New')
def fixture(cls,status='pending',eid='event'):
 s=cls();req=N(request_id='req',status=N(name='WAITING'));state=N(status=status,bound_request_id='req');s._wait_hint_request_bindings={'req':eid} if eid else {};s._wait_prefetch={eid:state} if eid else {};s._req_status={'req':N(req=req,transfer_jobs={})};s._preservation_events_enabled=False;s._bind_causal_scheduler_step=lambda rid:None;s._reclaim_wait_prefetch_for_demand=lambda *a,**k:None;s._expire_wait_prefetches=lambda now:None;s._refresh_ordered_key_positions=lambda rid:None;s.base_calls=0;s.result=(37,True);s.gpu_block_pool=N(get_num_free_blocks=lambda:3);s._prefetch_reserved_blocks=2;return s,req,state

@pytest.mark.parametrize('status',['pending','in_flight','ready',None])
@pytest.mark.parametrize('result',[(0,False),(None,False),(37,True)])
def test_actual_method_return_and_native_calls_unchanged(status,result):
 a,ar,_=fixture(Old,status,eid='event' if status else None);b,br,_=fixture(New,status,eid='event' if status else None);a.result=b.result=result
 assert a.get_num_new_matched_tokens(ar,0)==b.get_num_new_matched_tokens(br,0)
 assert a.base_calls==b.base_calls
 assert probe.snapshot(b)['blocked_calls']==(1 if status in ['pending','in_flight'] else 0)
 if status not in ['pending','in_flight']:assert b.get_num_new_matched_tokens(br,0) is result

def test_only_passive_calls_added_to_guard_method_ast():
 a=method(OLD/'sources/shared-lookup/semantic_offload/connector.py');b=copy.deepcopy(method(H/'sources/shared-lookup/semantic_offload/connector.py'))
 class Remove(ast.NodeTransformer):
  def visit_Expr(self,n):
   if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_demand_guard_probe':return None
   return self.generic_visit(n)
 b=Remove().visit(b);assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

def test_expired_state_after_reclaim_never_counted_as_guard():
 s,r,_=fixture(New);s._expire_wait_prefetches=lambda now:s._wait_prefetch.clear();assert s.get_num_new_matched_tokens(r,0)==s.result;assert probe.snapshot(s)['blocked_calls']==0

def test_nohint_native_pending_none_false_not_prefetch_guard():
 s,r,_=fixture(New,eid=None);s._req_status['req'].transfer_jobs={7:object()};s.result=(None,False);assert s.get_num_new_matched_tokens(r,0)==(None,False);assert probe.snapshot(s)['episodes']==[]

@pytest.mark.parametrize('bad',['nohint','wrongidentity','nativepending','ready'])
def test_reject_fake_guard_observation(bad):
 s,r,state=fixture(New)
 if bad=='nohint':s._wait_hint_request_bindings.clear()
 if bad=='wrongidentity':s._req_status['req'].req=N(request_id='req')
 if bad=='nativepending':state.status=None;s._req_status['req'].transfer_jobs={7:object()}
 if bad=='ready':state.status='ready'
 probe.observe(s,r,'event',state,True,at_ns=100);assert probe.snapshot(s)['episodes']==[] and probe.snapshot(s)['rejected_observations']==1

def test_single_hit_eligibility_and_poll_gap_not_zero_delay_claim():
 s,r,state=fixture(New);probe.observe(s,r,'event',state,True,at_ns=100);probe.on_event(s,{'event':'staging_retired','event_id':'event','retired_at_ns':300,'reason':'nomination_deadline_elapsed'});probe.observe(s,r,'event',None,False,at_ns=500);row=probe.snapshot(s)['episodes'][0];assert row['first_guard_ns']==row['last_guard_ns']==100 and row['eligible_at_ns']==300 and row['next_native_path_ns']==500

def test_transition_aggregate_and_completion_passthrough():
 s,r,state=fixture(New);probe.observe(s,r,'event',state,True,at_ns=100);state.status='in_flight';probe.observe(s,r,'event',state,True,at_ns=200);probe.on_event(s,{'event':'staging_ready','event_id':'event','ready_at_ns':250});probe.finished(s,'req',at_ns=400);x=probe.snapshot(s);assert x['episodes'][0]['eligible_at_ns']==250 and x['episodes'][0]['guard_calls']==2 and not x['active_request_ids']

def test_offline_binding_is_potential_not_actual_guard():
 raw={'wait_hint_events':[{'event':'resume_bound','event_id':'x','request_id':'r','staging_status':'pending','bound_at_ns':100},{'event':'staging_retired','event_id':'x','retired_at_ns':110,'reason':'expired'}]};x=analyze(raw,{});assert x['actual_guard_hits'] is None and x['sum_potential_span_ms']==.00001
