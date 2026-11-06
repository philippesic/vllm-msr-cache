"""Exercise actual candidate methods with deterministic queues and clocks."""
import ast,os,queue,types
from pathlib import Path
import pytest
H=Path(__file__).resolve().parent

def method(path,cls,name):
 tree=ast.parse(path.read_text());c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
 return next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name)

class Base:
 def has_pending_push_work(self):return self.parent_work

@pytest.fixture
def ready(monkeypatch):
 monkeypatch.setenv('SEMANTIC_READY_IDLE_WAIT','1')
 m=method(H/'sources/idle-pending/semantic_offload/connector.py','SemanticOffloadingConnectorScheduler','ready_staging_idle_timeout')
 c=ast.ClassDef(name='Candidate',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[m],decorator_list=[])
 ns={'Base':Base,'time':types.SimpleNamespace(monotonic_ns=lambda:1_000_000_000)}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[c],type_ignores=[])),'candidate','exec'),ns)
 x=ns['Candidate']();x.parent_work=False;x._wait_prefetch={'e':types.SimpleNamespace(status='ready',bound_request_id=None,lease=types.SimpleNamespace(expires_at_ns=61_000_000_000,nomination_deadline_ns=1_750_000_000,state='ready'))};return x

def test_ready_wait_bounded_and_exact_expiry(ready):
 assert ready.ready_staging_idle_timeout()==.05
 ready._wait_prefetch['e'].lease.expires_at_ns=1_007_000_000
 assert ready.ready_staging_idle_timeout()==.007
 ready._wait_prefetch['e'].lease.expires_at_ns=999_000_000
 assert ready.ready_staging_idle_timeout()==0

@pytest.mark.parametrize('status',['in_flight','bound_ready','splice_in_flight','retired_pending_fence'])
def test_nonready_needs_normal_poll(ready,status):
 ready._wait_prefetch['e'].status=status
 assert ready.ready_staging_idle_timeout() is None

@pytest.mark.parametrize('field,value',[('parent_work',True),('_wait_prefetch_jobs',{1:1}),('_pending_splice_releases',{1:1}),('_bounded_prefill_cuda_timing',True),('_wait_prefetch',{})])
def test_fences_cleanup_and_timing_disable_wait(ready,field,value):
 setattr(ready,field,value);assert ready.ready_staging_idle_timeout() is None

def test_owner_disables_wait(ready):
 ready._wait_prefetch['e'].bound_request_id='request';assert ready.ready_staging_idle_timeout() is None

def test_flag_off(ready,monkeypatch):
 monkeypatch.delenv('SEMANTIC_READY_IDLE_WAIT');assert ready.ready_staging_idle_timeout() is None

@pytest.fixture
def core():
 m=method(H/'runtime-idle-pending/vllm/v1/engine/core.py','EngineCoreProc','_process_input_queue')
 ns={'queue':queue,'logger':types.SimpleNamespace(isEnabledFor=lambda _:False),'DEBUG':10}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[m],type_ignores=[])),'core','exec'),ns)
 class Q:
  def __init__(self):self.calls=[];self.message=None
  def get(self,**kw):
   self.calls.append(kw)
   if self.message is not None:out=self.message;self.message=None;return out
   raise queue.Empty
  def empty(self):return self.message is None
 x=types.SimpleNamespace(process_input_queue_block=True,engines_running=False,batch_queue=[],is_running=lambda:True,has_work=lambda:True,input_queue=Q(),handled=[])
 x._handle_client_request=lambda *r:x.handled.append(r)
 x.scheduler=types.SimpleNamespace(requests={},get_kv_connector=lambda:types.SimpleNamespace(connector_scheduler=types.SimpleNamespace(ready_staging_idle_timeout=lambda:.05)))
 return x,ns['_process_input_queue']

def test_core_timeout_steps_after_bounded_wait(core):
 x,fn=core;fn(x);assert x.input_queue.calls==[{'timeout':.05}];assert x.handled==[]

def test_core_input_wakes_without_waiting_for_timer(core):
 x,fn=core;x.input_queue.message=('ADD','payload');fn(x)
 assert x.handled==[('ADD','payload')];assert x.input_queue.calls==[{'timeout':.05}]

@pytest.mark.parametrize('field,value',[('engines_running',True),('batch_queue',[1]),('process_input_queue_block',False)])
def test_core_active_work_never_waits(core,field,value):
 x,fn=core;setattr(x,field,value);fn(x);assert not x.input_queue.calls

def test_core_live_request_never_waits(core):
 x,fn=core;x.scheduler.requests={'live':object()};fn(x);assert not x.input_queue.calls


def test_pending_nomination_bounds_idle_wake(ready):
 ready._wait_prefetch['e'].status='pending'
 assert ready.ready_staging_idle_timeout()==.05
 ready._wait_prefetch['e'].lease.nomination_deadline_ns=1_003_000_000
 assert ready.ready_staging_idle_timeout()==.003
 ready._wait_prefetch['e'].lease.nomination_deadline_ns=999_000_000
 assert ready.ready_staging_idle_timeout()==0

@pytest.mark.parametrize('state',['cancelled','expired'])
def test_terminal_lease_needs_immediate_cleanup(ready,state):
 ready._wait_prefetch['e'].lease.state=state
 assert ready.ready_staging_idle_timeout()==0
