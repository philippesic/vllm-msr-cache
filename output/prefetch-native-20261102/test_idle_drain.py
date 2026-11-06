import ast,collections,contextlib,types
from pathlib import Path
import pytest
H=Path(__file__).resolve().parent
class NativeSchedule(Exception):pass
@pytest.fixture
def core():
 tree=ast.parse((H/'runtime-idle-drain/vllm/v1/engine/core.py').read_text())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EngineCore')
 fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step_with_batch_queue')
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
 ns={};exec(compile(ast.fix_missing_locations(module),'actual-core','exec'),ns)
 events=[]
 class F:
  def result(self):events.append('future-complete');return 'output'
 output=types.SimpleNamespace(total_num_scheduled_tokens=0)
 x=types.SimpleNamespace(batch_queue=collections.deque([(F(),output,None)]),batch_queue_size=3,engines_running=False)
 x.scheduler=types.SimpleNamespace(requests={},has_requests=lambda:True,get_kv_connector=lambda:types.SimpleNamespace(connector_scheduler=types.SimpleNamespace(ready_staging_idle_timeout=lambda:.05)))
 def schedule(*args):events.append('new-native-schedule');raise NativeSchedule
 x.scheduler.schedule=schedule;x._should_throttle_prefills=lambda:False
 def update(s,o):assert o=='output';events.append('ordinary-update');return {'response':1}
 x.scheduler.update_from_output=update
 x.log_error_detail=lambda *_:contextlib.nullcontext()
 x.capture_iteration_details=lambda *_:contextlib.nullcontext('iteration')
 x._process_aborts_queue=lambda:events.append('process-aborts')
 x._attach_iteration_details=lambda *_:events.append('attach-details')
 return x,ns['step_with_batch_queue'],events
def test_drain_preserves_future_abort_update_order(core):
 x,fn,events=core;assert fn(x)==({'response':1},False)
 assert events==['future-complete','process-aborts','ordinary-update','attach-details']
 assert not x.batch_queue and x.semantic_staging_idle_drain_calls==1
@pytest.mark.parametrize('guard',['live','engine','nonempty','pending_transfer','expired','missing_helper','empty_queue'])
def test_unsafe_conditions_keep_native_enqueue(core,guard):
 x,fn,events=core
 if guard=='live':x.scheduler.requests={'r':object()}
 if guard=='engine':x.engines_running=True
 if guard=='nonempty':x.batch_queue[0][1].total_num_scheduled_tokens=1
 if guard in {'pending_transfer','expired','missing_helper'}:
  helper=types.SimpleNamespace(ready_staging_idle_timeout=lambda:None if guard=='pending_transfer' else 0)
  if guard=='missing_helper':helper=object()
  x.scheduler.get_kv_connector=lambda:types.SimpleNamespace(connector_scheduler=helper)
 if guard=='empty_queue':x.batch_queue.clear()
 with pytest.raises(NativeSchedule):fn(x)
 assert events==['new-native-schedule']
def test_multiple_empty_futures_drain_oldest_only(core):
 x,fn,events=core
 class New:
  def result(self):raise AssertionError('newer future consumed first')
 x.batch_queue.appendleft((New(),types.SimpleNamespace(total_num_scheduled_tokens=0),None))
 fn(x);assert len(x.batch_queue)==1
def test_existing_future_error_propagates(core):
 x,fn,events=core
 class Failed:
  def result(self):raise RuntimeError('transfer execution failed')
 x.batch_queue[0]=(Failed(),x.batch_queue[0][1],None)
 with pytest.raises(RuntimeError,match='transfer execution failed'):fn(x)
 assert not events
