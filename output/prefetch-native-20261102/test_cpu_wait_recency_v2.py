"""Exercise actual publication and recency methods with a bounded CPU policy."""
import ast,types
from collections import OrderedDict
from types import SimpleNamespace as NS
import pytest
from test_cpu_wait_simple import state
from pathlib import Path
H=Path(__file__).resolve().parent
class ARC:
 def __init__(self,ready):
  self.t1=OrderedDict(ready);self.t2=OrderedDict();self.b1=OrderedDict([(b'ghost',None)]);self.b2=OrderedDict();self.target_t1_size=2
 def get(self,key):return self.t1.get(key) or self.t2.get(key)

def actual_method(filename,cls,name,ns):
 tree=ast.parse((H/'sources/cpu-wait-recency-v2/semantic_offload'/filename).read_text())
 c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 m=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
 # Substitute only the optional vLLM type import, not policy operations.
 m.body=[n for n in m.body if not isinstance(n,ast.ImportFrom)];m.decorator_list=[]
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),m],type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),'actual-recency-method','exec'),ns)
 return ns[name]

@pytest.fixture
def recency(state):
 s,raw,source,ready,calls,events=state;policy=ARC(ready);s.manager._policy=policy
 refresh=actual_method('manager.py','PreservationOffloadingManager','refresh_wait_hint_cpu_recency',{'ARCCachePolicy':ARC})
 s.manager.refresh_wait_hint_cpu_recency=lambda keys:refresh(s.manager,keys)
 s.publish=types.MethodType(actual_method('connector.py','SemanticOffloadingConnectorScheduler','publish_wait_hint',dict(s.publish.__func__.__globals__)),s)
 return s,raw,source,policy

def test_advice_refreshes_recency_without_promoting_into_frequency_queue(recency):
 s,raw,source,p=recency;before={k:p.get(k) for k in source.keys};out=s.publish(raw)
 assert out['reason']=='cpu_prefix_recency_refreshed' and out['ready_cpu_prefix_keys']==3
 assert list(p.t1)==list(reversed(source.keys)) and not p.t2
 assert {k:p.get(k) for k in source.keys}==before
 assert p.target_t1_size==2 and list(p.b1)==[b'ghost'] and not p.b2
 assert out['reserved_gpu_blocks']==out['speculative_transfer_bytes']==0

def test_existing_frequency_queue_membership_is_preserved(recency):
 s,raw,source,p=recency;p.t2[b'b']=p.t1.pop(b'b');s.publish(raw)
 assert set(p.t1)=={b'a',b'c'} and set(p.t2)=={b'b'}

@pytest.mark.parametrize('bad',['namespace','generation','source','future_clock','expired','future_field'])
def test_validation_precedes_any_advice(recency,bad):
 s,raw,source,p=recency;before=list(p.t1)
 if bad=='namespace':raw['namespace']=dict(raw['namespace'],model_revision='wrong')
 elif bad=='generation':s._observed_prefix_generations[('r','c','s')]=1
 elif bad=='source':s._observed_prefixes.clear()
 elif bad=='future_clock':raw['input_publication_ns']=2_100_000_000
 elif bad=='expired':raw['observation_available_ns']=raw['input_publication_ns']=1_100_000_000
 else:raw['actual_resume_ns']=3_000_000_000
 s.publish(raw);assert list(p.t1)==before and not p.t2

@pytest.mark.parametrize('gap',['missing','pending'])
def test_no_refresh_across_nonready_prefix_gap(recency,gap):
 s,raw,source,p=recency
 if gap=='missing':p.t1.pop(b'b')
 else:p.t1[b'b'].is_ready=False
 out=s.publish(raw);assert out['ready_cpu_prefix_keys']==1 and not p.t2
 assert list(p.t1)[-1]==b'a'

def test_unsupported_policy_is_untouched(recency):
 s,raw,source,p=recency;s.manager._policy=NS(get=p.get)
 assert s.publish(raw)['ready_cpu_prefix_keys']==0 and list(p.t1)==list(source.keys)

def test_idempotent_event_is_not_refreshed_again(recency):
 s,raw,source,p=recency;s._wait_hint_results[raw['event_id']]={'status':'no_action','reason':'cpu_prefix_recency_refreshed'}
 assert s.publish(raw)['idempotent_replay'] and list(p.t1)==list(source.keys)


def test_manager_without_arc_advice_api_is_untouched(recency):
 s,raw,source,p=recency;del s.manager.refresh_wait_hint_cpu_recency
 assert s.publish(raw)['ready_cpu_prefix_keys']==0 and list(p.t1)==list(source.keys)
