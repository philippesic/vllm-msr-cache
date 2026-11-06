import ast,types
from pathlib import Path
import pytest
from test_cpu_wait_simple import state
H=Path(__file__).resolve().parent

@pytest.fixture
def cold(state):
 s,raw,source,ready,calls,events=state;s.gpu_block_pool=object();ns=dict(s.publish.__func__.__globals__)
 ns['_local_gpu_prefix']=lambda pool,keys:(1,[7],True)
 tree=ast.parse((H/'sources/cpu-wait-cold/semantic_offload/connector.py').read_text())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SemanticOffloadingConnectorScheduler')
 m=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='publish_wait_hint');m.decorator_list=[]
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),m],type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),'actual-cold-cpu-publication','exec'),ns);s.publish=types.MethodType(ns['publish_wait_hint'],s)
 return s,raw,source,ready,calls,ns

def test_gpu_head_is_not_given_redundant_cpu_retention(cold):
 s,raw,source,ready,calls,ns=cold;out=s.publish(raw)
 assert calls[0][0]==list(source.keys[1:]) and out['ready_cpu_prefix_keys']==2
 assert out['reserved_gpu_blocks']==out['speculative_transfer_bytes']==0

def test_gpu_head_can_cover_a_missing_cpu_prefix(cold):
 s,raw,source,ready,calls,ns=cold;ready.pop(b'a');s.publish(raw)
 assert calls[0][0]==[b'b',b'c']

def test_no_cpu_touch_when_whole_source_is_gpu_cached(cold):
 s,raw,source,ready,calls,ns=cold;ns['_local_gpu_prefix']=lambda *a:(3,[7,8,9],True)
 assert s.publish(raw)['ready_cpu_prefix_keys']==0 and not calls

def test_unknown_native_gpu_index_does_not_guess(cold):
 s,raw,source,ready,calls,ns=cold;ns['_local_gpu_prefix']=lambda *a:(0,[],False)
 assert s.publish(raw)['ready_cpu_prefix_keys']==0 and not calls

@pytest.mark.parametrize('bad',['namespace','generation','source','future_clock','expired'])
def test_causal_validation_happens_before_native_gpu_query(cold,bad):
 s,raw,source,ready,calls,ns=cold
 def forbidden(*a):raise AssertionError('queried GPU before validation')
 ns['_local_gpu_prefix']=forbidden
 if bad=='namespace':raw['namespace']=dict(raw['namespace'],model_revision='wrong')
 elif bad=='generation':s._observed_prefix_generations[('r','c','s')]=1
 elif bad=='source':s._observed_prefixes.clear()
 elif bad=='future_clock':raw['input_publication_ns']=2_100_000_000
 else:raw['observation_available_ns']=raw['input_publication_ns']=1_100_000_000
 s.publish(raw);assert not calls

def test_cold_recency_composition_uses_the_tested_recency_refresh_call():
 def method(label):
  t=ast.parse((H/f'sources/{label}/semantic_offload/connector.py').read_text());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='SemanticOffloadingConnectorScheduler')
  return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='publish_wait_hint')
 old=method('cpu-wait-recency');new=method('cpu-wait-cold-recency')
 a=next(n for n in old.body if isinstance(n,ast.If) and '_wait_hint_cpu_retention_only' in ast.unparse(n.test))
 b=next(n for n in new.body if isinstance(n,ast.If) and '_wait_hint_cpu_retention_only' in ast.unparse(n.test))
 assert ast.dump(a.body[1])==ast.dump(b.body[2])
