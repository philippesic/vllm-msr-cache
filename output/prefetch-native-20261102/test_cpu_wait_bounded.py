import ast,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from test_cpu_wait_simple import state
H=Path(__file__).resolve().parent

@pytest.fixture(params=[('cpu-wait-bounded',0),('cpu-wait-bounded-cold',1)])
def bounded(state,request):
 s,raw,source,ready,calls,events=state;label,head=request.param;s._wait_hint_cpu_retention_max_blocks=1;s.gpu_block_pool=object()
 ns=dict(s.publish.__func__.__globals__);ns['_local_gpu_prefix']=lambda *a:(head,list(range(head)),True)
 tree=ast.parse((H/f'sources/{label}/semantic_offload/connector.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SemanticOffloadingConnectorScheduler')
 method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='publish_wait_hint');method.decorator_list=[]
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
 exec(compile(ast.fix_missing_locations(mod),'actual-bounded-cpu-publication','exec'),ns);s.publish=types.MethodType(ns['publish_wait_hint'],s)
 return s,raw,source,ready,calls,head

def test_bounded_retention_only_touches_first_eligible_page(bounded):
 s,raw,source,ready,calls,head=bounded;out=s.publish(raw)
 assert calls[0][0]==list(source.keys[head:head+1]) and out['ready_cpu_prefix_keys']==1
 assert out['reserved_gpu_blocks']==out['speculative_transfer_bytes']==0

@pytest.mark.parametrize('bad',['namespace','generation','source','future_clock','expired'])
def test_validation_still_precedes_bounded_retention(bounded,bad):
 s,raw,source,ready,calls,head=bounded
 if bad=='namespace':raw['namespace']=dict(raw['namespace'],model_revision='bad')
 elif bad=='generation':s._observed_prefix_generations[('r','c','s')]=1
 elif bad=='source':s._observed_prefixes.clear()
 elif bad=='future_clock':raw['input_publication_ns']=2_100_000_000
 else:raw['observation_available_ns']=raw['input_publication_ns']=1_100_000_000
 s.publish(raw);assert not calls

@pytest.mark.parametrize('value',[None,0,-1,True,1.5,'128'])
def test_cap_rejects_nonpositive_or_noninteger_configuration(value):
 t=ast.parse((H/'sources/cpu-wait-bounded/semantic_offload/connector.py').read_text());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='SemanticOffloadingConnectorScheduler');m=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
 nodes=[n for n in m.body if 'cpu_hint_limit' in ast.unparse(n)]
 assert len(nodes)==3;ns={'self':NS(),'extra_config':{'wait_hint_cpu_retention_max_blocks':value}}
 with pytest.raises(ValueError):exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual-cpu-hint-cap-validation','exec'),ns)
