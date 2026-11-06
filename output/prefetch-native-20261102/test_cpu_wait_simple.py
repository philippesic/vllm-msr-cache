import ast,types
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from semantic_offload.wait_hint import KnownPrefixWaitHint
H=Path(__file__).resolve().parent
@pytest.fixture
def state():
 tree=ast.parse((H/'sources/cpu-wait-simple/semantic_offload/connector.py').read_text())
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SemanticOffloadingConnectorScheduler')
 body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='publish_wait_hint']
 body += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ready_resident_prefix']
 for n in body:n.decorator_list=[]
 events=[];ns={'KnownPrefixWaitHint':KnownPrefixWaitHint,'time':NS(monotonic_ns=lambda:2_000_000_000),
  '_DISABLE_PREFETCH':False,'group_config_tokens_per_chunk':lambda g:16,'ReqContext':NS,'record_wait_hint_event':events.append}
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+body,type_ignores=[])
 exec(compile(ast.fix_missing_locations(mod),'actual-cpu-wait-retention','exec'),ns)
 namespace={k:'none' for k in ['model_revision','tokenizer_revision','adapter_namespace','layout_namespace']}
 raw=dict(schema='semantic-known-prefix-wait-hint-v1',event_id='1'*64,owner_id='owner',run_id='r',cell_id='c',session_id='s',generation=0,
  observed_prefix_id='2'*64,prefix_boundary_tokens=48,observation_available_ns=1_900_000_000,input_publication_ns=1_900_000_000,
  declared_wait_ms=750,wait_provenance='driver_declared_tool_wait_budget',namespace=namespace)
 source=NS(keys=(b'a',b'b',b'c'),observed_ns=1_000_000_000)
 identity=('r','c','s',0,'2'*64,48);ready={k:NS(is_ready=True) for k in source.keys};calls=[]
 manager=NS(_policy=NS(get=ready.get),touch=lambda keys,context:calls.append((list(keys),context)))
 s=NS(_wait_hint_results={},_wait_hint_ttl_ns=1_000_000_000,_prefetch_trigger='known_prefix_wait_hint',_prefetch_budget_fraction=.6,_wait_hint_namespace=namespace,
  config=NS(kv_group_configs=[NS()]),_observed_prefix_generations={('r','c','s'):0},_observed_prefixes={identity:source},
  _wait_hint_cpu_retention_only=True,_wait_hint_coalesce_exact_source=False,_prefetch_require_full_prefix=True,
  _admitted_prefill_in_progress=lambda:True,manager=manager,_store_wait_hint_result=lambda hint,**kw:kw)
 s.publish=types.MethodType(ns['publish_wait_hint'],s)
 return s,raw,source,ready,calls,events
def test_cpu_recency_can_be_refreshed_even_when_gpu_prefill_blocks_copying(state):
 s,raw,source,ready,calls,events=state;out=s.publish(raw)
 assert out['reason']=='cpu_prefix_retained' and out['reserved_gpu_blocks']==0 and out['speculative_transfer_bytes']==0
 assert calls[0][0]==list(source.keys) and calls[0][1].kv_transfer_params['generation']==0
 assert not events
@pytest.mark.parametrize('bad',['namespace','generation','source','future_clock','expired','invalid_hint','disabled'])
def test_no_retention_before_exact_hint_identity_and_causal_validation(state,bad):
 s,raw,source,ready,calls,events=state
 if bad=='namespace':raw['namespace']=dict(raw['namespace'],model_revision='different')
 if bad=='generation':s._observed_prefix_generations[('r','c','s')]=1
 if bad=='source':s._observed_prefixes.clear()
 if bad=='future_clock':raw['input_publication_ns']=2_100_000_000
 if bad=='expired':raw['observation_available_ns']=raw['input_publication_ns']=1_100_000_000
 if bad=='invalid_hint':raw['actual_resume_ns']=3_000_000_000
 if bad=='disabled':s._wait_hint_cpu_retention_only=False;s._prefetch_budget_fraction=0
 s.publish(raw);assert not calls and not events
@pytest.mark.parametrize('missing',[True,False])
def test_only_ready_contiguous_cpu_prefix_is_touched(state,missing):
 s,raw,source,ready,calls,events=state
 if missing:ready.pop(b'b')
 else:ready[b'b'].is_ready=False
 s.publish(raw);assert calls[0][0]==[b'a']

def test_same_event_replay_does_not_retouch_cpu_cache(state):
 s,raw,source,ready,calls,events=state
 s._wait_hint_results[raw['event_id']]={'status':'no_action','reason':'cpu_prefix_retained'}
 assert s.publish(raw)['idempotent_replay'] and not calls

def test_default_publish_path_ast_is_unchanged_after_removing_new_opt_in_branch():
 def method(path):
  tree=ast.parse((H/path).read_text())
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SemanticOffloadingConnectorScheduler')
  return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='publish_wait_hint')
 old=method('sources/candidate/semantic_offload/connector.py')
 new=method('sources/cpu-wait-simple/semantic_offload/connector.py')
 guards=[n for n in new.body if isinstance(n,ast.If) and '_wait_hint_cpu_retention_only' in ast.unparse(n.test)]
 assert len(guards)==1;new.body.remove(guards[0])
 assert ast.dump(new)==ast.dump(old)
