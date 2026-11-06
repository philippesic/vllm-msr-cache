import ast,copy,json
from pathlib import Path
import pytest
import replay_judge as judge
H=Path(__file__).resolve().parent
@pytest.fixture
def case():
 trace={'requests':[{'session_id':'s','turn':0,'prompt':[1,2]}]}
 raw={'requests':[{'session_id':'s','turn':0,'success':True,'payload':{'prompt':[1,2],'max_tokens':96,'ignore_eos':True},'output_token_ids':[4]*96}]}
 trace['requests'].append({'session_id':'s','turn':1,'prompt':[1,3]})
 second=copy.deepcopy(raw['requests'][0]);second['turn']=1;second['payload']['prompt']=[1,3];raw['requests'].append(second)
 return raw,trace

def test_integrity_and_actual_wrong_controls(case):
 raw,trace=case;assert judge.validate(raw,trace)['decode_tokens']==192
 assert judge.reject_invalid_controls(raw,trace)
def test_nonconstant_outputs_do_not_change_work(case):
 raw,trace=case;raw['requests'][0]['output_token_ids']=[7]*96;assert judge.validate(raw,trace)['passed']
def test_eos_shortening_rejected(case):
 raw,trace=case;raw['requests'][0]['payload']['ignore_eos']=False
 with pytest.raises(AssertionError):judge.validate(raw,trace)
def test_producer_inputs_unchanged_no_future_trace_access():
 def producer(p):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.AsyncFunctionDef) and n.name=='producer')
 assert ast.dump(producer(H/'fixed_workload.py'))==ast.dump(producer(H/'session_workload.py'))
 tree=producer(H/'fixed_workload.py')
 assert not any(isinstance(n,ast.Name) and n.id in {'fixed','fixed_rows','session','tokens'} for n in ast.walk(tree))
def test_prepared_exact_identities():
 for seed in [1103,2207,3301]:
  trace=json.loads((H/'prepared'/f'fixed-agent-{seed}.json').read_text())
  assert len(trace['requests'])==72 and len({(x['session_id'],x['turn']) for x in trace['requests']})==72
  assert all(x['prompt'] and all(type(t) is int for t in x['prompt']) for x in trace['requests'])
