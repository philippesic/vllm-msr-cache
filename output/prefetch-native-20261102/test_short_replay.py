import ast,types
from pathlib import Path
import pytest
from replay_judge import validate,reject_invalid_controls
H=Path(__file__).resolve().parent
def test_six_turn_trace_checks_every_identity_and_wrong_control():
 rows=[dict(index=s*16+t,session_id=f's{s}',turn=t,prompt=[s+1,t+1,5]) for s in range(4) for t in range(6)]
 raw={'requests':[dict(session_id=r['session_id'],turn=r['turn'],success=True,payload=dict(prompt=r['prompt'][:],max_tokens=96,ignore_eos=True),output_token_ids=[3]*96) for r in rows]}
 trace={'requests':rows}
 assert validate(raw,trace,clients=None,turns=6)['requests']==24
 assert reject_invalid_controls(raw,trace,clients=None,turns=6)
 raw['requests'].pop()
 with pytest.raises(AssertionError,match='incomplete request set'):validate(raw,trace,clients=None,turns=6)
def test_policy_producer_unchanged_and_source_excludes_outputs():
 def function(file,name):return next(n for n in ast.parse((H/file).read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
 assert ast.dump(function('short_fixed_workload.py','producer'))==ast.dump(function('fixed_input_workload.py','producer'))
 ns={};exec(compile(ast.Module(body=[function('short_fixed_workload.py','observed_prefix')],type_ignores=[]),'source','exec'),ns)
 class Forbidden:
  def __iter__(self):raise AssertionError('generated/future output read')
 assert ns['observed_prefix'](list(range(35)),Forbidden())==list(range(32))
