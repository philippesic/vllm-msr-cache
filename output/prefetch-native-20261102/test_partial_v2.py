import ast
from types import SimpleNamespace as NS
import pytest
from test_partial import H

def init_nodes():
 t=ast.parse((H/'sources/partial-native-v2/semantic_offload/connector.py').read_text())
 return next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='SemanticOffloadingConnectorScheduler').body

def test_only_constructor_changed_from_qualified_method_tests():
 def funcs(label):
  t=ast.parse((H/f'sources/{label}/semantic_offload/connector.py').read_text())
  cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='SemanticOffloadingConnectorScheduler')
  return {n.name:ast.dump(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
 a,b=funcs('partial-native'),funcs('partial-native-v2')
 assert {n for n in a if a[n]!=b[n]}=={'__init__'}

@pytest.mark.parametrize('bad',[None,'layout','workers','groups','chunk','disabled'])
def test_actual_startup_gate_keeps_layout_controls(bad):
 init=next(n for n in init_nodes() if isinstance(n,ast.FunctionDef) and n.name=='__init__')
 gate=next(n for n in init.body if isinstance(n,ast.If) and any(isinstance(b,ast.Raise) and isinstance(b.exc,ast.Call) and b.exc.args and isinstance(b.exc.args[0],ast.Constant) and str(b.exc.args[0].value).startswith('wait_hint_shared_local_prefix requires') for b in n.body))
 s=NS(_wait_hint_shared_local_prefix=True,_wait_hint_native_adoption_layout_supported=bad!='layout',_prefetch_require_full_prefix=False,_wait_hint_partial_native_publication=bad!='disabled',config=NS(num_workers=2 if bad=='workers' else 1,kv_group_configs=[0,1] if bad=='groups' else [0]))
 rejected=eval(compile(ast.Expression(gate.test),'<actual-startup-gate>','eval'),{'self':s,'config_blocks_per_chunk':lambda _:2 if bad=='chunk' else 1})
 assert rejected is (bad is not None)

def test_partial_option_rejects_string_boolean():
 init=next(n for n in init_nodes() if isinstance(n,ast.FunctionDef) and n.name=='__init__')
 loop=next(n for n in init.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple) and any(isinstance(e,ast.Constant) and e.value=='wait_hint_partial_native_publication' for e in n.iter.elts))
 with pytest.raises(TypeError,match='JSON boolean'):
  exec(compile(ast.Module(body=[loop],type_ignores=[]),'<actual-option-validation>','exec'),{'self':NS(),'extra_config':{'wait_hint_partial_native_publication':'false'}})
