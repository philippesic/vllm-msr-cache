import ast
from pathlib import Path
H=Path(__file__).resolve().parent
class CPU:
 def __init__(self,ids):self.block_ids=ids
class GPU:
 def __init__(self,ids,groups=None):self.block_ids=ids;self.group_sizes=groups

def method(label):
 t=ast.parse((H/f'sources/{label}/semantic_offload/connector.py').read_text())
 return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_physical_block_refs')

def refs(label):
 n=method(label);n.decorator_list=[];ns={'CPULoadStoreSpec':CPU}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),'actual-physical-refs','exec'),ns)
 return ns['_physical_block_refs']

def test_real_cpu_slot_zero_is_not_discarded():
 for label in ['arc-cpu-slot0','cpu-wait-recency-v3']:
  assert refs(label)(CPU([0,5,2]))==[(0,0),(0,5),(0,2)]

def test_gpu_null_block_still_filtered_with_and_without_groups():
 for label in ['arc-cpu-slot0','cpu-wait-recency-v3']:
  assert refs(label)(GPU([0,5]))==[(0,5)]
  assert refs(label)(GPU([0,5],[2]))==[(0,5)]

def test_hybrid_gpu_group_identity_still_qualified():
 assert refs('cpu-wait-recency-v3')(GPU([0,5,5,0],[2,2]))==[(0,5),(1,5)]

def test_both_arms_receive_identical_recording_fix():
 assert ast.dump(method('arc-cpu-slot0')) == ast.dump(method('cpu-wait-recency-v3'))

def test_old_cpu_zero_omission_is_reproduced_by_control():
 assert refs('candidate')(CPU([0])) == []
