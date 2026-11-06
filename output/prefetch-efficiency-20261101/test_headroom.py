import ast,math
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
H=Path(__file__).resolve().parent
s=ast.parse((H/'sources/demand-headroom/semantic_offload/connector.py').read_text())
method=next(n for n in ast.walk(s) if isinstance(n,ast.FunctionDef) and n.name=='_prefetch_free_capacity')
ns={'group_config_tokens_per_block':lambda c:16,'cdiv':lambda a,b:(a+b-1)//b}
exec(compile(ast.Module(body=[method],type_ignores=[]),'<headroom>','exec'),ns)
fn=ns['_prefetch_free_capacity']
def state(rows,free=200):
 return NS(gpu_block_pool=NS(get_num_free_blocks=lambda:free),config=NS(kv_group_configs=[None]),_req_status={str(i):NS(req=NS(status=NS(name=status),num_prompt_tokens=prompt,num_computed_tokens=done)) for i,(status,prompt,done) in enumerate(rows)})
@pytest.mark.parametrize('rows,free,expected',[
 ([],200,200),([('WAITING',1600,0)],200,100),([('PREEMPTED',1601,16)],200,100),
 ([('RUNNING',1600,0)],200,200),([('FINISHED_STOPPED',1600,0)],200,200),
 ([('WAITING',4000,0)],200,0),([('WAITING',800,0),('WAITING',1600,0)],200,100),
 ([('WAITING_FOR_REMOTE_KVS',1600,0)],200,100),([('WAITING',16,32)],200,200)])
def test_known_demand(rows,free,expected):assert fn(state(rows,free))==expected
def test_progress_releases_headroom():
 s=state([('WAITING',1600,0)]);assert fn(s)==100
 s._req_status['0'].req.status.name='RUNNING';assert fn(s)==200
 s.gpu_block_pool=None;assert fn(s)==0
