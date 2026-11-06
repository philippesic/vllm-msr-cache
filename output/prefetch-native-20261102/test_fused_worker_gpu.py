import ast,os,sys
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace as NS
import pytest
import torch
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'sources/fused-mean'))

@dataclass
class Mean:
 mean:torch.Tensor

def load():
 tree=ast.parse((H/'sources/fused-mean/semantic_offload/worker.py').read_text())
 names={'_build_durable_summaries_for_job','_summarize_worker_key_batches','_summarize_mean_key_batches'}
 nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
 for n in nodes:n.decorator_list=[]
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
 scope={'torch':torch,'os':os,'_MeanOnlySummary':Mean}
 exec(compile(ast.fix_missing_locations(module),'<actual-fused-worker>','exec'),scope)
 return scope

@pytest.mark.parametrize('factor_two',[False,True])
def test_exact_key_layout_last_write_and_owned_means(monkeypatch,factor_two):
 monkeypatch.setenv('SEMANTIC_FUSED_MEAN_SUMMARY','1')
 ns=load();kv=torch.randn((12,4,16,256),device='cuda',dtype=torch.float16)
 layout=[('a',[3]),('b',[7]),('a',[2]),('c',[1])]
 if factor_two:layout=[('a',[3,4]),('b',[7]),('a',[2,5]),('c',[1,6])]
 published=[]
 w=NS(_pending_job_blocks={9:layout},_probe_layer_name='middle',_attention_layers={'middle':NS(kv_cache=kv,head_size=128)},_check_layout=lambda *a:None,_summary_fields='required',_method='mean',_scoring_schedule='opportunity',durable_summaries={},_register_summary_publications=lambda j,k:published.append((j,k)),_mark_inserted_into_stack_cache=lambda _:None,_prune_durable_summaries=lambda:[])
 w._summarize_worker_key_batches=ns['_summarize_worker_key_batches'].__get__(w)
 w._summarize_mean_key_batches=ns['_summarize_mean_key_batches']
 expected={key:torch.cat([kv[i,:,:,:128] for i in ids],dim=1).float().mean(1) for key,ids in layout}
 before=kv.clone()
 ns['_build_durable_summaries_for_job'](w,9)
 assert not w._pending_job_blocks and published==[(9,['a','b','a','c'])]
 for key,value in expected.items():
  torch.testing.assert_close(torch.stack([s.mean for s in w.durable_summaries[key]]),value,rtol=0,atol=1e-6)
 assert torch.equal(kv,before)
 kv.fill_(100)
 for key,value in expected.items():
  torch.testing.assert_close(torch.stack([s.mean for s in w.durable_summaries[key]]),value,rtol=0,atol=1e-6)
