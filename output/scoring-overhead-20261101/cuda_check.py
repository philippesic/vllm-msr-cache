import json, torch
from pathlib import Path
from semantic_offload.worker import SemanticOffloadingWorker as W
from semantic_offload.index import build_summary
import importlib.util
spec=importlib.util.spec_from_file_location('mean_candidate',Path(__file__).parent/'sources/gemm/semantic_offload/mean_scoring.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);score_mean_heads_gemm=module.score_mean_heads_gemm
results=[]
for tokens,dtype in [(n,d) for n in [16,32,48] for d in [torch.float16,torch.float32]]:
 torch.manual_seed(814)
 x=torch.randn(7,2,tokens,128,device='cuda',dtype=dtype).float()
 actual=W._summarize_key_batches(x)
 maxerr=0.;equal=True
 for row in range(7):
  for head in range(2):
   expected=build_summary(x[row,head])
   for name in ['min','max','mean','mad']:
    a=getattr(actual[row][head],name);b=getattr(expected,name)
    equal=equal and torch.equal(a,b);maxerr=max(maxerr,(a-b).abs().max().item())
 results.append({'tokens':tokens,'input_dtype':str(dtype),'summary_bitwise_equal':equal,'max_error':maxerr})
q=torch.randn(3,2,128,device='cuda');m=torch.randn(199,2,128,device='cuda')
a=(q[:,None]*m[None]).sum(-1).mean(-1);b=score_mean_heads_gemm(q,m)
results.append({'gemm_error':(a-b).abs().max().item(),'full_ranks_equal':torch.equal(a.argsort(-1),b.argsort(-1))})
from tests.test_step_1_2_summaries import _make_layer,_make_worker
from tests.test_batched_summaries import _key,_reference,_assert_summary_equal
for dtype in [torch.float16,torch.float32]:
 layer=_make_layer(12,2,16,128,seed=923);layer.kv_cache=layer.kv_cache.to(device='cuda',dtype=dtype)
 worker=_make_worker({'probe':layer},block_size=16)
 layout=[(_key(1),[7,2]),(_key(2),[0]),(_key(3),[5,5,1]),(_key(4),[]),(_key(5),[4])]
 expected={key:_reference(layer,ids) for key,ids in layout if ids}
 worker.receive_job_blocks({19:layout});worker._build_durable_summaries_for_job(19)
 worker._method='mean';worker._rebuild_stack_cache()
 for index,key in enumerate(worker._stack_cache_keys):
  assert torch.equal(worker._stack_cache['mean'][index],torch.stack([s.mean for s in expected[key]]))
 layer.kv_cache.fill_(999)
 for key,value in expected.items():_assert_summary_equal(worker.durable_summaries[key],value)
 assert list(worker.durable_summaries)==list(expected)
 assert not torch.equal(expected[_key(1)][0].mean,expected[_key(2)][0].mean)
 results.append({'layout_input_dtype':str(dtype),'variable_key_mapping_bitwise_equal':True,'owned_after_kv_reuse':True,'wrong_mapping_rejected':True})
Path(__file__).with_name('cuda-correctness-final.json').write_text(json.dumps(results,indent=2));print(results)
assert all(x.get('summary_bitwise_equal',True) for x in results)
