"""Serialized GPU qualification; run only with the serving GPU lock free."""
import json,os
from pathlib import Path
import pytest
import torch
from fused_mean import owned_mean_pages

@pytest.mark.parametrize('count',[0,1,7,64,200])
@pytest.mark.parametrize('strided',[False,True])
def test_actual_fp16_kv_means_and_owned_storage(count,strided):
 torch.manual_seed(7401)
 if strided:
  backing=torch.randn((414,8,32,512),device='cuda',dtype=torch.float16)
  kv=backing[::2,::2,::2,::2]
 else:kv=torch.randn((207,4,16,256),device='cuda',dtype=torch.float16)
 kv[...,128:]=3000
 ids=[(i*31+3)%207 for i in range(count)]
 before=kv.clone()
 expected=torch.stack([kv[i,:,:,:128] for i in ids]).float().mean(2) if ids else torch.empty((0,4,128),device='cuda',dtype=torch.float32)
 actual=owned_mean_pages(kv,ids,128)
 torch.testing.assert_close(actual,expected,rtol=0,atol=1e-6)
 assert torch.equal(kv,before),'summary builder must never modify KV payload'
 error=float((actual-expected).abs().max()) if count else 0.
 path=Path(__file__).resolve().parent/'qualification/fused-mean-numerics.jsonl'
 with path.open('a') as f:f.write(json.dumps({'count':count,'strided':strided,'maximum_absolute_error':error,'kv_unchanged':True,'pid':os.getpid()})+'\n')
 kv.fill_(42)
 torch.testing.assert_close(actual,expected,rtol=0,atol=1e-6)

@pytest.mark.parametrize('bad',[-1,9])
def test_invalid_physical_block_rejected(bad):
 kv=torch.ones((9,4,16,256),device='cuda',dtype=torch.float16)
 with pytest.raises(IndexError):owned_mean_pages(kv,[bad],128)

def test_producer_and_summary_stream_ownership_fence():
 kv=torch.zeros((9,4,16,256),device='cuda',dtype=torch.float16)
 producer=torch.cuda.current_stream();summary=torch.cuda.Stream()
 kv.fill_(3)
 with torch.cuda.stream(summary):
  summary.wait_stream(producer)
  actual=owned_mean_pages(kv,[4,1,4],128)
  done=torch.cuda.Event();done.record()
 producer.wait_event(done);kv.fill_(9)
 torch.testing.assert_close(actual,torch.full_like(actual,3),rtol=0,atol=0)
