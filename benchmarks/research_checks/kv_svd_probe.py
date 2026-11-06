"""Codec cost/workspace diagnostic, supplemental to actual serving evidence."""
import json
import time
import argparse
from pathlib import Path
import torch
from harness.gpu_lock import GPULock
from semantic_offload.kv_svd import encode, decode


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with GPULock():
        torch.manual_seed(17011)
        x=torch.randn(2800,16,256,device='cuda',dtype=torch.float16)
        results=[]
        for chunk in (16,64,128,256,512,2800):
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            times=[]
            for repeat in range(4):
                torch.cuda.synchronize();start=time.perf_counter()
                factors=[encode(part,8) for part in x.split(chunk)]
                u=torch.cat([v[0] for v in factors]);proj=torch.cat([v[1] for v in factors])
                restored=decode(u,proj)
                torch.cuda.synchronize();times.append(time.perf_counter()-start)
                del restored,u,proj,factors
            results.append(dict(chunk=chunk,seconds=times,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
        a.output.write_text(json.dumps(results,indent=2))
        print(json.dumps(results))
if __name__=='__main__':main()
