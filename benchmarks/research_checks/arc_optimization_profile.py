"""CPU-only cost bounds for optional ARC implementations; no serving claims."""
import argparse
import json
import statistics
import time

from semantic_offload.arc_optimization import ARCAdvice, LinearARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from types import SimpleNamespace


def populated(cls, size):
    p = cls(size)
    for k in range(size):
        b = BlockStatus(k); b.ref_cnt = 0; p.t1[k] = b
    return p


def run():
    rows = []
    for size in (256, 1024, 4096):
        for victims in (1, 16, 128, 256):
            for cls in (ARCCachePolicy, LinearARCCachePolicy):
                values = []
                for _ in range(9):
                    p = populated(cls,size)
                    t = time.perf_counter_ns(); p.evict(victims,set()); values.append(time.perf_counter_ns()-t)
                rows.append(dict(kind='eviction',size=size,victims=victims,implementation=cls.__name__,median_ns=statistics.median(values)))
    for blocked in (512, 2048, 3584):
        for victims in (1, 16, 128):
            for cls in (ARCCachePolicy, LinearARCCachePolicy):
                values = []
                for _ in range(9):
                    p = populated(cls,4096)
                    protected = set(range(blocked))
                    t = time.perf_counter_ns(); p.evict(victims,protected); values.append(time.perf_counter_ns()-t)
                rows.append(dict(kind='protected_eviction',size=4096,protected=blocked,victims=victims,implementation=cls.__name__,median_ns=statistics.median(values)))
    for size in (256,1024,4096):
        for name,options in [('default',{}),('recent',{'queue':'recent'}),('half-boost',{'boost':.5}),('quarter-prefix',{'fraction':.25}),('cooldown',{'cooldown_ms':2000}),('budget',{'session_budget':512})]:
            p = populated(ARCCachePolicy,size); a = ARCAdvice(options); m=SimpleNamespace(_policy=p)
            keys=list(range(min(size,512))); values=[]
            for i in range(101):
                t=time.perf_counter_ns(); a.refresh(m,keys,('s',i),i*1_000_000_000); values.append(time.perf_counter_ns()-t)
            rows.append(dict(kind='advice',size=size,keys=len(keys),implementation=name,median_ns=statistics.median(values)))
    return dict(scope='CPU microbenchmark only; no GPU/serving speedup claim',rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    with open(args.output,'x') as f: json.dump(run(),f,indent=2)
