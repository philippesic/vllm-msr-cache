"""Actual installed native D2H/H2D handlers, all 28 packed Qwen KV pages.
Standalone transfer qualification, not serving/adoption or performance evidence.
"""
import hashlib,json
from pathlib import Path
import torch
import vllm.v1.kv_offload.cpu.gpu_worker as module
from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler
from vllm.v1.kv_offload.base import CanonicalKVCacheRef,GPULoadStoreSpec,BlockIDsLoadStoreSpec
H=Path(__file__).resolve().parent

def digest(t): return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def main():
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(1103)
    layers,page_size=28,16384
    gpu=[torch.empty((32,page_size),dtype=torch.int8,device='cuda') for _ in range(layers)]
    cpu=[torch.empty((32,page_size),dtype=torch.int8,pin_memory=True) for _ in range(layers)]
    refs=[[CanonicalKVCacheRef(i,page_size) for i in range(layers)]]
    store=SingleDirectionOffloadingHandler(list(gpu),list(cpu),1,refs,True)
    load=SingleDirectionOffloadingHandler(list(gpu),list(cpu),1,refs,False)
    records=[];total=0
    try:
        for case,ids in enumerate([[0],[0,1,2,3,4,5,6,7],[7,6,5,4,3,2,1,0],[31,0,7,7,2,15]]):
            golden=[torch.randint(-128,128,(32,page_size),dtype=torch.int8,pin_memory=True) for _ in range(layers)]
            producer=torch.cuda.Stream()
            with torch.cuda.stream(producer):
                for g,e in zip(gpu,golden):g.copy_(e,non_blocking=True)
                job=2*case+1
                assert store.transfer_async(job,GPULoadStoreSpec(ids,[len(ids)],[0]),BlockIDsLoadStoreSpec(list(range(len(ids)))))
            store._transfer_events[job].synchronize();done=store.get_finished()
            assert len(done)==1 and done[0].success and done[0].transfer_size==len(ids)*layers*page_size
            expected=[e[ids].clone() for e in golden]
            for c,e in zip(cpu,expected): assert torch.equal(c[:len(ids)],e)
            # Pressure may reuse original GPU pages only after the store fence.
            for g in gpu:g.fill_(90)
            torch.cuda.synchronize()
            dst=list(range(16,16+len(ids)))
            assert load.transfer_async(job+1,BlockIDsLoadStoreSpec(list(range(len(ids)))),GPULoadStoreSpec(dst,[len(ids)],[0]))
            load._transfer_events[job+1].synchronize();done=load.get_finished()
            assert len(done)==1 and done[0].success and done[0].transfer_size==len(ids)*layers*page_size
            actual=[g[dst].cpu() for g in gpu]
            assert all(torch.equal(a,e) for a,e in zip(actual,expected))
            assert all(bool((g[0]==90).all()) for g in gpu)
            corrupted=actual[0].clone();corrupted[0,0]^=1
            assert not torch.equal(corrupted,expected[0])
            size=len(ids)*layers*page_size;total+=2*size
            records.append({'source_gpu_ids':ids,'cpu_ids':list(range(len(ids))),'loaded_gpu_ids':dst,'bytes_per_direction':size,'all_28_layers_K_and_V_byte_equal':True,'source_reused_only_after_store_fence':True,'unrelated_gpu_pages_unchanged':True,'one_byte_corruption_control_rejected':True,'component_sha256':[digest(e) for e in expected]})
    finally:
        store.shutdown();load.shutdown();torch.cuda.synchronize()
    result={'passed':True,'device':torch.cuda.get_device_name(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'handler_path':module.__file__,'handler_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'layers':layers,'page_size_bytes':page_size,'block_bytes':layers*page_size,'total_transferred_bytes':total,'cases':records,'performance_eligible':False,'limitations':['Standalone native transport byte equality only; serving traces separately prove native physical adoption.','No model forward or generated-KV/logit comparison; does not explain the isolated no-H2D ARC output divergence.','Explicit completion fences and a controlled owner; cannot certify every uninstrumented concurrent schedule.']}
    (H/'native-byte-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':main()
