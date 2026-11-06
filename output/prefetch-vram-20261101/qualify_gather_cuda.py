"""Root-owned exact CUDA gather/reduction check; no serving-performance claims."""
import argparse,hashlib,json
from pathlib import Path
import torch
import semantic_offload.worker as module
from semantic_offload.worker import SemanticOffloadingWorker
from tests.test_mean_only_summaries import _worker, _key
from tests.test_step_1_2_summaries import _make_layer

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    source=a.source.resolve();assert Path(module.__file__).resolve()==source/'semantic_offload/worker.py'
    pins=json.loads((source/'pins.json').read_text())['files']
    assert all(sha(source/p)==v for p,v in pins.items())
    assert torch.cuda.is_available();torch.set_num_threads(1)
    layouts=[[0],[1,2,3],list(range(120)),list(reversed(range(120))),[5,2,2,3,0],[1,2,7,8,9,0,0],list(range(0,128,2))]
    records=[]
    for dtype in [torch.float16,torch.float32]:
        for ids in layouts:
            torch.manual_seed(1103)
            producer=torch.cuda.Stream();consumer=torch.cuda.Stream()
            with torch.cuda.stream(producer):
                packed=torch.randn(128,2,16,256,device='cuda',dtype=dtype)
                event=torch.cuda.Event();event.record()
            with torch.cuda.stream(consumer):
                consumer.wait_event(event)
                expected=torch.stack([packed[i,...,:128] for i in ids]).float()
                actual=SemanticOffloadingWorker._gather_prefix_mean_key_blocks(packed,ids,128)
                assert actual.dtype==torch.float32 and actual.is_contiguous()
                assert actual.stride()==expected.stride()
                assert actual.untyped_storage().data_ptr()!=packed.untyped_storage().data_ptr()
                actual_means=actual.mean(-2);expected_means=expected.mean(-2)
            consumer.synchronize()
            assert torch.equal(actual,expected) and torch.equal(actual_means,expected_means)
            packed.fill_(999);torch.cuda.synchronize()
            assert torch.equal(actual,expected)
            records.append({'dtype':str(dtype),'ids':ids,'gather_bitwise_equal':True,'mean_bitwise_equal':True,'owned_after_source_overwrite':True})
        layer=_make_layer(128,2,16,128,2207);layer.kv_cache=layer.kv_cache.to('cuda',dtype=dtype)
        candidate=_worker(layer);baseline=_worker(layer)
        candidate._score_transport_context={'prefix_mean_prefetch_scoring':True}
        candidate._summary_stream=torch.cuda.Stream();baseline._summary_stream=torch.cuda.Stream()
        layout=[(_key(1),[7,2]),(_key(2),[0]),(_key(3),[5,5,1]),(_key(4),[]),(_key(1),[8]),(_key(5),[3,4])]
        ready=torch.cuda.Event();ready.record()
        for worker in [candidate,baseline]:
            worker.receive_job_blocks({19:layout})
            with torch.cuda.stream(worker._summary_stream):
                worker._summary_stream.wait_event(ready)
                worker._build_durable_summaries_for_job(19)
        torch.cuda.synchronize()
        assert candidate._durable_summary_publications==baseline._durable_summary_publications
        assert candidate._summary_job_publications==baseline._summary_job_publications
        assert list(candidate.durable_summaries)==list(baseline.durable_summaries)
        owned={k:[s.mean.clone() for s in v] for k,v in baseline.durable_summaries.items()}
        layer.kv_cache.zero_();torch.cuda.synchronize()
        for k,rows in candidate.durable_summaries.items():
            assert all(torch.equal(s.mean,v) for s,v in zip(rows,owned[k]))
        records.append({'dtype':str(dtype),'grouped_mixed_counts_duplicate_last_write':True,'publication_identity_equal':True,'all_published_means_bitwise_equal_and_owned':True})
    torch.cuda.synchronize()
    assert all(sha(source/p)==v for p,v in pins.items())
    result={'passed':True,'device':torch.cuda.get_device_name(),'source':str(source),'source_manifest_sha256':sha(source/'pins.json'),'script_sha256':sha(__file__),'worker_sha256':sha(module.__file__),'checks':records,'source_unchanged':True,'maximum_observed_absolute_error':0.0,'performance_eligible':False,'limitations':['Same-input numerical/ownership qualification; native end-to-end exactness and performance require separate runs.']}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
if __name__=='__main__':main()
