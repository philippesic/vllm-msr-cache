import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_NVML_BASED_CUDA_CHECK='1',MATRIX_RUN_DIR=str(H/'cpu-observations'))
sys.path[:0]=[str(ROOT),str(H/'hooks')]
spec=importlib.util.spec_from_file_location('matrix_sitecustomize',H/'hooks/sitecustomize.py')
sitecustomize=importlib.util.module_from_spec(spec);spec.loader.exec_module(sitecustomize)
import torch
assert not torch.cuda.is_initialized()
def guard(name):
    def forbid(*a,**kw):raise AssertionError(name+': CUDA forbidden during CPU validation')
    return forbid
torch.cuda.init=guard('init');torch.cuda._lazy_init=guard('lazy');torch.cuda.set_device=guard('set_device');torch.cuda.is_available=lambda:False;torch.set_num_threads(1)
import pytest
paths=['tests/'+x for x in ['test_batched_summaries.py','test_config.py','test_step_1_3_scoring.py','test_step_1_4_eviction.py','test_query_capture.py','test_step_1_5_prefetch.py','test_wait_hint.py','test_wait_hint_endpoint.py','test_scoring_schedule.py','test_scoring_search.py','test_async_scoring.py','test_summary_offload_overlap.py','test_audit.py','test_round2.py','test_round4.py','test_throughput.py','test_agent_load_trace.py','test_mean_only_summaries.py','test_opportunity_metadata.py','test_full_prefix_nomination.py']]
pins={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sorted((ROOT/'semantic_offload').glob('*.py')),*[ROOT/p for p in paths],H/'hooks/sitecustomize.py',*sorted((H/'runtime').rglob('*.py'))]};t=time.monotonic();code=pytest.main(['-q','-p','no:cacheprovider',*paths]);assert not torch.cuda.is_initialized()
(H/'cpu-validation-final.json').write_text(json.dumps({'exit_code':int(code),'elapsed_s':time.monotonic()-t,'cuda_initialized':False,'cuda_forbidden':True,'task_runtime_overlay':True,'passive_matrix_wrappers':True,'paths':paths,'source_pins':pins,'source_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in pins.items())},indent=2))
raise SystemExit(code)
