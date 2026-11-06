import importlib.util, json, os, sys, time
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_NVML_BASED_CUDA_CHECK='1',MATRIX_RUN_DIR=str(H/'cpu-observations'))
sys.path[:0]=[str(H/'sources/asynccpu'),str(ROOT),str(H/'hooks')]
spec=importlib.util.spec_from_file_location('matrix_sitecustomize',H/'hooks/sitecustomize.py')
sitecustomize=importlib.util.module_from_spec(spec);spec.loader.exec_module(sitecustomize)
import torch
assert not torch.cuda.is_initialized()
def guard(name):
    def forbid(*a,**kw):raise AssertionError(name+': CUDA forbidden during CPU validation')
    return forbid
torch.cuda.init=guard('init');torch.cuda._lazy_init=guard('lazy');torch.cuda.set_device=guard('set_device');torch.cuda.is_available=lambda:False;torch.set_num_threads(1)
import semantic_offload.worker as selected_worker
assert Path(selected_worker.__file__).is_relative_to(H/'sources/asynccpu'), selected_worker.__file__
import pytest
paths=['tests/'+x for x in ['test_async_scoring.py','test_audit.py','test_round2.py','test_round4.py','test_step_1_3_scoring.py','test_batched_summaries.py']] + [str(H/'test_asyncpacked.py'), str(H/'test_asynccpu.py')]
t=time.monotonic();code=pytest.main(['-q','-p','no:cacheprovider',*paths]);assert not torch.cuda.is_initialized()
(H/'asynccpu-cpu-validation.json').write_text(json.dumps({'exit_code':int(code),'elapsed_s':time.monotonic()-t,'cuda_initialized':False,'cuda_forbidden':True,'task_runtime_overlay':True,'selected_worker':selected_worker.__file__,'passive_matrix_wrappers':True,'paths':paths},indent=2))
raise SystemExit(code)
