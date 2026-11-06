import importlib.util,json,os,sys
from pathlib import Path
H=Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTORCH_NVML_BASED_CUDA_CHECK='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SEMANTIC_PREFIX_POLICY_QUALITY='1',SEMANTIC_CAPTURE_STEPS='all',MATRIX_RUN_DIR=str(H/'quality-preflight-observations'))
sys.path[:0]=[str(H/'sources/shared-lookup'),str(H.parents[1])]
import torch
assert not torch.cuda.is_initialized()
def guard(name):
    def forbid(*a,**kw):raise AssertionError(name+':CUDA forbidden')
    return forbid
torch.cuda.init=guard('init');torch.cuda._lazy_init=guard('lazy');torch.cuda.set_device=guard('set_device');torch.cuda.is_available=lambda:False
spec=importlib.util.spec_from_file_location('quality_transport_check',H/'quality_hooks_readmission/sitecustomize.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
r=module.transport_selfcheck();assert not torch.cuda.is_initialized();r['cuda_initialized']=False
(H/'quality-transport-validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
