"""Remove only verified task scratch after all experiment owners have exited."""
import hashlib,json,os,shutil,time
from pathlib import Path
import psutil
H=Path(__file__).resolve().parent
owner=json.loads((H/'confirmation-owner.json').read_text())['process']
try:
 p=psutil.Process(owner['pid'])
 assert p.create_time()!=owner['create_time'] or p.status()==psutil.STATUS_ZOMBIE,'Confirmation is still active'
except psutil.NoSuchProcess:pass
items=[]
for base,dirs,files in os.walk(H,topdown=True,followlinks=False):
 for name in list(dirs):
  p=Path(base)/name
  if p.is_symlink():dirs.remove(name);continue
  if name=='__pycache__':
   dirs.remove(name);items.append((p,'Task-local disposable Python bytecode'))
for name in ['query-capture-child-before.py','query-capture-child-tests-before.py']:
 p=Path('/tmp')/name;backup=H/'baseline'/name
 if p.exists():
  assert not p.is_symlink() and p.stat().st_uid==os.getuid()
  assert p.read_bytes()==backup.read_bytes()
  items.append((p,'Capture owner confirmed exclusive unused duplicate; verified recovery copy retained'))
smoke=H.parent/'scoring-overhead-smoke-20261101'
if smoke.exists():
 assert not smoke.is_symlink()
 r=json.loads((smoke/'reproduction.json').read_text())
 assert r['origin']==str(H) and r['run_requested'] is False
 assert not (smoke/'runs').exists()
 items.append((smoke,'Disposable verified reproduction clone; original inputs retained'))
def size(path):
 if path.is_symlink():return 0
 if path.is_file():return path.stat().st_size
 return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
removed=[{'path':str(p),'logical_bytes':size(p),'reason':reason} for p,reason in items]
free_before=shutil.disk_usage(H).free
for p,_ in items:
 assert p.exists() and not p.is_symlink() and p.stat().st_uid==os.getuid()
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
free_after=shutil.disk_usage(H).free
retained=[]
for p in H.iterdir():
 if p.is_dir():
  retained.append({'path':str(p),'logical_bytes':size(p),'owner':'this research task','reason':{'runs':'Compressed positive/negative results, server trace and integrity evidence','sources':'Exact implementation variants for reproducing accepted and rejected candidates','prepared':'Fixed workload tokens and geometry','runtime':'Shared across arms; companion vLLM seams required for reproduction','baseline':'Recovery of original dirty source and tests','hooks':'Pinned primary confirmation measurement seam','hooks_future':'Exploratory thread/CPU quality instrumentation'}.get(p.name,'Reproduction or rejected-candidate evidence'),'next_review':'After user review or explicit evidence retention change'})
gpu=[];gpu_error=None
try:
 import pynvml
 pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
 gpu=[{'pid':p.pid,'used_gpu_memory':p.usedGpuMemory} for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]
 pynvml.nvmlShutdown()
except Exception as e:gpu_error=repr(e)
receipt={'at_ns':time.time_ns(),'host':'RTX2080Ti WSL','scope':str(H),'removed':removed,'filesystem_measurement':{'path':str(H),'free_bytes_before':free_before,'free_bytes_after':free_after,'observed_free_byte_delta':free_after-free_before,'note':'WSL filesystem observation only; concurrent activity can affect delta. No Windows VHDX reclaim or compaction claim.'},'retained':sorted(retained,key=lambda r:r['logical_bytes'],reverse=True),'retained_total_logical_bytes':size(H),'gpu_compute_processes':gpu,'gpu_inventory_error':gpu_error,'protected_shared_resources':['Existing HF model cache','Existing project venv and installed runtime','Existing Torch/Triton caches','Earlier experiment output and preexisting dirty/untracked source'],'process_cleanup':'Per-run drivers terminate/reap owned descendants; final-integrity.json and run-index.json retain shutdown receipts. No shared processes terminated.','no_downloads_checkpoints_worktrees_or_build_outputs_created':True,'no_commit_or_push':True}
(H/'STORAGE.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'removed_items':len(removed),'retained_evidence_mib':receipt['retained_total_logical_bytes']/2**20,'free_before':free_before,'free_after':free_after,'observed_delta':free_after-free_before,'gpu_compute_process_count':len(gpu),'gpu_inventory_error':gpu_error},indent=2))
