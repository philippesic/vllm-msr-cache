"""One finite serialized qualification/screen continuation, owned by root."""
import fcntl,json,subprocess,sys
from pathlib import Path
import psutil
from frontier import owned
from work_inventory import compare
H=Path(__file__).resolve().parent
try:psutil.Process(83873).wait(timeout=60)
except psutil.NoSuchProcess:pass
except psutil.TimeoutExpired:raise SystemExit('Prior confirmation still active; GPU tests not started')
progress=json.loads((H/'spaced-confirm-progress.json').read_text())
assert progress['status']=='complete',progress
work=[]
for seed in [1103,2207,3301]:
 row=compare(f'spaced-confirm-b524-arc-s{seed}-r1',f'spaced-confirm-b520-candidate-s{seed}-r1');work.append(row)
 print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)
(H/'spaced-confirm-work.json').write_text(json.dumps(work,indent=2)+'\n')
with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 logfile=H/'qualification/fused-gpu-unit.log'
 code=owned([sys.executable,'-B','-m','pytest','-q',str(H/'test_fused_mean_gpu.py'),str(H/'test_fused_worker_gpu.py')],logfile)
 print(logfile.read_text()[-12000:],flush=True)
 if code:raise SystemExit(code)
raise SystemExit(subprocess.call([sys.executable,'-B',str(H/'frontier.py'),'--plan',str(H/'FUSED-SPACED.json')]))
