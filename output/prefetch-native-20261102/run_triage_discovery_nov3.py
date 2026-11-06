"""Root-owned finite matched pair for a declared repository-triage service."""
from pathlib import Path
import json,hashlib,subprocess,sys,time,os,signal,gzip,shutil
H=Path(__file__).resolve().parent;ROOT=H.parents[1];EV=ROOT/'evidence/live-optimization-2080ti-20261103'
PYTHON='/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
original=json.loads((EV/'confirmation-frozen.json').read_text());pins=dict(original['file_pins'])
for name in ['run_live_service_nov3.py','live_service_workload_nov3.py']:
 p=H/name;pins[str(p.relative_to(ROOT))]=sha(p)
base=['--source-dir',str((H/'sources/nov3-hardened').relative_to(ROOT)),'--gpu-blocks','1024','--cpu-blocks','1024','--active','8','--batch-tokens','1024','--clients','24','--cohort','agent','--turns','3','--wait','bursty','--support-profile','--service-profile','repository_triage_v1','--hint-interval-ms','1000','--graphs','--timeout','1800','--startup-timeout','600','--seed','1103']
runs=[]
for arm in ['native','candidate']:
 name='live-nov3-triage-s1103-'+arm
 args=['--name',name,*base,*( ['--arm','arc','--native-arc'] if arm=='native' else ['--arm','candidate','--no-audit','--source-label','shared-lookup'])]
 runs.append({'name':name,'arm':arm,'args':args})
plan={'schema':'triage-discovery-pair-v1','scope':'New service profile discovery only, one matched pair, no performance inference/production claim. Public repository issue and recorded observations, not executed tools. Natural EOS and actual histories, no output cap.','created_ns':time.time_ns(),'pins':pins,'runs':runs}
p=H/'live-nov3-triage-discovery-plan.json';assert not p.exists();write(p,plan)
def verify():
 for name,pin in pins.items():
  p=Path(name);p=p if p.is_absolute() else ROOT/p
  assert sha(p)==pin,('pin drift',name)
progress=[];child=None
try:
 for item in runs:
  verify();name=item['name'];assert not(H/'runs'/name).exists();log=H/(name+'-driver.log');start=time.monotonic_ns();cmd=[PYTHON,str(H/'run_live_service_nov3.py'),*item['args']]
  print(json.dumps({'starting':name}),flush=True)
  with log.open('w') as f:
   child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   write(H/'live-nov3-triage-root-custody.json',{'owner':'root','supervisor_pid':os.getpid(),'child_pid':child.pid,'started_ns':start,'run':name})
   code=child.wait(timeout=2700);child=None
  raw=H/'runs'/name/'raw.json.gz';result=json.load(gzip.open(raw)) if raw.exists() else {}
  row={'name':name,'exit':code,'elapsed_s':(time.monotonic_ns()-start)/1e9,'status':result.get('status'),'requests':len(result.get('requests',[])),'EOS':sum(q.get('natural_eos',False) for q in result.get('requests',[])),'owned_descendants_remaining':result.get('owned_descendants_remaining'),'driver_log_sha256':sha(log)}
  with log.open('rb') as src,gzip.open(str(log)+'.gz','wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
  assert hashlib.sha256(gzip.open(str(log)+'.gz').read()).hexdigest()==row['driver_log_sha256'];log.unlink()
  verify();progress.append(row);write(H/'live-nov3-triage-discovery-progress.json',progress);print(json.dumps(row),flush=True)
finally:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGINT)
  try:child.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 verify();write(H/'live-nov3-triage-root-custody.json',{'owner':'root','status':'supervisor finished, check no children/GPU before handoff','supervisor_pid':os.getpid()})
