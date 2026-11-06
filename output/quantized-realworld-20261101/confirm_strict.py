"""Execute only the frozen finite public campaign, with owned child cleanup."""
import argparse,gzip,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
H=Path(__file__).resolve().parent

def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def check_pins(plan):
 for relative,expected in plan['pins'].items():
  p=(H/relative).resolve()
  if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise RuntimeError('frozen pin changed: '+relative)

def owned(command,log):
 started=time.time();before={p.pid:p.create_time() for p in psutil.Process().children(recursive=True)}
 with log.open('w') as out:
  p=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
  try:return p.wait()
  except BaseException:
   if p.poll() is None:p.send_signal(signal.SIGINT)
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=15)
   raise
  finally:
   children=[]
   for q in psutil.Process().children(recursive=True):
    try:
     if before.get(q.pid)!=q.create_time() and q.create_time()>=started:children.append(q)
    except psutil.Error:pass
   for q in children:
    try:q.terminate()
    except psutil.Error:pass
   _,alive=psutil.wait_procs(children,timeout=8)
   for q in alive:
    try:q.kill()
    except psutil.Error:pass
   psutil.wait_procs(alive,timeout=5)

def validate_summary(summary,name):
 if (summary.get('valid') is not True
     or summary.get('proofs',{}).get('all_passed') is not True
     or summary.get('proofs',{}).get('passed') != summary.get('all',{}).get('requests')
     or summary.get('all',{}).get('failures') != 0):
  raise RuntimeError('strict metrics/proof failure '+name)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args()
 planpath=Path(a.plan);plan=json.loads(planpath.read_text());check_pins(plan)
 progress={'plan':str(planpath.resolve()),'plan_sha256':hashlib.sha256(planpath.read_bytes()).hexdigest(),'pid':os.getpid(),'start_ns':time.monotonic_ns(),'status':'running','completed':[],'active':None}
 state=H/plan.get('progress_file','confirmation-readmission-progress.json');write(state,progress)
 try:
  for run in plan['runs']:
   check_pins(plan);progress['active']=run['name'];write(state,progress)
   print('START '+run['name'],flush=True)
   code=owned(run['command'],H/'qualification'/('driver-'+run['name']+'.log'))
   rawpath=H/'runs'/run['name']/'raw.json.gz'
   if code!=0 or not rawpath.exists():raise RuntimeError('run failed '+run['name']+' code '+str(code))
   raw=json.load(gzip.open(rawpath))
   if raw['status']!='complete' or raw['owned_descendants_remaining']:raise RuntimeError('run status/cleanup failure '+run['name'])
   summarypath=H/'runs'/run['name']/'summary.json'
   code=owned([sys.executable,'-B',str(H/'campaign_metrics.py'),'summarize',str(rawpath.parent),'--output',str(summarypath)],H/'qualification'/('metrics-'+run['name']+'.log'))
   summary=json.loads(summarypath.read_text()) if summarypath.exists() else {}
   if code:raise RuntimeError('metrics command failure '+run['name'])
   validate_summary(summary,run['name'])
   progress['completed'].append({'name':run['name'],'status':raw['status'],'output_tokens_s':summary['all']['output_tokens_s'],'requests':summary['all']['requests'],'strict_valid':summary['proofs']['all_passed'],'proofs_passed':summary['proofs']['passed']});progress['active']=None;write(state,progress)
   print('DONE '+json.dumps(progress['completed'][-1]),flush=True)
  progress['status']='complete'
 except BaseException as exc:
  progress['status']='failed';progress['error']=repr(exc);raise
 finally:
  progress['end_ns']=time.monotonic_ns();write(state,progress)
if __name__=='__main__':main()
