"""Finite frontier runner: capacity caps are censored; proof failures stop."""
import argparse,gzip,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
H=Path(__file__).resolve().parent

def write(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)

def check_pins(plan):
 for rel,expected in plan['pins'].items():
  if hashlib.sha256((H/rel).resolve().read_bytes()).hexdigest()!=expected:raise RuntimeError('frozen pin changed: '+rel)

def classify(raw,summary):
 if raw.get('owned_descendants_remaining'):raise RuntimeError('owned processes survived cleanup')
 rows=raw.get('requests',[]);proof=summary.get('proofs',{})
 # Every recorded response row must retain its strict proof; missing proofs
 # for unfinished submissions are NOT converted into response/proof rows.
 if (proof.get('requests')!=len(rows) or proof.get('passed')!=len(rows)
     or proof.get('failures') or any(r.get('success') is not True for r in rows)):
  raise RuntimeError('completed-response strict proof or response failure')
 n=raw.get('planned_requests');submitted=raw.get('submission_events',[])
 indices=[r['index'] for r in rows];sent=[r['index'] for r in submitted]
 if len(set(indices))!=len(indices) or len(set(sent))!=len(sent) or not set(indices)<=set(sent):raise RuntimeError('submission/completion identity mismatch')
 complete=(raw.get('status')=='complete')
 if complete:
  if (summary.get('valid') is not True or proof.get('all_passed') is not True
      or not isinstance(n,int) or n!=len(rows) or n!=len(sent)):
   raise RuntimeError('incomplete load marked complete')
  outcome='complete'
 elif raw.get('status')=='failed' and str(raw.get('error','')).startswith('TimeoutError('):
  outcome='traffic_time_capped' if raw.get('load_start_ns') else 'startup_time_capped'
 else:raise RuntimeError('unexpected runtime failure: '+str(raw.get('error',raw.get('status'))))
 return {'outcome':outcome,'complete_load_eligible':complete,'planned_requests':n,'submitted_requests':len(sent),'completed_requests':len(rows),'completed_strict_proofs':proof['passed'],'unfinished_submitted_indices':sorted(set(sent)-set(indices)),'unreleased_requests':n-len(sent) if isinstance(n,int) else None,'elapsed_load_s':summary.get('all',{}).get('elapsed_s'),'performance_claim':'descriptive complete workflow only; output/key inventory required' if complete else 'censored, not zero throughput and excluded from complete-load ratios'}

def owned(command,log):
 before={p.pid:p.create_time() for p in psutil.Process().children(recursive=True)}
 with log.open('w') as f:
  proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
  try:return proc.wait()
  except BaseException:
   if proc.poll() is None:proc.send_signal(signal.SIGINT)
   try:proc.wait(timeout=60)
   except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=15)
   raise
  finally:
   children=[]
   for p in psutil.Process().children(recursive=True):
    try:
     if before.get(p.pid)!=p.create_time():children.append(p)
    except psutil.Error:pass
   for p in children:
    try:p.terminate()
    except psutil.Error:pass
   _,alive=psutil.wait_procs(children,timeout=8)
   for p in alive:
    try:p.kill()
    except psutil.Error:pass
   psutil.wait_procs(alive,timeout=5)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();planpath=Path(a.plan);p=json.loads(planpath.read_text());check_pins(p)
 state=H/p['progress_file'];progress={'status':'running','pid':os.getpid(),'start_ns':time.monotonic_ns(),'plan':str(planpath.resolve()),'completed':[],'active':None};write(state,progress)
 try:
  for run in p['runs']:
   check_pins(p);progress['active']=run['name'];write(state,progress);print('START '+run['name'],flush=True)
   code=owned(run['command'],H/'qualification'/('driver-'+run['name']+'.log'));directory=H/'runs'/run['name'];path=directory/'raw.json.gz'
   if not path.exists():raise RuntimeError('missing recoverable raw after driver '+str(code))
   raw=json.load(gzip.open(path));summarypath=directory/'summary.json'
   code=owned([sys.executable,'-B',str(H/'campaign_metrics.py'),'summarize',str(directory),'--output',str(summarypath)],H/'qualification'/('metrics-'+run['name']+'.log'))
   if code:raise RuntimeError('metrics command failed')
   summary=json.loads(summarypath.read_text());disposition=classify(raw,summary);import replay_judge;trace=json.loads((H/'prepared'/f"fixed-agent-{run['seed']}.json").read_text());integrity=replay_judge.validate(raw,trace,clients=run.get("clients",4));integrity['invalid_control_rejected']=replay_judge.reject_invalid_controls(raw,trace,clients=run.get("clients",4));disposition['fixed_work_integrity']=integrity;write(directory/'frontier-disposition.json',disposition)
   row=dict(name=run['name'],arm=run['arm'],gpu_blocks=run['gpu_blocks'],seed=run['seed'],**disposition);progress['completed'].append(row);progress['active']=None;write(state,progress);print('DONE '+json.dumps(row),flush=True)
  progress['status']='complete'
 except BaseException as exc:progress.update(status='failed',error=repr(exc));raise
 finally:progress['end_ns']=time.monotonic_ns();write(state,progress)
if __name__=='__main__':main()
