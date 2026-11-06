"""Execute exactly three frozen spent-input observation-only invocations; no replacements."""
import ctypes,datetime,fcntl,gzip,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from http_confirmation_receipt_astra_nov3 import verify_http
from coverage_receipt_astra_nov3 import verify_coverage
ROOT=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent
MANIFEST=H/'gpu-coverage-frozen-plan-nov3.json'
RECEIPT=H/'live-nov3-gpu-coverage-custody.json';PROGRESS=H/'live-nov3-gpu-coverage-progress.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def compress(p):
 original=sha(p);q=p.with_suffix(p.suffix+'.gz')
 with p.open('rb') as src,gzip.open(q,'wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
 with gzip.open(q,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==original
 p.unlink();return {'path':str(q.relative_to(ROOT)),'uncompressed_sha256':original,'bytes':q.stat().st_size}
def clean_children():
 owned=psutil.Process().children(recursive=True)
 for p in reversed(owned):
  try:p.terminate()
  except psutil.Error:pass
 _,alive=psutil.wait_procs(owned,timeout=8)
 for p in alive:
  try:p.kill()
  except psutil.Error:pass
 psutil.wait_procs(alive,timeout=5)
 return [p.pid for p in alive if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE]
def idle():
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
 processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
 if processes:raise RuntimeError('GPU not idle: '+processes)
def main():
 os.chdir(ROOT);ctypes.CDLL(None).prctl(36,1,0,0,0)
 manifest_bytes=MANIFEST.read_bytes();manifest_hash=hashlib.sha256(manifest_bytes).hexdigest();m=json.loads(manifest_bytes)
 assert len(m['execution_order'])==3 and len(m['file_pins'])==191
 assert not PROGRESS.exists() and not RECEIPT.exists(),'Refuse to overwrite prior custody/progress'
 assert all(not (H/'runs'/x['name']).exists() for x in m['execution_order'])
 (H/'gpu-coverage-frozen-astra-copy.json').write_bytes(manifest_bytes)
 state={'schema':'astra-gpu-coverage-custody-v1','status':'starting','accepted_utc':now(),'manifest_sha256':manifest_hash,'supervisor_sha256':sha(Path(__file__)),'coverage_verifier_sha256':sha(H/'coverage_receipt_astra_nov3.py'),'cadence_verifier_sha256':sha(H/'cadence_receipt_astra_nov3.py'),'HTTP_verifier_sha256':sha(H/'http_confirmation_receipt_astra_nov3.py'),'supervisor_pid':os.getpid(),'supervisor_create_time':psutil.Process().create_time(),'free_bytes_before':shutil.disk_usage(H).free,'runs':[],'pin_checks':[]};write(RECEIPT,state);write(PROGRESS,[])
 def pins(stage):
  bad=[]
  if sha(MANIFEST)!=manifest_hash:bad.append(str(MANIFEST))
  for name,expected in m['file_pins'].items():
   p=ROOT/name
   if not p.is_file() or sha(p)!=expected:bad.append(name)
  check={'stage':stage,'at_utc':now(),'checked_files':len(m['file_pins']),'mismatches':bad};state['pin_checks'].append(check);write(RECEIPT,state)
  if bad:raise RuntimeError('Frozen pin drift: '+repr(bad))
 failures=0
 try:
  pins('before_first');idle();state['status']='running';write(RECEIPT,state)
  for index,item in enumerate(m['execution_order']):
   pins('before_'+item['name']);idle()
   cmd=[m['python'],str(ROOT/item.get('runner',m['runner'])),*item['args']]
   row={**item,'command':cmd,'started_utc':now(),'start_monotonic_ns':time.monotonic_ns(),'sequence':index};log=H/(item['name']+'-driver-console.log')
   print('START '+item['name'],flush=True)
   with log.open('x') as f:
    child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    row['runner_pid']=child.pid;state['current']=row;write(RECEIPT,state);code=child.wait()
   row.update(exit_code=code,completed_utc=now(),wall_s=(time.monotonic_ns()-row['start_monotonic_ns'])/1e9)
   remaining=psutil.Process().children(recursive=True)
   if remaining:
    row['unexpected_descendants_after_runner']=[p.pid for p in remaining];row['cleanup_remaining']=clean_children()
   raw=H/'runs'/item['name']/'raw.json.gz'
   if raw.exists():
    r=json.load(gzip.open(raw,'rt'));requests=r.get('requests',[]);row.update(raw_path=str(raw.relative_to(ROOT)),raw_sha256=sha(raw),status=r.get('status'),error=r.get('error'),requests=len(requests),planned=r.get('planned_requests'),natural_eos=sum(bool(q.get('natural_eos')) for q in requests),load_s=(r['load_end_ns']-r['load_start_ns'])/1e9 if 'load_end_ns' in r else None,owned_descendants_remaining=r.get('owned_descendants_remaining'))
    row['HTTP_receipt']=verify_http(r)
    row['workflow_success']=row['HTTP_receipt']['qualified'] and code==0 and r.get('status')=='complete' and len(requests)==r.get('planned_requests') and bool(requests) and all(q.get('success') and q.get('natural_eos') for q in requests)
    s=r.get('postload_export',{}).get('scheduler',{});row['executed_configuration']=s.get('measurement_configuration')
    # Narrow explicit corruption indicators; do not mistake optional deep_gemm
    # import warnings for a serving correctness fault.
    body=(r.get('error','') or '')+' '+(r.get('error_traceback','') or '')
    row['correctness_fault']=any(x in body.lower() for x in ['actual kv transfer mismatch','invalid one-byte oracle failed','device-side assert','illegal memory access'])
    for server_log in raw.parent.glob('server*.log.gz'):
     with gzip.open(server_log,'rt',errors='replace') as f:server_text=f.read().lower()
     row['correctness_fault'] |= any(x in server_text for x in ['actual kv transfer mismatch','invalid one-byte oracle failed','device-side assert triggered','cuda error: an illegal memory access'])
    if row['workflow_success']:
     expected={'lean':True,'residency_scans':False,'readmission_observer':False,'lookup_deferral_observer':False,'finished_token_clamp':True}
     row['measurement_fault']=r.get('http_client_configuration',{}).get('force_close') is not True or bool(row['HTTP_receipt'].get('counts',{}).get('connection_reuse')) or s.get('measurement_configuration')!=expected or bool(s.get('samples')) or bool(r.get('readmission_observer_events')) or bool(r.get('native_lookup_deferral_events'))
    if row['workflow_success']:
     row['coverage_receipt']=verify_coverage(r)
     row['measurement_fault'] |= not row['coverage_receipt']['qualified']
   else:row.update(status='missing_raw',workflow_success=False,error='Runner exited without raw evidence')
   row['console']=compress(log);state['runs'].append(row);state.pop('current',None);write(PROGRESS,state['runs']);write(RECEIPT,state)
   print('END '+json.dumps({k:row.get(k) for k in ['name','exit_code','status','workflow_success','natural_eos','requests','load_s','error']}),flush=True)
   failures=0 if row['workflow_success'] else failures+1
   if 'pair' in item:
    pair=m['pairs'][item['pair']];done={x['name'] for x in state['runs']}
    if set(pair['runs'].values())<=done:print('PAIR_COMPLETE '+str(item['pair']),flush=True)
   if row.get('correctness_fault') or row.get('measurement_fault') or row.get('unexpected_descendants_after_runner') or row.get('owned_descendants_remaining'):
    state['stop_reason']='correctness_measurement_or_cleanup_fault';break
   if failures>=3:state['stop_reason']='three_consecutive_failed_invocations';break
  pins('end');idle();state['status']='complete' if len(state['runs'])==3 and 'stop_reason' not in state else 'stopped_by_manifest_condition'
 except BaseException as e:
  state.update(status='stopped_by_external_or_pin_blocker',stop_reason=repr(e),exception_traceback=traceback.format_exc());print('STOP '+repr(e),flush=True)
 finally:
  state['owned_descendants_remaining']=clean_children();state['completed_utc']=now();state['free_bytes_after']=shutil.disk_usage(H).free;write(RECEIPT,state)
  print('SUPERVISOR_FINISHED '+json.dumps({'status':state['status'],'runs':len(state['runs']),'reason':state.get('stop_reason'),'remaining':state['owned_descendants_remaining']}),flush=True)
 return 0 if state['status']=='complete' else 1
if __name__=='__main__':sys.exit(main())
