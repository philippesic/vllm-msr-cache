"""Serial frozen operational custody. Requires a separate ROOT launch certificate."""
import argparse,ctypes,datetime,fcntl,gzip,hashlib,json,os,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
import psutil
from operational_case_receipt_v2_astra_nov3 import verify_case
ROOT=Path(__file__).resolve().parents[2]
H=Path(__file__).resolve().parent
REVIEW=H/'operational-v3-review-plan-nov3.json'
RECEIPT=H/'live-nov3-operational-v3-custody.json'
PROGRESS=H/'live-nov3-operational-v3-progress.json'
EVENTS=H/'live-nov3-operational-v3-events.jsonl'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def resolve(path):return ROOT/Path(path)


def merged_pins(manifest):
    result={}
    for field in ('existing_frozen_inputs','new_review_file_pins','file_pins'):
        for name,pin in manifest.get(field,{}).items():
            path=str(resolve(name).resolve())
            if path in result and result[path]!=pin:raise ValueError('Conflicting file pins: '+path)
            result[path]=pin
    if str((ROOT/'README.md').resolve()) in result:raise ValueError('README is provenance-only in v3, not a live stop pin')
    if not result:raise ValueError('No frozen file pins')
    return result


def document_provenance():
    path=ROOT/'README.md'
    result={'path':str(path),'sha256':None,'at_utc':now(),'scope':'Nonexecuted documentation provenance only; changes do not stop execution.'}
    try:result['sha256']=sha(path) if path.is_file() else None
    except OSError as exc:result['read_error']=repr(exc)
    return result


def pin_mismatches(pins):
    return [path for path,expected in pins.items() if not Path(path).is_file() or sha(path)!=expected]


def validate_authorization(manifest_path,manifest,authorization,pins):
    if authorization.get('launch_authorized') is not True or authorization.get('gpu_custody')!='ASTRA':
        raise ValueError('Separate ROOT launch authorization and ASTRA custody are required')
    if authorization.get('manifest_sha256')!=sha(manifest_path):raise ValueError('Launch certificate manifest mismatch')
    if authorization.get('supervisor_sha256')!=sha(__file__):raise ValueError('Launch certificate supervisor mismatch')
    for p in [Path(__file__),H/'operational_case_receipt_v2_astra_nov3.py',REVIEW]:
        if pins.get(str(p.resolve()))!=sha(p):raise ValueError('Required supervisor/review file absent from freeze: '+str(p))
    review=json.loads(REVIEW.read_text())
    signature=lambda x:tuple(x[k] for k in ('name','phase','arm','args','normal_natural_eos_planned','deliberate_abort_planned','health_natural_eos_planned'))
    if [signature(x) for x in manifest['execution_order']]!=[signature(x) for x in review['execution_order']]:
        raise ValueError('Frozen invocations differ from reviewed exact three-case plan')
    if len(manifest['execution_order'])!=3:raise ValueError('Expected exactly three server cases')
    if resolve(manifest['runner']).resolve()!=resolve(review['runner']).resolve() or manifest['python']!=review['python']:raise ValueError('Frozen runner/python differs from reviewed plan')


def compress(path):
    before=sha(path);target=path.with_suffix(path.suffix+'.gz')
    if target.exists():raise FileExistsError('Refuse to overwrite compressed evidence '+str(target))
    with path.open('rb') as src,gzip.open(target,'wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
    with gzip.open(target,'rb') as src:
        if hashlib.file_digest(src,'sha256').hexdigest()!=before:raise RuntimeError('Compressed evidence mismatch')
    size=path.stat().st_size;path.unlink()
    return {'path':str(target.relative_to(ROOT) if target.is_relative_to(ROOT) else target),'uncompressed_sha256':before,'uncompressed_bytes':size,'compressed_bytes':target.stat().st_size}


def descendants():
    rows=[]
    for p in psutil.Process().children(recursive=True):
        try:
            if p.status()!=psutil.STATUS_ZOMBIE:rows.append(p)
        except psutil.Error:pass
    return rows


def cleanup():
    # This standalone supervisor owns its entire child tree; it starts with none.
    children=descendants();inventory=[]
    for p in children:
        try:inventory.append({'pid':p.pid,'created':p.create_time(),'command':p.cmdline()})
        except psutil.Error:pass
    for p in reversed(children):
        try:p.terminate()
        except psutil.Error:pass
    _,alive=psutil.wait_procs(children,timeout=8)
    for p in alive:
        try:p.kill()
        except psutil.Error:pass
    psutil.wait_procs(alive,timeout=5)
    return {'terminated_owned':inventory,'remaining_owned':[p.pid for p in descendants()]}


def idle_inventory():
    lock_ok=False
    with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock_ok=True;fcntl.flock(lock,fcntl.LOCK_UN)
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True,timeout=20).strip()
    if output:raise RuntimeError('GPU has active compute processes; will not touch them: '+output)
    return {'gpu_compute_apps':[],'gpu_lock_available':lock_ok,'at_utc':now()}


def resources():
    memory=psutil.virtual_memory();disk=shutil.disk_usage(H)
    return {'available_host_bytes':memory.available,'total_host_bytes':memory.total,
            'swap':dict(psutil.swap_memory()._asdict()),'disk_free_bytes':disk.free}


class ExportEvents:
    """Observe existing export files; never send a monitoring RPC during work."""
    def __init__(self,run_dir,emit):
        self.run_dir=run_dir;self.emit=emit;self.stop=threading.Event();self.seen={}
        self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.stop.wait(2):
            for path in self.run_dir.glob('scheduler-*.json'):
                try:
                    signature=(path.stat().st_mtime_ns,path.stat().st_size)
                    if self.seen.get(str(path))==signature:continue
                    raw=json.loads(path.read_text());scalar=raw.get('operational_scalar_state')
                    if scalar is None:continue
                    self.seen[str(path)]=signature
                    self.emit({'event':'scalar_export_observed','case':self.run_dir.name,'at_utc':now(),
                               'snapshot':scalar,'scope':'An existing quiescent export; not proof of wave completion or a new RPC.'})
                except (OSError,ValueError):pass # Atomicity of runtime export writes is not assumed.
    def close(self):self.stop.set();self.thread.join()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True);ap.add_argument('--authorization',required=True)
    ap.add_argument('--execute',action='store_true',help='Required in addition to the ROOT certificate; no implicit execution')
    args=ap.parse_args(argv)
    if not args.execute:ap.error('No GPU execution without explicit --execute and ROOT launch certificate')
    os.chdir(ROOT)
    manifest_path=resolve(args.manifest);auth_path=resolve(args.authorization)
    manifest_bytes=manifest_path.read_bytes();manifest=json.loads(manifest_bytes);pins=merged_pins(manifest)
    authorization=json.loads(auth_path.read_text())
    validate_authorization(manifest_path,manifest,authorization,pins)
    manifest_hash=sha(manifest_path);auth_hash=sha(auth_path)
    if any(p.exists() for p in [RECEIPT,PROGRESS,EVENTS,H/'operational-v3-frozen-astra-copy.json']):raise FileExistsError('Prior operational custody evidence exists; no replacement or overwrite')
    if any((H/'runs'/x['name']).exists() for x in manifest['execution_order']):raise FileExistsError('One or more planned run names already exist')
    if descendants():raise RuntimeError('Supervisor started with unexpected child processes')
    ctypes.CDLL(None).prctl(36,1,0,0,0)
    state={'schema':'astra-operational-v3-custody-v1','status':'preflight','accepted_utc':now(),
           'manifest_sha256':manifest_hash,'authorization_sha256':auth_hash,'supervisor_sha256':sha(__file__),
           'supervisor_pid':os.getpid(),'supervisor_create_time':psutil.Process().create_time(),
           'runs':[],'pin_checks':[],'document_provenance_start':document_provenance(),'resources_before':resources()}
    write(RECEIPT,state);write(PROGRESS,[])
    (H/'operational-v3-frozen-astra-copy.json').write_bytes(manifest_bytes)
    event_lock=threading.Lock()
    def emit(event):
        with event_lock:
            with EVENTS.open('a') as f:f.write(json.dumps(event,sort_keys=True)+'\n')
            print(json.dumps(event,sort_keys=True),flush=True)
    def check_pins(stage):
        bad=pin_mismatches(pins)
        if sha(manifest_path)!=manifest_hash:bad.append(str(manifest_path))
        if sha(auth_path)!=auth_hash:bad.append(str(auth_path))
        row={'stage':stage,'at_utc':now(),'checked_files':len(pins),'mismatches':bad}
        state['pin_checks'].append(row);write(RECEIPT,state)
        if bad:raise RuntimeError('Frozen pin drift: '+repr(bad))
    failures=0
    try:
        check_pins('before_first');state['initial_idle']=idle_inventory();state['status']='running';write(RECEIPT,state)
        for index,item in enumerate(manifest['execution_order']):
            check_pins('before_'+item['name']);idle_inventory()
            row={**item,'sequence':index,'started_utc':now(),'start_monotonic_ns':time.monotonic_ns(),'resources_before':resources()}
            command=[manifest['python'],str(resolve(manifest['runner'])),*item['args']]
            row['command']=command;log=H/(item['name']+'-driver-console.log')
            row['console_uncompressed_path']=str(log.relative_to(ROOT));state['current']=row;write(RECEIPT,state)
            emit({'event':'case_start','case':item['name'],'sequence':index,'at_utc':now()})
            watcher=ExportEvents(H/'runs'/item['name'],emit);child=None
            try:
                with log.open('x') as console:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=console,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                    row['runner_pid']=child.pid;write(RECEIPT,state);watcher.thread.start()
                    row['exit_code']=child.wait()
            finally:
                if watcher.thread.ident:watcher.close()
                row['wall_s']=(time.monotonic_ns()-row['start_monotonic_ns'])/1e9
                row['completed_utc']=now()
                if child is not None and child.poll() is None:row['interrupted_cleanup']=cleanup();row['exit_code']=child.poll()
                if log.exists():row['console']=compress(log)
                write(RECEIPT,state)
            if descendants():row['unexpected_descendants_after_runner']=[p.pid for p in descendants()];row['unexpected_cleanup']=cleanup()
            raw_path=H/'runs'/item['name']/'raw.json.gz'
            row['case_success']=False
            if raw_path.exists():
                raw=json.load(gzip.open(raw_path,'rt'));row.update(raw_path=str(raw_path.relative_to(ROOT)),raw_sha256=sha(raw_path),status=raw.get('status'),error=raw.get('error'),owned_descendants_remaining=raw.get('owned_descendants_remaining'))
                try:
                    receipt=verify_case(raw,item);receipt_path=H/(item['name']+'-basic-receipt.json')
                    write(receipt_path,receipt);row['basic_receipt']={'path':str(receipt_path.relative_to(ROOT)),'sha256':sha(receipt_path),'qualified':receipt['qualified'],'issues':receipt['issues']}
                    row['case_success']=row['exit_code']==0 and raw.get('status')=='complete' and receipt['qualified']
                    # Failed/incomplete serving cases remain failures; a complete
                    # server result with contradictory proof is a correctness blocker.
                    row['correctness_receipt_fault']=raw.get('status')=='complete' and not receipt['qualified']
                    for wave in receipt['waves']:
                        emit({'event':'wave_completed_receipt','case':item['name'],'wave':wave['wave_index'],'requests':wave['requests'],'natural_eos':wave['natural_eos'],'qualified':wave['qualified'],'observed_after_case':True})
                except Exception as exc:
                    row['receipt_exception']=repr(exc);row['receipt_traceback']=traceback.format_exc();row['correctness_receipt_fault']=True
                fault_text=(raw.get('error') or '')+' '+(raw.get('error_traceback') or '')
                for path in raw_path.parent.glob('server*.log.gz'):
                    with gzip.open(path,'rt',errors='replace') as f:fault_text+=' '+f.read()
                low=fault_text.lower()
                row['safety_fault']=any(t in low for t in ['actual kv transfer mismatch','invalid one-byte oracle failed','device-side assert triggered','cuda error: an illegal memory access','cuda out of memory','cannot allocate memory','no space left on device'])
            else:row.update(status='missing_raw',error='Runner exited without compressed raw evidence')
            row['resources_after']=resources();state['runs'].append(row);state.pop('current',None);write(PROGRESS,state['runs']);write(RECEIPT,state)
            emit({'event':'case_complete','case':item['name'],'status':row.get('status'),'qualified':row['case_success'],'exit_code':row.get('exit_code'),'wall_s':row['wall_s'],'at_utc':now()})
            failures=0 if row['case_success'] else failures+1
            if row.get('safety_fault') or row.get('correctness_receipt_fault') or row.get('unexpected_descendants_after_runner') or row.get('owned_descendants_remaining') or row.get('exit_code',0)<0:
                state['stop_reason']='resource_correctness_external_signal_or_cleanup_blocker';break
            if failures>=3:state['stop_reason']='three_consecutive_failed_cases';break
        state['status']='complete' if len(state['runs'])==3 and 'stop_reason' not in state else 'stopped_by_manifest_condition'
    except BaseException as exc:
        state.update(status='stopped_by_external_or_pin_blocker',stop_reason=repr(exc),exception_traceback=traceback.format_exc())
        emit({'event':'block_stopped','reason':repr(exc),'at_utc':now()})
    finally:
        state['cleanup']=cleanup()
        state['document_provenance_end']=document_provenance()
        try:check_pins('end')
        except Exception as exc:state['end_pin_error']=repr(exc);state['status']='stopped_by_end_pin_blocker'
        try:state['final_idle']=idle_inventory()
        except Exception as exc:state['final_idle_error']=repr(exc);state['status']='cleanup_or_external_GPU_blocker'
        state['resources_after']=resources();state['completed_utc']=now();state['gpu_custody_return_ready']=not state['cleanup']['remaining_owned'] and 'final_idle' in state
        write(RECEIPT,state);write(PROGRESS,state['runs'])
        emit({'event':'supervisor_finished','status':state['status'],'cases':len(state['runs']),'cleanup_remaining':state['cleanup']['remaining_owned'],'gpu_custody_return_ready':state['gpu_custody_return_ready'],'at_utc':now()})
    return 0 if state['status']=='complete' and all(r['case_success'] for r in state['runs']) else 1

if __name__=='__main__':sys.exit(main())
