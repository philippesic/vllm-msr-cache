"""Finite single/multi-agent full-MSR matrix on the current host."""
import argparse, asyncio, ctypes, fcntl, gzip, hashlib, json, os, signal, subprocess, sys, threading, time
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sys.path[:0]=[str(ROOT),str(H)]
import psutil
from harness.server import launch_server
from harness.policies import kv_transfer_config
from semantic_offload.config import msr_kv_transfer_config
from benchmarks.agent_load_trace import build_trial
from prefetch_workload import exercise
SEEDS=[1103,2207,3301]
MODEL='Qwen/Qwen2.5-1.5B-Instruct'
BLOCK_BYTES=28*2*2*128*2*16
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def config(policy,p,threshold=0):
    candidate=policy!='arc'
    if os.getenv('EXPERIMENT_ORIGINAL','0')=='1':
        assert candidate
        original={'preservation_events':True,'admission_namespace':p['namespace'],'admission_mode':'disabled','store_threshold':threshold,'prefetch_trigger':'known_prefix_wait_hint','soft_retention':'off','staging_mode':'owned_reclaimable','wait_hint_native_adoption':False,'prefetch_preserve_gpu_prefix_cache':True,'prefetch_score_max_age_ms':1000,'wait_hint_ttl_ms':2000,'wait_hint_coalescing_min_sources':2,'wait_hint_coalescing_max_ms':50,'method':'mean','probe_layer':'middle','capture_stride':1,'scoring_mode':'synchronous','scoring_schedule':'continuous','scoring_search':'flat','head_aggregation':'mean','rank_ema_mode':'default','decision_feature_trace':False,'bounded_prefill_cuda_timing':False,'prefetch_budget_fraction':.05}
        return msr_kv_transfer_config(p['geometry']['cpu_kv_bytes'],**original)
    common={'preservation_events':True,'admission_namespace':p['namespace'],'admission_mode':'disabled','store_threshold':threshold,'prefetch_trigger':'known_prefix_wait_hint','soft_retention':'off','staging_mode':'owned_reclaimable','wait_hint_native_adoption':candidate and os.getenv('EXPERIMENT_ADOPTION','1')=='1','prefetch_preserve_gpu_prefix_cache':os.getenv('EXPERIMENT_PRESERVE','0')=='1','prefetch_score_max_age_ms':10000,'wait_hint_ttl_ms':5000,'wait_hint_coalescing_min_sources':int(os.getenv('EXPERIMENT_COALESCING_MIN','1')),'wait_hint_coalescing_max_ms':int(os.getenv('EXPERIMENT_COALESCING_MS','0')),'method':'mean','probe_layer':'middle','capture_stride':int(os.getenv('EXPERIMENT_STRIDE','1')),'scoring_mode':os.getenv('EXPERIMENT_SCORING','synchronous'),'scoring_schedule':os.getenv('EXPERIMENT_SCHEDULE','opportunity') if candidate else 'continuous','scoring_search':'flat','head_aggregation':'mean','rank_ema_mode':'default','shared_prefetch_scorer':candidate,'query_snapshot_max_age_ms':10000,'async_scoring_max_snapshot_age_ms':10000,'wait_hint_coalesce_exact_source':candidate and os.getenv('EXPERIMENT_COALESCE','1')=='1','prefetch_require_full_prefix':candidate and os.getenv('EXPERIMENT_FULL_ONLY','1')=='1','summary_offload_overlap':os.getenv('EXPERIMENT_OVERLAP','0')=='1','decision_feature_trace':False,'bounded_prefill_cuda_timing':False,'prefetch_selector':os.getenv('EXPERIMENT_SELECTOR','request_order'),'prefetch_budget_fraction':float(os.getenv('EXPERIMENT_PREFETCH_FRACTION','.6')) if candidate else 0.,'prefetch_headroom_blocks':1,'wait_hint_diagnostic_no_dispatch':not candidate}
    common['prefix_mean_prefetch_scoring']=candidate and os.getenv('EXPERIMENT_PREFIX_SCORE','0')=='1'
    common['offload_prompt_only']=os.getenv('EXPERIMENT_GENERATION_OFFLOAD','0')!='1'
    common['wait_hint_completed_transcript']=candidate and os.getenv('EXPERIMENT_COMPLETED_TRANSCRIPT','0')=='1'
    return kv_transfer_config('arc',p['geometry']['cpu_kv_bytes'],common)
def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    rev=(Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/refs/main').read_text().strip()
    pins={str(x.relative_to(ROOT)):sha(x) for directory in [ROOT/'semantic_offload',ROOT/'harness',ROOT/'benchmarks'] for x in directory.glob('*.py')}
    for seed in SEEDS:
        t=build_trial(seed)
        sessions=[]
        for s in t['sessions']:
            tokens=tok.encode(s['context_text'],add_special_tokens=False)[:s['target_initial_context_tokens']]
            assert len(tokens)==s['target_initial_context_tokens']
            sessions.append({'initial_tokens':tokens,'turn_append_tokens':[tok.encode('\n'+x['tool_result_for_this_turn']+'\n'+x['user_instruction']+'\nFinding:',add_special_tokens=False) for x in s['turns']]})
        backgrounds=[]
        for b in t['background']:
            ids=tok.encode((b['prompt']+'\n')*100,add_special_tokens=False)[:b['target_prompt_tokens']]
            assert len(ids)==b['target_prompt_tokens'];backgrounds.append({'tokens':ids})
        for workload in ['single','multi']:
            trace=json.loads(json.dumps(t));selected=sessions
            if workload=='single':trace['sessions']=trace['sessions'][:1];selected=sessions[:1]
            p={'model':MODEL,'namespace':{'model_revision':rev,'tokenizer_revision':rev,'adapter_namespace':'none','layout_namespace':'qwen2-dense-tp1-fp16-block16'},'trace':trace,'sessions':selected,'background':backgrounds,'geometry':{'gpu_blocks':640,'gpu_kv_bytes':640*BLOCK_BYTES,'cpu_blocks':199,'cpu_kv_bytes':199*BLOCK_BYTES,'block_bytes':BLOCK_BYTES,'max_model_len':8192,'max_num_seqs':3,'max_num_batched_tokens':1024}}
            write(H/'prepared'/f'{workload}-{seed}.json',p)
    frozen={'source_pins':pins,'task_pins':{str(p.relative_to(H)):sha(p) for p in H.rglob('*.py') if 'preflight' not in p.parts},'model':MODEL,'revision':rev,'seeds':SEEDS,'main_order':[{ 'workload':w,'seed':s,'policies':['arc','semantic'] if i%2==0 else ['semantic','arc']} for i,s in enumerate(SEEDS) for w in ['single','multi']],'primary_threshold':2,'component_diagnostics':'one matched pair per workload at seed1103, excluded from headline throughput','conditional_admission_stress':'one matched pair per workload at seed1103 and threshold0 if primary has no CPU evictions or no prefetch consumption; all other geometry unchanged','synthetic_not_production':True,'no_statistical_equivalence_claim':True}
    write(H/'frozen.json',frozen);print('PREPARED',json.dumps({'model':MODEL,'seeds':SEEDS,'geometry':p['geometry']}),flush=True)
def parse(path,marker):
    result=[]
    for segment in Path(path).read_text(errors='replace').split(marker+' ')[1:]:
        try:result.append(json.JSONDecoder().raw_decode(segment)[0])
        except json.JSONDecodeError:pass
    return result
def control(handle,operation):
    import urllib.request
    body=json.dumps({'operation':operation}).encode()
    with urllib.request.urlopen(urllib.request.Request(handle.base_url()+'/_matrix/control',data=body,headers={'Content-Type':'application/json'}),timeout=90) as response:return json.load(response)
class Monitor:
    def __init__(self):self.stop=threading.Event();self.samples=[];self.proc=psutil.Process();self.known={};self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        try:
            import pynvml as nv
            nv.nvmlInit();gpu=nv.nvmlDeviceGetHandleByIndex(0)
        except Exception:nv=None
        while not self.stop.is_set():
            row={'at_ns':time.monotonic_ns(),'processes':[]}
            for proc in self.proc.children(recursive=True):
                try:
                    c=proc.cpu_times();mem=proc.memory_info();# /proc start ticks are stable when WSL adjusts its boot-time
                    # wall-clock estimate; psutil.create_time() is not.
                    start_ticks=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                    ident=f'{proc.pid}:{start_ticks}'
                    self.known[ident]=proc
                    row['processes'].append({'identity':ident,'cpu_s':c.user+c.system,'rss_bytes':mem.rss})
                except (psutil.Error, OSError, ValueError, IndexError):pass
            if nv:
                try:
                    u=nv.nvmlDeviceGetUtilizationRates(gpu);mem=nv.nvmlDeviceGetMemoryInfo(gpu)
                    row['gpu']={'utilization_pct':u.gpu,'memory_utilization_pct':u.memory,'memory_used_bytes':mem.used,'power_w':nv.nvmlDeviceGetPowerUsage(gpu)/1000,'temperature_c':nv.nvmlDeviceGetTemperature(gpu,nv.NVML_TEMPERATURE_GPU),'clock_mhz':nv.nvmlDeviceGetClockInfo(gpu,nv.NVML_CLOCK_SM)}
                except Exception as e:row['gpu_error']=str(e)
            self.samples.append(row);self.stop.wait(.25)
        if nv:nv.nvmlShutdown()
    def close(self):self.stop.set();self.thread.join()
def compress(path):
    out=path.with_suffix(path.suffix+'.gz');h=sha(path)
    with path.open('rb') as src,gzip.open(out,'wb',compresslevel=3) as dst:
        import shutil;shutil.copyfileobj(src,dst)
    with gzip.open(out,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
    before=path.stat().st_size;path.unlink();return {'path':str(out),'uncompressed_sha256':h,'uncompressed_bytes':before,'compressed_bytes':out.stat().st_size}
def run(workload,seed,policy,phase='main',threshold=2):
    ctypes.CDLL(None).prctl(36,1,0,0,0) # Own orphaned descendants as Linux child subreaper.
    frozen=json.loads((H/'frozen.json').read_text())
    # Source identity recorded per arm below.
    p=json.loads((H/'prepared'/f'{workload}-{seed}.json').read_text());g=p['geometry'];g['gpu_blocks']=int(os.getenv('EXPERIMENT_GPU_BLOCKS','640'));g['gpu_kv_bytes']=g['gpu_blocks']*BLOCK_BYTES
    name=f'{phase}-{workload}-{seed}-{policy}';out=H/'runs'/name
    assert not out.exists(),name
    out.mkdir(parents=True)
    result={'name':name,'workload':workload,'seed':seed,'policy':policy,'phase':phase,'threshold':threshold,'geometry':g,'requests':[],'hints':[],'cancellations':[],'config':config(policy,p,threshold),'status':'started','lifecycle_start_ns':time.monotonic_ns()}
    source=Path(os.getenv('EXPERIMENT_SOURCE',str(ROOT)))/'semantic_offload'
    result['experiment']={'source':str(source),'source_pins':{p.name:sha(p) for p in source.glob('*.py')},'capture_backend':os.getenv('SEMANTIC_CAPTURE_BACKEND','dispatch'),'capture_steps':os.getenv('SEMANTIC_CAPTURE_STEPS','all'),'stride':os.getenv('EXPERIMENT_STRIDE','1'),'scoring_mode':os.getenv('EXPERIMENT_SCORING','synchronous')}
    result['experiment'].update(original_configuration=os.getenv('EXPERIMENT_ORIGINAL','0')=='1',summary_fields=os.getenv('SEMANTIC_SUMMARY_FIELDS','all'),harness_pins={str(f.relative_to(H)):sha(f) for f in [H/'run.py',H/('competing_workload.py' if workload.startswith('competing') else 'prefetch_workload.py'),H/'hooks/sitecustomize.py']},runtime_pins={str(f.relative_to(H)):sha(f) for f in (H/'runtime').rglob('*.py')},model_revision=frozen['revision'])
    handle=None;monitor=Monitor();monitor.thread.start();before={q.pid:q.create_time() for q in psutil.Process().children(recursive=True)}
    with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            handle=launch_server(MODEL,19371,str(out),gpu_memory_utilization=.75,max_model_len=g['max_model_len'],num_gpu_blocks_override=g['gpu_blocks'],kv_transfer_config=result['config'],extra_args=['--served-model-name','matrix-model','--dtype','half','--enforce-eager','--attention-backend','TRITON_ATTN','--tensor-parallel-size','1','--block-size','16','--max-num-seqs',str(g['max_num_seqs']),'--max-num-batched-tokens','1024','--seed',str(seed)],extra_env={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':os.pathsep.join([str(H/'hooks'),os.environ.get('EXPERIMENT_SOURCE',str(ROOT)),str(ROOT)]),'MATRIX_RUN_DIR':str(out),'MATRIX_QUALITY':os.getenv('MATRIX_QUALITY','0'),'MATRIX_DIAGNOSTIC':'1' if phase=='diagnostic' else '0','SEMANTIC_PROFILE_DIR':str(out/'components') if phase=='diagnostic' else '','SEMANTIC_PROFILE_CUDA_STRIDE':'16','PYTHONHASHSEED':'4109','PYTHONDONTWRITEBYTECODE':'1','VLLM_BATCH_INVARIANT':'0','VLLM_PLUGINS':'semantic_wait_hint','VLLM_WSL2_ENABLE_PIN_MEMORY':'1','VLLM_USE_V2_MODEL_RUNNER':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','SEMANTIC_CAPTURE_STEPS':os.getenv('SEMANTIC_CAPTURE_STEPS','all'),'SEMANTIC_CAPTURE_BACKEND':os.getenv('SEMANTIC_CAPTURE_BACKEND','dispatch'),'SEMANTIC_SUMMARY_FIELDS':os.getenv('SEMANTIC_SUMMARY_FIELDS','all'),'SEMANTIC_OFFLOAD_DEBUG':'0','SEMANTIC_OFFLOAD_TIMING':'0','SEMANTIC_OFFLOAD_DISABLE_PREFETCH':'0'},ready_timeout_s=180,log_label=name)
            result.update(server_command=handle.command,server_log=handle.log_path,ready_ns=time.monotonic_ns())
            import urllib.request
            body={'model':'matrix-model','prompt':[(p['rounds'][0]['pressure_a'][0]+1)%151000,*p['rounds'][0]['pressure_a'][1:]],'max_tokens':16,'temperature':0,'ignore_eos':True}
            with urllib.request.urlopen(urllib.request.Request(handle.base_url()+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=90) as response:result['warmup']=json.load(response)
            result['before_load_export']=control(handle,'reset')
            with urllib.request.urlopen(handle.metrics_url()) as response:result['metrics_initial']=response.read().decode()
            exercise_fn=exercise
            if workload.startswith('competing'):
                from competing_workload import exercise as exercise_fn
            asyncio.run(asyncio.wait_for(exercise_fn(handle,p,policy,result),timeout=600))
            result['postload_export']=control(handle,'export')
            result['status']='complete'
            if not all(r['success'] for r in result['requests']):result['status']='request_failures'
        except BaseException as e:
            result.update(status='failed',error=repr(e));print('RUN FAILED',name,repr(e),flush=True)
        finally:
            result['cleanup_start_ns']=time.monotonic_ns()
            if handle:handle.shutdown();result['server_exit_code']=handle.proc.poll()
            # Only descendants born during this run; preserve all pre-existing work.
            owned=[]
            for proc in psutil.Process().children(recursive=True):
                try:
                    if before.get(proc.pid)!=proc.create_time():owned.append(proc)
                except psutil.Error:pass
            for proc in owned:
                try:proc.terminate()
                except psutil.Error:pass
            _,alive=psutil.wait_procs(owned,timeout=8)
            for proc in alive:
                try:proc.kill()
                except psutil.Error:pass
            psutil.wait_procs(alive,timeout=5)
            monitor.close();result['resource_samples']=monitor.samples
            result['owned_descendants_remaining']=[q.pid for q in psutil.Process().children(recursive=True) if before.get(q.pid)!=q.create_time() and q.status()!=psutil.STATUS_ZOMBIE]
            if handle:
                result['terminals']=parse(handle.log_path,'SEMANTIC_PRESERVATION_EVENT')
                result['wait_hint_events']=parse(handle.log_path,'SEMANTIC_WAIT_HINT_EVENT')
                result['runtime_resolution']=parse(handle.log_path,'SEMANTIC_RUNTIME_RESOLUTION')
                result['configuration']=parse(handle.log_path,'MATRIX_CONFIGURATION')
            result['lifecycle_end_ns']=time.monotonic_ns();write(out/'raw.json',result)
            receipts=[compress(x) for x in [out/'raw.json',*out.glob('server*.log')]]
            write(out/'compression.json',receipts)
    print(json.dumps({'run':name,'status':result['status'],'requests':len(result['requests']),'duration_s':(result['lifecycle_end_ns']-result['lifecycle_start_ns'])/1e9}),flush=True)
    return result['status']=='complete'
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);ap.add_argument('--workload',choices=['qualify','causal','competing','competing_long','competing_long_dram']);ap.add_argument('--seed',type=int,default=1103);ap.add_argument('--policy',choices=['arc','semantic']);ap.add_argument('--phase',default='main');ap.add_argument('--threshold',type=int,default=0);a=ap.parse_args()
    if a.mode=='prepare':subprocess.run([sys.executable,'-B',str(H/'prepare.py')],check=True)
    else:sys.exit(0 if run(a.workload,a.seed,a.policy,a.phase,a.threshold) else 1)
