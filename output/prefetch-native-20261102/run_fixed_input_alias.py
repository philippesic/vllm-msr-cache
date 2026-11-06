"""Finite quantized public-workload campaign. Every invocation owns one server."""
import argparse,asyncio,ctypes,fcntl,json,os,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1];sys.path[:0]=[str(ROOT),str(H)]
import psutil
from harness.server import launch_server
from runtime_support import Monitor,config,control,parse,sha,write,compress
BLOCK_BYTES=917504

def run(a):
 ctypes.CDLL(None).prctl(36,1,0,0,0)
 selection=json.loads((H/'model-selection.json').read_text());model=str(Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots'/selection['revision'])
 out=H/'runs'/a.name;out.mkdir(parents=True,exist_ok=False)
 namespace={'model_revision':selection['revision'],'tokenizer_revision':selection['revision'],'adapter_namespace':'none','layout_namespace':'qwen2-dense-tp1-fp16-block16'}
 geometry={'gpu_blocks':a.gpu_blocks,'gpu_kv_bytes':a.gpu_blocks*BLOCK_BYTES,'cpu_blocks':a.cpu_blocks,'cpu_kv_bytes':a.cpu_blocks*BLOCK_BYTES,'block_bytes':BLOCK_BYTES,'max_model_len':7680 if not a.qualification else 3328,'max_num_seqs':a.active,'max_num_batched_tokens':1024}
 p={'namespace':namespace,'geometry':geometry,'trace':{'seed':a.seed}}
 os.environ.update(EXPERIMENT_ORIGINAL='1' if a.arm=='original' else '0',EXPERIMENT_OVERLAP='1',EXPERIMENT_PREFIX_SCORE='1',EXPERIMENT_GENERATION_OFFLOAD='1',EXPERIMENT_COMPLETED_TRANSCRIPT='1',EXPERIMENT_PREFETCH_FRACTION=str(a.prefetch_fraction),EXPERIMENT_SELECTOR='semantic')
 cfg=config('arc' if a.arm=='arc' else 'semantic',p,0)
 if a.arm=='candidate':cfg['kv_connector_extra_config']['wait_hint_ttl_ms']=60000
 from fixed_input_workload import validate_hint_contract
 hint_preflight=validate_hint_contract(namespace)
 source=H/'sources'/('full-input-alias' if a.arm=='candidate' else a.source_label)
 runtime_root=H/'runtime-idle-ready'
 if a.source_label.startswith('shared-'):cfg['kv_connector_extra_config']['wait_hint_shared_local_prefix']=True
 if a.source_label=='shared-lookup':cfg['kv_connector_extra_config']['wait_hint_publish_at_lookup']=not a.publish_at_bind
 env={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':os.pathsep.join([str(H/('quality_hooks_readmission' if a.quality else 'hooks-defer')),str(source),str(ROOT)]),'MATRIX_RUN_DIR':str(out),'MATRIX_RUNTIME_ROOT':str(runtime_root),'MATRIX_QUALITY':'0','PYTHONHASHSEED':'4109','PYTHONDONTWRITEBYTECODE':'1','VLLM_BATCH_INVARIANT':'0','VLLM_PLUGINS':'semantic_wait_hint','VLLM_WSL2_ENABLE_PIN_MEMORY':'1','VLLM_USE_V2_MODEL_RUNNER':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','SEMANTIC_CAPTURE_STEPS':'multi_token','SEMANTIC_CAPTURE_BACKEND':'attention','SEMANTIC_SUMMARY_FIELDS':'required' if a.arm!='original' else 'all','REAL_KV_BYTE_AUDIT':'1' if a.audit else '0','REAL_KV_LOCAL_PREFIX_IDS':a.audit_local_prefix_ids}
 env['SEMANTIC_FUSED_MEAN_SUMMARY']='1' if a.arm=='candidate' else '0'
 env['SEMANTIC_REGISTER_FULL_INPUT_PREFIX']='1' if a.arm=='candidate' else '0'
 if a.observe_native_queries:env['MATRIX_OBSERVE_NATIVE_QUERIES']='1'
 if a.boundary_diagnostic:env['MATRIX_BOUNDARY_DIAGNOSTIC']='1'
 if a.quality:env.update(SEMANTIC_CAPTURE_STEPS='all',SEMANTIC_PREFIX_POLICY_QUALITY='1')
 if a.arm=='original':env.update(SEMANTIC_CAPTURE_STEPS='all',SEMANTIC_CAPTURE_BACKEND='dispatch')
 result={'name':a.name,'arm':a.arm,'policy':a.arm,'seed':a.seed,'geometry':geometry,'config':cfg,'namespace':namespace,'args':vars(a),'requests':[],'hints':[],'cancellations':[],'status':'started','performance_eligible':not (a.audit or a.quality or a.boundary_diagnostic or a.feasibility_only or a.observe_native_queries),'lifecycle_start_ns':time.monotonic_ns(),'pins':{str(f.relative_to(H)):sha(f) for f in [H/'run_frontier_defer.py',H/'session_workload.py',H/'runtime_support.py',H/'hooks-defer/sitecustomize.py',H/'readmission_observer.py',H/'lookup_deferral_observer.py',H/'byte_audit.py'] if f.exists()},'source_pins':json.loads((source/'pins.json').read_text())}
 result['local_prefix_byte_audit']={'configured':bool(a.audit and a.audit_local_prefix_ids),'status':'requested' if a.audit and a.audit_local_prefix_ids else 'not_configured_not_run','prior_separate_qualification':'runs/qualify-frozen-hooks-native-bytes/qualification-summary.json'}
 result['speculative_completion_observation']='passive snapshot of actual wait job transfer_record after native completion handling; field present even for zero observed jobs'
 result['hint_contract_preflight']=hint_preflight
 if a.quality:result['quality_hook_pin']=sha(H/'quality_hooks/sitecustomize.py')
 result['runtime_pins']={str(f.relative_to(H)):sha(f) for f in runtime_root.rglob('*.py')}
 result['harness_pins']={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'harness/server.py',ROOT/'harness/policies.py',ROOT/'semantic_offload/config.py',ROOT/'benchmarks/eviction_prefetch_campaign.py',ROOT/'benchmarks/preservation.py']}
 result['artifact_pins']={str(f.relative_to(H)):sha(f) for f in [H/'prepared/session-corpus.json',H/'model-selection.json',H/'corpus.py']}
 result['model_quantization_config']=json.loads((Path(model)/'config.json').read_text())['quantization_config']
 result['corpus_identity']={'sha256':sha(H/'prepared/session-corpus.json'),'schema':'quantized-realworld-sessions-v1','tokenizer_revision':selection['revision']}
 result['model_identity']={'model':selection['model'],'revision':selection['revision'],'selection_sha256':sha(H/'model-selection.json'),'quantization_config':result['model_quantization_config']}
 result['reset_scope']={'prefix_mean_scoring_stats':'explicit clear in worker and scheduler at pre-load reset; absent Counter keys therefore zero','passive_counts_times_samples':'explicit clear','model_kv_caches':'preserved'}
 try:
  import pynvml as nv
  nv.nvmlInit();g=nv.nvmlDeviceGetHandleByIndex(0);m=nv.nvmlDeviceGetMemoryInfo(g)
  result['gpu_before']={'used_bytes':m.used,'free_bytes':m.free,'total_bytes':m.total,'compute_processes':[{'pid':q.pid,'used_gpu_memory':q.usedGpuMemory} for q in nv.nvmlDeviceGetComputeRunningProcesses(g)]};nv.nvmlShutdown()
 except Exception as e:result['gpu_before_error']=repr(e)
 result['host_memory_before']=dict(psutil.virtual_memory()._asdict());handle=None;monitor=Monitor();before={q.pid:q.create_time() for q in psutil.Process().children(recursive=True)}
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   handle=launch_server(model,19372,str(out),gpu_memory_utilization=.85,max_model_len=geometry['max_model_len'],num_gpu_blocks_override=a.gpu_blocks,kv_transfer_config=cfg,extra_args=['--served-model-name','matrix-model','--dtype','half','--quantization','awq','--enforce-eager','--attention-backend','TRITON_ATTN','--tensor-parallel-size','1','--block-size','16','--max-num-seqs',str(a.active),'--max-num-batched-tokens','1024','--seed',str(a.seed)],extra_env=env,ready_timeout_s=a.startup_timeout,log_label=a.name)
   result.update(server_command=handle.command,server_log=handle.log_path,ready_ns=time.monotonic_ns())
   import urllib.request
   body={'model':'matrix-model','prompt':('An ordinary neutral sentence used to warm the serving engine. ' * 100),'max_tokens':32,'temperature':0}
   result['warmup_configuration']=body
   with urllib.request.urlopen(urllib.request.Request(handle.base_url()+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=120) as r:result['warmup']=json.load(r)
   result['before_load_export']=control(handle,'reset')
   with urllib.request.urlopen(handle.metrics_url()) as r:result['metrics_initial']=r.read().decode()
   monitor.thread.start()
   if a.qualification:
    import qualification_workload
    q=json.loads((H/'cases'/a.qualification_case).read_text());q.update(namespace=namespace,geometry=geometry)
    asyncio.run(asyncio.wait_for(qualification_workload.exercise(handle,q,a.arm,result),600))
   else:
    from fixed_input_workload import exercise
    asyncio.run(asyncio.wait_for(exercise(handle,p,a,result),a.timeout))
   result['postload_export']=control(handle,'export')
   with urllib.request.urlopen(handle.metrics_url()) as r:result['metrics_final']=r.read().decode()
   result['status']='complete' if all(r['success'] for r in result['requests']) else 'request_failures'
  except BaseException as e:result.update(status='failed',error=repr(e));print('FAILED',a.name,repr(e),flush=True)
  finally:
   if monitor.thread.ident:monitor.close()
   if handle:handle.shutdown();result['server_exit_code']=handle.proc.poll()
   owned=[]
   for q in psutil.Process().children(recursive=True):
    try:
     if before.get(q.pid)!=q.create_time():owned.append(q)
    except psutil.Error:pass
   for q in owned:
    try:q.terminate()
    except psutil.Error:pass
   _,alive=psutil.wait_procs(owned,timeout=8)
   for q in alive:
    try:q.kill()
    except psutil.Error:pass
   psutil.wait_procs(alive,timeout=5)
   result['resource_samples']=monitor.samples;result['owned_descendants_remaining']=[q.pid for q in alive if q.is_running() and q.status()!=psutil.STATUS_ZOMBIE]
   if handle:
    for label,marker in [('terminals','SEMANTIC_PRESERVATION_EVENT'),('wait_hint_events','SEMANTIC_WAIT_HINT_EVENT'),('speculative_transfer_completions','MATRIX_SPECULATIVE_TRANSFER_COMPLETION'),('readmission_observer_events','MATRIX_READMISSION_OBSERVER'),('native_lookup_deferral_events','MATRIX_NATIVE_LOOKUP_DEFERRED'),('runtime_resolution','SEMANTIC_RUNTIME_RESOLUTION'),('configuration','MATRIX_CONFIGURATION')]:result[label]=parse(handle.log_path,marker)
   result['lifecycle_end_ns']=time.monotonic_ns();write(out/'raw.json',result)
   write(out/'compression.json',[compress(f) for f in [out/'raw.json',*out.glob('server*.log')]])
 print(json.dumps({'name':a.name,'status':result['status'],'error':result.get('error'),'requests':len(result['requests']),'load_s':(result.get('load_end_ns',0)-result.get('load_start_ns',0))/1e9,'owned_descendants_remaining':result['owned_descendants_remaining']}),flush=True)
 return result['status']=='complete'
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--publish-at-bind',action='store_true');ap.add_argument('--prefetch-fraction',type=float,default=.6);ap.add_argument('--name',required=True);ap.add_argument('--arm',choices=['arc','candidate','original'],required=True);ap.add_argument('--source-label',choices=['candidate','shared-prefix','shared-lookup'],default='candidate');ap.add_argument('--seed',type=int,default=1103);ap.add_argument('--gpu-blocks',type=int,required=True);ap.add_argument('--cpu-blocks',type=int,default=4096);ap.add_argument('--active',type=int,default=4);ap.add_argument('--clients',type=int,default=8);ap.add_argument('--cohort',choices=['chat','agent'],default='chat');ap.add_argument('--turns',type=int,default=3);ap.add_argument('--max-output',type=int,default=96);ap.add_argument('--wait',choices=['zero','bursty','longtool'],default='zero');ap.add_argument('--open-rate',type=float,default=0);ap.add_argument('--abandon-every',type=int,default=0);ap.add_argument('--qualification-case',default='native-qualification.json');ap.add_argument('--qualification',action='store_true');ap.add_argument('--audit',action='store_true');ap.add_argument('--audit-local-prefix-ids',default='');ap.add_argument('--quality',action='store_true');ap.add_argument('--boundary-diagnostic',action='store_true');ap.add_argument('--timeout',type=int,default=240);ap.add_argument('--startup-timeout',type=int,default=180);ap.add_argument('--feasibility-only',action='store_true');ap.add_argument('--observe-native-queries',action='store_true')
 sys.exit(0 if run(ap.parse_args()) else 1)
