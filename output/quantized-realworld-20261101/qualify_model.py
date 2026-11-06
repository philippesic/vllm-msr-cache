"""Qualify an actual quantized model server and record its runtime; no benchmark claim."""
import ctypes,fcntl,hashlib,json,os,sys,time,urllib.request
from pathlib import Path
import psutil
H=Path(__file__).resolve().parent;ROOT=H.parents[1];sys.path.insert(0,str(ROOT))
from harness.server import launch_server
from harness.policies import kv_transfer_config

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ctypes.CDLL(None).prctl(36,1,0,0,0)
 selection=json.loads((H/'model-selection.json').read_text());model=str(Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots'/selection['revision'])
 out=H/'qualification'/os.environ.get('QUAL_LABEL','model-server');out.mkdir(exist_ok=False)
 namespace={'model_revision':selection['revision'],'tokenizer_revision':selection['revision'],'adapter_namespace':'none','layout_namespace':'qwen2-dense-tp1-fp16-block16'}
 config=kv_transfer_config('arc',4096*917504,{'preservation_events':True,'admission_namespace':namespace,'admission_mode':'disabled','store_threshold':0,'shared_prefetch_scorer':False,'wait_hint_diagnostic_no_dispatch':True})
 before={p.pid:p.create_time() for p in psutil.Process().children(recursive=True)};handle=None;result={'status':'started','selection':selection,'script_sha256':sha(__file__),'started_ns':time.monotonic_ns()}
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   handle=launch_server(model,19372,str(out),gpu_memory_utilization=.85,max_model_len=8192,num_gpu_blocks_override=1024,kv_transfer_config=config,extra_args=['--served-model-name','quant-model','--dtype','half','--quantization','awq','--enforce-eager','--attention-backend','TRITON_ATTN','--tensor-parallel-size','1','--block-size','16','--max-num-seqs','8','--max-num-batched-tokens','1024','--seed','1103'],extra_env={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':os.pathsep.join([str(H/'hooks'),str(H/'sources/candidate'),str(ROOT)]),'MATRIX_RUN_DIR':str(out),'MATRIX_QUALITY':'0','PYTHONHASHSEED':'4109','PYTHONDONTWRITEBYTECODE':'1','VLLM_BATCH_INVARIANT':'0','VLLM_PLUGINS':'semantic_wait_hint','VLLM_WSL2_ENABLE_PIN_MEMORY':'1','VLLM_USE_V2_MODEL_RUNNER':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','SEMANTIC_CAPTURE_BACKEND':'attention'},ready_timeout_s=360,log_label='quant-model')
   result.update(command=handle.command,log=handle.log_path)
   body={'model':'quant-model','prompt':'Explain why a database index can speed up a query in two sentences.','max_tokens':48,'temperature':0,'seed':1103}
   with urllib.request.urlopen(urllib.request.Request(handle.base_url()+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=180) as resp:result['response']=json.load(resp)
   result['status']='passed'
  except BaseException as e:result.update(status='failed',error=repr(e));raise
  finally:
   if handle:handle.shutdown();result['server_exit_code']=handle.proc.poll()
   owned=[]
   for p in psutil.Process().children(recursive=True):
    try:
     if before.get(p.pid)!=p.create_time():owned.append(p)
    except psutil.Error:pass
   for p in owned:
    try:p.terminate()
    except psutil.Error:pass
   _,alive=psutil.wait_procs(owned,timeout=8)
   for p in alive:
    try:p.kill()
    except psutil.Error:pass
   psutil.wait_procs(alive,timeout=5)
   result.update(finished_ns=time.monotonic_ns(),owned_processes_remaining=[p.pid for p in alive if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE])
   (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
   print(json.dumps({k:result[k] for k in ['status','error','owned_processes_remaining'] if k in result}))
if __name__=='__main__':main()
