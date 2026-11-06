import gzip,hashlib,json,os,threading,time
from pathlib import Path
import psutil
from harness.policies import kv_transfer_config
from semantic_offload.config import msr_kv_transfer_config

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
            own=self.proc.cpu_times();row={'at_ns':time.monotonic_ns(),'processes':[],'driver_cpu_s':own.user+own.system,'driver_rss_bytes':self.proc.memory_info().rss}
            for proc in self.proc.children(recursive=True):
                try:
                    c=proc.cpu_times();mem=proc.memory_info();# /proc start ticks are stable when WSL adjusts its boot-time
                    # wall-clock estimate; psutil.create_time() is not.
                    proc_stat=Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split();start_ticks=proc_stat[19]
                    ident=f'{proc.pid}:{start_ticks}'
                    self.known[ident]=proc
                    row['processes'].append({'identity':ident,'cpu_s':c.user+c.system,'rss_bytes':mem.rss,'minor_faults':int(proc_stat[7]),'major_faults':int(proc_stat[9])})
                except (psutil.Error, OSError, ValueError, IndexError):pass
            if nv:
                try:
                    u=nv.nvmlDeviceGetUtilizationRates(gpu);mem=nv.nvmlDeviceGetMemoryInfo(gpu)
                    row['gpu']={'utilization_pct':u.gpu,'memory_utilization_pct':u.memory,'memory_used_bytes':mem.used,'power_w':nv.nvmlDeviceGetPowerUsage(gpu)/1000,'temperature_c':nv.nvmlDeviceGetTemperature(gpu,nv.NVML_TEMPERATURE_GPU),'clock_mhz':nv.nvmlDeviceGetClockInfo(gpu,nv.NVML_CLOCK_SM)}
                except Exception as e:row['gpu_error']=str(e)
            swap=psutil.swap_memory();row['host_memory']={'available_bytes':psutil.virtual_memory().available,'swap_used_bytes':swap.used,'swap_in_bytes':swap.sin,'swap_out_bytes':swap.sout}
            self.samples.append(row);self.stop.wait(.25)
        if nv:nv.nvmlShutdown()
    def close(self):self.stop.set();self.thread.join()

def compress(path):
    out=path.with_suffix(path.suffix+'.gz');h=sha(path)
    with path.open('rb') as src,gzip.open(out,'wb',compresslevel=3) as dst:
        import shutil;shutil.copyfileobj(src,dst)
    with gzip.open(out,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
    before=path.stat().st_size;path.unlink();return {'path':str(out),'uncompressed_sha256':h,'uncompressed_bytes':before,'compressed_bytes':out.stat().st_size}
