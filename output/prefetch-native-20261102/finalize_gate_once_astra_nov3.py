"""Summarize evidence and owned-process cleanup; judgment supplied separately."""
import collections,datetime,fcntl,hashlib,json,shutil,statistics,subprocess
from pathlib import Path
import psutil
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bounds(vals):
 present=[v for v in vals if v is not None]
 return {'observed':len(present),'missing':len(vals)-len(present),'min':min(present),'max':max(present),'mean':statistics.mean(present),'values':vals} if present else {'observed':0,'missing':len(vals),'values':vals}
def main():
 mp=H/'gate-once-frozen-plan-nov3.json'
 m=json.loads(mp.read_text());s=json.loads((H/'live-nov3-gate-once-custody.json').read_text());a=json.loads((H/'live-nov3-gate-once-independent-analysis.json').read_text())
 assert sha(mp)==s['manifest_sha256']
 bad=[p for p,v in m['file_pins'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=v];assert not bad
 if psutil.pid_exists(s['supervisor_pid']):assert psutil.Process(s['supervisor_pid']).create_time()!=s['supervisor_create_time']
 assert not s['owned_descendants_remaining'];assert all(r.get('owned_descendants_remaining',[])==[] for r in s['runs'])
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();assert not gpu
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
 before=shutil.disk_usage(H).free;files=[]
 for item in m['execution_order']:
  d=H/'runs'/item['name']
  if d.exists():files.extend(p for p in d.rglob('*') if p.is_file())
  p=H/(item['name']+'-driver-console.log.gz')
  if p.exists():files.append(p)
 assert not [p for p in files if p.suffix in ['.log','.jsonl']]
 rows=[r for r in a['rows'].values() if r['attempted']];http={'all_qualified':all(r['HTTP_receipt']['qualified'] for r in rows),'total_HTTP_attempts':sum(r['HTTP_receipt'].get('expected_total_HTTP_requests',0) for r in rows),'generations':sum(r['HTTP_receipt'].get('generation_rows',0) for r in rows),'hints':sum(r['HTTP_receipt'].get('hint_rows',0) for r in rows),'intentional_hint_cancellations':sum(len(r['HTTP_receipt'].get('intentional_hint_cancellations',[])) for r in rows),'unexpected_transport_exceptions':sum(len(r['HTTP_receipt'].get('unexpected_transport_exceptions',[])) for r in rows),'connection_reuse':sum(r['HTTP_receipt'].get('counts',{}).get('connection_reuse',0) for r in rows),'scope':'Missing verifier fields do not establish absence of errors; all_qualified is mandatory.'}
 resources={};mechanism={}
 for arm in ['native','periodic','once_positive']:
  rr=[r for r in rows if r['arm']==arm]
  resources[arm]={'runs':len(rr),'configured_cpu_kv_mib':sorted(set(r['host_resources']['configured_cpu_kv_bytes']/2**20 for r in rr)),'actual_locked_pinned_host_bytes':None,'CUDA_load_peaks_unique':list({json.dumps(r['resources']['cuda_load_peaks'],sort_keys=True):r['resources']['cuda_load_peaks'] for r in rr}.values())}
  for key in ['serving_nvml_peak_mib','startup_nvml_peak_mib','sampled_server_child_cpu_s','sampled_server_summed_rss_peak_mib','sampled_driver_cpu_s','sampled_driver_rss_peak_mib']:
   resources[arm][key]=bounds([r['resources'].get(key) for r in rr])
  resources[arm]['host_system']={stage:{key:bounds([(r['host_resources'][stage].get(key) or {}).get('delta' if key.startswith('swap_') else 'min') for r in rr]) for key in ['swap_in_bytes','swap_out_bytes','swap_used_bytes','available_bytes']} for stage in ['startup','serving']}
 for part in [7703,8803,9901,11003,12007,13001]:
  mechanism[str(part)]={}
  for arm in ['native','periodic','once_positive']:
   rr=[r for r in rows if r['partition']==part and r['arm']==arm]
   mechanism[str(part)][arm]={'local_compute_first_admission_tokens':[r['first_prefill_attribution']['tokens']['local_compute'] for r in rr],'local_GPU_cache_first_admission_tokens':[r['first_prefill_attribution']['tokens']['local_cache_hit'] for r in rr],'external_first_admission_tokens':[r['first_prefill_attribution']['tokens']['external_kv_transfer'] for r in rr],'CPU_evicted_blocks':[r['scheduler_counts'].get('cpu_evicted_blocks') for r in rr],'H2D_bytes':[r['transfers']['CPU_to_GPU'] for r in rr],'D2H_bytes':[r['transfers']['GPU_to_CPU'] for r in rr],'preemptions':[r['first_prefill_attribution']['preemptions'] for r in rr],'gate_per_hint':[r.get('gate_once_receipt',{}).get('per_hint') for r in rr] if arm!='native' else None}
 summary={'schema':'astra-gated-once-positive-discovery-summary-v1','manifest_sha256':sha(mp),'supervisor_status':s['status'],'completed_invocations':len(s['runs']),'all_18_eligible':len(rows)==18 and all(r['eligible'] for r in rows),'natural_eos':sum(r['natural_eos'] for r in rows),'http':http,'all_executed_lean_configuration_matches':all(r['measurement']['matches_frozen_lean'] for r in rows),'all_optional_observer_counts_explicit_zero':all(all(r['measurement'][k]==0 for k in ['residency_samples','readmission_events','lookup_observer_events']) for r in rows),'all_first_admission_attribution_reconciles':all(r['first_prefill_attribution']['reconciles'] is True for r in rows),'all_zero_preemptions':all(r['first_prefill_attribution']['preemptions']==0 for r in rows),'all_store_accept_complete_balanced':all(r['store_balance']['balanced'] is True for r in rows),'missing_store_failed_counter':'Unavailable; not assumed zero.','descriptive_comparisons_no_CI':a['comparisons'],'output_totals':a['arm_outcomes'],'resources':resources,'mechanism_by_partition':mechanism,'production_endorsement':False,'judgment':'See separate independent discovery assessment; no confirmation or endurance qualification.','workload_provenance_caveat':'Raw pins list periodic workload in once mode; selected workload.py hash, frozen manifest, rawargs and executed stop policy independently verified in gate_once_receipt.'}
 (H/'live-nov3-gate-once-independent-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 inv={'schema':'astra-gated-once-positive-discovery-handback-v1','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'custody':'ASTRA -> ROOT','manifest_pins_checked':len(m['file_pins']),'pin_mismatches':bad,'manifest_matches_frozen_copy':True,'supervisor_exited':True,'all_runner_descendants_exited':True,'owned_processes_remaining':[],'gpu_compute_processes':gpu,'gpu_lock_available':True,'free_bytes_before_cleanup':before,'free_bytes_after_cleanup':shutil.disk_usage(H).free,'removed_files':[],'retained_bytes':sum(p.stat().st_size for p in files),'retained_files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size} for p in files],'retention_reason':'Full compressed raw/server/console evidence and per-run pin/source archives retained for independent verification. No disposable worktrees/downloads/checkpoints or owned processes remain. Shared caches and earlier results untouched. WSL filesystem free space is not Windows VHDX reclamation; concurrent activity can affect the delta.'}
 (H/'live-nov3-gate-once-handback-inventory.json').write_text(json.dumps(inv,indent=2)+'\n')
 print(json.dumps({'HTTP':http,'pins':len(m['file_pins']),'retained_bytes':inv['retained_bytes'],'all_18_eligible':summary['all_18_eligible'],'cleanup_complete':True},indent=2))
if __name__=='__main__':main()
