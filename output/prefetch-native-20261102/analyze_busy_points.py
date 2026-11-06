"""Small summary reads; full raw/work comparisons only while serving is idle."""
import argparse,fcntl,json,statistics
from pathlib import Path
from work_inventory import compare
H=Path(__file__).resolve().parent
def point(name):
 d=json.loads((H/'runs'/name/'summary.json').read_text())
 assert d['status']=='complete' and d['valid'] and d['proofs']['all_passed']
 assert d['planned_requests']==d['all']['requests']==d['proofs']['passed']==72
 r=d['resources'];u=d['offline_staging_utility'];c=d['counters']
 assert u['explicit_completion_collection_present'] and not u['unknown_completed_attempts'] and not u['unknown_consumption_attempts']
 used=u['global_named_demand_consumed_bytes'];copied=u['global_explicit_successful_completion_bytes_observed']
 return dict(name=name,throughput_tokens_s=d['all']['output_tokens_s'],
  all_mean_ttft_ms=d['all']['release_ttft_ms']['mean'],all_p95_ttft_ms=d['all']['release_ttft_ms']['p95'],
  return_mean_ttft_ms=statistics.mean(v['release_ttft_ms'] for v in d['requests'] if v['identity'][1]>0),
  output_tokens=d['all']['output_tokens'],cuda_reserved_peak_bytes=r['load_cuda_reserved_peak_bytes_sum_workers'],
  cuda_allocated_peak_bytes=r['load_cuda_allocated_peak_bytes_sum_workers'],nvml_devicewide_peak_bytes=r['nvml_device_memory_peak_bytes_sampled'],
  cpu_seconds=r['cpu_s_sampled'],copied_bytes=copied,used_bytes=used,useful_fraction=used/copied if copied else None,
  demand_h2d_bytes=c['demand_h2d_bytes'],d2h_bytes=c['d2h_bytes'],preemptions=c['preemptions'],
  cpu_evicted_blocks=c['manager_counts_delta'].get('cpu_evicted_blocks'),
  native_adopted_jobs=sum(v['native_adopted'] is True for v in u['event_rows']),strict_proofs=d['proofs']['passed'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',action='append',required=True);ap.add_argument('--arc',action='append',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cs=[point(n) for n in a.candidate];arcs=[point(n) for n in a.arc]
  work=[compare(c['name'],arc['name']) for c in cs for arc in arcs]
  assert all(w['identical_input_token_ids'] and w['identical_output_token_ids'] for w in work)
  result={'candidate_points':cs,'arc_points':arcs,'work':[dict(w,rows=None) for w in work],
   'scope':'Exploratory exact recorded-token serving replay. No statistical proof from a pilot. Rates include all private waits.',
   'pilots_dominated_on_throughput_and_reserved':{c['name']:any(arc['cuda_reserved_peak_bytes']<=c['cuda_reserved_peak_bytes'] and arc['throughput_tokens_s']>=c['throughput_tokens_s'] for arc in arcs) for c in cs}}
  (H/a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
