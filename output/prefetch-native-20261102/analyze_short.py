"""Offline only: the serving lock prevents parsing contention with a trial."""
import argparse,fcntl,json,statistics
from pathlib import Path
from work_inventory import compare
H=Path(__file__).resolve().parent
def point(name):
 d=json.loads((H/'runs'/name/'summary.json').read_text());assert d['valid'] and d['proofs']['all_passed']
 assert d['planned_requests']==d['all']['requests']==d['proofs']['passed']==24
 r=d['resources'];u=d['offline_staging_utility'];assert not u['unknown_completed_attempts'] and not u['unknown_consumption_attempts']
 used=u['global_named_demand_consumed_bytes'];copied=u['global_explicit_successful_completion_bytes_observed']
 return dict(name=name,all_mean_ttft_ms=d['all']['release_ttft_ms']['mean'],all_p95_ttft_ms=d['all']['release_ttft_ms']['p95'],return_mean_ttft_ms=statistics.mean(v['release_ttft_ms'] for v in d['requests'] if v['identity'][1]>0),output_tokens=d['all']['output_tokens'],throughput_tokens_s=d['all']['output_tokens_s'],cuda_reserved_peak_bytes=r['load_cuda_reserved_peak_bytes_sum_workers'],cuda_allocated_peak_bytes=r['load_cuda_allocated_peak_bytes_sum_workers'],nvml_devicewide_peak_bytes=r['nvml_device_memory_peak_bytes_sampled'],cpu_seconds=r['cpu_s_sampled'],used_bytes=used,copied_bytes=copied,useful_fraction=used/copied if copied else None,native_adopted_jobs=sum(v['native_adopted'] is True for v in u['event_rows']),strict_proofs=d['proofs']['passed'])
def pair(candidate,arc):
 c,a=point(candidate),point(arc);work=compare(candidate,arc)
 return dict(candidate=c,arc=a,return_saving_ms=a['return_mean_ttft_ms']-c['return_mean_ttft_ms'],all_saving_ms=a['all_mean_ttft_ms']-c['all_mean_ttft_ms'],reserved_saving_bytes=a['cuda_reserved_peak_bytes']-c['cuda_reserved_peak_bytes'],work=work)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('arc');ap.add_argument('--output',required=True);args=ap.parse_args()
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);result=pair(args.candidate,args.arc);(H/args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='work'},indent=2));print(json.dumps({k:v for k,v in result['work'].items() if k!='rows'}))
if __name__=='__main__':main()
