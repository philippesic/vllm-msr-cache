"""Predeclared twelve-pair inference, never run while serving is active."""
import fcntl,itertools,json,math,statistics
from pathlib import Path
from analyze_busy_points import point
from work_inventory import compare
H=Path(__file__).resolve().parent
T95_DF11=2.200985160082949

def ci12(values):
 assert len(values)==12,'Twelve complete fresh workflow pairs are required'
 mean=statistics.mean(values);half=T95_DF11*statistics.stdev(values)/math.sqrt(12)
 return {'n_independent_workflow_pairs':12,'mean':mean,'lower95':mean-half,'upper95':mean+half,'method':'paired Student t; df11; whole-workflow pairs, not requests; inference conditional on three fixed public traces'}

def inference(log_ratios):
 ci=ci12(log_ratios);observed=statistics.mean(log_ratios)
 null=[statistics.mean(s*x for s,x in zip(signs,log_ratios)) for signs in itertools.product((-1,1),repeat=12)]
 p=sum(x>=observed-1e-14 for x in null)/len(null)
 return {'geometric_mean_gain_fraction':math.expm1(ci['mean']),'lower95_gain_fraction':math.expm1(ci['lower95']),'upper95_gain_fraction':math.expm1(ci['upper95']),'one_sided_exact_sign_flip_p':p,'paired_log_ci':ci,'passed_statistics':ci['lower95']>0 and p<=.05}

def normalized_log_ratio(candidate,arc,memory_key):
 assert candidate['throughput_tokens_s']>0 and arc['throughput_tokens_s']>0
 assert candidate[memory_key]>0 and arc[memory_key]>0
 return math.log(candidate['throughput_tokens_s']/arc['throughput_tokens_s'])+math.log(arc[memory_key]/candidate[memory_key])

def returning_p95(name):
 from campaign_metrics_warm import dist
 summary=json.loads((H/'runs'/name/'summary.json').read_text())
 return dist(v['release_ttft_ms'] for v in summary['requests'] if v['identity'][1]>0)['p95']

def main():
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=json.loads((H/'CPU-RECENCY-V3-CONFIRM.json').read_text());p=json.loads((H/plan['progress_file']).read_text())
  assert p['status']=='complete' and len(p['completed'])==24
  rows=[];work=[]
  for pair in plan['pairs']:
   c,a=point(pair['candidate']),point(pair['arc'])
   for name in (c['name'],a['name']):
    d=json.loads((H/'runs'/name/'frontier-disposition.json').read_text());i=d['fixed_work_integrity']
    assert d['planned_requests']==d['submitted_requests']==d['completed_requests']==d['completed_strict_proofs']==72
    assert i['passed'] and i['exact_recorded_output_ids'] and i['invalid_control_rejected']
   w=compare(c['name'],a['name']);assert w['identical_input_token_ids'] and w['identical_output_token_ids'];work.append(w)
   memory=all(c[k]<=a[k] for k in ('cuda_reserved_peak_bytes','cuda_allocated_peak_bytes','nvml_devicewide_peak_bytes'))
   rows.append(dict(pair,candidate=c,arc=a,log_throughput_ratio=math.log(c['throughput_tokens_s']/a['throughput_tokens_s']),memory_no_greater=memory,all_ttft_saving_ms=a['all_mean_ttft_ms']-c['all_mean_ttft_ms'],return_ttft_saving_ms=a['return_mean_ttft_ms']-c['return_mean_ttft_ms'],all_p95_ttft_saving_ms=a['all_p95_ttft_ms']-c['all_p95_ttft_ms'],return_p95_ttft_saving_ms=returning_p95(a['name'])-returning_p95(c['name'])))
  stats=inference([r['log_throughput_ratio'] for r in rows]);memory=all(r['memory_no_greater'] for r in rows)
  result={'plan':'CPU-RECENCY-V3-CONFIRM.json','scope':plan['primary']['scope'],'all_pilots_and_prior_confirmation_excluded':True,'statistics':stats,'memory_no_greater_every_pair':memory,'statistical_serving_criterion_before_separate_byte_and_integration_checks':stats['passed_statistics'] and memory,'all_ttft_saving_ms':ci12([r['all_ttft_saving_ms'] for r in rows]),'return_ttft_saving_ms':ci12([r['return_ttft_saving_ms'] for r in rows]),'all_p95_ttft_saving_ms':ci12([r['all_p95_ttft_saving_ms'] for r in rows]),'return_p95_ttft_saving_ms':ci12([r['return_p95_ttft_saving_ms'] for r in rows]),'throughput_per_VRAM_secondary':{key:inference([normalized_log_ratio(r['candidate'],r['arc'],key) for r in rows]) for key in ['nvml_devicewide_peak_bytes','cuda_reserved_peak_bytes','cuda_allocated_peak_bytes']},'normalized_metric_does_not_replace_primary':'Pure throughput lower CI >0 and memory no greater in every pair remain mandatory; normalized efficiency is secondary.', 'all_exact_prompt_and_output_ids':True,'pairs':rows}
  (H/'cpu-recency-v3-confirm-work.json').write_text(json.dumps(work,indent=2)+'\n');(H/'cpu-recency-v3-confirm-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
