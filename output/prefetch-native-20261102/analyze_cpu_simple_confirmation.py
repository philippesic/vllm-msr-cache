"""Whole workflow inference; never perform raw analysis during serving."""
import fcntl,itertools,json,math,statistics
from pathlib import Path
from analyze_busy_points import point
from analyze_warm_confirmation import ci6
from work_inventory import compare
H=Path(__file__).resolve().parent

def inference(log_ratios):
 ci=ci6(log_ratios)
 observed=statistics.mean(log_ratios)
 null=[statistics.mean(s*x for s,x in zip(signs,log_ratios)) for signs in itertools.product((-1,1),repeat=6)]
 p=sum(v>=observed-1e-14 for v in null)/len(null)
 return {'geometric_mean_gain_fraction':math.expm1(ci['mean']),'lower95_gain_fraction':math.expm1(ci['lower95']),'upper95_gain_fraction':math.expm1(ci['upper95']),'one_sided_exact_sign_flip_p':p,'paired_log_ci':ci,'passed_statistics':ci['lower95']>0 and p<=.05}

def main():
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=json.loads((H/'CPU-SIMPLE-CONFIRM.json').read_text());progress=json.loads((H/plan['progress_file']).read_text())
  assert progress['status']=='complete' and len(progress['completed'])==12
  rows=[];work=[]
  for pair in plan['pairs']:
   c,a=point(pair['candidate']),point(pair['arc'])
   for n in (c['name'],a['name']):
    d=json.loads((H/'runs'/n/'frontier-disposition.json').read_text())
    assert d['fixed_work_integrity']['passed'] and d['fixed_work_integrity']['exact_recorded_output_ids'] and d['fixed_work_integrity']['invalid_control_rejected']
   w=compare(c['name'],a['name']);assert w['identical_input_token_ids'] and w['identical_output_token_ids'];work.append(w)
   memory_ok=all(c[k]<=a[k] for k in ('cuda_reserved_peak_bytes','cuda_allocated_peak_bytes','nvml_devicewide_peak_bytes'))
   rows.append(dict(pair,candidate=c,arc=a,log_throughput_ratio=math.log(c['throughput_tokens_s']/a['throughput_tokens_s']),memory_no_greater=memory_ok,all_ttft_saving_ms=a['all_mean_ttft_ms']-c['all_mean_ttft_ms'],return_ttft_saving_ms=a['return_mean_ttft_ms']-c['return_mean_ttft_ms']))
  stats=inference([r['log_throughput_ratio'] for r in rows]);memory=all(r['memory_no_greater'] for r in rows)
  result={'plan':'CPU-SIMPLE-CONFIRM.json','scope':plan['primary']['scope'],'pilots_excluded':True,'statistics':stats,'memory_no_greater_every_pair':memory,'statistical_serving_criterion_before_separate_byte_qualification':stats['passed_statistics'] and memory,'all_ttft_saving_ms':ci6([r['all_ttft_saving_ms'] for r in rows]),'return_ttft_saving_ms':ci6([r['return_ttft_saving_ms'] for r in rows]),'all_exact_prompt_and_output_ids':True,'pairs':rows}
  (H/'cpu-simple-confirm-work.json').write_text(json.dumps(work,indent=2)+'\n');(H/'cpu-simple-confirm-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
