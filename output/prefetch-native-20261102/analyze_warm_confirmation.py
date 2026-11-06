"""Offline confirmation analysis. Takes the serving lock; never runs during load."""
import csv,fcntl,json,math,statistics
from pathlib import Path
from work_inventory import compare
H=Path(__file__).resolve().parent
T95_DF5=2.570581835636314

def ci6(values):
 assert len(values)==6,'Only the predeclared six complete workflow pairs qualify'
 mean=statistics.mean(values);half=T95_DF5*statistics.stdev(values)/math.sqrt(6)
 return {'n_independent_workflow_pairs':6,'mean':mean,'lower95':mean-half,'upper95':mean+half,'method':'paired Student t; df5; complete workflow mean deltas, not per-request pseudo-replication; conditional on fixedthree public traces'}

def point(name):
 d=json.loads((H/'runs'/name/'summary.json').read_text())
 assert d['valid'] and d['status']=='complete' and d['proofs']['all_passed']
 assert d['planned_requests']==d['all']['requests']==d['proofs']['passed']==12
 r=d['resources'];u=d['offline_staging_utility'];used=u['global_named_demand_consumed_bytes'];copied=u['global_explicit_successful_completion_bytes_observed']
 assert u['explicit_completion_collection_present'] and not u['unknown_completed_attempts'] and not u['unknown_consumption_attempts']
 return {'name':name,'all_mean_ttft_ms':d['all']['release_ttft_ms']['mean'],'all_p95_ttft_ms':d['all']['release_ttft_ms']['p95'],'return_mean_ttft_ms':statistics.mean(v['release_ttft_ms'] for v in d['requests'] if v['identity'][1]>0),'throughput_output_tokens_s':d['all']['output_tokens_s'],'output_tokens':d['all']['output_tokens'],'cuda_allocated_peak_bytes':r['load_cuda_allocated_peak_bytes_sum_workers'],'cuda_reserved_peak_bytes':r['load_cuda_reserved_peak_bytes_sum_workers'],'nvml_devicewide_peak_bytes':r['nvml_device_memory_peak_bytes_sampled'],'cpu_seconds_sampled':r['cpu_s_sampled'],'copied_bytes':copied,'used_bytes':used,'useful_fraction':used/copied if copied else None,'native_adopted_jobs':sum(v['native_adopted'] is True for v in u['event_rows']),'strict_proofs':d['proofs']['passed']}

def main():
 with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  progress=json.loads((H/'warm-confirm-v2-repeated-progress.json').read_text());assert progress['status']=='complete' and len(progress['completed'])==12
  pairs=[];work=[]
  for rep in (1,2):
   for seed in (1103,2207,3301):
    names={arm:f'warm-confirm-v2-b{512 if arm=="candidate" else 576}-{arm}-s{seed}-r{rep}' for arm in ('candidate','arc')}
    c,a=(point(names[arm]) for arm in ('candidate','arc'));w=compare(names['candidate'],names['arc']);work.append(w)
    equal_lengths=all(v['left_input_tokens']==v['right_input_tokens'] and v['left_output_tokens']==v['right_output_tokens'] for v in w['rows'])
    pairs.append({'seed':seed,'repetition':rep,'candidate':c,'arc':a,'return_saving_ms':a['return_mean_ttft_ms']-c['return_mean_ttft_ms'],'all_saving_ms':a['all_mean_ttft_ms']-c['all_mean_ttft_ms'],'return_saving_fraction':1-c['return_mean_ttft_ms']/a['return_mean_ttft_ms'],'cuda_reserved_saving_bytes':a['cuda_reserved_peak_bytes']-c['cuda_reserved_peak_bytes'],'work_per_request_lengths_equal':equal_lengths,'input_ids_equal':w['identical_input_token_ids'],'output_ids_equal':w['identical_output_token_ids']})
  ret=ci6([p['return_saving_ms'] for p in pairs]);allci=ci6([p['all_saving_ms'] for p in pairs])
  candidate_max=max(p['candidate']['cuda_reserved_peak_bytes'] for p in pairs);arc_min=min(p['arc']['cuda_reserved_peak_bytes'] for p in pairs)
  accepted=ret['lower95']>0 and candidate_max<arc_min and all(p['work_per_request_lengths_equal'] for p in pairs)
  result={'schema':'prefetch-confirmation-analysis-v1','plan':'WARM-CONFIRM-V2-REPEATED.json','screening_pilots_excluded':True,'scope':'Qwen2.5-7B-InstructAWQ FP16/Turing, 4 concurrent publicsoftwareagent sessions, actual generated3turn histories, private seeded10/20/30/45/60s tool waits, constantcausal750ms nomination estimate, 4096CPUblocks, common7680 contextlimit, neutralwarmup32decode. Fixed3 publiccase partitions,2freshrepetitions each. No originalbusy24 throughputclaim.','return_ttft_saving_ms':ret,'all_ttft_saving_ms':allci,'candidate_max_reserved_bytes':candidate_max,'arc_min_reserved_bytes':arc_min,'memory_strictly_lower_all_runs':candidate_max<arc_min,'serving_statistical_criterion_passed_before_byte_qualification':accepted,'byte_qualification_separate_required':True,'all_pairs_exact_prompt_and_output_ids':all(p['input_ids_equal'] and p['output_ids_equal'] for p in pairs),'pairs':pairs}
  (H/'warm-confirmation-work.json').write_text(json.dumps(work,indent=2)+'\n');(H/'warm-confirmation-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
  with (H/'warm-confirmation-points.csv').open('w') as f:
   points=[dict(seed=p['seed'],repetition=p['repetition'],arm=arm,**p[arm]) for p in pairs for arm in ('candidate','arc')]
   w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
  print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
