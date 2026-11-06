"""Offline analysis of every frozen paired run; never feeds serving policy."""
import csv,gzip,itertools,json,math,statistics
from pathlib import Path
import campaign_metrics as metrics
H=Path(__file__).resolve().parent

def get(d,path,default=None):
 for k in path.split('.'):
  if not isinstance(d,dict) or k not in d:return default
  d=d[k]
 return d

def main():
 plan=json.loads((H/'CONFIRMATION_PLAN.json').read_text());rows=[];summaries={};comparisons=[]
 for run in plan['runs']:
  p=H/'runs'/run['name'];s=json.loads((p/'summary.json').read_text());summaries[run['name']]=s;a=s['all'];c=s['counters'];u=s['offline_staging_utility'];res=s['resources'];con=s['achieved_concurrency']
  rows.append({'name':run['name'],'cohort':run['cohort'],'arm':run['arm'],'seed':run['seed'],'requests':a['requests'],'valid':s['valid'],'strict_proofs_passed':s['proofs']['passed'],'TPS':a['output_tokens_s'],'requests_s':a['requests_s'],'goodput_requests_s':a['goodput_requests_s'],'goodput_TPS':a['goodput_output_tokens_s'],'output_tokens':a['output_tokens'],'input_tokens_mean':get(a,'prompt_token_lengths.mean'),'output_tokens_mean':get(a,'output_token_lengths.mean'),'TTFT_p50_ms':get(a,'release_ttft_ms.p50'),'TTFT_p95_ms':get(a,'release_ttft_ms.p95'),'ITL_p50_ms':get(a,'itl_ms.p50'),'ITL_p95_ms':get(a,'itl_ms.p95'),'E2E_p50_ms':get(a,'release_e2e_ms.p50'),'E2E_p95_ms':get(a,'release_e2e_ms.p95'),'queue_p95_ms':get(a,'server_queue_ms.p95'),'aggregate_SLO_pass':get(a,'aggregate_latency_slo.passed'),'server_CPU_s':res.get('cpu_s_sampled'),'driver_CPU_s':res.get('driver_cpu_s_sampled'),'server_RSS_peak_bytes':res.get('rss_peak_bytes_sampled'),'CUDA_allocated_peak_bytes':res.get('load_cuda_allocated_peak_bytes_sum_workers'),'CUDA_reserved_peak_bytes':res.get('load_cuda_reserved_peak_bytes_sum_workers'),'NVML_device_peak_bytes':res.get('nvml_device_memory_peak_bytes_sampled'),'host_swapin_bytes':get(res,'host_memory.swap_in_bytes_delta'),'host_swapout_bytes':get(res,'host_memory.swap_out_bytes_delta'),'process_major_faults':get(res,'process_fault_totals_sampled.major_faults'),'scheduled_peak':con.get('scheduled_peak'),'running_peak':con.get('running_peak'),'queued_peak':con.get('queued_peak'),'CPU_cache_resident_peak_blocks':c.get('cpu_resident_peak_blocks'),'CPU_cache_evicted_blocks':get(c,'manager_counts_delta.cpu_evicted_blocks'),'candidate_numerical_calls':get(c,'worker_prefix_mean_scoring_delta.numerical_calls',0),'candidate_numerical_candidates':get(c,'worker_prefix_mean_scoring_delta.numerical_candidates',0),'legacy_mean_numerical_enqueues':get(c,'worker_counts_delta.legacy_mean_numerical_calls',0),'constrained_nominations':c.get('constrained_nominations'),'native_adoptions':c.get('native_adoption_true_requests'),'preemptions':c.get('preemptions'),'demand_H2D_bytes':c.get('demand_h2d_bytes'),'D2H_bytes':c.get('d2h_bytes'),'explicit_completed_prefetch_H2D_bytes':u.get('global_explicit_successful_completion_bytes_observed'),'named_demand_consumed_prefetch_bytes':u.get('global_named_demand_consumed_bytes'),'completed_unconsumed_by_named_demand_bytes':u.get('global_completed_unconsumed_by_named_demand_bytes'),'all_reservation_peak_blocks':get(u,'reservation_highwater.known_closed_interval_peak_unique_blocks'),'bound_reservation_peak_blocks':get(u,'request_owned_highwater.known_closed_interval_peak_unique_blocks'),'all_reservation_p95_ms':get(u,'reservation_lifetime_ms.p95'),'bound_reservation_p95_ms':get(u,'request_owned_reservation_lifetime_ms.p95'),'unknown_completed_attempts':len(u.get('unknown_completed_attempts',[])),'unknown_consumption_attempts':len(u.get('unknown_consumption_attempts',[]))})
 (H/'comparisons').mkdir(exist_ok=True)
 for cohort in ['chat','agent']:
  for seed in [1103,2207,3301]:
   group={r['arm']:r for r in plan['runs'] if r['cohort']==cohort and r['seed']==seed}
   raw={k:json.load(gzip.open(H/'runs'/v['name']/'raw.json.gz')) for k,v in group.items()}
   for left,right in [('largeARC','smallARC'),('smallARC','candidate'),('largeARC','candidate'),('smallARC','original'),('original','candidate')]:
    comp=metrics.compare(raw[left],raw[right]);name=f'{cohort}-s{seed}-{left}-v-{right}';out=H/'comparisons'/(name+'.json');out.write_text(json.dumps(comp,indent=2,allow_nan=False)+'\n')
    ratio=summaries[group[right]['name']]['all']['output_tokens_s']/summaries[group[left]['name']]['all']['output_tokens_s']
    comparisons.append({'cohort':cohort,'seed':seed,'left':left,'right':right,'TPS_ratio_right_over_left':ratio,'descriptive_within1pct':ratio>=.99,'exact_matched_eligible':comp['performance_claim_eligible'],'inventory':{k:len(v) for k,v in comp['inventory'].items()},'path':str(out.relative_to(H)),'rates':comp['rates_descriptive_only']})
   del raw
 aggregates=[]
 for cohort in ['chat','agent']:
  for left,right in [('largeARC','smallARC'),('smallARC','candidate'),('largeARC','candidate'),('smallARC','original'),('original','candidate')]:
   pairs=[p for p in comparisons if p['cohort']==cohort and p['left']==left and p['right']==right];ratios=[p['TPS_ratio_right_over_left'] for p in pairs]
   aggregates.append({'cohort':cohort,'left':left,'right':right,'paired_partitions':len(pairs),'TPS_ratio_geomean_descriptive':math.exp(sum(math.log(r) for r in ratios)/len(ratios)),'TPS_ratio_min':min(ratios),'TPS_ratio_max':max(ratios),'pairs_within1pct_descriptive':sum(p['descriptive_within1pct'] for p in pairs),'exact_matched_eligible_pairs':sum(p['exact_matched_eligible'] for p in pairs),'note':'three paired corpus partitions, no confidence interval or exact-work waiver'})
 output={'plan':'CONFIRMATION_PLAN.json','status':'all_frozen_runs_analyzed','runs':rows,'paired_comparisons':comparisons,'aggregates':aggregates,'interpretation':'all TPS ratios are descriptive unless exact matched eligibility and manual budget/source review also pass; output mismatch means closed-loop downstream histories are unequal work; originalMSR-vs-candidateARCretention is an integrated configuration comparison, not pure capture/scoring ablation'}
 (H/'CONFIRMATION_RESULTS.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
 with (H/'confirmation-results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'runs':len(rows),'strict_passed':sum(r['strict_proofs_passed'] for r in rows),'requests':sum(r['requests'] for r in rows),'aggregates':aggregates}),flush=True)
if __name__=='__main__':main()
