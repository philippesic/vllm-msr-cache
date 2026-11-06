"""Offline measured frontier, threshold readouts and exact mismatch inventories."""
import argparse,csv,gzip,json,math
from pathlib import Path
import campaign_metrics as metrics
H=Path(__file__).resolve().parent

def get(d,path,default=None):
 for key in path.split('.'):
  if not isinstance(d,dict) or key not in d:return default
  d=d[key]
 return d

def row(run,phase):
 d=H/'runs'/run['name'];s=json.loads((d/'summary.json').read_text());out=json.loads((d/'frontier-disposition.json').read_text());a=s['all'];c=s['counters'];u=s['offline_staging_utility'];r=s['resources'];n=s['achieved_concurrency']
 result=dict(name=run['name'],observer_revision=run.get('observer_revision','old-V2readmission-only'),phase=phase,arm=run['arm'],seed=run['seed'],gpu_blocks=run['gpu_blocks'],GPU_KV_MiB=run['gpu_blocks']*.875,CPU_KV_MiB=3584,**out)
 fields={'TPS':a.get('output_tokens_s'),'requests_s':a.get('requests_s'),'goodput_requests_s':a.get('goodput_requests_s'),'goodput_TPS':a.get('goodput_output_tokens_s'),'output_tokens':a.get('output_tokens'),'output_mean':get(a,'output_token_lengths.mean'),'prompt_mean':get(a,'prompt_token_lengths.mean'),'TTFT_p50_ms':get(a,'release_ttft_ms.p50'),'TTFT_p95_ms':get(a,'release_ttft_ms.p95'),'ITL_p50_ms':get(a,'itl_ms.p50'),'ITL_p95_ms':get(a,'itl_ms.p95'),'E2E_p50_ms':get(a,'release_e2e_ms.p50'),'E2E_p95_ms':get(a,'release_e2e_ms.p95'),'queue_p95_ms':get(a,'server_queue_ms.p95'),'absolute_SLO_pass':get(a,'aggregate_latency_slo.passed'),'CPU_s':r.get('cpu_s_sampled'),'driver_CPU_s':r.get('driver_cpu_s_sampled'),'RSS_peak_bytes':r.get('rss_peak_bytes_sampled'),'CUDA_allocated_peak_bytes':r.get('load_cuda_allocated_peak_bytes_sum_workers'),'CUDA_reserved_peak_bytes':r.get('load_cuda_reserved_peak_bytes_sum_workers'),'NVML_device_peak_bytes':r.get('nvml_device_memory_peak_bytes_sampled'),'host_swap_in_bytes':get(r,'host_memory.swap_in_bytes_delta'),'host_swap_out_bytes':get(r,'host_memory.swap_out_bytes_delta'),'process_major_faults':get(r,'process_fault_totals_sampled.major_faults'),'running_peak':n.get('running_peak'),'scheduled_peak':n.get('scheduled_peak'),'queued_peak':n.get('queued_peak'),'preemptions':c.get('preemptions'),'D2H_bytes':c.get('d2h_bytes'),'demand_H2D_bytes':c.get('demand_h2d_bytes'),'prefetch_H2D_bytes':u.get('global_explicit_successful_completion_bytes_observed'),'named_consumed_prefetch_bytes':u.get('global_named_demand_consumed_bytes'),'completed_unconsumed_named_bytes':u.get('global_completed_unconsumed_by_named_demand_bytes'),'numerical_calls':get(c,'worker_prefix_mean_scoring_delta.numerical_calls',0),'numerical_candidates':get(c,'worker_prefix_mean_scoring_delta.numerical_candidates',0),'constrained_nominations':c.get('constrained_nominations'),'native_adoptions':c.get('native_adoption_true_requests'),'reservation_peak_blocks':get(u,'reservation_highwater.known_closed_interval_peak_unique_blocks'),'bound_reservation_peak_blocks':get(u,'request_owned_highwater.known_closed_interval_peak_unique_blocks'),'reservation_p95_ms':get(u,'reservation_lifetime_ms.p95'),'bound_reservation_p95_ms':get(u,'request_owned_reservation_lifetime_ms.p95'),'unknown_completion_attempts':len(u.get('unknown_completed_attempts',[])),'unknown_consumption_attempts':len(u.get('unknown_consumption_attempts',[])),'strict_completed_proofs_pass':s['proofs']['passed']==len(s['requests'])}
 result.update(fields);result['absolute15s_TTFT_pass']=isinstance(result.get('TTFT_p95_ms'),(int,float)) and result['TTFT_p95_ms']<=15000;return result

def thresholds(rows,upper):
 result=[]
 for seed in sorted({r['seed'] for r in rows}):
  anchors=[r for r in rows if r['seed']==seed and r['arm']=='arc' and r['gpu_blocks']==upper and r['complete_load_eligible']]
  if not anchors:continue
  anchor=anchors[0]
  for r in rows:
   if r['seed']==seed and r['complete_load_eligible']:
    r['TPS_fraction_highARC']=r['TPS']/anchor['TPS'];r['TTFT_p95_ratio_highARC']=r['TTFT_p95_ms']/anchor['TTFT_p95_ms']
  for fraction in [.99,.97,.95]:
   for tolerance in [1.05,1.10]:
    for arm in ['arc','candidate']:
     qualifying=[r for r in rows if r['seed']==seed and r['arm']==arm and r['complete_load_eligible'] and r['TPS_fraction_highARC']>=fraction and r['TTFT_p95_ratio_highARC']<=tolerance]
     selected=min(qualifying,key=lambda r:r['gpu_blocks']) if qualifying else None
     result.append({'seed':seed,'arm':arm,'target_TPS_fraction':fraction,'TTFT_tolerance':tolerance,'smallest_measured_budget_blocks':selected['gpu_blocks'] if selected else None,'point_name':selected['name'] if selected else None,'actual_scoring_and_adoption':bool(selected and selected['numerical_calls']>0 and selected['native_adoptions']>0),'absolute15s_TTFT_pass':selected['absolute15s_TTFT_pass'] if selected else None,'absolute_agent_SLO_pass':selected['absolute_SLO_pass'] if selected else None,'interpretation':'descriptive workflow only, not exact-work/capacity claim'})
 return result

def nondominated(rows):
 points=[r for r in rows if r['complete_load_eligible'] and r['CUDA_allocated_peak_bytes'] is not None];result=[]
 for r in points:
  others=[o for o in points if o['seed']==r['seed'] and o['name']!=r['name']]
  dominates=lambda o:(o['CUDA_allocated_peak_bytes']<=r['CUDA_allocated_peak_bytes'] and o['TPS']>=r['TPS'] and o['TTFT_p95_ms']<=r['TTFT_p95_ms'] and (o['CUDA_allocated_peak_bytes']<r['CUDA_allocated_peak_bytes'] or o['TPS']>r['TPS'] or o['TTFT_p95_ms']<r['TTFT_p95_ms']))
  if not any(dominates(o) for o in others):result.append(r['name'])
 return result

def highest_observed_sensitivity(rows):
 """Posthoc alternative anchor; retain the predeclared normalization intact."""
 result=[]
 for seed in sorted({r['seed'] for r in rows}):
  partition=[dict(r) for r in rows if r['seed']==seed]
  anchors=[r for r in partition if r['arm']=='arc' and r['complete_load_eligible']]
  if not anchors:continue
  anchor=max(anchors,key=lambda r:r['TPS']);table=thresholds(partition,anchor['gpu_blocks'])
  for r in rows:
   if r['seed']==seed and r['complete_load_eligible']:
    r['TPS_fraction_highest_observedARC']=r['TPS']/anchor['TPS'];r['TTFT_p95_ratio_highest_observedARC']=r['TTFT_p95_ms']/anchor['TTFT_p95_ms']
  result.append({'seed':seed,'anchor':{k:anchor[k] for k in ['name','gpu_blocks','TPS','TTFT_p95_ms']},'posthoc':True,'interpretation':'highest observed single ARC point; sensitivity only, no fitted capacity or unique optimum','thresholds':table})
 return result

def selection(rows):
 screen=[r for r in rows if r['phase']=='screen' and r['complete_load_eligible']];pairs={b:{r['arm']:r for r in screen if r['gpu_blocks']==b} for b in sorted({r['gpu_blocks'] for r in screen})}
 for fraction in [.99,.97,.95]:
  choices=[]
  for b,p in pairs.items():
   c=p.get('candidate');a=p.get('arc')
   if not c or not a or 'TPS_fraction_highARC' not in c:continue
   passes=lambda r:r['TPS_fraction_highARC']>=fraction and r['TTFT_p95_ratio_highARC']<=1.05
   if c['numerical_calls']>0 and c['native_adoptions']>0 and passes(c) and not passes(a):choices.append(c)
  if choices:
   x=min(choices,key=lambda r:r['gpu_blocks']);return {'proposed_budget_blocks':x['gpu_blocks'],'reason':f'predeclared candidate-enabled advantage at{fraction:.0%}retention/+5%TTFT','root_review_required_before_confirmation':True}
 # No prefetch-qualified threshold extension: choose a measured boundary/null,
 # not the largest noisy candidate/ARC ratio.
 candidates=[]
 for arm in ['arc','candidate']:
  points=sorted([r for r in screen if r['arm']==arm and 'TPS_fraction_highARC' in r],key=lambda r:r['gpu_blocks'])
  for lo,hi in zip(points,points[1:]):
   drop=hi['TPS_fraction_highARC']-lo['TPS_fraction_highARC'];cross=lo['TPS_fraction_highARC']<.95<=hi['TPS_fraction_highARC'];pair=pairs.get(lo['gpu_blocks'],{});c=pair.get('candidate');activity=bool(c and c['numerical_calls']>0 and c['native_adoptions']>0)
   candidates.append({'budget':lo['gpu_blocks'],'crossing95':cross,'activity':activity,'drop':drop,'arm':arm,'high_budget':hi['gpu_blocks']})
 if candidates:
  x=max(candidates,key=lambda z:(z['crossing95'],z['activity'],z['drop'],-z['budget']));return {'proposed_budget_blocks':x['budget'],'reason':'predeclared informative95%boundary or adjacentdrop; no qualified semantic advantage','boundary_evidence':x,'root_review_required_before_confirmation':True}
 return {'proposed_budget_blocks':None,'reason':'no sufficient complete paired frontier/anchor; preserve censored failures','root_review_required_before_confirmation':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--include-confirmation',action='store_true');ap.add_argument('--reuse-comparisons',action='store_true');a=ap.parse_args();plan=json.loads((H/('ANALYSIS_PLAN.json' if (H/'ANALYSIS_PLAN.json').exists() else 'SWEEP_PLAN.json')).read_text());plans=[('screen',plan)]
 if a.include_confirmation:plans.append(('confirmation',json.loads((H/'CONFIRMATION_PLAN.json').read_text())))
 rows=[row(run,phase) for phase,p in plans for run in p['runs']];upper=max(plan['budgets']['GPU_blocks_sorted']);targets=thresholds(rows,upper);sensitivity=highest_observed_sensitivity(rows);comparisons=[];(H/'comparisons').mkdir(exist_ok=True)
 for phase,p in plans:
  for seed,budget in sorted({(r['seed'],r['gpu_blocks']) for r in p['runs']}):
   members={r['arm']:r for r in p['runs'] if r['seed']==seed and r['gpu_blocks']==budget}
   if set(members)!=set(['arc','candidate']):continue
   name=f'{phase}-b{budget}-s{seed}';out=H/'comparisons'/(name+'.json')
   if a.reuse_comparisons and out.exists():c=json.loads(out.read_text())
   else:
    l=json.load(gzip.open(H/'runs'/members['arc']['name']/'raw.json.gz'));r=json.load(gzip.open(H/'runs'/members['candidate']['name']/'raw.json.gz'));c=metrics.compare(l,r);out.write_text(json.dumps(c,indent=2,allow_nan=False)+'\n')
   comparisons.append({'phase':phase,'seed':seed,'gpu_blocks':budget,'exact_work_eligible':c['performance_claim_eligible'],'inventory_counts':{k:len(v) for k,v in c['inventory'].items()},'comparison_path':str(out.relative_to(H)),'rates_descriptive':c['rates_descriptive_only']})
 result={'schema':'measured-vram-frontier-v1','coverage':{'measured_points':len(rows),'complete_valid_points':sum(r['complete_load_eligible'] for r in rows),'invalid_proof_points':sum(r['outcome']=='strict_proof_invalid' for r in rows),'time_capped_points':sum(r['outcome'].endswith('time_capped') for r in rows),'accepted_response_proofs':sum(r['completed_strict_proofs'] for r in rows if r['complete_load_eligible'])},'rows':rows,'thresholds':targets,'secondary_highest_observed_ARC_sensitivity':sensitivity,'nondominated_measured_points':nondominated(rows),'confirmation_selection':selection(rows),'comparisons':comparisons,'rejected_attempts':json.loads((H/'REJECTED_ATTEMPTS.json').read_text()) if (H/'REJECTED_ATTEMPTS.json').exists() else [],'limits':['finite actual-output closedloop workflow, simulated waits, no universal fitted optimum','generatedhistory differences preserved; exact-work threshold never waived','cappedpoints are incomplete/censored, excluded from completed-load ratios','single seed screen and heldoutpartition confirmation shown separately, no confidenceinterval','original scoringquality unknown; no new scoringquality equivalence claim']};name='FINAL_RESULTS' if a.include_confirmation else 'SCREEN_RESULTS';(H/(name+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (H/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 thresholdrows=[dict(reference='primary_predeclared_largest_budget_ARC',**t) for t in targets]+[dict(reference='secondary_posthoc_highest_observed_ARC',**t) for part in sensitivity for t in part['thresholds']]
 with (H/'THRESHOLDS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(thresholdrows[0]));w.writeheader();w.writerows(thresholdrows)
 (H/'THRESHOLDS.json').write_text(json.dumps({'primary':targets,'secondary_posthoc':sensitivity},indent=2)+'\n')
 print(json.dumps({'points':len(rows),'ineligible_points':sum(not r['complete_load_eligible'] for r in rows),'completed_requests':sum(r['completed_requests'] for r in rows),'strict_proofs':sum(r['completed_strict_proofs'] for r in rows),'selection':result['confirmation_selection']}))
if __name__=='__main__':main()
