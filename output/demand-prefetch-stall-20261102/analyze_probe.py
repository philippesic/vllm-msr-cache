"""Direct passive observations: postponement bounds, never causal TPS loss."""
import gzip,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));import campaign_metrics as metrics
from offline import analyze

def delta_ms(end,start):return (end-start)/1e6 if isinstance(end,int) and isinstance(start,int) and end>=start else None

def summarize(raw,summary):
 p=raw.get('postload_export',{}).get('scheduler',{}).get('demand_guard_probe')
 if not isinstance(p,dict):raise ValueError('missing direct probe export')
 if p['rejected_observations']:raise ValueError('rejected direct observation identities: '+str(p['rejected_observations']))
 requests={r['payload']['request_id']:r for r in raw['requests']};episodes=[];lo=raw['load_start_ns'];hi=raw['load_end_ns']
 for e in p['episodes']:
  if not lo<=e['first_guard_ns']<=hi:continue
  r=requests.get(e['request_id']);row=dict(e);row.update(session_id=r.get('session_id') if r else None,turn=r.get('turn') if r else None,first_to_last_guard_ms=delta_ms(e['last_guard_ns'],e['first_guard_ns']),first_guard_to_eligible_ms=delta_ms(e.get('eligible_at_ns'),e['first_guard_ns']),first_guard_to_next_native_path_ms=delta_ms(e.get('next_native_path_ns'),e['first_guard_ns']),eligible_to_next_native_path_ms=delta_ms(e.get('next_native_path_ns'),e.get('eligible_at_ns')),request_TTFT_ms=delta_ms(r.get('first_token_ns'),r.get('intended_release_ns',r.get('request_start_ns'))) if r else None);episodes.append(row)
 dist=lambda field:metrics.dist([e[field] for e in episodes if e[field] is not None]);potential=analyze(raw,{})
 return {'name':raw['name'],'GPU_blocks':raw['geometry']['gpu_blocks'],'strict_proofs':summary['proofs'],'diagnostic_only':True,'planned_requests':raw['planned_requests'],'completed_requests':len(raw['requests']),'direct_guard_calls':sum(e['guard_calls'] for e in episodes),'affected_requests':len({e['request_id'] for e in episodes}),'pending_guard_calls':sum(e['status_calls'].get('pending',0) for e in episodes),'inflight_guard_calls':sum(e['status_calls'].get('in_flight',0) for e in episodes),'all_branch_observer_calls_including_warmup':p['calls'],'rejected_observation_count':p['rejected_observations'],'active_at_export':p['active_request_ids'],'first_to_last_guard_ms':dist('first_to_last_guard_ms'),'first_guard_to_eligible_ms':dist('first_guard_to_eligible_ms'),'first_guard_to_next_native_path_ms':dist('first_guard_to_next_native_path_ms'),'eligible_to_next_native_path_ms':dist('eligible_to_next_native_path_ms'),'sum_first_to_last_guard_ms':sum(e['first_to_last_guard_ms'] or 0 for e in episodes),'sum_first_guard_to_eligible_ms':sum(e['first_guard_to_eligible_ms'] or 0 for e in episodes),'native_pending_transfer_deferral_events':len(raw.get('native_lookup_deferral_events',[])),'numerical_calls':summary['counters']['worker_prefix_mean_scoring_delta'].get('numerical_calls',0),'native_adoptions':summary['counters']['native_adoption_true_requests'],'observed_TPS':summary['all']['output_tokens_s'],'observed_p95TTFT_ms':summary['all']['release_ttft_ms']['p95'],'load_elapsed_s':summary['all']['elapsed_s'],'same_run_bind_states':potential['bind_states'],'same_run_potential_ms':potential['sum_potential_span_ms'],'episodes':episodes,'limits':['zero hits applies to this measured execution, not all other budgets or future scheduling','first-to-last hit0 for a single call does not mean zero postponement','eligibility span and nextpoll gap may contain unrelated scheduler/queue time','summed per-request spans can overlap; not GPUidle or throughput loss','no causal policy ablation; direct throughput effect remains unmeasured']}

def main():
 plan=json.loads((H/'PLAN.json').read_text());rows=[]
 for r in plan['runs']:
  d=H/'runs'/r['name'];s=json.loads((d/'summary.json').read_text());raw=json.load(gzip.open(d/'raw.json.gz'));assert s['proofs']['all_passed'],s['proofs'];row=summarize(raw,s);rows.append(row);print(row['GPU_blocks'],row['direct_guard_calls'],row['affected_requests'],row['first_guard_to_eligible_ms'])
 result={'schema':'direct-demand-prefetch-guard-v1','runs':rows,'offline_all8':'OFFLINE.json','causal_ablation_run':False,'interpretation':('Exact branch instrumented; ZERO pending/inflight guard returns observed in both measured runs.' if sum(r['direct_guard_calls'] for r in rows)==0 else 'Exact branch instrumented; '+str(sum(r['direct_guard_calls'] for r in rows))+' pending/inflight guard returns observed.')+' No counterfactual throughput attribution;'+str(sum(r['all_branch_observer_calls_including_warmup'] for r in rows))+'observer calls include warmup.'};(H/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
