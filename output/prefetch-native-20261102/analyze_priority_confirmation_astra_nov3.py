"""All-outcome three-arm discovery analysis; no inferential intervals."""
import collections,gzip,hashlib,json,math,re
from pathlib import Path
from analyze_confirmation_astra_nov3 import read,paired,delta
from analyze_triage_confirmation_astra_nov3 import host,observed_gpu
from return_priority_receipt_astra_nov3 import verify_priority
from http_confirmation_receipt_astra_nov3 import verify_http
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 mp=ROOT/'evidence/live-optimization-2080ti-20261103/priority-confirmation-frozen.json';m=json.loads(mp.read_text());rows={}
 for item in m['execution_order']:
  r=read(item)
  if r['attempted']:
   raw=json.load(gzip.open(ROOT/r['raw'],'rt'));r['HTTP_receipt']=verify_http(raw);r['priority_receipt']=verify_priority(raw,item) if r['eligible'] else {'qualified':False,'issues':['incomplete_workflow']}
   r['eligible']=r['eligible'] and r['HTTP_receipt']['qualified'] and r['priority_receipt']['qualified'] and r['owned_descendants_remaining']==[] and r['measurement']['matches_frozen_lean']
   r['host_resources']={'before':raw.get('host_memory_before'),'startup':host(raw.get('startup_resource_samples',[])),'serving':host(raw.get('resource_samples',[])),'configured_cpu_kv_bytes':raw['geometry']['cpu_kv_bytes'],'actual_locked_pinned_host_bytes':None}
   r['thermal_clock_power']={k:{'startup':observed_gpu(raw.get('startup_resource_samples',[]),k),'serving':observed_gpu(raw.get('resource_samples',[]),k)} for k in ['temperature_c','clock_mhz','power_w']}
   r['native_transfer_jobs']={d:delta(raw,'vllm:kv_offload_size_count',f'transfer_type="{d}"') for d in ['GPU_to_CPU','CPU_to_GPU']}
   r['restoration_scope']='First-admission local/CPU/GPU source attribution, not cumulative physical executed prefill/GPU time. Total native transfer bytes/jobs cover measured load. Per-transfer preemption reason unavailable; observed preemption count reported separately.'
   r['measurement']['readmission_events']=len(raw['readmission_observer_events']) if 'readmission_observer_events' in raw else None
   r['measurement']['lookup_observer_events']=len(raw['native_lookup_deferral_events']) if 'native_lookup_deferral_events' in raw else None
   qs=raw.get('requests',[]);r['service_format']={'responses':len(qs),'starts_status':sum(x.get('generated_text','').startswith('Status:') for x in qs),'single_blank_line_block':sum(bool(x.get('generated_text')) and len(re.split(r'\n\s*\n',x['generated_text'].strip()))==1 for x in qs),'with_code_fence':sum('```' in x.get('generated_text','') for x in qs),'completed_action_phrase_indices':[x['index'] for x in qs if re.search(r'\bI (?:have |had )?(?:modified|changed|ran|executed|updated|applied|fixed)\b',x.get('generated_text',''),re.I)],'scope':'Mechanical inspection flags, not factual accuracy or task success; single blank-line block does not establish paragraph compliance.'}
   counts=r['scheduler_counts'];r['store_balance']={'accepted':counts.get('store_accepted_keys'),'completed':counts.get('store_completed_keys'),'explicit_failed':counts.get('store_failed_keys'),'balanced':counts['store_accepted_keys']==counts['store_completed_keys'] if 'store_accepted_keys' in counts and 'store_completed_keys' in counts else None,'missing_failed_counter':'Unavailable, not assumed zero.'}
  rows[item['name']]=r
 comparisons=[]
 for group in m['pairs']:
  for candidate,control in [('candidate','native8'),('native8','native0'),('candidate','native0')]:
   a=rows[group['runs'][candidate]];b=rows[group['runs'][control]];x={'pair':group['pair'],'partition':group['partition'],'candidate':candidate,'control':control,'candidate_run':a['name'],'control_run':b['name'],'both_eligible':a['eligible'] and b['eligible']}
   if a['attempted'] and b['attempted']:x.update(first_prompts_equal=a['first_prompt_digests']==b['first_prompt_digests'],private_waits_equal=a['private_wait_ms']==b['private_wait_ms'],source_ids_equal=a['source_ids']==b['source_ids'])
   if x['both_eligible']:
    x['differences']=paired(a,b);x['rate_changes_percent']={k:100*math.expm1(v) for k,v in x['differences'].items() if k.endswith('log_ratio')}
   comparisons.append(x)
 outcomes={arm:{'planned':sum(r['arm']==arm for r in rows.values()),'attempted':sum(r['arm']==arm and r['attempted'] for r in rows.values()),'eligible':sum(r['arm']==arm and r['eligible'] for r in rows.values()),'observed_natural_EOS':sum(r.get('natural_eos',0) for r in rows.values() if r['arm']==arm)} for arm in ['native0','native8','candidate']}
 out={'schema':'astra-gated-once-positive-discovery-analysis-v1','manifest_sha256':sha(mp),'analyzer_sha256':sha(Path(__file__)),'rows':rows,'comparisons':comparisons,'arm_outcomes':outcomes,'all36eligible':all(r['eligible'] for r in rows.values()),'confidence_intervals':None,'scope':m['scope'],'provenance_caveat':'Runner raw pins bind helper and workload; supervisor separately archives helper because runner archives only workload. Both archives checked against frozen source and raw pins.','production_endorsement':False,'assessment':'Pending full metrics/source-grounded output review. No single-trace discovery constitutes confirmation.'}
 from analyze_confirmation_astra_nov3 import interval,T975
 T975[3]=3.182446305284263
 inference={}
 if out['all36eligible'] and len(rows)==36:
  for candidate,control in [('candidate','native8'),('native8','native0'),('candidate','native0')]:
   cc=[x for x in comparisons if x['candidate']==candidate and x['control']==control];assert len(cc)==12 and all(x['both_eligible'] and x['first_prompts_equal'] and x['private_waits_equal'] and x['source_ids_equal'] for x in cc)
   inference[candidate+'_vs_'+control]={}
   for label,subset in [('all',cc)]+[('partition_'+str(p),[x for x in cc if x['partition']==p]) for p in [14009,15013,16001]]:
    inference[candidate+'_vs_'+control][label]={k:interval([x['differences'][k] for x in subset],k.endswith('log_ratio')) for k in cc[0]['differences'] if k.endswith(('log_ratio','difference_ms'))}
 out['confidence_intervals']=inference
 out['inference_scope']='Nominaltwo-sided95%pairedwholeworkflow Studentt df11pooled/df3partition, conditional3fixedtraces; no requestpseudoreplication or multiplicityadjustment. Inference withheld unlessall36plannedworkflows eligible.'
 (H/'live-nov3-priority-confirmation-independent-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'all36eligible':out['all36eligible'],'outcomes':outcomes,'once_vs_controls':[{'partition':x['partition'],'control':x['control'],'changes':x.get('rate_changes_percent'),'meanTTFT_ms':x.get('differences',{}).get('mean_ttft_difference_ms'),'p95TTFT_ms':x.get('differences',{}).get('p95_ttft_difference_ms')} for x in comparisons if x['candidate'].startswith('gate')]},indent=2))
if __name__=='__main__':main()
