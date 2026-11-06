"""All-outcome three-arm discovery analysis; no inferential intervals."""
import collections,gzip,hashlib,json,math,re
from pathlib import Path
from analyze_confirmation_astra_nov3 import read,paired,delta
from analyze_triage_confirmation_astra_nov3 import host,observed_gpu
from gate_once_receipt_astra_nov3 import verify_gate_once
from http_confirmation_receipt_astra_nov3 import verify_http
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 mp=H/'gate-once-frozen-plan-nov3.json';m=json.loads(mp.read_text());rows={}
 for item in m['execution_order']:
  r=read(item)
  if r['attempted']:
   raw=json.load(gzip.open(ROOT/r['raw'],'rt'));r['HTTP_receipt']=verify_http(raw);r['gate_once_receipt']=verify_gate_once(raw,item['arm']) if r['eligible'] else {'qualified':False,'issues':['incomplete_workflow']}
   r['eligible']=r['eligible'] and r['HTTP_receipt']['qualified'] and r['gate_once_receipt']['qualified'] and r['owned_descendants_remaining']==[] and r['measurement']['matches_frozen_lean']
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
  for candidate,control in [('periodic','native'),('once_positive','native'),('once_positive','periodic')]:
   a=rows[group['runs'][candidate]];b=rows[group['runs'][control]];x={'partition':group['partition'],'candidate':candidate,'control':control,'candidate_run':a['name'],'control_run':b['name'],'both_eligible':a['eligible'] and b['eligible']}
   if a['attempted'] and b['attempted']:x.update(first_prompts_equal=a['first_prompt_digests']==b['first_prompt_digests'],private_waits_equal=a['private_wait_ms']==b['private_wait_ms'],source_ids_equal=a['source_ids']==b['source_ids'])
   if x['both_eligible']:
    x['differences']=paired(a,b);x['rate_changes_percent']={k:100*math.expm1(v) for k,v in x['differences'].items() if k.endswith('log_ratio')}
   comparisons.append(x)
 outcomes={arm:{'planned':sum(r['arm']==arm for r in rows.values()),'attempted':sum(r['arm']==arm and r['attempted'] for r in rows.values()),'eligible':sum(r['arm']==arm and r['eligible'] for r in rows.values()),'observed_natural_EOS':sum(r.get('natural_eos',0) for r in rows.values() if r['arm']==arm)} for arm in ['native','periodic','once_positive']}
 out={'schema':'astra-gated-once-positive-discovery-analysis-v1','manifest_sha256':sha(mp),'analyzer_sha256':sha(Path(__file__)),'rows':rows,'comparisons':comparisons,'arm_outcomes':outcomes,'all18eligible':all(r['eligible'] for r in rows.values()),'confidence_intervals':None,'scope':m['scope'],'provenance_caveat':'Raw driver pins retain periodic workload entry for once; explicit selected archived workload hash and frozen manifest/args/executed producer receipt are checked separately.','production_endorsement':False,'assessment':'Pending full metrics/source-grounded output review. No single-trace discovery constitutes confirmation.'}
 (H/'live-nov3-gate-once-independent-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'all18eligible':out['all18eligible'],'outcomes':outcomes,'once_vs_controls':[{'partition':x['partition'],'control':x['control'],'changes':x.get('rate_changes_percent'),'meanTTFT_ms':x.get('differences',{}).get('mean_ttft_difference_ms'),'p95TTFT_ms':x.get('differences',{}).get('p95_ttft_difference_ms')} for x in comparisons if x['candidate']=='once_positive']},indent=2))
if __name__=='__main__':main()
