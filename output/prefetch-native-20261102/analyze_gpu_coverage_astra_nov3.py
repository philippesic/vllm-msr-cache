"""Raw whole-workflow descriptive analysis; no model/tokenizer calls."""
import collections,gzip,hashlib,json,math,re
from pathlib import Path
from analyze_confirmation_astra_nov3 import read,paired
from http_confirmation_receipt_astra_nov3 import verify_http
from coverage_receipt_astra_nov3 import verify_coverage
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def host(samples):
 vals=[s['host_memory'] for s in samples if 'host_memory' in s]
 def bounds(k):
  a=[v[k] for v in vals if k in v]
  return {'first':a[0],'last':a[-1],'min':min(a),'max':max(a),'delta':a[-1]-a[0]} if a else None
 return {'samples':len(vals),'available_bytes':bounds('available_bytes'),'swap_used_bytes':bounds('swap_used_bytes'),'swap_in_bytes':bounds('swap_in_bytes'),'swap_out_bytes':bounds('swap_out_bytes'),'scope':'System-wide sampled RAM/swap counters; not uniquely attributed to this server.'}
def observed_gpu(samples,field):
 vals=[s['gpu'][field] for s in samples if isinstance(s.get('gpu'),dict) and s['gpu'].get(field) is not None]
 return {'n':len(vals),'min':min(vals),'max':max(vals),'mean':sum(vals)/len(vals)} if vals else None
def bridge_acknowledgements(raw):
 hints=raw.get('hints')
 if not isinstance(hints,list):return {'available':False}
 ack=[h['response'] for h in hints if isinstance(h.get('response'),dict) and h['response'].get('reason')=='cpu_prefix_recency_refreshed']
 fields=['cpu_only_prefix_keys','local_gpu_prefix_keys','local_gpu_coverage_available','gpu_prefix_lookup_performed','additional_ready_cpu_suffix_keys','refreshed_cpu_key_count']
 valid=[x for x in ack if all(k in x for k in fields)];scope='Acknowledgement counts include repeated touches, not unique keys or retained lifetime. GPU lookup performed means helper attempted; unavailable index performs no native scan. Eligibility is not guaranteed future restoration.'
 summary={'available':True,'hint_rows':len(hints),'response_reasons':dict(collections.Counter(h.get('response',{}).get('reason','missing_response') for h in hints)),'mode_acknowledgements':len(ack),'bridge_acknowledgements':len(valid),'all_mode_ack_fields_present':len(valid)==len(ack) if ack else None,'scope':scope}
 if not valid:
  summary['bridge_counts']=None;return summary
 summary['bridge_counts']={'helper_attempts':sum(x['gpu_prefix_lookup_performed'] is True for x in valid),'helper_not_attempted':sum(x['gpu_prefix_lookup_performed'] is False for x in valid),'coverage_available_true':sum(x['local_gpu_coverage_available'] is True for x in valid),'coverage_unavailable_false':sum(x['local_gpu_coverage_available'] is False for x in valid),'coverage_not_observed_null':sum(x['local_gpu_coverage_available'] is None for x in valid),'positive_suffix_acknowledgements':sum(x['additional_ready_cpu_suffix_keys']>0 for x in valid),'eligible_suffix_key_touch_sum':sum(x['additional_ready_cpu_suffix_keys'] for x in valid),'actual_refreshed_key_touch_sum':sum(x['refreshed_cpu_key_count'] for x in valid),'cpu_only_prefix_key_touch_sum':sum(x['cpu_only_prefix_keys'] for x in valid),'GPU_prefix_key_observation_sum':sum(x['local_gpu_prefix_keys'] for x in valid if x['local_gpu_prefix_keys'] is not None),'idempotent_acknowledgements':sum(x.get('idempotent_replay') is True for x in valid)}
 summary['positive_suffix_rows']=[{'event_id':h['hint'].get('event_id'),'session_id':h['hint'].get('session_id'),'response':h['response']} for h in hints if h.get('response',{}).get('additional_ready_cpu_suffix_keys',0)>0]
 return summary
def main():
 planpath=H/'gpu-coverage-frozen-plan-nov3.json';plan=json.loads(planpath.read_text());rows={}
 for item in plan['execution_order']:
  row=read(item)
  if row['attempted']:
   raw=json.load(gzip.open(ROOT/row['raw'],'rt'));row['HTTP_receipt']=verify_http(raw);row['base_workflow_eligible']=row['eligible'];row['eligible']=row['eligible'] and row['HTTP_receipt']['qualified'];row['measurement_scope']='Complete eligible workflow' if row['eligible'] else 'Failed or ineligible workflow: partial metrics descriptive only, no successful-performance comparison';row['final_export_available']='postload_export' in raw;row['final_prometheus_available']='metrics_final' in raw;row['host_resources']={'before':raw.get('host_memory_before'),'startup':host(raw.get('startup_resource_samples',[])),'serving':host(raw.get('resource_samples',[])),'configured_cpu_kv_bytes':raw['geometry']['cpu_kv_bytes'],'actual_locked_pinned_host_bytes':None}
   attrs=row['first_prefill_attribution'];total=row['transfers']['CPU_to_GPU'];external=attrs['tokens']['external_kv_transfer'];blockbytes=raw['geometry']['block_bytes']
   row['restoration_accounting']={'first_admission_external_tokens':external,'total_H2D_bytes':total,'total_H2D_block_token_equivalent':total/blockbytes*16 if total is not None else None,'preemptions':attrs['preemptions'],'H2D_bytes_attributed_specifically_to_preemption':None,'scope':'First-admission external tokens are reported by PrefillStats; total transfer bytes count all measured H2D. No per-transfer reason observer enabled, so preemption-specific bytes unavailable. Do not equate first-admission attribution to total physical prefill work.'}
   row['measurement']['readmission_events']=len(raw['readmission_observer_events']) if 'readmission_observer_events' in raw else None
   row['measurement']['lookup_observer_events']=len(raw['native_lookup_deferral_events']) if 'native_lookup_deferral_events' in raw else None
   row['thermal_clock_power']={field:{'startup':observed_gpu(raw.get('startup_resource_samples',[]),field),'serving':observed_gpu(raw.get('resource_samples',[]),field)} for field in ['temperature_c','clock_mhz','power_w']}
   q=raw['requests'];row['service_format']={'responses':len(q),'texts_available':sum(bool(x.get('generated_text')) for x in q),'starts_status':sum(x.get('generated_text','').startswith('Status:') for x in q),'single_blank_line_block':sum(bool(x.get('generated_text')) and len(re.split(r'\n\s*\n',x.get('generated_text','').strip()))==1 for x in q),'with_code_fence':sum('```' in x.get('generated_text','') for x in q),'completed_action_phrase_indices':[x['index'] for x in q if re.search(r'\bI (?:have |had )?(?:modified|changed|ran|executed|updated|applied|fixed)\b',x.get('generated_text',''),re.I)],'scope':'Mechanical indicators, not factual-quality or task-success scoring.'}
   counts=row['scheduler_counts'];row['store_balance']={'accepted':counts.get('store_accepted_keys'),'completed':counts.get('store_completed_keys'),'explicit_failed':counts.get('store_failed_keys'),'balanced':counts['store_accepted_keys']==counts['store_completed_keys'] if 'store_accepted_keys' in counts and 'store_completed_keys' in counts else None,'missing_failed_counter':'Unavailable explicitly; not assumed zero.'}
   row['coverage_receipt']=verify_coverage(raw)
   row['hint_response_reasons']=dict(collections.Counter(h.get('response',{}).get('reason','missing_response') for h in raw.get('hints',[])))
   row['cadence_receipt']=row['coverage_receipt']['cadence']
   row['eligible']=row['eligible'] and row['coverage_receipt']['qualified']
   row['configuration_activation']={'requested_gpu_aware':raw['args'].get('gpu_aware_retention'),'configured_gpu_aware':raw['config']['kv_connector_extra_config'].get('wait_hint_cpu_retention_gpu_aware',False)}
  rows[item['name']]=row
 report={'schema':'astra-spent-input-observation-only-GPU-coverage-v1','plan_sha256':sha(planpath),'extractor_sha256':sha(H/'analyze_confirmation_astra_nov3.py'),'analysis_sha256':sha(Path(__file__)),'rows':rows,'performance_eligible':False,'comparisons':[],'scope':'Three candidate-only observations on spent inputs. Queue scan changes measurement overhead; no policy, speed or production inference.','limitations':['Coverage is contemporaneous, repeated per-hint observations; not unique data or future cache survival.','Free/active counts partition the contiguous cached GPU prefix, not necessarily contiguous subprefixes.','Full CPU/GPU overlap is duplication at observation time, not evidence that CPU protection is wasteful after later GPU eviction.','Missing GPU coverage remains unavailable, not zero; cached replay excluded from fresh snapshots.','First-admission tokens are not cumulative executed GPU work.']}
 (H/'live-nov3-gpu-coverage-independent-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'rows':{k:{'status':v.get('status'),'natural_eos':v.get('natural_eos'),'coverage':v.get('coverage_receipt',{}).get('per_hint')} for k,v in rows.items()}},indent=2))
if __name__=='__main__':main()
