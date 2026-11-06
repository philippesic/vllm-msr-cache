"""Raw whole-workflow descriptive analysis; no model/tokenizer calls."""
import collections,gzip,hashlib,json,math,re
from pathlib import Path
from analyze_confirmation_astra_nov3 import read,paired
from http_receipt_astra_nov3 import verify_http
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def host(samples):
 vals=[s['host_memory'] for s in samples if 'host_memory' in s]
 def bounds(k):
  a=[v[k] for v in vals if k in v]
  return {'first':a[0],'last':a[-1],'min':min(a),'max':max(a),'delta':a[-1]-a[0]} if a else None
 return {'samples':len(vals),'available_bytes':bounds('available_bytes'),'swap_used_bytes':bounds('swap_used_bytes'),'swap_in_bytes':bounds('swap_in_bytes'),'swap_out_bytes':bounds('swap_out_bytes'),'scope':'System-wide sampled RAM/swap counters; not uniquely attributed to this server.'}
def main():
 planpath=H/'triage-http-grid-frozen-plan-nov3.json';plan=json.loads(planpath.read_text());rows={}
 for item in plan['execution_order']:
  row=read(item)
  if row['attempted']:
   raw=json.load(gzip.open(ROOT/row['raw'],'rt'));row['HTTP_receipt']=verify_http(raw);row['base_workflow_eligible']=row['eligible'];row['eligible']=row['eligible'] and row['HTTP_receipt']['qualified'];row['measurement_scope']='Complete eligible workflow' if row['eligible'] else 'Failed or ineligible workflow: partial metrics descriptive only, no successful-performance comparison';row['final_export_available']='postload_export' in raw;row['final_prometheus_available']='metrics_final' in raw;row['host_resources']={'before':raw.get('host_memory_before'),'startup':host(raw.get('startup_resource_samples',[])),'serving':host(raw.get('resource_samples',[])),'configured_cpu_kv_bytes':raw['geometry']['cpu_kv_bytes'],'actual_locked_pinned_host_bytes':None}
   attrs=row['first_prefill_attribution'];total=row['transfers']['CPU_to_GPU'];external=attrs['tokens']['external_kv_transfer'];blockbytes=raw['geometry']['block_bytes']
   row['restoration_accounting']={'first_admission_external_tokens':external,'total_H2D_bytes':total,'total_H2D_block_token_equivalent':total/blockbytes*16 if total is not None else None,'preemptions':attrs['preemptions'],'H2D_bytes_attributed_specifically_to_preemption':None,'scope':'First-admission external tokens are reported by PrefillStats; total transfer bytes count all measured H2D. No per-transfer reason observer enabled, so preemption-specific bytes unavailable. Do not equate first-admission attribution to total physical prefill work.'}
   q=raw['requests'];row['service_format']={'responses':len(q),'texts_available':sum(bool(x.get('generated_text')) for x in q),'starts_status':sum(x.get('generated_text','').startswith('Status:') for x in q),'single_blank_line_block':sum(bool(x.get('generated_text')) and len(re.split(r'\n\s*\n',x.get('generated_text','').strip()))==1 for x in q),'with_code_fence':sum('```' in x.get('generated_text','') for x in q),'completed_action_phrase_indices':[x['index'] for x in q if re.search(r'\bI (?:have |had )?(?:modified|changed|ran|executed|updated|applied|fixed)\b',x.get('generated_text',''),re.I)],'scope':'Mechanical indicators, not factual-quality or task-success scoring.'}
   counts=row['scheduler_counts'];row['store_balance']={'accepted':counts.get('store_accepted_keys'),'completed':counts.get('store_completed_keys'),'explicit_failed':counts.get('store_failed_keys'),'balanced':counts['store_accepted_keys']==counts['store_completed_keys'] if 'store_accepted_keys' in counts and 'store_completed_keys' in counts else None,'missing_failed_counter':'Unavailable explicitly; not assumed zero.'}
  rows[item['name']]=row
 pairs=[]
 for p in plan['pairs']:
  a=rows[p['runs']['candidate']];b=rows[p['runs']['native']];out={**p,'both_eligible':a['eligible'] and b['eligible']}
  if a['attempted'] and b['attempted']:out.update(first_prompts_equal=a['first_prompt_digests']==b['first_prompt_digests'],private_waits_equal=a['private_wait_ms']==b['private_wait_ms'],source_ids_equal=a['source_ids']==b['source_ids'])
  if out['both_eligible']:
   out['differences']=paired(a,b);out['rate_changes_percent']={k:100*math.expm1(v) for k,v in out['differences'].items() if k.endswith('log_ratio')}
  pairs.append(out)
 report={'schema':'astra-triage-fresh-HTTP-discovery-analysis-v1','plan_sha256':sha(planpath),'extractor_sha256':sha(H/'analyze_confirmation_astra_nov3.py'),'analysis_sha256':sha(Path(__file__)),'rows':rows,'pairs':pairs,'scope':'One matched workflow pergeometry on spent seed1103; all outcomes retained, no inferential interval or production claim.','quality_receipt':'live-nov3-triage-v2-smoke-astra-review.json','limitations':['Natural output lengths vary; interpret workflow rate separately from token rate.','No executed tool task success or autonomous factual correctness.','Human-reviewed draft usefulness/correction cost unmeasured.','First-prefill attribution is not authoritative physical work; preemption-specific transfer bytes unavailable with lean observers disabled.','Matched source/geometry/common normalized control required; cross-geometry memory changes are generic resource changes.']}
 (H/'live-nov3-triage-http-grid-independent-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'pairs':pairs,'rows':{n:{k:r.get(k) for k in ['eligible','requests','natural_eos','load_s','tokens','ttft_ms','restoration_accounting','service_format']} for n,r in rows.items()}},indent=2))
if __name__=='__main__':main()
