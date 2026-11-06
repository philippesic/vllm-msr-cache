"""Independent frozen-manifest analysis. No request-level pseudo replication."""
import collections,gzip,hashlib,json,math,re,statistics
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
M=ROOT/'evidence/live-optimization-2080ti-20261103/confirmation-frozen.json'
T975={5:2.570581835636314,11:2.200985160082949}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dist(values):
 if not values:return {'n':0,'mean':None,'p95':None}
 v=sorted(values);x=(len(v)-1)*.95;i=int(x)
 return {'n':len(v),'mean':statistics.mean(v),'p95':v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)}
def counter(text,name,label=None):
 values=[]
 for line in text.splitlines():
  if not (line.startswith(name+'{') or line.startswith(name+' ')):continue
  if label and label not in line:continue
  values.append(float(line.rsplit(' ',1)[1]))
 return sum(values) if values else None
def delta(r,name,label=None):
 a=counter(r.get('metrics_initial',''),name,label);b=counter(r.get('metrics_final',''),name,label)
 return b-a if a is not None and b is not None else None
def resource(r):
 samples=r.get('resource_samples',[]);startup=r.get('startup_resource_samples',[])
 identities=collections.defaultdict(list)
 for sample in samples:
  for p in sample.get('processes',[]):identities[p['identity']].append(p['cpu_s'])
 return {'scope':'Observed serving samples only, 250ms polling; CPU time differences within each process observed window; summed RSS may double count shared pages.',
 'sample_count':len(samples),'sample_window_s':(samples[-1]['at_ns']-samples[0]['at_ns'])/1e9 if samples else None,
 'sampled_server_child_cpu_s':sum(max(v)-min(v) for v in identities.values()) if samples else None,
 'sampled_driver_cpu_s':samples[-1]['driver_cpu_s']-samples[0]['driver_cpu_s'] if samples else None,
 'sampled_server_summed_rss_peak_mib':max((sum(p['rss_bytes'] for p in x.get('processes',[])) for x in samples),default=0)/2**20 if samples else None,
 'sampled_driver_rss_peak_mib':max((x.get('driver_rss_bytes',0) for x in samples),default=0)/2**20 if samples else None,
 'serving_nvml_peak_mib':max((x.get('gpu',{}).get('memory_used_bytes',0) for x in samples),default=0)/2**20 if samples else None,
 'startup_nvml_peak_mib':max((x.get('gpu',{}).get('memory_used_bytes',0) for x in startup),default=0)/2**20 if startup else None,
 'cuda_load_peaks':[{k:w.get(k) for k in ['gpu_allocated_bytes','gpu_reserved_bytes','load_peak_allocated_bytes','load_peak_reserved_bytes']} for w in r.get('postload_export',{}).get('workers',[])]}
def degeneracy(q):
 text=q.get('generated_text','');words=text.split();grams=collections.Counter(tuple(words[i:i+8]) for i in range(max(0,len(words)-7)));lines=collections.Counter(x.strip() for x in text.splitlines() if len(x.strip())>=12)
 repeat=max(grams.values(),default=0);line_repeat=max(lines.values(),default=0);ids=q.get('output_token_ids',[]);token_repeat=max(collections.Counter(tuple(ids[i:i+8]) for i in range(max(0,len(ids)-7))).values(),default=0)
 return {'index':q.get('index'),'turn':q.get('turn'),'session_id':q.get('session_id'),'tokens':len(q.get('output_token_ids',[])),'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'max_repeated_8_word_phrase':repeat,'max_repeated_nontrivial_line':line_repeat,'max_repeated_8_token_phrase':token_repeat,'screen_flag':len(q.get('output_token_ids',[]))>=256 and (repeat>=10 or line_repeat>=8 or token_repeat>=20),'screen_scope':'Heuristic inspection aid; neither a quality score nor an automatic exclusion.'}
def read(item):
 p=H/'runs'/item['name']/'raw.json.gz'
 if not p.exists():return {**item,'attempted':False,'eligible':False}
 r=json.load(gzip.open(p,'rt'));q=r.get('requests',[]);planned=r.get('planned_requests');duration=(r['load_end_ns']-r['load_start_ns'])/1e9 if 'load_end_ns' in r and 'load_start_ns' in r else None
 eligible=bool(r.get('performance_eligible') and r.get('status')=='complete' and len(q)==planned and q and all(x.get('success') and x.get('natural_eos') for x in q))
 outputs=sum(len(x.get('output_token_ids',[])) for x in q);logical=sum(len(x['payload']['prompt']) for x in q);res=resource(r);nvml=res['serving_nvml_peak_mib'];tokens_s=outputs/duration if duration and duration>0 else None
 attrs={s:delta(r,'vllm:prompt_tokens_by_source_total',f'source="{s}"') for s in ['local_compute','local_cache_hit','external_kv_transfer']};present=all(v is not None for v in attrs.values());s=r.get('postload_export',{}).get('scheduler',{});counts=s.get('counts',{})
 expected={'lean':True,'residency_scans':False,'readmission_observer':False,'lookup_deferral_observer':False,'finished_token_clamp':True}
 screens=[degeneracy(x) for x in q]
 return {**item,'attempted':True,'raw':str(p.relative_to(ROOT)),'raw_sha256':digest(p),'status':r.get('status'),'error':r.get('error'),'eligible':eligible,'requests':len(q),'planned':planned,'natural_eos':sum(bool(x.get('natural_eos')) for x in q),'length_terminations':sum(x.get('finish_reasons')==['length'] for x in q),'load_s':duration,'tokens':outputs,'logical_prompt_tokens':logical,'workflow_rate_s':1/duration if duration else None,'tokens_s':tokens_s,'tokens_s_per_gib':tokens_s/(nvml/1024) if tokens_s and nvml else None,
 'ttft_ms':dist([(x['first_token_ns']-x['request_start_ns'])/1e6 for x in q if 'first_token_ns' in x]),'e2e_ms':dist([(x['request_end_ns']-x['request_start_ns'])/1e6 for x in q if 'request_end_ns' in x]),'resources':res,
 'first_prefill_attribution':{'tokens':attrs,'all_series_present':present,'reconciles':sum(attrs.values())==logical if present else None,'preemptions':delta(r,'vllm:num_preemptions_total'),'scope':'Initial admission attribution only; not cumulative physical executed work or GPU time.'},
 'transfers':{d:delta(r,'vllm:kv_offload_total_bytes_total',f'transfer_type="{d}"') for d in ['GPU_to_CPU','CPU_to_GPU']},
 'scheduler_counts':{k:v for k,v in counts.items() if k.startswith(('store_','lookup_','cpu_','finished_token_clamp'))},
 'measurement':{'executed':s.get('measurement_configuration'),'matches_frozen_lean':s.get('measurement_configuration')==expected,'residency_samples':len(s.get('samples',[])) if 'samples' in s else None,'readmission_events':len(r.get('readmission_observer_events',[])),'lookup_observer_events':len(r.get('native_lookup_deferral_events',[]))},
 'pins':{k:r.get(k) for k in ['source_pins','runtime_pins','pins','harness_pins','artifact_pins','native_scheduler_source','model_identity']},'owned_descendants_remaining':r.get('owned_descendants_remaining'),
 'first_prompt_digests':{str(x['index']):hashlib.sha256(json.dumps(x['payload']['prompt']).encode()).hexdigest() for x in q if x.get('turn')==0},
 'private_wait_ms':r.get('workload',{}).get('private_wait_ms'),'source_ids':r.get('workload',{}).get('source_ids'),
 'output_screen':{'flagged':sum(x['screen_flag'] for x in screens),'rows':screens,'longest':sorted(screens,key=lambda x:x['tokens'],reverse=True)[:5]}}
def paired(a,b):
 return {'workflow_rate_log_ratio':math.log(b['load_s']/a['load_s']),'token_rate_log_ratio':math.log(a['tokens_s']/b['tokens_s']),'efficiency_log_ratio':math.log(a['tokens_s_per_gib']/b['tokens_s_per_gib']),'generated_tokens_difference':a['tokens']-b['tokens'],'mean_ttft_difference_ms':a['ttft_ms']['mean']-b['ttft_ms']['mean'],'p95_ttft_difference_ms':a['ttft_ms']['p95']-b['ttft_ms']['p95'],'mean_e2e_difference_ms':a['e2e_ms']['mean']-b['e2e_ms']['mean'],'p95_e2e_difference_ms':a['e2e_ms']['p95']-b['e2e_ms']['p95'],
 'serving_nvml_difference_mib':a['resources']['serving_nvml_peak_mib']-b['resources']['serving_nvml_peak_mib'],'startup_nvml_difference_mib':a['resources']['startup_nvml_peak_mib']-b['resources']['startup_nvml_peak_mib'],
 'sampled_server_cpu_difference_s':a['resources']['sampled_server_child_cpu_s']-b['resources']['sampled_server_child_cpu_s'],'sampled_driver_cpu_difference_s':a['resources']['sampled_driver_cpu_s']-b['resources']['sampled_driver_cpu_s'],'sampled_server_rss_difference_mib':a['resources']['sampled_server_summed_rss_peak_mib']-b['resources']['sampled_server_summed_rss_peak_mib']}
def interval(values,log=False):
 n=len(values);mu=statistics.mean(values);sd=statistics.stdev(values);half=T975[n-1]*sd/math.sqrt(n);lo=mu-half;hi=mu+half
 d={'n':n,'df':n-1,'mean':mu,'sample_sd':sd,'two_sided_95_percent':[lo,hi],'values':values,'method':'Student t on paired whole-workflow observations'}
 if log:d.update(geometric_mean_ratio=math.exp(mu),relative_change_percent=100*math.expm1(mu),relative_change_95_percent=[100*math.expm1(lo),100*math.expm1(hi)])
 return d
def main():
 m=json.loads(M.read_text());rows={x['name']:read(x) for x in m['execution_order']};pairs=[]
 for p in m['pairs']:
  a=rows[p['runs']['candidate']];b=rows[p['runs']['native']];out={**p,'both_eligible':a['eligible'] and b['eligible']}
  if a['attempted'] and b['attempted']:out.update(first_prompts_equal=a['first_prompt_digests']==b['first_prompt_digests'],private_waits_equal=a['private_wait_ms']==b['private_wait_ms'])
  if out['both_eligible']:out['differences']=paired(a,b)
  pairs.append(out)
 clean=len(pairs)==12 and all(p['both_eligible'] for p in pairs);analysis={}
 if clean:
  for label,subset in [('all',pairs),('partition_2207',[p for p in pairs if p['partition']==2207]),('partition_3301',[p for p in pairs if p['partition']==3301])]:
   analysis[label]={}
   for key in pairs[0]['differences']:
    values=[p['differences'][key] for p in subset]
    analysis[label][key]=interval(values,key.endswith('log_ratio')) if key.endswith(('log_ratio','difference_ms')) else {'n':len(values),'mean':statistics.mean(values),'min':min(values),'max':max(values),'values':values,'scope':'Descriptive resource/output differences; no additional inferential claim.'}
 anchors=[]
 for x in m['execution_order']:
  if x['kind']!='descriptive_historical_anchor':continue
  a=rows[x['candidate_reference']];b=rows[x['name']];out={'candidate':a['name'],'anchor':b['name'],'eligible':a['eligible'] and b['eligible'],'scope':'Single predeclared comparison, no confidence interval; combines generic graph/resource/runtime changes with extension.'}
  if out['eligible']:out['differences']=paired(a,b)
  anchors.append(out)
 arms={}
 for arm in ['native','candidate','original_eager']:
  subset=[r for r in rows.values() if r['arm']==arm];attempted=[r for r in subset if r['attempted']]
  arms[arm]={'planned_runs':len(subset),'attempted_runs':len(attempted),'eligible_runs':sum(r['eligible'] for r in subset),'observed_natural_eos':sum(r.get('natural_eos',0) for r in subset),'observed_requests':sum(r.get('requests',0) for r in subset),'all_attempted_load_s':sum(r.get('load_s') or 0 for r in subset),'all_attempted_output_tokens':sum(r.get('tokens',0) for r in subset)}
 report={'schema':'astra-independent-frozen-confirmation-analysis-v1','manifest_sha256':digest(M),'analysis_generator_sha256':digest(Path(__file__)),'clean_12_pair_confirmation':clean,'primary_intervals_available':clean,'interval_policy':'Predeclared 12-pair and 6-pair partition t intervals only when every planned pair is eligible; incomplete comparisons are not selectively pooled.','scope':m['analysis']['inference_scope'],'rows':rows,'pairs':pairs,'paired_analysis':analysis,'descriptive_anchors':anchors,'arm_outcomes':arms,'production_assessment':'Pending independent inspection of completed results; successful execution alone is not endorsement.'}
 (H/'live-nov3-confirmation-independent-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'clean':clean,'arm_outcomes':arms,'primary':analysis.get('all',{}).get('workflow_rate_log_ratio'),'ttft_mean':analysis.get('all',{}).get('mean_ttft_difference_ms'),'ttft_p95':analysis.get('all',{}).get('p95_ttft_difference_ms')},indent=2))
if __name__=='__main__':main()
