"""Independent all-outcome persistent-engine operational analysis; no CI."""
import collections,gzip,hashlib,json,re,statistics
from pathlib import Path
from analyze_confirmation_astra_nov3 import dist,delta,resource,degeneracy
from analyze_triage_confirmation_astra_nov3 import host,observed_gpu
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
M=ROOT/'evidence/live-optimization-2080ti-20261103/operational-v2-frozen.json'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def distribution(xs):
    result=dist(xs);result['max']=max(xs) if xs else None;return result

def scalar_exports(raw):
    exports=[]
    for wave in raw.get('waves',[]):
        for stage in ('immediate_export','settled_export'):
            value=wave.get(stage)
            if isinstance(value,dict):exports.append({'wave':wave['wave_index'],'stage':stage,'export':value})
    for stage in ('overlap_settled_export','postload_export'):
        if isinstance(raw.get(stage),dict):exports.append({'wave':None,'stage':stage,'export':raw[stage]})
    probe=raw.get('cancellation_probe',{})
    for stage in ('after_health_export','settled_after_health_export'):
        if isinstance(probe.get(stage),dict):exports.append({'wave':None,'stage':stage,'export':probe[stage]})
    return exports


def request_rows(wave):
    rows=[];issues=[];byindex={x['index']:x for x in wave.get('requests',[])}
    contiguous={x['index']:x for x in wave.get('prefix_contiguity',[])}
    for q in wave.get('requests',[]):
        turn=q['turn'];wait=None;same=None;category='initial'
        if turn:
            wait=wave.get('workload',{}).get('private_wait_ms',{}).get(f"{q['index']//4}:{turn-1}")
            category='return_zero_wait' if wait==0 else 'return_positive_wait' if isinstance(wait,int) and wait>0 else 'return_wait_unavailable'
            previous=byindex.get(q['index']-1)
            if previous:
                prefix=previous['payload']['prompt']+previous.get('output_token_ids',[])[:-1]
                prefix=prefix[:len(prefix)//16*16]
                same=q['payload']['prompt'][:len(prefix)]==prefix
                recorded=contiguous.get(q['index'])
                if not recorded or recorded['contiguous'] is not same or recorded['previous_computed_fullblock_tokens']!=len(prefix):issues.append('contiguity_receipt_mismatch:'+str(q['index']))
            else:issues.append('missing_prior_request:'+str(q['index']))
        rows.append({'wave':wave['wave_index'],'index':q['index'],'turn':turn,'category':category,'private_wait_ms':wait,'return_contiguous':same,
                     'natural_eos':q.get('natural_eos'),'output_tokens':len(q.get('output_token_ids',[])),
                     'prompt_tokens':len(q.get('payload',{}).get('prompt',[])),
                     'TTFT_ms':(q['first_token_ns']-q['request_start_ns'])/1e6 if q.get('first_token_ns') is not None else None,
                     'E2E_ms':(q['request_end_ns']-q['request_start_ns'])/1e6 if q.get('request_end_ns') is not None else None})
    return rows,issues


def summarize_requests(rows):
    return {'requests':len(rows),'natural_eos':sum(q['natural_eos'] is True for q in rows),
            'output_tokens':sum(q['output_tokens'] for q in rows),'logical_prompt_tokens':sum(q['prompt_tokens'] for q in rows),
            'TTFT_ms':distribution([q['TTFT_ms'] for q in rows if q['TTFT_ms'] is not None]),
            'E2E_ms':distribution([q['E2E_ms'] for q in rows if q['E2E_ms'] is not None]),
            'return_contiguity':dict(collections.Counter(str(q['return_contiguous']) for q in rows if q['turn'])),
            'categories':{c:{'requests':len(xs),'TTFT_ms':distribution([q['TTFT_ms'] for q in xs if q['TTFT_ms'] is not None]),'E2E_ms':distribution([q['E2E_ms'] for q in xs if q['E2E_ms'] is not None])} for c in ['initial','return_zero_wait','return_positive_wait','return_wait_unavailable'] if (xs:=[q for q in rows if q['category']==c])}}


def process_trends(raw,exports):
    samples=raw.get('resource_samples',[]);series=collections.defaultdict(list)
    for sample in samples:
        for proc in sample.get('processes',[]):series[proc['identity']].append({'at_ns':sample['at_ns'],**proc})
    identities={};quiescent=[]
    for identity,values in series.items():
        lastquarter=values[len(values)*3//4:];rss=[v['rss_bytes']/2**20 for v in values]
        def slope(points):
            if len(points)<2:return None
            x=[(v['at_ns']-points[0]['at_ns'])/60e9 for v in points];y=[v['rss_bytes']/2**20 for v in points]
            xm=statistics.mean(x);ym=statistics.mean(y);den=sum((t-xm)**2 for t in x)
            return sum((t-xm)*(r-ym) for t,r in zip(x,y))/den if den else None
        identities[identity]={'samples':len(values),'first_RSS_MiB':rss[0],'last_RSS_MiB':rss[-1],'min_RSS_MiB':min(rss),'peak_RSS_MiB':max(rss),'first_to_last_MiB':rss[-1]-rss[0],
                              'whole_sample_slope_MiB_per_minute':slope(values),'last_quarter_slope_MiB_per_minute':slope(lastquarter),
                              'last_quarter_range_MiB':(max(v['rss_bytes'] for v in lastquarter)-min(v['rss_bytes'] for v in lastquarter))/2**20,
                              'observed_CPU_seconds':values[-1]['cpu_s']-values[0]['cpu_s']}
    for e in exports:
        if 'settled' not in e['stage']:continue
        scalar=e['export'].get('scheduler',{}).get('operational_scalar_state',{});stamp=scalar.get('at_ns')
        nearest=min(samples,key=lambda x:abs(x['at_ns']-stamp)) if samples and stamp is not None else None
        quiescent.append({'wave':e['wave'],'stage':e['stage'],'export_at_ns':stamp,'nearest_sample_offset_ms':(nearest['at_ns']-stamp)/1e6 if nearest else None,
                          'process_RSS_MiB':{p['identity']:p['rss_bytes']/2**20 for p in nearest.get('processes',[])} if nearest else None,
                          'driver_RSS_MiB':nearest['driver_rss_bytes']/2**20 if nearest and 'driver_rss_bytes' in nearest else None,
                          'NVML_MiB':nearest['gpu']['memory_used_bytes']/2**20 if nearest and isinstance(nearest.get('gpu'),dict) else None})
    return {'per_process':identities,'quiescent_samples':quiescent,'scope':'Descriptive finite RSS slopes/ranges, not a proof of asymptotic plateau. Separate process RSS may share pages; driver retains benchmark evidence intentionally.'}


def metadata(raw,exports):
    timelines=[];values=collections.defaultdict(list);issues=[]
    for e in exports:
        scheduler=e['export'].get('scheduler',{}).get('operational_scalar_state')
        workers=[w.get('operational_worker_state') for w in e['export'].get('workers',[])]
        row={'wave':e['wave'],'stage':e['stage'],'scheduler':scheduler,'workers':workers};timelines.append(row)
        for role,record in [('scheduler',scheduler)]+[('worker'+str(i),w) for i,w in enumerate(workers)]:
            if not isinstance(record,dict):issues.append('missing_scalar:'+role+':'+e['stage']);continue
            for group in ('scheduler_lengths','manager_lengths','worker_lengths'):
                for k,v in record.get(group,{}).items():
                    if v is not None:values[role+'/'+group+'/'+k].append(v)
    return {'timeline':timelines,'observed_ranges':{k:{'first':v[0],'last':v[-1],'min':min(v),'max':max(v)} for k,v in values.items()},'issues':issues,
            'scope':'Missing fields stay unavailable; no residency enumeration. Nonzero operational metadata and audit history are distinguished in assessment.'}


def main():
    manifest=json.loads(M.read_text());reports={}
    for item in manifest['execution_order']:
        path=H/'runs'/item['name']/'raw.json.gz';row={**item,'attempted':path.exists()}
        if not path.exists():reports[item['name']]=row;continue
        raw=json.load(gzip.open(path,'rt'));exports=scalar_exports(raw);waves=[];all_requests=[];issues=[]
        for wave in raw.get('waves',[]):
            qs,wi=request_rows(wave);issues.extend(wi);all_requests.extend(qs);w=summarize_requests(qs)
            duration=(wave['load_end_ns']-wave['load_start_ns'])/1e9 if 'load_end_ns' in wave and 'load_start_ns' in wave else None
            w.update(wave=wave['wave_index'],seed=wave['seed'],status=wave.get('status'),load_s=duration,tokens_per_s=w['output_tokens']/duration if duration else None,
                     source_ids=wave.get('workload',{}).get('source_ids'),private_waits=wave.get('workload',{}).get('private_wait_ms'),
                     first_prompt_digests={q['index']:hashlib.sha256(json.dumps(q['payload']['prompt']).encode()).hexdigest() for q in wave.get('requests',[]) if q['turn']==0})
            waves.append(w)
        normal=summarize_requests(all_requests)
        starts=[w['load_start_ns'] for w in raw.get('waves',[]) if 'load_start_ns' in w];ends=[w['load_end_ns'] for w in raw.get('waves',[]) if 'load_end_ns' in w]
        normal_window=(max(ends)-min(starts))/1e9 if starts and ends else None
        rs=resource(raw)
        if not any(isinstance(s.get('gpu'),dict) and s['gpu'].get('memory_used_bytes') is not None for s in raw.get('resource_samples',[])):rs['serving_nvml_peak_mib']=None
        if not any(isinstance(s.get('gpu'),dict) and s['gpu'].get('memory_used_bytes') is not None for s in raw.get('startup_resource_samples',[])):rs['startup_nvml_peak_mib']=None
        final_export=raw.get('postload_export')
        final_export_scope='postload_export'
        if not isinstance(final_export,dict) and exports:
            final_export=max(exports,key=lambda e:e['export'].get('scheduler',{}).get('operational_scalar_state',{}).get('at_ns',-1))['export']
            final_export_scope='latest retained boundary export; final Prometheus metrics remain unavailable if omitted after case failure'
            rs['cuda_load_peaks']=[{k:w.get(k) for k in ['gpu_allocated_bytes','gpu_reserved_bytes','load_peak_allocated_bytes','load_peak_reserved_bytes']} for w in final_export.get('workers',[])]
        nvml=rs['serving_nvml_peak_mib']
        normal.update(normal_arrival_to_final_completion_s=normal_window,
                      normal_tokens_per_s=normal['output_tokens']/normal_window if normal_window else None,
                      normal_requests_per_s=normal['requests']/normal_window if normal_window else None,
                      normal_workflows_per_s=len(waves)/normal_window if normal_window else None,
                      normal_tokens_per_s_per_GiB=normal['output_tokens']/normal_window/(nvml/1024) if normal_window and nvml else None)
        probe=raw.get('cancellation_probe',{});health=probe.get('health_requests',[]);abort=probe.get('abort_request')
        all_submitted_prompts=normal['logical_prompt_tokens']+sum(len(q['payload']['prompt']) for q in health)+(len(abort['payload']['prompt']) if abort else 0)
        attribution={s:delta(raw,'vllm:prompt_tokens_by_source_total',f'source="{s}"') for s in ['local_compute','local_cache_hit','external_kv_transfer']}
        cumulative_scope='Whole measured case including health and deliberate-abort admissions where present; per-wave Prometheus counters unavailable. First-admission attribution is not physical cumulative prefill or GPU time.'
        screens=[dict(degeneracy(q),wave=w['wave_index']) for w in raw.get('waves',[]) for q in w.get('requests',[])]
        text_rows=[q for w in raw.get('waves',[]) for q in w.get('requests',[])]
        receipt_path=H/(item['name']+'-basic-receipt.json')
        row.update(raw_sha256=sha(path),status=raw.get('status'),error=raw.get('error'),owned_descendants_remaining=raw.get('owned_descendants_remaining'),
                   normal=normal,waves=waves,contiguity_issues=issues,resources=rs,process_memory=process_trends(raw,exports),metadata=metadata(raw,exports),
                   host_resources={'startup':host(raw.get('startup_resource_samples',[])),'serving':host(raw.get('resource_samples',[])),'configured_CPU_KV_bytes':raw['geometry']['cpu_kv_bytes'],'actual_locked_host_bytes':None},
                   thermal={f:observed_gpu(raw.get('resource_samples',[]),f) for f in ['temperature_c','clock_mhz','power_w']},
                   total_native_transfers={d:{'bytes':delta(raw,'vllm:kv_offload_total_bytes_total',f'transfer_type="{d}"'),'jobs':delta(raw,'vllm:kv_offload_size_count',f'transfer_type="{d}"')} for d in ['GPU_to_CPU','CPU_to_GPU']},
                   first_admission={'tokens':attribution,'all_submitted_prompt_tokens':all_submitted_prompts,'reconciles_all_submitted':sum(attribution.values())==all_submitted_prompts if all(v is not None for v in attribution.values()) else None,'preemptions':delta(raw,'vllm:num_preemptions_total'),'scope':cumulative_scope},
                   scheduler_counts=(final_export or {}).get('scheduler',{}).get('counts'),final_export_scope=final_export_scope,final_Prometheus_available='metrics_final' in raw,
                   basic_receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None,
                   cancellation_probe=probe,
                   format={'responses':len(text_rows),'starts_status':sum(q.get('generated_text','').startswith('Status:') for q in text_rows),'code_fences':sum('```' in q.get('generated_text','') for q in text_rows),'multiple_blank_blocks':sum(len(re.split(r'\n\s*\n',q.get('generated_text','').strip()))>1 for q in text_rows),'degeneracy_flags':[x for x in screens if x['screen_flag']],'longest':sorted(screens,key=lambda x:x['tokens'],reverse=True)[:5]},
                   performance_inference=False)
        reports[item['name']]=row
    comparisons=[]
    for phase in ('B','C','E'):
        pairs={r['arm']:r for r in reports.values() if r['phase']==phase}
        a=pairs['candidate'];b=pairs['native8'];x={'phase':phase,'candidate':a['name'],'control':b['name'],'both_attempted':a['attempted'] and b['attempted']}
        if x['both_attempted']:
            x['complete_qualified_pair']=all(r.get('status')=='complete' and r.get('basic_receipt',{}).get('qualified') for r in (a,b))
            x['descriptive_changes']={}
            for key in ('normal_tokens_per_s','normal_workflows_per_s','normal_tokens_per_s_per_GiB'):
                av=a['normal'][key];bv=b['normal'][key]
                x['descriptive_changes'][key+'_percent']=100*(av/bv-1) if av and bv else None
            for label in ('TTFT_ms','E2E_ms'):
                for stat in ('mean','p95','max'):
                    av=a['normal'][label][stat];bv=b['normal'][label][stat]
                    x['descriptive_changes'][label+'_'+stat+'_difference']=av-bv if av is not None and bv is not None else None
            x['wave_input_checks']=[{'wave':aa['wave'],'source_ids_equal':aa['source_ids']==bb['source_ids'],'first_prompts_equal':aa['first_prompt_digests']==bb['first_prompt_digests'],'private_waits_equal':aa['private_waits']==bb['private_waits']} for aa,bb in zip(a['waves'],b['waves'])]
        comparisons.append(x)
    out={'schema':'astra-operational-v2-analysis-v1','manifest_sha256':sha(M),'analysis_sha256':sha(Path(__file__)),
         'cases':reports,'descriptive_comparisons':comparisons,'confidence_intervals':None,'production_endorsement':False,
         'scope':'All outcomes retained. Finite repeated-input operational checks; no independent fresh performance confirmation, no request-level or wave-level inferential replication. Existing fresh confirmation intervals unchanged.'}
    (H/'live-nov3-operational-v2-independent-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'attempted':sum(x['attempted'] for x in reports.values()),'normal_EOS':sum(x.get('normal',{}).get('natural_eos',0) for x in reports.values()),'comparisons':comparisons},indent=2))
if __name__=='__main__':main()
