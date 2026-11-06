"""Derive bounded comparison tables from preserved raw matrix evidence."""
import collections, csv, gzip, hashlib, json, math, statistics
from pathlib import Path
H=Path(__file__).resolve().parent
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def csvwrite(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def percentile(xs,q):
    if not xs:return None
    xs=sorted(xs);pos=(len(xs)-1)*q;i=int(pos);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(pos-i)
def distribution(xs):
    xs=[x for x in xs if x is not None]
    return {'n':len(xs),'mean':statistics.mean(xs) if xs else None,'median':percentile(xs,.5),'p90':percentile(xs,.9),'p95':percentile(xs,.95),'max':max(xs) if xs else None}
def safe_sum(rows,key):
    xs=[x[key] for x in rows if isinstance(x.get(key),(int,float))]
    return sum(xs) if xs else None
def resources(raw):
    start=raw.get('load_start_ns');end=raw.get('load_end_ns')
    if not start or not end:return {}
    samples=raw.get('resource_samples',[]);inside=[x for x in samples if start<=x['at_ns']<=end]
    before=[x for x in samples if x['at_ns']<=start];after=[x for x in samples if x['at_ns']>=end]
    first=before[-1] if before else (inside[0] if inside else None);last=after[0] if after else (inside[-1] if inside else None)
    initial={p['identity']:p['cpu_s'] for p in first['processes']} if first else {}
    final={p['identity']:p['cpu_s'] for p in last['processes']} if last else {}
    cpu=sum(max(0,v-initial.get(k,v)) for k,v in final.items())
    identities=collections.defaultdict(set)
    for sample in ([first] if first else [])+inside+([last] if last else []):
        for proc in sample['processes']:
            identity=proc['identity'];pid,creation=identity.split(':',1)
            identities[pid].add(creation)
    unstable_cpu_ids=sorted(pid for pid,creation in identities.items() if len(creation)>1)
    if unstable_cpu_ids: cpu=None

    rss=lambda s:sum(p['rss_bytes'] for p in s['processes'])
    integral=0;energy=0;gpu_seconds=0;gpu_observed_s=0;sample_seconds=0
    for a,b in zip(samples,samples[1:]):
        lo=max(start,a['at_ns']);hi=min(end,b['at_ns'])
        if hi<=lo:continue
        sec=(hi-lo)/1e9;sample_seconds+=sec
        integral+=sec*(rss(a)+rss(b))/2
        if 'gpu' in a and 'gpu' in b:
            ga,gb=a['gpu'],b['gpu'];energy+=sec*(ga['power_w']+gb['power_w'])/2
            gpu_seconds+=sec*(ga['utilization_pct']+gb['utilization_pct'])/200;gpu_observed_s+=sec
    gpu=[x['gpu'] for x in inside if 'gpu' in x]
    return {'sample_count':len(inside),'cpu_s_sampled':cpu,'cpu_process_identity_stable':not unstable_cpu_ids,'cpu_unstable_process_ids':unstable_cpu_ids,'cpu_window_overhang_ms':((start-first['at_ns'])+(last['at_ns']-end))/1e6 if first and last else None,'rss_peak_bytes_sampled':max(map(rss,inside),default=None),'rss_byte_seconds_sampled':integral if sample_seconds else None,'rss_mean_bytes_sampled':integral/sample_seconds if sample_seconds else None,'device_gpu_utilization_mean_pct':gpu_seconds/gpu_observed_s*100 if gpu_observed_s else None,'device_energy_j_sampled':energy if gpu_observed_s else None,'device_memory_used_peak_bytes_sampled':max((x['memory_used_bytes'] for x in gpu),default=None),'device_power_mean_w':statistics.mean(x['power_w'] for x in gpu) if gpu else None,'device_temperature_max_c':max((x['temperature_c'] for x in gpu),default=None),'device_sm_clock_mean_mhz':statistics.mean(x['clock_mhz'] for x in gpu) if gpu else None,'gpu_sample_errors':dict(collections.Counter(x.get('gpu_error') for x in inside if x.get('gpu_error')))}
def aggregate(rows,elapsed):
    ok=[r for r in rows if r['success']];tokens=sum(r['output_tokens'] for r in ok)
    span=sum(r['token_span_ms'] for r in ok);intervals=sum(r['itl_intervals'] for r in ok)
    return {'requests':len(rows),'completed':len(ok),'failures':len(rows)-len(ok),'output_tokens':tokens,'elapsed_s':elapsed,'output_tokens_s':tokens/elapsed if elapsed else None,'requests_s':len(ok)/elapsed if elapsed else None,'release_ttft_ms':distribution([r['release_ttft_ms'] for r in ok]),'submit_ttft_ms':distribution([r['submit_ttft_ms'] for r in ok]),'request_elapsed_ms':distribution([r['request_elapsed_ms'] for r in ok]),'release_lateness_ms':distribution([r['release_lateness_ms'] for r in ok]),'mean_itl_ms':span/intervals if intervals else None,'itl_intervals':intervals,'executed_prompt_tokens':safe_sum(ok,'executed_prompt_tokens'),'gpu_hit_tokens':safe_sum(ok,'gpu_hit_tokens'),'external_reused_tokens':safe_sum(ok,'external_reused_tokens'),'prompt_tokens':sum(r['input_tokens'] for r in ok),'queue_ms':distribution([r['server_queue_ms'] for r in ok]),'service_ms':distribution([r['server_service_ms'] for r in ok])}
def analyze(path):
    raw=json.loads(gzip.decompress(path.read_bytes()));name=raw['name'];starts=raw.get('load_start_ns',0);end=raw.get('load_end_ns',starts)
    terminal={t.get('external_request_id'):t for t in raw.get('terminals',[]) if t.get('external_request_id')}
    rows=[];transfers={};prefetch_jobs={}
    for r in sorted(raw['requests'],key=lambda r:r['index']):
        rid=r['payload']['request_id'];t=terminal.get(rid,{});p=t.get('prefetch_attribution',{});a=t.get('speculative_transfer_accounting',{});receipts=r.get('token_receipt_ns',[]);timing=t.get('server_timing',{})
        delta=lambda target,origin:(target-origin)/1e6 if target is not None and origin is not None else None
        row={'run':name,'phase':raw['phase'],'workload':raw['workload'],'seed':raw['seed'],'policy':raw['policy'],'threshold':raw['threshold'],'index':r['index'],'role':r['role'],'session_id':r['session_id'],'success':r['success'],'error':r.get('error'),'input_tokens':len(r['payload']['prompt']),'output_tokens':len(r.get('output_token_ids',[])),'release_offset_ms':delta(r['intended_release_ns'],starts),'release_lateness_ms':delta(r['request_start_ns'],r['intended_release_ns']),'release_ttft_ms':delta(r.get('first_token_ns'),r['intended_release_ns']),'submit_ttft_ms':delta(r.get('first_token_ns'),r['request_start_ns']),'request_elapsed_ms':delta(r['request_end_ns'],r['request_start_ns']),'token_span_ms':delta(receipts[-1],receipts[0]) if receipts else 0,'itl_intervals':max(0,len(receipts)-1),'terminal_observed':bool(t),'terminal_authoritative':t.get('authoritative'),'event_integrity':t.get('event_integrity'),'gpu_hit_tokens':t.get('admitted_local_tokens'),'external_reused_tokens':t.get('admitted_external_tokens'),'executed_prompt_tokens':t.get('actual_prompt_tokens_locally_executed'),'demand_h2d_bytes':t.get('load_bytes'),'d2h_bytes':t.get('store_bytes'),'demand_h2d_worker_s':t.get('load_worker_seconds'),'d2h_worker_s':t.get('store_worker_seconds'),'prefetch_consumed_blocks':t.get('prefetch_counts',{}).get('consumed',{}).get('blocks'),'prefetch_attempted_blocks':t.get('prefetch_counts',{}).get('attempted',{}).get('blocks'),'prefetch_discarded_blocks':t.get('prefetch_counts',{}).get('discarded',{}).get('blocks'),'splice_bytes':p.get('gpu_splice_bytes'),'prefetch_completed_bytes':a.get('completed_bytes'),'prefetch_consumed_bytes':a.get('consumed_bytes'),'prefetch_unused_completed_bytes':a.get('unconsumed_completed_bytes'),'prefetch_waste_bytes':t.get('speculative_transfer_costs',{}).get('waste_completed_bytes'),'native_adoption':p.get('native_adoption'),'prefetch_ready_lead_ms':p.get('staging_ready_lead_time_s')*1000 if isinstance(p.get('staging_ready_lead_time_s'),(int,float)) else None,'prefetch_request_usable_offset_ms':delta(p.get('request_usable_ns'),r['intended_release_ns']),'prefetch_fence_before_forward':p.get('fence_before_first_forward'),'prefetch_allocation_failures':p.get('allocation_failures'),'preemptions':t.get('preemption_count'),'server_queue_ms':timing.get('queue_delay_ns')/1e6 if timing.get('queue_delay_ns') is not None else None,'server_service_ms':timing.get('service_time_ns')/1e6 if timing.get('service_time_ns') is not None else None,'ordered_keys_complete':t.get('ordered_key_positions_complete'),'physical_identity_complete':t.get('physical_block_identity_complete'),'block_table_complete':(t.get('physical_block_table_provenance') or {}).get('complete')}
        row['mean_itl_ms']=row['token_span_ms']/row['itl_intervals'] if row['itl_intervals'] else None;rows.append(row)
        for job in t.get('transfer_job_statuses',[]):transfers[(job['job_id'],job['request_id'])]=job
        for job in t.get('speculative_transfer_job_statuses',[]):prefetch_jobs[(job['job_id'],job['request_id'])]=job
    elapsed=(end-starts)/1e9
    post=raw.get('postload_export',{});scheduler=post.get('scheduler',{});before=raw.get('before_load_export',{}).get('scheduler',{});worker=(post.get('workers') or [{}])[0];before_worker=(raw.get('before_load_export',{}).get('workers') or [{}])[0]
    counts={k:v-before.get('counts',{}).get(k,0) for k,v in scheduler.get('counts',{}).items()}
    samples=[s for s in scheduler.get('samples',[]) if starts<=s['at_ns']<=end]
    waits=raw.get('wait_hint_events',[]);nominations=[x for x in waits if x.get('event')=='nomination_decision'];nonempty=[x for x in nominations if x.get('candidates')]
    ready=[x for x in waits if x.get('event')=='staging_ready'];submitted=[x for x in waits if x.get('event')=='staging_submitted']
    hint_counts=collections.Counter((h.get('response') or {}).get('reason') or (h.get('response') or {}).get('status') or h.get('http_status') or h.get('error') for h in raw.get('hints',[]))
    summary={'name':name,'phase':raw['phase'],'workload':raw['workload'],'seed':raw['seed'],'policy':raw['policy'],'threshold':raw['threshold'],'status':raw['status'],'error':raw.get('error'),'model':raw.get('server_command',[None,None,None])[2] if raw.get('server_command') else None,'configuration':raw.get('configuration'),'runtime_resolution':raw.get('runtime_resolution'),'all':aggregate(rows,elapsed),'groups':{kind:aggregate([x for x in rows if x['role'].startswith('background')==(kind=='background')],elapsed) for kind in ['agent','background']},'sessions':{},'manager_counts':counts,'manager_type':scheduler.get('manager_type'),'policy_type':scheduler.get('policy_type'),'worker_capture_calls':worker.get('capture',{}).get('calls',0)-before_worker.get('capture',{}).get('calls',0),'worker_capture_rows':worker.get('capture',{}).get('rows',0)-before_worker.get('capture',{}).get('rows',0),'durable_summary_blocks':worker.get('durable_summary_blocks'),'async_stats':worker.get('async_stats'),'pending_async_observations':worker.get('pending_async_observations'),'pending_profile_cuda':worker.get('pending_profile_cuda'),'hint_rpc_count':len(raw.get('hints',[])),'hint_result_counts':dict(hint_counts),'wait_event_counts':dict(collections.Counter(x.get('event') for x in waits)),'nonempty_nominations':len(nonempty),'scored_prefetch_candidates':sum(x.get('scored_candidate_count',0) for x in nonempty),'constrained_prefetch_nominations':sum(bool(x.get('shared_budget_constrained')) for x in nonempty),'staging_jobs_submitted':len(submitted),'staging_jobs_ready':len(ready),'global_ready_speculative_h2d_bytes':sum(x.get('transfer_bytes',0) for x in ready),'prefetch_consumed_blocks':safe_sum(rows,'prefetch_consumed_blocks'),'prefetch_consumed_bytes':safe_sum(rows,'prefetch_consumed_bytes'),'prefetch_unused_completed_bytes':safe_sum(rows,'prefetch_unused_completed_bytes'),'prefetch_waste_bytes':safe_sum(rows,'prefetch_waste_bytes'),'splice_bytes':safe_sum(rows,'splice_bytes'),'prefetch_allocation_failures':safe_sum(rows,'prefetch_allocation_failures'),'preemptions':safe_sum(rows,'preemptions'),'demand_h2d_bytes':safe_sum(rows,'demand_h2d_bytes'),'d2h_bytes':safe_sum(rows,'d2h_bytes'),'demand_h2d_worker_s':safe_sum(rows,'demand_h2d_worker_s'),'d2h_worker_s':safe_sum(rows,'d2h_worker_s'),'pool_samples':len(samples),'cpu_peak_resident_blocks':max((x.get('cpu_resident',0) for x in samples),default=None),'cpu_peak_ready_blocks':max((x.get('cpu_ready',0) for x in samples),default=None),'gpu_min_free_blocks':min((x.get('gpu_free',0) for x in samples),default=None),'gpu_peak_reserved_blocks':max((x.get('gpu_reserved',0) for x in samples),default=None),'gpu_peak_referenced_blocks':max((x.get('gpu_referenced',0) for x in samples),default=None),'load_gpu_peak_allocated_bytes':worker.get('load_peak_allocated_bytes'),'load_gpu_peak_reserved_bytes':worker.get('load_peak_reserved_bytes'),'resources':resources(raw),'transfer_jobs':len(transfers),'transfer_failed_jobs':sum(not x.get('success',False) for x in transfers.values()),'speculative_transfer_failed_jobs':sum(not x.get('success',False) for x in prefetch_jobs.values()),'terminal_events_matching_requests':sum(x['terminal_observed'] for x in rows),'terminal_events_authoritative':sum(x['terminal_authoritative'] is True for x in rows),'ordered_key_complete_requests':sum(x['ordered_keys_complete'] is True for x in rows),'physical_identity_complete_requests':sum(x['physical_identity_complete'] is True for x in rows),'block_table_complete_requests':sum(x['block_table_complete'] is True for x in rows),'owned_descendants_remaining':raw.get('owned_descendants_remaining'),'lifecycle_s':(raw['lifecycle_end_ns']-raw['lifecycle_start_ns'])/1e9}
    for field,origin,key in [('prefix_mean_evaluations','worker','numerical_calls'),('prefix_mean_evaluated_candidates','worker','numerical_candidates'),('prefix_mean_centroid_builds','worker','centroid_builds'),('prefix_mean_centroid_hits','worker','centroid_hits'),('prefix_mean_applied_observations','scheduler','accepted_observations'),('prefix_mean_applied_candidates','scheduler','accepted_candidates'),('prefix_mean_applied_key_count','scheduler','accepted_prefix_key_count')]:
        after_stats=(worker if origin=='worker' else scheduler).get('prefix_mean_scoring',{})
        before_stats=(before_worker if origin=='worker' else before).get('prefix_mean_scoring',{})
        summary[field]=after_stats.get(key,0)-before_stats.get(key,0)
    for sid in sorted({x['session_id'] for x in rows}):
        rs=[x for x in rows if x['session_id']==sid];rraw=[x for x in raw['requests'] if x['session_id']==sid]
        summary['sessions'][sid]={**aggregate(rs,elapsed),'last_completion_offset_ms':(max(x['request_end_ns'] for x in rraw)-starts)/1e6}
    profile=[]
    for file in path.parent.glob('components/components-*.json'):
        data=json.loads(file.read_text())
        for component,values in data.get('metrics',{}).items():profile.append({'run':name,'pid':data['pid'],'component':component,**values,'log2_ns_histogram':json.dumps(values['log2_ns_histogram'])})
    write(path.parent/'summary.json',summary);csvwrite(path.parent/'requests.csv',rows)
    return raw,summary,rows,profile
def pct(a,b):return (a/b-1)*100 if a is not None and b else None
def compare(arc,candidate):
    ar,asum,arows,_=arc;cr,csum,crows,_=candidate
    baseline={x['index']:x for x in ar['requests']};current={x['index']:x for x in cr['requests']}
    pairs=set(baseline)&set(current);at={t['external_request_id']:t for t in ar.get('terminals',[]) if t.get('external_request_id')};ct={t['external_request_id']:t for t in cr.get('terminals',[]) if t.get('external_request_id')}
    key=lambda t:[(p['group_idx'],p['chunk_index'],p['key']) for p in t.get('ordered_key_positions',[])]
    result={'phase':asum['phase'],'workload':asum['workload'],'seed':asum['seed'],'arc_run':asum['name'],'semantic_run':csum['name'],'matched_indices':len(pairs),'request_sets_equal':set(baseline)==set(current),'prompt_mismatch_indices':[i for i in pairs if baseline[i]['payload']['prompt']!=current[i]['payload']['prompt']],'output_mismatch_indices':[i for i in pairs if baseline[i].get('output_token_ids')!=current[i].get('output_token_ids')],'native_key_mismatch_indices':[],'missing_native_key_indices':[],'arc_tokens_s':asum['all']['output_tokens_s'],'semantic_tokens_s':csum['all']['output_tokens_s'],'throughput_delta_pct':pct(csum['all']['output_tokens_s'],asum['all']['output_tokens_s']),'release_ttft_mean_delta_pct':pct(csum['all']['release_ttft_ms']['mean'],asum['all']['release_ttft_ms']['mean']),'release_ttft_median_delta_pct':pct(csum['all']['release_ttft_ms']['median'],asum['all']['release_ttft_ms']['median']),'release_ttft_p95_delta_pct':pct(csum['all']['release_ttft_ms']['p95'],asum['all']['release_ttft_ms']['p95']),'itl_delta_pct':pct(csum['all']['mean_itl_ms'],asum['all']['mean_itl_ms']),'cpu_delta_pct':pct(csum['resources'].get('cpu_s_sampled'),asum['resources'].get('cpu_s_sampled')),'executed_prompt_delta_pct':pct(csum['all']['executed_prompt_tokens'],asum['all']['executed_prompt_tokens'])}
    for i in pairs:
        rid=baseline[i]['payload']['request_id']
        if not at.get(rid,{}).get('ordered_key_positions_complete') or not ct.get(rid,{}).get('ordered_key_positions_complete'):result['missing_native_key_indices'].append(i)
        elif key(at[rid])!=key(ct[rid]):result['native_key_mismatch_indices'].append(i)
    return result
def main():
    runs={}
    for file in sorted((H/'runs').glob('*/raw.json.gz')):
        result=analyze(file);runs[result[1]['name']]=result
    pairs=[]
    for name,result in runs.items():
        if name.endswith('-arc') and name[:-3]+'semantic' in runs:pairs.append(compare(result,runs[name[:-3]+'semantic']))
    summaries=[v[1] for v in runs.values()];rows=[r for v in runs.values() for r in v[2]];components=[r for v in runs.values() for r in v[3]]
    flat=[]
    for s in summaries:
        a=s['all'];flat.append({'run':s['name'],'phase':s['phase'],'workload':s['workload'],'seed':s['seed'],'policy':s['policy'],'threshold':s['threshold'],'status':s['status'],'requests':a['requests'],'failures':a['failures'],'elapsed_s':a['elapsed_s'],'output_tokens_s':a['output_tokens_s'],'ttft_mean_ms':a['release_ttft_ms']['mean'],'ttft_median_ms':a['release_ttft_ms']['median'],'ttft_p95_ms':a['release_ttft_ms']['p95'],'mean_itl_ms':a['mean_itl_ms'],'executed_prompt_tokens':a['executed_prompt_tokens'],'gpu_hit_tokens':a['gpu_hit_tokens'],'external_reused_tokens':a['external_reused_tokens'],**{k:s[k] for k in ['demand_h2d_bytes','d2h_bytes','prefetch_consumed_blocks','staging_jobs_ready','scored_prefetch_candidates','constrained_prefetch_nominations','worker_capture_calls','cpu_peak_resident_blocks','gpu_peak_reserved_blocks','preemptions','prefetch_allocation_failures','load_gpu_peak_allocated_bytes','load_gpu_peak_reserved_bytes']},**{'manager_'+k:v for k,v in s['manager_counts'].items()},**s['resources']})
    pooled=[]
    for phase in ['main','stress','diagnostic']:
        for workload in ['single','multi']:
            for policy in ['arc','semantic']:
                selected=[s for s in summaries if s['phase']==phase and s['workload']==workload and s['policy']==policy];selectedrows=[r for r in rows if r['phase']==phase and r['workload']==workload and r['policy']==policy]
                if selected:pooled.append({'phase':phase,'workload':workload,'policy':policy,'runs':len(selected),'all':aggregate(selectedrows,sum(s['all']['elapsed_s'] for s in selected))})
    write(H/'measurements.json',{'summaries':summaries,'comparisons':pairs,'pooled':pooled,'limitations':['generated representative sessions, not production traces','closed-loop arrivals and batch/cache histories differ as outcomes','small repeated sample; descriptive percentiles and seed ranges, no significance or equivalence claim','observer overhead included; component diagnostic runs excluded from main','sampled device metrics include display/other activity; RSS may double-count shared mappings','no isolated attention-kernel time, production-price model, capacity frontier, or broad model qualification']})
    csvwrite(H/'requests.csv',rows);csvwrite(H/'runs.csv',flat);csvwrite(H/'comparisons.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in x.items()} for x in pairs]);csvwrite(H/'components.csv',components)
    print(json.dumps({'runs':len(summaries),'comparisons':[{k:v for k,v in p.items() if k in ['phase','workload','seed','throughput_delta_pct','prompt_mismatch_indices','output_mismatch_indices','native_key_mismatch_indices','missing_native_key_indices']} for p in pairs]},indent=2))
if __name__=='__main__':main()
