"""Conservative stdlib metrics for finite quantized serving campaigns.

All rates use the complete load interval, including offered idle/think time.
Native proofs delegate to the pinned project's authoritative event validator;
cross-arm identity is joined by (session_id, turn), never row order or run IDs.
"""
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import shutil

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
SLOS = {'chat': {'ttft_ms': 5000, 'itl_ms': 250, 'e2e_ms': 60000},
        'agent': {'ttft_ms': 15000, 'itl_ms': 500, 'e2e_ms': 120000}}
METRIC_DEFINITION = 'v4-evictable-warm-reservation-lifetime'
EVENT_TIMESTAMPS = {'observed_prefix_registered':'observed_at_ns','hint_result':'observed_at_ns',
    'prefix_mean_scores_applied':'consumer_ns','staging_submitted':'submitted_at_ns',
    'nomination_decision':'decision_ns','staging_ready':'ready_at_ns',
    'native_prefix_published':'published_at_ns','resume_bound':'bound_at_ns',
    'native_prefix_adopted':'observed_at_ns','native_prefix_missed':'observed_at_ns',
    'staging_bound_request_owned':'owned_at_ns','staging_retired':'retired_at_ns',
    'staging_retired_pending_fence':'retired_at_ns','resume_binding_rejected':'observed_at_ns',
    'demand_reclaim':'observed_at_ns','staging_cache_warmed':'observed_at_ns'}


def event_stamp(event):
    value=event.get('at_ns')
    return value if number(value) else event.get(EVENT_TIMESTAMPS.get(event.get('event')))


def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def dist(xs):
    xs = sorted(x for x in xs if number(x))
    def q(p):
        if not xs: return None
        z = (len(xs)-1)*p; i = int(z)
        return xs[i] + (xs[min(i+1,len(xs)-1)]-xs[i])*(z-i)
    return {'n':len(xs), 'mean':statistics.mean(xs) if xs else None,
            'p50':q(.5), 'p95':q(.95), 'p99':q(.99),
            'min':min(xs) if xs else None, 'max':max(xs) if xs else None}


def nested(value, path):
    for key in path.split('.'):
        if not isinstance(value, dict): return None
        value = value.get(key)
    return value


def total(values):
    values = list(values)
    return sum(values) if values and all(number(x) for x in values) else None


def peak(values):
    values = [x for x in values if number(x)]
    return max(values) if values else None


def histogram_dist(histogram):
    pairs=sorted((int(k),v) for k,v in histogram.items() if str(k).isdigit() and isinstance(v,int) and not isinstance(v,bool) and v>0)
    count=sum(v for _,v in pairs)
    if not count:return dist([])
    def at(index):
        seen=0
        for value,n in pairs:
            seen+=n
            if seen>index:return value
    def q(p):
        z=(count-1)*p;i=int(z)
        return at(i)+(at(min(i+1,count-1))-at(i))*(z-i)
    return {'n':count,'mean':sum(k*v for k,v in pairs)/count,'p50':q(.5),'p95':q(.95),'p99':q(.99),'min':pairs[0][0],'max':pairs[-1][0]}


def load(path):
    path = Path(path)
    if path.is_dir():
        path = path / ('raw.json.gz' if (path/'raw.json.gz').exists() else 'raw.json')
    body = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    raw = json.loads(body)
    return raw, {'path':str(path.resolve()), 'uncompressed_sha256':hashlib.sha256(body).hexdigest()}


def identity(row):
    sid, turn = row.get('session_id'), row.get('turn')
    if not isinstance(sid,str) or not sid or not isinstance(turn,int) or isinstance(turn,bool) or turn<0:
        raise ValueError('missing/invalid stable (session_id, turn) identity')
    return (sid,turn)


def index_rows(raw):
    result = {}
    for row in raw.get('requests',[]):
        key = identity(row)
        if key in result: raise ValueError('duplicate logical identity: '+repr(key))
        result[key] = row
    if not result: raise ValueError('no requests to align')
    return result


def validator(path=None):
    path = Path(path or ROOT/'benchmarks/preservation.py')
    spec = importlib.util.spec_from_file_location('_campaign_preservation',path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, {'path':str(path.resolve()), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def key_chain(event):
    rows = event.get('ordered_key_positions')
    if event.get('ordered_key_positions_complete') is not True or not isinstance(rows,list) or not rows:
        return None
    chain = []
    for row in rows:
        if not isinstance(row,dict) or not all(k in row for k in ('group_idx','chunk_index','key')):
            return None
        if not isinstance(row['key'],str) or not row['key']: return None
        chain.append((row['group_idx'],row['chunk_index'],row['key']))
    positions = [r[:2] for r in chain]
    if len(set(positions)) != len(positions) or positions != sorted(positions): return None
    return chain


def terminal_index(raw):
    result = collections.defaultdict(list)
    for event in raw.get('terminals',[]):
        if isinstance(event,dict): result[event.get('external_request_id')].append(event)
    return result


def proof(row, events, module):
    rid = nested(row,'payload.request_id')
    matched = events.get(rid,[])
    if len(matched)!=1: return {'passed':False,'reason':'terminal_count_'+str(len(matched))}
    event = matched[0]
    try:
        mapping = module.map_preservation_events([event], expected_request_ids=[rid])
        if mapping.get('preservation_outcome_basis') != module.AUTHORITATIVE_REQUEST_EVENTS_BASIS:
            return {'passed':False,'reason':mapping.get('preservation_reason','invalid_native_proof')}
        if key_chain(event) is None: return {'passed':False,'reason':'missing_or_invalid_ordered_keys'}
        prompt=nested(row,'payload.prompt')
        digest=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()
        if event.get('prompt_token_digest')!=digest or event.get('prompt_token_count')!=len(prompt):
            return {'passed':False,'reason':'server_prompt_identity_mismatch_or_missing'}
        params = nested(row,'payload.kv_transfer_params') or {}
        for field in ('sample_index','run_id','cell_id','session_id'):
            actual=event.get('policy_session_id') if field=='session_id' else event.get(field)
            if field in params and params[field]!=actual:
                return {'passed':False,'reason':'terminal_identity_mismatch_'+field}
        if 'session_id' in params and params['session_id']!=row.get('session_id'):
            return {'passed':False,'reason':'logical_session_identity_mismatch'}
        table = event.get('physical_block_table_provenance')
        if not isinstance(table,dict) or table.get('complete') is not True or table.get('errors'):
            return {'passed':False,'reason':'missing_or_invalid_physical_block_table'}
        return {'passed':True,'reason':None}
    except (TypeError,ValueError,KeyError,IndexError) as error:
        return {'passed':False,'reason':'validator_exception_'+type(error).__name__}


def request_metrics(row, start, end, events, module):
    errors=[]; key=None
    try: key=identity(row)
    except ValueError as e: errors.append(str(e))
    release=row.get('intended_release_ns'); submit=row.get('request_start_ns')
    finish=row.get('request_end_ns',row.get('end_ns')); first=row.get('first_token_ns')
    receipts=row.get('token_receipt_ns',[]); outputs=row.get('output_token_ids')
    prompt=nested(row,'payload.prompt')
    if not isinstance(prompt,list) or not all(isinstance(x,int) and not isinstance(x,bool) for x in prompt):
        errors.append('invalid_prompt_token_ids')
    if not isinstance(outputs,list) or not all(isinstance(x,int) and not isinstance(x,bool) for x in outputs):
        errors.append('invalid_output_token_ids'); outputs=[]
    if not all(number(x) for x in (release,submit,finish)):
        errors.append('missing_request_timing')
    elif not start<=release<=submit<=finish<=end:
        errors.append('request_timing_outside_or_nonmonotonic_load')
    if not isinstance(receipts,list) or any(not number(x) for x in receipts):
        errors.append('invalid_receipt_times'); receipts=[]
    elif any(a>b for a,b in zip(receipts,receipts[1:])):
        errors.append('nonmonotonic_receipts')
    if len(receipts)!=len(outputs): errors.append('receipt_token_count_mismatch')
    if outputs and (not number(first) or not receipts or first!=receipts[0]): errors.append('first_token_mismatch')
    if receipts and number(submit) and number(finish) and not submit<=receipts[0]<=receipts[-1]<=finish:
        errors.append('receipts_outside_request')
    success=row.get('success') is True
    if success and not outputs: errors.append('successful_request_without_tokens')
    delta=lambda a,b:(a-b)/1e6 if number(a) and number(b) else None
    itls=[(b-a)/1e6 for a,b in zip(receipts,receipts[1:])] if not errors else []
    ttft=delta(first,release); e2e=delta(finish,release)
    cohort=row.get('cohort') or str(row.get('role','')).split('-')[0]
    slo=SLOS.get(cohort)
    itl_p95=dist(itls)['p95']
    # One-token completions have no inter-token interval to constrain.
    good=success and not errors and slo is not None and ttft is not None and ttft<=slo['ttft_ms'] and e2e<=slo['e2e_ms'] and (not itls or itl_p95<=slo['itl_ms'])
    terminal=(events.get(nested(row,'payload.request_id')) or [{}])[0]
    return {'identity':list(key) if key else None,'cohort':cohort,'success':success,
            'timing_valid':not errors,'errors':errors,'input_tokens':len(prompt) if isinstance(prompt,list) else None,
            'output_tokens':len(outputs),'release_ttft_ms':ttft if not errors else None,
            'submit_ttft_ms':delta(first,submit) if not errors else None,
            'release_e2e_ms':e2e if not errors else None,'submit_e2e_ms':delta(finish,submit) if not errors else None,
            'release_lateness_ms':delta(submit,release) if not errors else None,
            'server_queue_ms':(nested(terminal,'server_timing.queue_delay_ns')/1e6 if number(nested(terminal,'server_timing.queue_delay_ns')) else None),
            'itl_ms':itls,'request_itl_p95_ms':itl_p95,'request_itl_max_ms':max(itls) if itls else None,
            'itl_gaps_above_slo':sum(x>slo['itl_ms'] for x in itls) if slo else None,
            'every_itl_gap_within_slo':all(x<=slo['itl_ms'] for x in itls) if slo else None,
            'slo_pass':good,'slo_defined':slo is not None,
            'finish_reasons':row.get('finish_reasons'), 'proof':proof(row,events,module)}


def aggregate(rows, elapsed):
    completed=[r for r in rows if r['success'] and r['timing_valid']]
    good=[r for r in completed if r['slo_pass']]
    rate=lambda n:n/elapsed if number(elapsed) and elapsed>0 else None
    result={'requests':len(rows),'successful_responses':sum(r['success'] for r in rows),
            'completed_valid':len(completed),'failures':sum(not r['success'] for r in rows),
            'invalid_rows':sum(not r['timing_valid'] for r in rows),'elapsed_s':elapsed,
            'output_tokens':sum(r['output_tokens'] for r in completed),
            'output_tokens_s':rate(sum(r['output_tokens'] for r in completed)),
            'requests_s':rate(len(completed)), 'slo_good_requests':len(good),
            'goodput_requests_s':rate(len(good)), 'goodput_output_tokens_s':rate(sum(r['output_tokens'] for r in good)),
            'slo_unknown_requests':sum(not r['slo_defined'] for r in rows),
            'prompt_token_lengths':dist(r['input_tokens'] for r in rows),
            'output_token_lengths':dist(r['output_tokens'] for r in completed),
            'itl_ms':dist(x for r in completed for x in r['itl_ms']),
            'finish_reasons':dict(collections.Counter(str(x) for r in rows for x in (r['finish_reasons'] or ['missing'])))}
    for key in ('release_ttft_ms','submit_ttft_ms','release_e2e_ms','submit_e2e_ms','release_lateness_ms','server_queue_ms'):
        result[key]=dist(r[key] for r in completed)
    result['request_itl_p95_ms']=dist(r.get('request_itl_p95_ms') for r in completed)
    result['itl_gaps_above_slo']=total(r.get('itl_gaps_above_slo') for r in completed)
    result['requests_with_any_itl_gap_violation']=sum(r.get('every_itl_gap_within_slo') is False for r in completed)
    cohorts={r.get('cohort') for r in rows};slo=SLOS.get(next(iter(cohorts))) if len(cohorts)==1 else None
    checks={'ttft':result['release_ttft_ms']['p95'],'itl':result['itl_ms']['p95'],'e2e':result['release_e2e_ms']['p95']}
    available=bool(completed) and slo is not None and checks['ttft'] is not None and checks['e2e'] is not None
    result['aggregate_latency_slo']={'scope':'cohort p95 release-TTFT, token-weighted receipt ITL, release-E2E; separate from per-request goodput and failure/backlog criteria',
        'thresholds_ms':slo,'observed_p95_ms':checks,
        'passed':all(value is None or value<=slo[key+'_ms'] for key,value in checks.items()) if available else None,
        'itl_no_intervals':not result['itl_ms']['n']}
    return result


def resources(raw, start, end):
    samples=sorted((x for x in raw.get('resource_samples',[]) if number(x.get('at_ns'))),key=lambda x:x['at_ns'])
    inside=[s for s in samples if start<=s['at_ns']<=end]
    before=[s for s in samples if s['at_ns']<=start]; after=[s for s in samples if s['at_ns']>=end]
    first=before[-1] if before else (inside[0] if inside else None)
    last=after[0] if after else (inside[-1] if inside else None)
    window=([first] if first else [])+inside+([last] if last else [])
    ids=collections.defaultdict(set); malformed=[]
    for s in window:
        for p in s.get('processes',[]):
            ident=p.get('identity',''); split=ident.split(':') if isinstance(ident,str) else []
            if len(split)!=2 or not all(x.isdigit() for x in split): malformed.append(ident)
            else: ids[split[0]].add(split[1])
    unstable=sorted(pid for pid,values in ids.items() if len(values)!=1)
    initial={p.get('identity'):p.get('cpu_s') for p in (first or {}).get('processes',[])}
    final={p.get('identity'):p.get('cpu_s') for p in (last or {}).get('processes',[])}
    missing=sorted(set(initial)^set(final))
    cpu=None
    if first is not None and last is not None and first['at_ns']<last['at_ns'] and initial and set(initial)==set(final) and not unstable and not malformed:
        diffs=[final[k]-initial[k] for k in initial if number(final[k]) and number(initial[k])]
        if len(diffs)==len(initial) and all(x>=0 for x in diffs): cpu=sum(diffs)
    fault_deltas={};fault_totals={}
    begin_procs={p.get('identity'):p for p in (first or {}).get('processes',[])}
    end_procs={p.get('identity'):p for p in (last or {}).get('processes',[])}
    for field in ('minor_faults','major_faults'):
        values={}
        for ident in sorted(set(begin_procs)|set(end_procs),key=str):
            a=begin_procs.get(ident,{}).get(field);b=end_procs.get(ident,{}).get(field)
            values[ident]=b-a if first and last and first['at_ns']<last['at_ns'] and str(ident).split(':')[0] not in unstable and ident not in malformed and number(a) and number(b) and b>=a else None
        fault_deltas[field]=values
        fault_totals[field]=total(values.values())
    host={}
    for field in ('swap_in_bytes','swap_out_bytes'):
        a=nested(first or {},'host_memory.'+field);b=nested(last or {},'host_memory.'+field)
        host[field+'_delta']=b-a if first and last and first['at_ns']<last['at_ns'] and number(a) and number(b) and b>=a else None
    for field in ('available_bytes','swap_used_bytes'):
        values=[nested(s,'host_memory.'+field) for s in inside];values=[v for v in values if number(v)]
        host[field+'_min_sampled']=min(values) if values else None
        host[field+'_peak_sampled']=max(values) if values else None
        a=nested(first or {},'host_memory.'+field);b=nested(last or {},'host_memory.'+field)
        host[field+'_delta']=b-a if first and last and first['at_ns']<last['at_ns'] and number(a) and number(b) else None
    host['source']='host-wide psutil virtual_memory/swap_memory; swap deltas use CPU endpoint samples, not server attribution'
    host['sample_count']=sum(isinstance(s.get('host_memory'),dict) for s in inside)
    driver_cpu=None
    if first and last and first['at_ns']<last['at_ns']:
        a=first.get('driver_cpu_s');b=last.get('driver_cpu_s')
        if number(a) and number(b) and b>=a:driver_cpu=b-a
    rss=lambda s:total(p.get('rss_bytes') for p in s.get('processes',[]))
    gpu=[s.get('gpu',{}) for s in inside]
    coverage=(max(0,min(end,last['at_ns'])-max(start,first['at_ns']))/(end-start) if first and last and end>start else 0)
    before_workers=(raw.get('before_load_export') or {}).get('workers',[])
    post_workers=(raw.get('postload_export') or {}).get('workers',[])
    return {'source':'resource_samples /proc pid:start_ticks; NVML device-wide; CUDA worker reset/export',
            'sample_count':len(inside),'first_inside_offset_ms':(inside[0]['at_ns']-start)/1e6 if inside else None,
            'last_inside_offset_ms':(inside[-1]['at_ns']-start)/1e6 if inside else None,
            'sample_span_coverage_fraction':coverage,'max_sample_gap_ms':peak((b['at_ns']-a['at_ns'])/1e6 for a,b in zip(inside,inside[1:])),
            'cpu_s_sampled':cpu,'cpu_interval_start_ns':first['at_ns'] if first else None,'cpu_interval_end_ns':last['at_ns'] if last else None,
            'driver_cpu_s_sampled':driver_cpu,'driver_rss_peak_bytes_sampled':peak(s.get('driver_rss_bytes') for s in inside),
            'host_memory':host,'process_fault_deltas_sampled':fault_deltas,'process_fault_totals_sampled':fault_totals,
            'cpu_window_overhang_ms':((max(0,start-first['at_ns'])+max(0,last['at_ns']-end))/1e6 if first and last else None),
            'cpu_unstable_pids':unstable,'cpu_malformed_identities':malformed,'cpu_unmatched_identities':missing,
            'rss_peak_bytes_sampled':peak(rss(s) for s in inside),
            'nvml_device_memory_peak_bytes_sampled':peak(g.get('memory_used_bytes') for g in gpu),
            'nvml_utilization_pct':dist(g.get('utilization_pct') for g in gpu),
            'gpu_sample_count':sum(bool(g) for g in gpu),'gpu_sample_errors':dict(collections.Counter(s['gpu_error'] for s in inside if s.get('gpu_error'))),
            'startup_monitor_sample_count':sum(s['at_ns']<start for s in samples),
            'startup_nvml_device_memory_peak_bytes_sampled':peak(nested(s,'gpu.memory_used_bytes') for s in samples if s['at_ns']<start),
            'before_load_cuda_allocated_bytes':total(w.get('gpu_allocated_bytes') for w in before_workers),
            'before_load_cuda_reserved_bytes':total(w.get('gpu_reserved_bytes') for w in before_workers),
            'load_cuda_allocated_peak_bytes_sum_workers':total(w.get('load_peak_allocated_bytes') for w in post_workers),
            'load_cuda_reserved_peak_bytes_sum_workers':total(w.get('load_peak_reserved_bytes') for w in post_workers),
            'cuda_peak_note':'sum of per-worker peaks; not necessarily simultaneous; reset before load, exported after load'}


def concurrency(raw,start,end):
    # Client in-flight and server running/queued are distinct quantities.
    changes=[]
    for r in raw.get('requests',[]):
        a=r.get('request_start_ns');b=r.get('request_end_ns',r.get('end_ns'))
        if number(a) and number(b) and start<=a<=b<=end: changes.extend([(a,1),(b,-1)])
    active=maximum=0
    for _,delta in sorted(changes,key=lambda p:(p[0],p[1])): active+=delta;maximum=max(maximum,active)
    samples=nested(raw,'postload_export.scheduler.samples') or []
    samples=[s for s in samples if number(s.get('at_ns')) and start<=s['at_ns']<=end]
    result={'client_inflight_peak':maximum if changes else None,'client_inflight_source':'request_start_ns to request_end_ns, all HTTP in-flight including server queue',
            'server_sample_count':len(samples),'source':'postload_export.scheduler.samples; declared fields only'}
    for name,field in [('scheduled','scheduled_requests'),('running','running_requests'),('queued','waiting_requests')]:
        values=[s[field] for s in samples if number(s.get(field))]
        result[name+'_peak']=peak(values);result[name+'_distribution']=dist(values)
        result[name+'_samples']=len(values);result[name+'_field']=field
    counts=nested(raw,'postload_export.scheduler.counts') or {}
    calls=counts.get('native_scheduler_calls')
    if number(calls) and calls>0:
        result['source']='postload_export.scheduler.counts; all Scheduler.schedule outputs after reset; histograms, no per-step timestamps'
        result['server_schedule_calls']=calls
        for name,label in [('scheduled','scheduled'),('running','running'),('queued','waiting')]:
            prefix='native_'+label+'_hist_'
            histogram={k[len(prefix):]:v for k,v in counts.items() if k.startswith(prefix) and number(v)}
            observed=sum(histogram.values());summed=counts.get('native_'+label+'_sum');maximum=counts.get('native_'+label+'_max')
            result[name+'_histogram']=histogram;result[name+'_samples']=observed
            result[name+'_coverage_complete']=observed==calls
            result[name+'_peak']=maximum if observed==calls and number(maximum) else None
            result[name+'_mean']=summed/calls if observed==calls and number(summed) else None
            result[name+'_distribution']=histogram_dist(histogram)
            result[name+'_field']='native_'+label+'_{hist_N,max,sum}'
    return result


def staging_utility(raw,start,end):
    """Offline joins of observed choices to actual named demand; never policy input.

    Released VRAM bytes are not transferred bytes. Native local-prefix blocks
    are not H2D suffix blocks. Missing cleanup/completion evidence stays unknown.
    """
    recorded=raw.get('wait_hint_events');declared=isinstance(recorded,list)
    unique={};duplicates=0
    for e in recorded or []:
        key=json.dumps(e,sort_keys=True,separators=(',',':'))
        if key in unique:duplicates+=1
        else:unique[key]=e
    events=list(unique.values());inside=[e for e in events if number(event_stamp(e)) and start<=event_stamp(e)<=end]
    groups=collections.defaultdict(lambda:collections.defaultdict(list));choices=collections.defaultdict(list)
    for e in events:
        if isinstance(e.get('event_id'),str):groups[e['event_id']][e.get('event')].append(e)
    for e in inside:
        if e.get('event')=='nomination_decision':
            for chosen in e.get('chosen',[]):
                if isinstance(chosen,dict) and isinstance(chosen.get('event_id'),str):choices[chosen['event_id']].append(e.get('decision_ns'))
    active_ids=set(choices)
    kinds=('staging_cache_warmed','staging_submitted','staging_ready','staging_bound_request_owned','native_prefix_published','native_prefix_adopted','native_prefix_missed')
    for e in inside:
        if e.get('event') in kinds and isinstance(e.get('event_id'),str):active_ids.add(e['event_id'])
    terminals=collections.defaultdict(list)
    for t in raw.get('terminals',[]):
        event_id=nested(t,'prefetch_attribution.wait_hint.event_id')
        if isinstance(event_id,str):terminals[event_id].append(t)
    completions=raw.get('speculative_transfer_completions');completion_declared=isinstance(completions,list)
    completion_jobs=collections.defaultdict(list)
    for c in completions or []:
        if isinstance(c,dict) and c.get('job_id') is not None:
            if c not in completion_jobs[c['job_id']]:completion_jobs[c['job_id']].append(c)
    rows=[];reservation_intervals=[];owned_intervals=[];unknown_intervals=[]
    def one(es):return es[0] if len(es)==1 else {}
    def amount(event,field):
        value=event.get(field)
        return value if number(value) and value>=0 else None
    for eid in sorted(active_ids):
        g=groups[eid];sub=one(g['staging_submitted']);ready=one(g['staging_ready'])
        owned=one(g['staging_bound_request_owned']);published=one(g['native_prefix_published'])
        adopted=one(g['native_prefix_adopted']);missed=one(g['native_prefix_missed']);bound=one(g['resume_bound'])
        # Pending-fence retirement is not release; only final retirement or
        # native publication or fenced cache warming releases staging reservations.
        endpoints=[e for name in ('staging_cache_warmed','native_prefix_published','staging_retired') for e in g[name] if number(event_stamp(e))]
        submitted=event_stamp(sub);close=None
        if number(submitted):
            candidates=[e for e in endpoints if event_stamp(e)>=submitted]
            if candidates:close=min(candidates,key=event_stamp)
        close=close or {};closed=event_stamp(close);retired=one(g['staging_retired'])
        problems=[]
        for name in kinds+('resume_bound',):
            if len(g[name])>1:problems.append('ambiguous_'+name)
        pending=bool(g['staging_retired_pending_fence'])
        ids=sub.get('reserved_block_ids');ids_valid=isinstance(ids,list) and bool(ids) and all(isinstance(x,int) and not isinstance(x,bool) for x in ids) and len(set(ids))==len(ids)
        blocks=len(ids) if ids_valid else None
        duration=(closed-submitted)/1e6 if number(closed) and number(submitted) and closed>=submitted else None
        owned_at=event_stamp(owned)
        owned_duration=(closed-owned_at)/1e6 if number(closed) and number(owned_at) and closed>=owned_at else None
        if sub:
            if duration is not None and ids_valid:reservation_intervals.append((submitted,closed,tuple(ids),eid))
            else:unknown_intervals.append(eid)
        if owned:
            if owned_duration is not None and ids_valid:owned_intervals.append((owned_at,closed,tuple(ids),eid))
            else:problems.append('unknown_owned_reservation_interval')
        ts=terminals[eid];terminal=one(ts)
        if len(ts)>1:problems.append('ambiguous_terminal_join')
        ready_at=event_stamp(ready);enqueue=event_stamp(bound);admission=nested(terminal,'first_admission.admitted_at_ns')
        source_bytes=amount(sub,'submitted_bytes');ready_bytes=amount(ready,'transfer_bytes')
        job_id=sub.get('job_id',ready.get('job_id'));cs=completion_jobs.get(job_id,[]);c=one(cs)
        if len(cs)>1:problems.append('conflicting_completion_records')
        if c and c.get('event_id')!=eid:problems.append('completion_event_identity_mismatch');c={}
        explicit_completed=(amount(c,'bytes') if c.get('success') is True and c.get('cancelled') is False and number(c.get('completed_at_ns')) else None)
        if explicit_completed is not None and ready_bytes is not None and explicit_completed!=ready_bytes:problems.append('ready_completion_byte_mismatch')
        completed=explicit_completed if explicit_completed is not None else ready_bytes
        suffix=published.get('prefetched_suffix_block_ids');local_count=published.get('local_prefix_count')
        adoption=nested(terminal,'prefetch_attribution.native_adoption')
        suffix_count=len(suffix) if isinstance(suffix,list) else None
        if suffix_count is None and isinstance(adoption,dict):suffix_count=adoption.get('prefetched_suffix_block_count')
        byte_per_block=amount(sub,'bytes_per_key')
        native_ok=bool(adopted) and isinstance(adopted.get('published_block_ids'),list) and bool(adopted['published_block_ids']) and adopted.get('published_block_ids')==adopted.get('actual_block_ids')
        adopted_bytes=suffix_count*byte_per_block if native_ok and number(suffix_count) and byte_per_block is not None else (0 if missed or retired else None)
        consumed=nested(terminal,'speculative_transfer_accounting.consumed_bytes')
        consumed=consumed if number(consumed) and consumed>=0 else (adopted_bytes if number(adopted_bytes) else (0 if retired and completed is not None else None))
        unconsumed=completed-consumed if number(completed) and number(consumed) and completed>=consumed else None
        if number(completed) and number(consumed) and consumed>completed:problems.append('consumed_exceeds_observed_completed')
        reason=retired.get('reason');category=('cancelled' if reason and any(x in reason.lower() for x in ('cancel','abort')) else 'expired' if reason and any(x in reason.lower() for x in ('expir','ttl')) else 'other_retirement' if retired else None)
        rows.append({'event_id':eid,'job_id':job_id,'chosen_decisions':len(choices[eid]),'submitted':bool(sub),
            'submitted_ns':submitted,'ready_ns':ready_at,'request_enqueue_ns':enqueue,'first_admission_ns':admission,
            'owned_ns':owned_at,'reservation_end_ns':closed,'reservation_end_event':close.get('event'),
            'reservation_lifetime_ms':duration,'request_owned_reservation_lifetime_ms':owned_duration,
            'reserved_suffix_blocks':blocks,'submitted_bytes':source_bytes,'ready_completed_bytes':ready_bytes,
            'explicit_successful_completion_bytes':explicit_completed,'observed_completed_bytes':completed,
            'native_adopted':native_ok if adopted or missed else None,'native_missed':bool(missed) if adopted or missed else None,
            'published_local_prefix_blocks':local_count,'published_transferred_suffix_blocks':suffix_count,
            'native_adopted_transferred_suffix_bytes':adopted_bytes,
            'named_demand_consumed_bytes':consumed,'completed_unconsumed_by_named_demand_bytes':unconsumed,
            'completed_unconsumed_retired_bytes':unconsumed if retired else None,
            'retirement_reason':reason,'retirement_category':category,'released_reservation_bytes':amount(retired,'released_bytes'),
            'pending_fence_retirement_observed':pending,
            'ready_before_enqueue':ready_at<=enqueue if number(ready_at) and number(enqueue) else None,
            'ready_before_first_admission':ready_at<=admission if number(ready_at) and number(admission) else None,
            'ready_lead_before_enqueue_ms':(enqueue-ready_at)/1e6 if number(ready_at) and number(enqueue) else None,
            'join_problems':problems})
    def water(intervals):
        changes=collections.defaultdict(list)
        for lo,hi,ids,eid in intervals:
            lo=max(lo,start);hi=min(hi,end)
            if hi<=lo:continue
            changes[lo].append((1,ids,eid));changes[hi].append((-1,ids,eid))
        active=collections.defaultdict(set);maximum=0;collisions=[]
        for at in sorted(changes):
            for delta,ids,eid in sorted(changes[at],key=lambda x:x[0]):
                for block in ids:
                    if delta>0:
                        if active[block] and eid not in active[block]:collisions.append({'at_ns':at,'block_id':block})
                        active[block].add(eid)
                    else:
                        active[block].discard(eid)
                        if not active[block]:active.pop(block,None)
            maximum=max(maximum,len(active))
        return {'known_closed_interval_peak_unique_blocks':maximum if declared else None,'physical_overlap_errors':collisions,'closed_intervals':len(intervals)}
    ready_jobs={};ready_conflicts=[]
    for e in inside:
        if e.get('event')=='staging_ready':
            key=e.get('job_id')
            if key is None:ready_conflicts.append('missing_job_id');continue
            if key in ready_jobs and (ready_jobs[key].get('event_id'),ready_jobs[key].get('transfer_bytes'))!=(e.get('event_id'),e.get('transfer_bytes')):ready_conflicts.append(str(key))
            else:ready_jobs[key]=e
    completed_records=[one(cs) for cs in completion_jobs.values()]
    completed_inside=[c for c in completed_records if number(c.get('completed_at_ns')) and start<=c['completed_at_ns']<=end and c.get('success') is True and c.get('cancelled') is False]
    completion_valid=completion_declared and all(len(cs)==1 and isinstance(cs[0].get('success'),bool) and isinstance(cs[0].get('cancelled'),bool) and number(cs[0].get('completed_at_ns')) for cs in completion_jobs.values())
    global_completed=total(c.get('bytes') for c in completed_inside) if completed_inside else (0 if completion_valid else None)
    global_ready=total(e.get('transfer_bytes') for e in ready_jobs.values()) if ready_jobs else (0 if declared else None)
    if ready_conflicts:global_ready=None
    submitted_jobs=[r for r in rows if r['submitted']]
    job_counts=collections.Counter(r['job_id'] for r in submitted_jobs)
    submitted_conflicts=[str(k) for k,v in job_counts.items() if v>1 or k is None]
    def observed_sum(field):return total(r.get(field) for r in submitted_jobs) if submitted_jobs else (0 if declared else None)
    def known_subtotal(field):
        values=[r[field] for r in submitted_jobs if number(r.get(field))]
        return sum(values) if values else (0 if declared and not submitted_jobs else None)
    missing_explicit=[r['event_id'] for r in submitted_jobs if r['explicit_successful_completion_bytes'] is None]
    reservation_water=water(reservation_intervals);owned_water=water(owned_intervals)
    return {'scope':'offline chosen-hint / named-demand usefulness; future demand is evaluation only, never policy input; not task quality or unrelated cache-reuse credit',
        'unique_chosen_event_ids':len(choices),'chosen_without_submission':[eid for eid in choices if not groups[eid]['staging_submitted']],
        'event_rows':rows,'exact_duplicate_event_records_removed':duplicates,
        'global_ready_completed_bytes_observed':global_ready,'ready_unique_transfer_jobs':len(ready_jobs),'ready_job_conflicts':ready_conflicts,
        'global_explicit_successful_completion_bytes_observed':global_completed,'explicit_completion_collection_present':completion_declared,
        'completion_collection_conflicting_jobs':[str(k) for k,v in completion_jobs.items() if len(v)>1],
        'submitted_without_explicit_successful_completion':missing_explicit,
        'submitted_job_identity_conflicts':submitted_conflicts,
        'explicit_completion_coverage_for_submitted_jobs':completion_declared and all(len(completion_jobs.get(r['job_id'],[]))==1 and completion_jobs[r['job_id']][0].get('event_id')==r['event_id'] for r in submitted_jobs),
        'completion_limit':'staging_ready omits successful DMA completed after expiry/pending-fence retirement; old raws only bound ready-completed traffic from below; released_bytes is reservation capacity, not transferred bytes',
        'global_named_demand_consumed_bytes':observed_sum('named_demand_consumed_bytes') if not submitted_conflicts else None,
        'global_submitted_transfer_bytes':observed_sum('submitted_bytes') if not submitted_conflicts else None,
        'named_demand_consumed_bytes_known_subtotal':known_subtotal('named_demand_consumed_bytes') if not submitted_conflicts else None,
        'global_native_adopted_transferred_suffix_bytes':observed_sum('native_adopted_transferred_suffix_bytes') if not submitted_conflicts else None,
        'native_adopted_transferred_suffix_bytes_known_subtotal':known_subtotal('native_adopted_transferred_suffix_bytes') if not submitted_conflicts else None,
        'global_completed_unconsumed_by_named_demand_bytes':observed_sum('completed_unconsumed_by_named_demand_bytes') if not submitted_conflicts else None,
        'completed_unconsumed_by_named_demand_bytes_known_subtotal':known_subtotal('completed_unconsumed_by_named_demand_bytes') if not submitted_conflicts else None,
        'completed_unconsumed_retired_bytes_by_category':{category:total(r['completed_unconsumed_retired_bytes'] for r in rows if r['retirement_category']==category) for category in ('expired','cancelled','other_retirement')},
        'unknown_completed_attempts':[r['event_id'] for r in rows if r['submitted'] and r['observed_completed_bytes'] is None],
        'unknown_consumption_attempts':[r['event_id'] for r in rows if r['submitted'] and r['named_demand_consumed_bytes'] is None],
        'reservation_lifetime_ms':dist(r['reservation_lifetime_ms'] for r in rows),
        'request_owned_reservation_lifetime_ms':dist(r['request_owned_reservation_lifetime_ms'] for r in rows),
        'reservation_highwater':reservation_water,'request_owned_highwater':owned_water,
        'unknown_reservation_intervals':unknown_intervals,
        'highwater_complete_for_submitted_attempts':bool(declared) and not unknown_intervals and not any(r['join_problems'] for r in rows) and not submitted_conflicts and not reservation_water['physical_overlap_errors']}


def counters(raw,start,end):
    post=raw.get('postload_export') or {};before=raw.get('before_load_export') or {}
    scheduler=post.get('scheduler',{});bs=before.get('scheduler',{})
    workers=post.get('workers',[]);bw={w.get('pid'):w for w in before.get('workers',[])}
    prefix_reset=nested(raw,'reset_scope.prefix_mean_scoring_stats')=='explicit clear in worker and scheduler at pre-load reset; absent Counter keys therefore zero'
    passive_reset=nested(raw,'reset_scope.passive_counts_times_samples')=='explicit clear'
    result={'manager_counts_delta':{},'worker_capture_calls_delta':None,'worker_capture_rows_delta':None,
            'worker_prefix_mean_scoring_delta':{},'scheduler_prefix_mean_scoring_delta':{},
            'source':'explicit reset/export deltas; missing before/after fields stay null'}
    result['manager_counts_after']=scheduler.get('counts')
    result['worker_capture_after']=[w.get('capture') for w in workers]
    result['worker_prefix_mean_scoring_after']=[w.get('prefix_mean_scoring') for w in workers]
    result['scheduler_prefix_mean_scoring_after']=scheduler.get('prefix_mean_scoring')
    result['after_counter_note']='raw after values can include pre-load lifetime work; deltas alone isolate measurement unless reset semantics prove otherwise'
    for key,value in scheduler.get('counts',{}).items():
        old=bs.get('counts',{}).get(key,0 if passive_reset and isinstance(bs.get('counts'),dict) else None)
        result['manager_counts_delta'][key]=value-old if number(value) and number(old) and value>=old else None
    for field in ('calls','rows'):
        deltas=[]
        for w in workers:
            a=nested(w,'capture.'+field);b=nested(bw.get(w.get('pid'),{}),'capture.'+field)
            deltas.append(a-b if number(a) and number(b) and a>=b else None)
        result['worker_capture_'+field+'_delta']=total(deltas)
    for origin,after,prior in [('scheduler',scheduler,bs)]:
        for key,value in after.get('prefix_mean_scoring',{}).items():
            old=prior.get('prefix_mean_scoring',{}).get(key,0 if prefix_reset and isinstance(prior.get('prefix_mean_scoring'),dict) else None)
            result[origin+'_prefix_mean_scoring_delta'][key]=value-old if number(value) and number(old) and value>=old else None
    for key in set(k for w in workers for k in w.get('prefix_mean_scoring',{})):
        deltas=[]
        for w in workers:
            prior=bw.get(w.get('pid'),{})
            a=nested(w,'prefix_mean_scoring.'+key);b=prior.get('prefix_mean_scoring',{}).get(key,0 if prefix_reset and nested(prior,'metric_reset_scope.missing_counter_after_explicit_reset_is_zero') is True and isinstance(prior.get('prefix_mean_scoring'),dict) else None)
            deltas.append(a-b if number(a) and number(b) and a>=b else None)
        result['worker_prefix_mean_scoring_delta'][key]=total(deltas)
    result['worker_counts_delta']={}
    for key in set(k for w in workers for k in w.get('counts',{})):
        deltas=[]
        for w in workers:
            a=w.get('counts',{}).get(key);prior=bw.get(w.get('pid'),{})
            b=prior.get('counts',{}).get(key,0 if passive_reset and isinstance(prior.get('counts'),dict) else None)
            deltas.append(a-b if number(a) and number(b) and a>=b else None)
        result['worker_counts_delta'][key]=total(deltas)
    events=[e for e in raw.get('wait_hint_events',[]) if number(event_stamp(e)) and start<=event_stamp(e)<=end]
    unknown=[e for e in raw.get('wait_hint_events',[]) if not number(event_stamp(e))]
    result['wait_event_timestamp_coverage']={'total':len(raw.get('wait_hint_events',[])),'in_load':len(events),'missing_timestamp':len(unknown),'timestamp_fields':EVENT_TIMESTAMPS,
        'missing_timestamp_event_counts':dict(collections.Counter(e.get('event') for e in unknown)),
        'missing_timestamp_event_ids':[{'event':e.get('event'),'event_id':e.get('event_id')} for e in unknown]}
    result['wait_event_counts_all_recorded']=dict(collections.Counter(e.get('event') for e in raw.get('wait_hint_events',[])))
    result['wait_event_counts']=dict(collections.Counter(e.get('event') for e in events))
    result['scored_candidates']=total(e.get('scored_candidate_count') for e in events if e.get('event')=='nomination_decision')
    result['constrained_nominations']=sum(e.get('shared_budget_constrained') is True for e in events if e.get('event')=='nomination_decision') if events else None
    fields={'demand_h2d_bytes':'load_bytes','d2h_bytes':'store_bytes','preemptions':'preemption_count',
            'prefetch_consumed_bytes':'speculative_transfer_accounting.consumed_bytes',
            'prefetch_completed_bytes':'speculative_transfer_accounting.completed_bytes',
            'prefetch_unused_completed_bytes':'speculative_transfer_accounting.unconsumed_completed_bytes',
            'prefetch_waste_bytes':'speculative_transfer_costs.waste_completed_bytes',
            'splice_bytes':'prefetch_attribution.gpu_splice_bytes'}
    terminals=terminal_index(raw);matched=[]
    for row in raw.get('requests',[]):
        es=terminals.get(nested(row,'payload.request_id'),[])
        matched.append(es[0] if len(es)==1 else {})
    for name,field in fields.items():result[name]=total(nested(t,field) for t in matched)
    adoptions=[nested(t,'prefetch_attribution.native_adoption') for t in matched]
    result['native_adoption_true_requests']=sum(isinstance(a,dict) and a.get('same_physical_blocks') is True and bool(a.get('published_block_ids')) and a.get('actual_block_ids')==a.get('published_block_ids') for a in adoptions) if matched else None
    result['native_adoption_field_coverage']=sum(isinstance(a,dict) for a in adoptions)
    for field in ('transfer_job_statuses','speculative_transfer_job_statuses'):
        jobs={}
        for t in matched:
            for job in t.get(field,[]):jobs[(job.get('job_id'),job.get('request_id'))]=job
        result[field+'_unique_jobs']=len(jobs) if matched and all(field in t for t in matched) else None
        result[field+'_failed_jobs']=sum(j.get('success') is False for j in jobs.values()) if jobs else (0 if result[field+'_unique_jobs']==0 else None)
        result[field+'_unknown_success_jobs']=sum(not isinstance(j.get('success'),bool) for j in jobs.values()) if jobs else None
    pool=[s for s in scheduler.get('samples',[]) if number(s.get('at_ns')) and start<=s['at_ns']<=end]
    result['pool_sample_count']=len(pool)
    for field in ('cpu_resident','cpu_ready','cpu_pending','gpu_reserved','gpu_referenced','gpu_hashed'):
        result[field+'_peak_blocks']=peak(s.get(field) for s in pool)
    result['cpu_resident_peak_bytes_estimate']=(result['cpu_resident_peak_blocks']*nested(raw,'geometry.block_bytes') if number(result['cpu_resident_peak_blocks']) and number(nested(raw,'geometry.block_bytes')) else None)
    result['worker_summary_memory_after']=[w.get('summary_memory') for w in workers]
    result['reset_scope']=raw.get('reset_scope')
    result['counter_missing_zero_authorized']={'prefix_mean_scoring':prefix_reset,'passive_counts':passive_reset,'resources':False}
    return result


def summarize(raw, validator_path=None):
    errors=[]
    try: index_rows(raw)
    except ValueError as e:errors.append(str(e))
    start=raw.get('load_start_ns');end=raw.get('load_end_ns')
    valid_interval=number(start) and number(end) and end>start
    if not valid_interval:errors.append('missing_or_invalid_load_interval');start=end=0
    module,pin=validator(validator_path)
    terminals=terminal_index(raw)
    rows=[request_metrics(r,start,end,terminals,module) for r in raw.get('requests',[])]
    if any(r['errors'] for r in rows):errors.append('invalid_request_rows')
    planned=raw.get('planned_requests')
    if isinstance(planned,int) and not isinstance(planned,bool) and planned!=len(rows):errors.append('planned_request_count_mismatch')
    elapsed=(end-start)/1e9 if valid_interval else None
    arrivals=sum(number(r.get('intended_release_ns')) and start<=r['intended_release_ns']<=end for r in raw.get('requests',[]))
    backlog=sum(number(r.get('intended_release_ns')) and r['intended_release_ns']<=end and (not number(r.get('request_end_ns')) or r['request_end_ns']>end) for r in raw.get('requests',[]))
    return {'schema':'quantized-campaign-metrics-v3','metric_definition_version':METRIC_DEFINITION,'name':raw.get('name'),'status':raw.get('status'),
            'valid':not errors,'errors':errors,'validator':pin,'slo_origin':'investigator choices; per-request p95 ITL for request goodput, separate aggregate p95 latency SLO; receipt time includes coalesced chunk zero intervals',
            'slos':SLOS,'all':aggregate(rows,elapsed),'cohorts':{c:aggregate([r for r in rows if r['cohort']==c],elapsed) for c in sorted({r['cohort'] for r in rows})},
            'recorded_arrivals':arrivals,'recorded_backlog_at_load_end':backlog,
            'arrival_coverage_note':'only recorded request rows; unrecorded/cancelled tasks require planned_requests manifest',
            'planned_requests':raw.get('planned_requests'),'resources':resources(raw,start,end),
            'achieved_concurrency':concurrency(raw,start,end),'counters':counters(raw,start,end),'offline_staging_utility':staging_utility(raw,start,end),
            'proofs':{'requests':len(rows),'passed':sum(r['proof']['passed'] for r in rows),
                      'all_passed':bool(rows) and all(r['proof']['passed'] for r in rows),
                      'failures':[{'identity':r['identity'],**r['proof']} for r in rows if not r['proof']['passed']]},
            'configuration':{k:raw.get(k) for k in ('arm','seed','geometry','config','namespace','workload','corpus_identity','model_identity','artifact_pins','model_quantization_config','configuration','runtime_resolution','performance_eligible','server_command')},
            'requests':rows}


def compare(left,right,validator_path=None):
    a=index_rows(left);b=index_rows(right);sa=summarize(left,validator_path);sb=summarize(right,validator_path)
    common=sorted(set(a)&set(b));ea=terminal_index(left);eb=terminal_index(right)
    inventory={'left_only':[list(k) for k in sorted(set(a)-set(b))],'right_only':[list(k) for k in sorted(set(b)-set(a))],
               'prompt_mismatches':[],'output_mismatches':[],'key_mismatches':[],'missing_key_proofs':[],'cohort_mismatches':[]}
    for k in common:
        x,y=a[k],b[k]
        if nested(x,'payload.prompt')!=nested(y,'payload.prompt'):inventory['prompt_mismatches'].append(list(k))
        if x.get('output_token_ids')!=y.get('output_token_ids'):inventory['output_mismatches'].append(list(k))
        if x.get('cohort')!=y.get('cohort'):inventory['cohort_mismatches'].append(list(k))
        ts=[(events.get(nested(row,'payload.request_id')) or [{}]) for row,events in [(x,ea),(y,eb)]]
        chains=[key_chain(t[0]) if len(t)==1 else None for t in ts]
        if None in chains:inventory['missing_key_proofs'].append(list(k))
        elif chains[0]!=chains[1]:inventory['key_mismatches'].append(list(k))
    exact=all(not values for values in inventory.values())
    valid=exact and sa['valid'] and sb['valid'] and sa['proofs']['all_passed'] and sb['proofs']['all_passed'] and left.get('status')=='complete' and right.get('status')=='complete' and not sa['all']['failures'] and not sb['all']['failures']
    configuration_mismatches=[]
    for field in ('namespace','corpus_identity','model_identity'):
        if not left.get(field) or not right.get(field) or left[field]!=right[field]:configuration_mismatches.append(field)
    for field in ('source_ids','cohort','clients','turn_limit','output_cap','respect_eos','history','wait_origin','arrival_origin','private_wait_ms'):
        if nested(left,'workload.'+field)!=nested(right,'workload.'+field):configuration_mismatches.append('workload.'+field)
    metrics=('output_tokens_s','requests_s','goodput_requests_s','goodput_output_tokens_s')
    rates={m:{'left':sa['all'][m],'right':sb['all'][m], 'delta_pct':(sb['all'][m]/sa['all'][m]-1)*100 if number(sa['all'][m]) and sa['all'][m]>0 and number(sb['all'][m]) else None} for m in metrics}
    return {'schema':'quantized-campaign-comparison-v3','metric_definition_version':METRIC_DEFINITION,'left':sa['name'],'right':sb['name'],'matched_logical_requests':len(common),
            'identity_exact':exact,'matched_correctness_passed':valid,'inventory':inventory,
            'configuration_mismatches':configuration_mismatches,
            'rates_descriptive_only':rates,'performance_claim_eligible':valid and not configuration_mismatches and left.get('planned_requests')==len(a) and right.get('planned_requests')==len(b) and left.get('performance_eligible') is True and right.get('performance_eligible') is True,
            'eligibility_note':'no output waivers; proof validity does not compare KV bytes; source/budget/workload matching still requires campaign review',
            'left_summary':sa,'right_summary':sb}


def archive_previous_output(path):
    if not path.exists():return None
    previous=json.loads(path.read_text())
    prior_version=previous.get('metric_definition_version')
    if prior_version==METRIC_DEFINITION:return None
    tag='v2-p95-itl' if prior_version=='v2-per-request-p95-itl' else 'v1-maxgap' if prior_version is None else prior_version
    if not all(c.isalnum() or c in '-_' for c in tag):raise ValueError('invalid prior metric version archive tag')
    saved=path.with_name(path.stem+'-'+tag+path.suffix)
    if saved.exists() and saved.read_bytes()!=path.read_bytes():raise ValueError('refusing to overwrite different archived prior summary: '+str(saved))
    if not saved.exists():shutil.copy2(path,saved)
    return {'archive':str(saved),'sha256':hashlib.sha256(saved.read_bytes()).hexdigest(),'definition':prior_version or 'v1 max-gap ITL request-goodput','raw_data_unchanged':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validator',type=Path)
    subs=parser.add_subparsers(dest='command',required=True)
    for cmd in ('summarize','compare'):
        p=subs.add_parser(cmd);p.add_argument('left',type=Path)
        if cmd=='compare':p.add_argument('right',type=Path)
        p.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();left,lp=load(args.left)
    result=summarize(left,args.validator) if args.command=='summarize' else compare(left,load(args.right)[0],args.validator)
    result['left_raw']=lp
    if args.command=='compare':result['right_raw']=load(args.right)[1]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    archived=archive_previous_output(args.output)
    if archived:result['replaces_prior_metric_definition']=archived
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
