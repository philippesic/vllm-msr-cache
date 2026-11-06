"""Independent operational-case checks, respecting per-wave HTTP identities."""
import hashlib
from pathlib import Path
from http_confirmation_receipt_astra_nov3 import verify_http
from gate_once_receipt_astra_nov3 import verify_gpu_gate
H=Path(__file__).resolve().parent
ROOT=H.parents[1]


def verify_case(raw,item):
    issues=[];waves=[];gate=item['arm']=='candidate'
    args=raw.get('args',{});cfg=raw.get('config',{}).get('kv_connector_extra_config',{})
    contract=raw.get('operational_contract',{})
    origin=contract.get('gateway_origin_ns');limit=args.get('registry_limit')
    if raw.get('performance_eligible') is not False or raw.get('operational_qualification') is not True:issues.append('operational_scope_not_executed')
    if args.get('audit') is not False:issues.append('unexpected_intrusive_audit')
    if gate and (args.get('no_audit') is not True or cfg.get('preservation_audit') is not False):issues.append('bounded_candidate_mode_mismatch')
    if args.get('return_priority_ms')!=8000:issues.append('priority_credit_mismatch')
    if contract.get('between_wave_cache_reset') is not False or contract.get('between_wave_counter_reset') is not False:issues.append('between_wave_reset_contract')
    if origin!=raw.get('load_start_ns') or type(origin) is not int:issues.append('gateway_origin_mismatch')
    expected_wave_count=int(item['args'][item['args'].index('--waves')+1])
    rows=raw.get('waves',[])
    if len(rows)!=expected_wave_count:issues.append('wave_count_mismatch')
    if [w.get('wave_index') for w in rows]!=list(range(expected_wave_count)):issues.append('wave_index_order_mismatch')
    all_ids=set();all_sids=set();capacity=0
    for wave in rows:
        wi=[];identity=raw['name']+'-wave-'+str(wave.get('wave_index'))
        requests=wave.get('requests',[])
        expected=int(args.get('clients',0))*int(args.get('turns',0))
        if len(requests)!=expected or not all(q.get('success') and q.get('natural_eos') for q in requests):wi.append('not_all_requests_natural_EOS')
        if wave.get('application_namespace')!=identity:wi.append('wave_namespace_mismatch')
        local_sids=set()
        for q in requests:
            b=q.get('payload',{});rid=b.get('request_id');sid=q.get('session_id');t=q.get('turn');submitted=q.get('priority_clock_ns')
            if rid in all_ids:wi.append('duplicate_request_id')
            all_ids.add(rid);local_sids.add(sid)
            if not isinstance(rid,str) or not rid.startswith('sc|'+identity+'|persistent-wave|fixed|sample-'):wi.append('request_namespace_mismatch')
            if not isinstance(sid,str) or not sid.startswith(identity+'::'):wi.append('session_namespace_mismatch')
            if type(t) is not int or type(submitted) is not int or type(origin) is not int or submitted<origin or q.get('priority_origin_ns')!=origin or q.get('request_start_ns')!=submitted or q.get('priority_credit_ms')!=8000:
                wi.append('priority_clock_mismatch')
            elif b.get('priority')!=(submitted-origin)//1000000-(8000 if t>0 else 0):wi.append('priority_arithmetic_mismatch')
            if b.get('max_tokens')!=raw['geometry']['max_model_len']-len(b.get('prompt',[])) or b.get('ignore_eos',False) is not False or b.get('stop') or b.get('stop_token_ids'):wi.append('generation_contract_mismatch')
        if local_sids & all_sids:wi.append('cross_wave_session_identity_reused')
        all_sids.update(local_sids)
        http=verify_http(wave)
        if not http['qualified']:wi.append('HTTP_accounting_failure')
        for h in wave.get('hints',[]):
            hint=h.get('hint',{})
            if hint.get('run_id')!=identity or hint.get('cell_id')!='persistent-wave' or hint.get('session_id') not in local_sids:wi.append('hint_wave_namespace_mismatch')
        merged={**wave,'config':raw.get('config'),'args':args,'namespace':raw.get('namespace')}
        gating=verify_gpu_gate(merged,'gate' if gate else 'native')
        exhausted=[h for h in wave.get('hints',[]) if h.get('response',{}).get('reason')=='session_registry_capacity']
        capacity+=len(exhausted)
        allowed_no_gate=(gate and limit==128 and bool(exhausted) and len(exhausted)==len(wave.get('hints',[])) and gating['issues']==['no_fresh_gate_ACKs'] and gating['cadence']['qualified'])
        if not gating['qualified'] and not allowed_no_gate:wi.append('gate_or_causal_receipt_failure')
        if exhausted and limit!=128:wi.append('unexpected_registry_exhaustion_below_production_limit')
        wi=sorted(set(wi))
        waves.append({'wave_index':wave.get('wave_index'),'seed':wave.get('seed'),'status':wave.get('status'),
                      'requests':len(requests),'natural_eos':sum(q.get('natural_eos') is True for q in requests),
                      'HTTP':http,'gate':gating,'capacity_no_action_ACKs':len(exhausted),
                      'capacity_only_gate_unavailable_expected':allowed_no_gate,'issues':wi,'qualified':not wi})
        issues.extend('wave'+str(wave.get('wave_index'))+':'+x for x in wi)
    scalar_checks=[]
    exports=[w.get(k) for w in rows for k in ('immediate_export','settled_export')]
    exports += [raw.get('overlap_settled_export'),raw.get('postload_export')]
    for exported in exports:
        if not isinstance(exported,dict):continue
        scalar=exported.get('scheduler',{}).get('operational_scalar_state')
        if not isinstance(scalar,dict):issues.append('missing_scalar_export');continue
        entry={'pid':scalar.get('pid'),'at_ns':scalar.get('at_ns'),'issues':[]}
        if gate:
            lengths=scalar.get('scheduler_lengths',{})
            for field,bound in [('_observed_prefix_generations',limit),('_observed_prefixes',1024),('_wait_hint_results',4096)]:
                value=lengths.get(field)
                if type(value) is not int or type(bound) is not int or not 0<=value<=bound:entry['issues'].append('bounded_metadata:'+field)
            if scalar.get('session_registry_limit')!=limit:entry['issues'].append('actual_registry_limit_mismatch')
            if scalar.get('preservation_audit_enabled') is not False:entry['issues'].append('actual_lifetime_audit_enabled')
            for field in ['_preservation_requests','_preservation_finished','_preservation_jobs','_preservation_job_events']:
                if lengths.get(field)!=0:entry['issues'].append('lifetime_audit_history_not_empty:'+field)
        scalar_checks.append(entry)
        issues.extend(entry['issues'])
    if item['phase']=='D':
        observed=[x.get('scheduler',{}).get('operational_scalar_state',{}).get('scheduler_lengths',{}).get('_observed_prefix_generations') for x in exports if isinstance(x,dict)]
        if 128 not in observed:issues.append('accelerated_registry_capacity_not_observed')
    archives=[]
    for archive,source in [('driver.py',H/'run_live_operational_v2_nov3.py'),('workload.py',H/'live_operational_workload_v2_nov3.py'),('hooks.py',H/'hooks-operational-v2/sitecustomize.py'),('public_priority_helper.py',ROOT/'semantic_offload/request_priority.py'),('wave_driver.py',H/'operational_waves_v2_nov3.py'),('scalar_telemetry.py',H/'operational_state_v2_nov3.py'),('abort_observer.py',H/'operational_abort_v2_nov3.py'),('cancel_probe.py',H/'operational_cancel_probe_v2_nov3.py'),('failure_capture.py',H/'operational_failure_capture_v2_nov3.py')]:
        q=H/'runs'/raw['name']/archive
        valid=q.is_file() and hashlib.sha256(q.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
        archives.append({'archive':archive,'matches_frozen_source':valid})
        if not valid:issues.append('archive_mismatch:'+archive)
    probe=raw.get('cancellation_probe');abort_receipt=None
    if item.get('deliberate_abort_planned'):
        abort_receipt=verify_abort_http(probe,raw['geometry']['max_model_len'],origin)
        if not abort_receipt['qualified']:issues.append('abort_or_health_qualification_failure')
    elif probe is not None:issues.append('unplanned_cancel_probe')
    if item['phase']=='D' and capacity==0:issues.append('accelerated_capacity_path_not_exercised')
    if len(raw.get('requests',[]))!=item['normal_natural_eos_planned']:issues.append('flattened_request_count_mismatch')
    return {'qualified':not issues,'issues':issues,'waves':waves,'scalar_bound_checks':scalar_checks,'archive_receipts':archives,'abort_receipt':abort_receipt,
            'natural_eos':sum(w['natural_eos'] for w in waves),'capacity_no_action_ACKs':capacity,
            'scope':'Operational basic correctness, not a memory-plateau/performance/production endorsement. Metadata bounds and resource trends require full analysis.'}


def verify_abort_http(probe,max_model_len=7680,origin_ns=None):
    if not isinstance(probe,dict):return {'qualified':False,'issues':['missing_probe']}
    issues=[];cfg=probe.get('http_client_configuration',{})
    if any(cfg.get(k)!=v for k,v in {'force_close':True,'automatic_POST_retry':False,'application_retries':0,'connection_trace':True}.items()):issues.append('HTTP_configuration')
    abort=probe.get('abort_request',{});health=probe.get('health_requests',[])
    if len(health)!=4 or not all(q.get('http_status')==200 and q.get('natural_eos') and q.get('terminal_complete') and q.get('finish_reasons')==['stop'] for q in health):issues.append('health_not_all_EOS')
    if abort.get('terminal_before_close') is not False or not abort.get('disconnect_ns') or abort.get('http_status')!=200:issues.append('abort_not_exercised')
    for q in [abort,*health]:
        body=q.get('payload',{})
        if body.get('max_tokens')!=max_model_len-len(body.get('prompt',[])) or body.get('ignore_eos',False) is not False or body.get('stop') or body.get('stop_token_ids'):issues.append('probe_generation_contract')
    expected_id=abort.get('payload',{}).get('request_id')
    records=probe.get('settled_after_health_export',{}).get('scheduler',{}).get('operational_abort_transitions',{}).get('records',[])
    matching=[r for r in records if r.get('external_mapping_available') is True and r.get('external_request_id')==expected_id and r.get('status')=='FINISHED_ABORTED']
    if len(matching)!=1 or len(records)!=1 or probe.get('settled_after_health_export',{}).get('scheduler',{}).get('operational_abort_transitions',{}).get('count')!=1:issues.append('actual_abort_transition_missing_foreign_or_duplicate')
    for q in [abort,*health]:
        stamp=q.get('priority_clock_ns');origin=q.get('priority_origin_ns');credit=q.get('priority_credit_ms');turn=q.get('priority_turn');start=q.get('request_start_ns')
        if not all(type(v) is int for v in (stamp,origin,credit,turn,start)) or stamp<origin or start<stamp or origin!=origin_ns or credit!=8000 or turn!=0:
            issues.append('probe_priority_clock_contract')
        elif q.get('payload',{}).get('priority')!=(stamp-origin)//1000000:
            issues.append('probe_priority_arithmetic')
    expected={('intentional_disconnect',abort.get('payload',{}).get('request_id'))}
    expected.update(('health_generation',q.get('payload',{}).get('request_id')) for q in health)
    groups={}
    for event in probe.get('http_connection_events',[]):
        req=event.get('request') or {};groups.setdefault((req.get('kind'),req.get('request_id')),[]).append(event)
    if set(groups)!=expected or len(expected)!=5:issues.append('HTTP_event_identity_count')
    for key,events in groups.items():
        if [e.get('event') for e in events]!=['request_start','connection_create_start','connection_create_end','request_headers_received'] or events[-1].get('status')!=200:issues.append('unexpected_HTTP_trace:'+str(key))
    if probe.get('qualified') is not True:issues.append('driver_probe_not_qualified')
    return {'qualified':not issues,'issues':issues,'natural_eos_health':sum(q.get('natural_eos') is True for q in health),
            'actual_abort_matches':matching,'deliberate_disconnect_count':1,'HTTP_attempts':len(expected),
            'scope':'Deliberate stream close is separate from completed natural EOS and unexpected transport errors.'}
