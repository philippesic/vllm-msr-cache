"""Create new operational v2 files without changing any stopped-block inputs."""
from pathlib import Path
H=Path(__file__).resolve().parent

def once(text,old,new):
 assert text.count(old)==1,(old[:80],text.count(old))
 return text.replace(old,new)

def create(name,text):
 p=H/name
 if p.exists() and p.read_text()!=text:raise FileExistsError('Preserve independently edited v2 file: '+str(p))
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)

abort=(H/'operational_abort_nov3.py').read_text()
abort=once(abort,"ABORTS.append({'request_id': req.request_id, 'status': status.name,", "params = getattr(req, 'kv_transfer_params', None)\n                external_id = params.get('external_request_id') if isinstance(params, dict) else None\n                ABORTS.append({'request_id': req.request_id, 'external_request_id': external_id if type(external_id) is str else None, 'external_mapping_available': type(external_id) is str, 'status': status.name,")
abort=abort.replace("Only requests returned by native finish_requests in FINISHED_ABORTED state", "Only actual returned FINISHED_ABORTED Requests; external_request_id read directly from Request.kv_transfer_params, never parsed from internal ID")
create('operational_abort_v2_nov3.py',abort)

probe=(H/'operational_cancel_probe_nov3.py').read_text()
probe=once(probe,"body['priority'] = gateway_priority(time.monotonic_ns(), origin_ns, 0, 8000)", "priority_clock_ns = time.monotonic_ns()\n    body['priority'] = gateway_priority(priority_clock_ns, origin_ns, 0, 8000)")
probe=once(probe,'    return body\n', "    return body, {'priority_clock_ns': priority_clock_ns, 'priority_origin_ns': origin_ns, 'priority_credit_ms': 8000, 'priority_turn': 0}\n")
probe=once(probe,"body = probe_payload(sources[0], a.name, 0, origin_ns)", "body, priority_receipt = probe_payload(sources[0], a.name, 0, origin_ns)")
probe=once(probe,"abort = {'payload': body, 'request_start_ns':", "abort = {'payload': body, **priority_receipt, 'request_start_ns':")
probe=once(probe,"payload = probe_payload(source, a.name, ordinal, origin_ns)", "payload, priority_receipt = probe_payload(source, a.name, ordinal, origin_ns)")
probe=once(probe,"row = {'payload': payload, 'request_start_ns':", "row = {'payload': payload, **priority_receipt, 'request_start_ns':")
start=probe.index("    # Pinned installed completion serving")
end=probe.index("    outcome['actual_abort_transition_matches']",start)
probe=probe[:start]+'''    # Exact direct mapping from the actual returned aborted Request; internal
    # randomization and prefix parsing are irrelevant to acceptance.
    abort['expected_external_request_id'] = abort['payload']['request_id']
    observed = outcome['settled_after_health_export']['scheduler'].get('operational_abort_transitions')
    outcome['abort_observer'] = observed
    matches = [r for r in (observed or {}).get('records', []) if r.get('external_mapping_available') is True and r.get('external_request_id') == abort['expected_external_request_id'] and r.get('status') == 'FINISHED_ABORTED']
''' + probe[end:]
probe=probe.replace("and len(matches) == 1 and len(outcome['health_requests'])", "and len(matches) == 1 and (observed or {}).get('count') == 1 and len((observed or {}).get('records', [])) == 1 and len(outcome['health_requests'])")
create('operational_cancel_probe_v2_nov3.py',probe)

waves=(H/'operational_waves_nov3.py').read_text().replace('live_operational_workload_nov3','live_operational_workload_v2_nov3').replace('operational_cancel_probe_nov3','operational_cancel_probe_v2_nov3')
create('operational_waves_v2_nov3.py',waves)
create('live_operational_workload_v2_nov3.py',(H/'live_operational_workload_nov3.py').read_text())
create('operational_state_v2_nov3.py',(H/'operational_state_nov3.py').read_text())
hook=(H/'hooks-operational/sitecustomize.py').read_text().replace('operational_state_nov3','operational_state_v2_nov3').replace('operational_abort_nov3','operational_abort_v2_nov3')
create('hooks-operational-v2/sitecustomize.py',hook)

failure='''"""Best-effort post-failure diagnostics, preserving original failure status/error."""
import time
import urllib.request
from runtime_support import control

def capture(handle,result):
    receipt={'started_ns':time.monotonic_ns(),'operations':{},'scope':'After workload task cleanup only; no reset, no new generation, original failure unchanged.'}
    result['failure_final_capture']=receipt
    if 'postload_export' not in result:
        try:
            result['postload_export']=control(handle,'export')
            receipt['operations']['export']={'success':True}
        except Exception as exc:receipt['operations']['export']={'success':False,'error':repr(exc)}
    if 'metrics_final' not in result:
        try:
            with urllib.request.urlopen(handle.metrics_url(),timeout=30) as response:result['metrics_final']=response.read().decode()
            receipt['operations']['metrics']={'success':True}
        except Exception as exc:receipt['operations']['metrics']={'success':False,'error':repr(exc)}
    receipt['completed_ns']=time.monotonic_ns()
'''
create('operational_failure_capture_v2_nov3.py',failure)
driver=(H/'run_live_operational_nov3.py').read_text()
for old,new in [('live_operational_workload_nov3','live_operational_workload_v2_nov3'),('operational_waves_nov3','operational_waves_v2_nov3'),('operational_state_nov3','operational_state_v2_nov3'),('operational_abort_nov3','operational_abort_v2_nov3'),('operational_cancel_probe_nov3','operational_cancel_probe_v2_nov3'),('hooks-operational','hooks-operational-v2')]:driver=driver.replace(old,new)
driver=once(driver,"('cancel_probe.py',H/'operational_cancel_probe_v2_nov3.py')", "('cancel_probe.py',H/'operational_cancel_probe_v2_nov3.py'),('failure_capture.py',H/'operational_failure_capture_v2_nov3.py')")
driver=once(driver,"H/'operational_cancel_probe_v2_nov3.py',H/'hooks-operational-v2", "H/'operational_cancel_probe_v2_nov3.py',H/'operational_failure_capture_v2_nov3.py',H/'hooks-operational-v2")
driver=once(driver,"   if handle:handle.shutdown();result['server_exit_code']=handle.proc.poll()", "   if handle and result.get('status')=='failed':\n    from operational_failure_capture_v2_nov3 import capture\n    capture(handle,result)\n   if handle:handle.shutdown();result['server_exit_code']=handle.proc.poll()")
create('run_live_operational_v2_nov3.py',driver)
