"""New v2 receipt/supervisor; stopped-block files are read-only templates."""
from pathlib import Path
H=Path(__file__).resolve().parent

def create(name,text):
 p=H/name
 if p.exists() and p.read_text()!=text:raise FileExistsError(name)
 p.write_text(text)

receipt=(H/'operational_case_receipt_astra_nov3.py').read_text()
for old,new in [('run_live_operational_nov3','run_live_operational_v2_nov3'),('live_operational_workload_nov3','live_operational_workload_v2_nov3'),('operational_waves_nov3','operational_waves_v2_nov3'),('operational_state_nov3','operational_state_v2_nov3'),('operational_abort_nov3','operational_abort_v2_nov3'),('operational_cancel_probe_nov3','operational_cancel_probe_v2_nov3'),('hooks-operational/','hooks-operational-v2/')]:receipt=receipt.replace(old,new)
receipt=receipt.replace("('cancel_probe.py',H/'operational_cancel_probe_v2_nov3.py')", "('cancel_probe.py',H/'operational_cancel_probe_v2_nov3.py'),('failure_capture.py',H/'operational_failure_capture_v2_nov3.py')")
receipt=receipt.replace("abort_receipt=verify_abort_http(probe,raw['geometry']['max_model_len'])", "abort_receipt=verify_abort_http(probe,raw['geometry']['max_model_len'],origin)")
receipt=receipt.replace('def verify_abort_http(probe,max_model_len=7680):', 'def verify_abort_http(probe,max_model_len=7680,origin_ns=None):')
receipt=receipt.replace("expected_id=(abort.get('response_id') or '')+'-0'", "expected_id=abort.get('payload',{}).get('request_id')")
receipt=receipt.replace("r.get('request_id')==expected_id and r.get('status')=='FINISHED_ABORTED'", "r.get('external_mapping_available') is True and r.get('external_request_id')==expected_id and r.get('status')=='FINISHED_ABORTED'")
receipt=receipt.replace("if len(matching)!=1:issues.append('actual_abort_transition_missing_or_duplicate')", "if len(matching)!=1 or len(records)!=1 or probe.get('settled_after_health_export',{}).get('scheduler',{}).get('operational_abort_transitions',{}).get('count')!=1:issues.append('actual_abort_transition_missing_foreign_or_duplicate')")
marker="    expected={('intentional_disconnect',abort.get('payload',{}).get('request_id'))}"
replacement="""    for q in [abort,*health]:
        stamp=q.get('priority_clock_ns');origin=q.get('priority_origin_ns');credit=q.get('priority_credit_ms');turn=q.get('priority_turn');start=q.get('request_start_ns')
        if not all(type(v) is int for v in (stamp,origin,credit,turn,start)) or stamp<origin or start<stamp or origin!=origin_ns or credit!=8000 or turn!=0:
            issues.append('probe_priority_clock_contract')
        elif q.get('payload',{}).get('priority')!=(stamp-origin)//1000000:
            issues.append('probe_priority_arithmetic')
"""+marker
assert marker in receipt;receipt=receipt.replace(marker,replacement)
create('operational_case_receipt_v2_astra_nov3.py',receipt)

supervisor=(H/'supervise_operational_astra_nov3.py').read_text().replace('operational_case_receipt_astra_nov3','operational_case_receipt_v2_astra_nov3').replace('operational-review-plan-nov3.json','operational-v2-review-plan-nov3.json').replace('live-nov3-operational-custody','live-nov3-operational-v2-custody').replace('live-nov3-operational-progress','live-nov3-operational-v2-progress').replace('live-nov3-operational-events','live-nov3-operational-v2-events').replace('operational-frozen-astra-copy','operational-v2-frozen-astra-copy')
supervisor=supervisor.replace("!=9:raise ValueError('Expected exactly nine server cases')", "!=7:raise ValueError('Expected exactly seven server cases')").replace("len(state['runs'])==9", "len(state['runs'])==7").replace('exact nine-case plan','exact seven-case plan').replace('astra-operational-custody-v1','astra-operational-v2-custody-v1')
create('supervise_operational_v2_astra_nov3.py',supervisor)
