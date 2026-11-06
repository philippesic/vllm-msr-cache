"""Prospective remaining-seven-case review plan; no GPU launch authorization."""
import copy,hashlib,json,sysconfig
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1];E=ROOT/'evidence/live-optimization-2080ti-20261103'
old_path=E/'operational-frozen.json';old=json.loads(old_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
historical={}
for field in ('existing_frozen_inputs','new_review_file_pins','file_pins'):
 for name,pin in old.get(field,{}).items():
  path=str((ROOT/name).resolve())
  assert path not in historical or historical[path]==pin
  historical[path]=pin
readme=str((ROOT/'README.md').resolve());live=dict(historical);old_readme=live[readme];live[readme]=sha(ROOT/'README.md')
transition={'schema':'operational-v2-one-file-readme-pin-transition-v1','original_manifest_sha256':sha(old_path),'path':readme,'historical_sha256':old_readme,'new_live_sha256':live[readme],
 'reason':'Concurrent user local commit/merge changed README during original block. Preserve merged content; no serving/module pin changed. ROOT provenance independently retained.',
 'root_provenance':str(E/'operational-readme-drift-root-verification.json'),'root_provenance_sha256':sha(E/'operational-readme-drift-root-verification.json'),
 'serving_runtime_harness_change_due_to_this_transition':False,'old_failed_block_preserved':True}
(H/'operational-v2-readme-pin-transition.json').write_text(json.dumps(transition,indent=2)+'\n')
cases=[]
for prior in old['execution_order']:
 if prior['phase']=='A':continue
 case=copy.deepcopy(prior);case['name']=prior['name'].replace('live-nov3-operational-','live-nov3-operational-v2-',1)
 case['args'][case['args'].index('--name')+1]=case['name'];case['original_unattempted_case']=prior['name']
 if case['phase']=='C':
  case['args'].append('--cancel-probe');case['deliberate_abort_planned']=1;case['health_natural_eos_planned']=4
 cases.append(case)
assert len(cases)==7 and sum(c['normal_natural_eos_planned'] for c in cases)==1872
names=['run_live_operational_v2_nov3.py','live_operational_workload_v2_nov3.py','operational_waves_v2_nov3.py','operational_state_v2_nov3.py','operational_abort_v2_nov3.py','operational_cancel_probe_v2_nov3.py','operational_failure_capture_v2_nov3.py','hooks-operational-v2/sitecustomize.py','supervise_operational_v2_astra_nov3.py','operational_case_receipt_v2_astra_nov3.py','test_operational_v2_harness_nov3.py','test_operational_v2_supervisor_nov3.py','test_operational_v2_additions_nov3.py','prepare_operational_v2_harness_nov3.py','prepare_operational_v2_control_nov3.py','prepare_operational_v2_plan_nov3.py','operational-v2-readme-pin-transition.json','live-nov3-operational-v2-cpu-tests.xml']
files=[H/n for n in names if (H/n).exists()]
base=Path(sysconfig.get_paths()['purelib'])/'vllm'
files += [base/'v1/request.py',base/'v1/engine/input_processor.py',base/'entrypoints/openai/completion/serving.py',E/'operational-readme-drift-root-verification.json',E/'operational-readme-concurrent-change.patch']
newpins={str(p.resolve()):sha(p) for p in files}
for name,pin in live.items():
 assert Path(name).is_file() and sha(Path(name))==pin,('Additional old pin drift',name)
plan={'schema':'astra-operational-v2-review-plan-v1','launch_authorized':False,'freeze_pending_root_review':True,
 'python':old['python'],'runner':str(H/'run_live_operational_v2_nov3.py'),'execution_order':cases,
 'planned_server_cases':7,'planned_waves':26,'normal_natural_eos_planned':1872,'health_natural_eos_planned':8,'total_natural_eos_planned':1880,'deliberate_aborts_planned':2,
 'scope':'Unattempted original B–E configurations only under new names; two new direct-mapped cancellation qualifications added to C. No A reruns/replacements/relabeling. All rates descriptive; existing fresh confirmation unchanged.',
 'historical_original_manifest':str(old_path),'historical_original_manifest_sha256':sha(old_path),'historical_original_frozen_inputs':historical,
 'existing_frozen_inputs':live,'new_review_file_pins':newpins,
 'readme_pin_transition':transition,
 'measurement_contract':{**old['measurement_contract'],
  'cancellation':'Exact equality between submitted payload request_id and external_request_id read directly from actual returned native FINISHED_ABORTED Request.kv_transfer_params. Exactly one actual aborted record/count; missing/foreign/duplicate direct mapping fails. No suffix parsing for acceptance.',
  'probe_priority_receipt':'Every probe/health request records exact helper clock, shared case epoch, credit8000 and turn0, independently recomputed by receipt verifier.',
  'failure_metrics':'Best-effort final worker/scheduler export and Prometheus capture after workload/overlap task cleanup only; retain original failed status/error/traceback. No reset or additional generation.',
  'new_hook_overhead':'Common bounded32 direct abort records and scalar exports; candidate/native same v2 driver/hooks, no serving-cache changes.'},
 'acceptance_checks':old['acceptance_checks'],
 'stop_conditions':old['stop_conditions'],
 'failure_count_scope':'Three consecutive failures within this new seven-case block; the two prior A failures remain historical failures and are not replaced.',
 'prelaunch_requirements':['ROOT independent review of direct mapping/clock/failure capture and CPU controls.','New freeze and separate manifest+supervisor-bound ROOT GPU authorization.','All live pins before every launch/end; current README explicitly certified one-file transition; no other original pin drift.','All seven new run names absent, GPUidle/lockavailable.'],
 'production_endorsement':False}
(H/'operational-v2-review-plan-nov3.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'cases':7,'waves':26,'normal_EOS':1872,'health_EOS':8,'old_pins':len(historical),'only_original_live_pin_change':'README.md','newpins':len(newpins),'launch_authorized':False}))
