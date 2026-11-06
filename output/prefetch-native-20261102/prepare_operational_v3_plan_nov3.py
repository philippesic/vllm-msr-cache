"""Prospective D/E-only continuation; preparation does not authorize GPU use."""
import copy,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1];E=ROOT/'evidence/live-optimization-2080ti-20261103'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old_path=E/'operational-v2-frozen.json';old=json.loads(old_path.read_text());historical={}
for field in ('existing_frozen_inputs','new_review_file_pins','file_pins'):
 for name,pin in old.get(field,{}).items():
  path=str((ROOT/name).resolve());assert path not in historical or historical[path]==pin
  historical[path]=pin
readme=str((ROOT/'README.md').resolve());live=dict(historical);old_readme=live.pop(readme)
assert len(historical)==813 and len(live)==812
for name,pin in live.items():assert Path(name).is_file() and sha(Path(name))==pin,('Execution/input drift',name)
transition={'schema':'operational-v3-document-provenance-v1','path':readme,'historical_v2_sha256':old_readme,'current_at_preparation_sha256':sha(ROOT/'README.md'),
 'historical_v1_sha256':json.loads((H/'operational-v2-readme-pin-transition.json').read_text())['historical_sha256'],
 'root_drift_receipt':str(E/'operational-v2-readme-drift-root-verification.json'),'root_drift_receipt_sha256':sha(E/'operational-v2-readme-drift-root-verification.json'),
 'policy':'Assistant-chosen prospective correction: README is nonexecuted provenance only, sampled at launch/end without mutation stop. Every other existing execution/input pin is unchanged and remains a hard stop.',
 'historical_failures_and_stops_unchanged':True,'serving_changes':False}
(H/'operational-v3-document-provenance.json').write_text(json.dumps(transition,indent=2)+'\n')
cases=[]
for prior in old['execution_order']:
 if prior['phase'] not in ('D','E'):continue
 case=copy.deepcopy(prior);case['name']=prior['name'].replace('operational-v2-','operational-v3-',1)
 case['args'][case['args'].index('--name')+1]=case['name'];case['original_unattempted_v2_case']=prior['name'];cases.append(case)
assert len(cases)==3 and sum(c['normal_natural_eos_planned'] for c in cases)==1008
assert sum(c['deliberate_abort_planned']+c['health_natural_eos_planned'] for c in cases)==0
names=['supervise_operational_v3_astra_nov3.py','prepare_operational_v3_plan_nov3.py','test_operational_v3_supervisor_nov3.py','operational-v3-document-provenance.json','live-nov3-operational-v3-cpu-tests.xml','live-nov3-operational-v3-root-race-controls.xml','live-nov3-operational-v3-post-race-cpu-tests.xml']
files=[H/n for n in names if (H/n).exists()]+[E/'operational-v2-readme-drift-root-verification.json',E/'operational-bc-root-verification.json',H/'root_operational_case_analysis_nov3.py']
newpins={str(p.resolve()):sha(p) for p in files}
plan={'schema':'astra-operational-v3-review-plan-v1','launch_authorized':False,'freeze_pending_root_review':True,
 'python':old['python'],'runner':old['runner'],'execution_order':cases,'planned_server_cases':3,'planned_waves':14,
 'normal_natural_eos_planned':1008,'health_natural_eos_planned':0,'total_natural_eos_planned':1008,'deliberate_aborts_planned':0,
 'scope':'Only never-attempted D/E under new names: candidate128registry eightwaves and matched candidate/native three finite overlapping bursts. No A/B/C reruns, replacement, or relabeling. All rates descriptive.',
 'historical_v2_manifest':str(old_path),'historical_v2_manifest_sha256':sha(old_path),'historical_v2_frozen_inputs':historical,
 'existing_frozen_inputs':live,'new_review_file_pins':newpins,'document_provenance':transition,
 'measurement_contract':old['measurement_contract'],'acceptance_checks':old['acceptance_checks'],'stop_conditions':old['stop_conditions'],
 'failure_count_scope':'Three consecutive failed cases within this new block; earlier failures/stops preserved.',
 'prelaunch_requirements':['ROOT review and separate manifest/supervisor-bound launch certificate.','Every execution/input pin before each launch/end; only README recorded as nonexecuted provenance without stopping.','Three new names absent, idle GPU and available lock.'],
 'production_endorsement':False}
(H/'operational-v3-review-plan-nov3.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'cases':3,'waves':14,'normal_EOS':1008,'preserved_execution_pins':len(live),'newpins':len(newpins),'launch_authorized':False}))
