"""ROOT independent review/freeze of remaining B-E cases and direct abort repair."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

R=Path('/home/philip/vllm-semantic-cache-ext')
H=R/'output/prefetch-native-20261102'
E=R/'evidence/live-optimization-2080ti-20261103'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def normalized(group):return {str((R/k).resolve()):v for k,v in group.items()}
def merged(m):
    result={}
    for key in ('existing_frozen_inputs','new_review_file_pins','file_pins'):
        for path,pin in normalized(m.get(key,{})).items():
            assert path not in result or result[path]==pin,path
            result[path]=pin
    return result

if __name__=='__main__':
    freeze=E/'operational-v2-frozen.json';auth=E/'operational-v2-launch-root-authorization.json'
    assert not freeze.exists() and not auth.exists()
    old=json.loads((E/'operational-frozen.json').read_text())
    plan_path=H/'operational-v2-review-plan-nov3.json'
    plan=json.loads(plan_path.read_text())
    inventory=json.loads((H/'live-nov3-operational-v2-review-inventory.json').read_text())
    historical=merged(old)
    assert normalized(plan['historical_original_frozen_inputs'])==historical
    current=normalized(plan['existing_frozen_inputs'])
    assert current.keys()==historical.keys()
    changed=[p for p in current if current[p]!=historical[p]]
    assert changed==[str(R/'README.md')]
    assert sha(R/'README.md')==current[str(R/'README.md')]
    original=old['execution_order'][2:]
    cases=plan['execution_order'];assert len(cases)==len(original)==7
    for previous,case in zip(original,cases):
        assert (case['phase'],case['arm'])==(previous['phase'],previous['arm'])
        expected=copy.deepcopy(previous['args'])
        expected[expected.index('--name')+1]=case['name']
        assert case['name']==previous['name'].replace('operational-','operational-v2-',1)
        if case['phase']=='C':expected.append('--cancel-probe')
        assert case['args']==expected,case['name']
        assert case['normal_natural_eos_planned']==previous['normal_natural_eos_planned']
        assert case['deliberate_abort_planned']==(1 if case['phase']=='C' else 0)
        assert case['health_natural_eos_planned']==(4 if case['phase']=='C' else 0)
        assert not (H/'runs'/case['name']).exists()
    assert sum(c['normal_natural_eos_planned'] for c in cases)==1872
    assert sum(c['health_natural_eos_planned'] for c in cases)==8
    assert sum(c['deliberate_abort_planned'] for c in cases)==2
    prospective=merged(plan)
    for path,pin in inventory['file_pins'].items():
        assert str(Path(path).resolve()) not in prospective or prospective[str(Path(path).resolve())]==pin
        prospective[str(Path(path).resolve())]=pin
    for path,pin in prospective.items():assert sha(Path(path))==pin,path
    with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with socket.socket() as s:
            s.settimeout(1);assert s.connect_ex(('127.0.0.1',19372))!=0
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    report={'schema':'root-operational-v2-review-v1','at_ns':time.time_ns(),
            'review_passed':True,'CPU_controls_root_rerun':60,'CUDA_visible_devices':'',
            'exact_original_B_E_settings_verified':True,'only_new_C_cancel_probes_added':True,
            'old783_pins_only_README_prospectively_changed':True,'all_prospective_pins_checked':len(prospective),
            'reviewed':['direct actual returned aborted Request parameter mapping, no suffix parsing',
                        'bounded32 record retention and preserved native args/results/exceptions',
                        'precise shared-epoch priority receipts on abort/health calls',
                        'best-effort failure-only export and metric capture preserves original failure',
                        'unchanged workload, isolated overlap globals, no active-sibling export',
                        'remaining seven names absent, exact argv and counts, no A replacements',
                        'explicit one-file README transition, retained original failed block',
                        'supervisor exact plan/certificate binding, per-case pin checks and cleanup'],
            'GPU_idle':True,'GPU_lock_available':True,'port_closed':True,
            'plan_sha256':sha(plan_path),'production_endorsement':False}
    report_path=E/'operational-v2-harness-root-review.json'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    manifest=copy.deepcopy(plan)
    manifest.update(schema='root-frozen-operational-v2-block-v1',freeze_pending_root_review=False,
                    launch_authorized=False,root_review='passed',frozen_at_ns=time.time_ns())
    extras=dict(inventory['file_pins'])
    for path in [Path(__file__).resolve(),report_path,H/'live-nov3-operational-v2-review-inventory.json',
                 E/'operational-a-root-verification.json',E/'operational-a-waves.csv',H/'root_verify_operational_a_nov3.py']:
        extras[str(path)]=sha(path)
    manifest['file_pins']=extras
    allpins=merged(manifest)
    for path,pin in allpins.items():assert sha(Path(path))==pin,path
    manifest['ROOT_unique_input_pins']=len(allpins)
    freeze.write_text(json.dumps(manifest,indent=2)+'\n')
    certificate={'schema':'root-operational-v2-launch-authorization-v1','at_ns':time.time_ns(),
                 'launch_authorized':True,'gpu_custody':'ASTRA','manifest_sha256':sha(freeze),
                 'supervisor_sha256':sha(H/'supervise_operational_v2_astra_nov3.py'),
                 'authority':'Existing user sustained optimization request; ROOT verified routine harness repair and remaining scope',
                 'scope':'Exactly these seven new-name B-E cases. No A replacement, retries or unreviewed GPU follow-ons.',
                 'original_failed_A_cases_preserved':True,'production_endorsement':False}
    auth.write_text(json.dumps(certificate,indent=2)+'\n')
    print(json.dumps({'review':'passed','all_pins':len(allpins),'cases':7,'normal_EOS_planned':1872,
                      'health_EOS_planned':8,'aborts_planned':2,'GPU_custody':'ASTRA',
                      'manifest':str(freeze),'authorization':str(auth)}))
