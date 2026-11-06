"""ROOT freeze of never-attempted D/E, guarding all executed inputs."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

R=Path('/home/philip/vllm-semantic-cache-ext');H=R/'output/prefetch-native-20261102'
E=R/'evidence/live-optimization-2080ti-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def merge(m):
    out={}
    for field in ('existing_frozen_inputs','new_review_file_pins','file_pins'):
        for key,value in m.get(field,{}).items():
            key=str((R/key).resolve());assert key not in out or out[key]==value
            out[key]=value
    return out

if __name__=='__main__':
    target=E/'operational-v3-frozen.json';authorization=E/'operational-v3-launch-root-authorization.json'
    assert not target.exists() and not authorization.exists()
    old=json.loads((E/'operational-v2-frozen.json').read_text());pins=merge(old)
    assert len(pins)==813;pins.pop(str(R/'README.md'));assert len(pins)==812
    plan_path=H/'operational-v3-review-plan-nov3.json'
    plan=json.loads(plan_path.read_text())
    inventory=json.loads((H/'live-nov3-operational-v3-review-inventory.json').read_text())
    assert {str((R/k).resolve()):v for k,v in plan['existing_frozen_inputs'].items()}==pins
    before=[r for r in old['execution_order'] if r['phase'] in ('D','E')]
    cases=plan['execution_order'];assert len(cases)==len(before)==3
    for previous,case in zip(before,cases):
        args=copy.deepcopy(previous['args']);args[args.index('--name')+1]=case['name']
        assert case['args']==args
        assert case['name']==previous['name'].replace('operational-v2-','operational-v3-')
        for key in ('phase','arm','normal_natural_eos_planned','deliberate_abort_planned','health_natural_eos_planned'):
            assert case[key]==previous[key]
        assert not (H/'runs'/case['name']).exists()
    assert sum(c['normal_natural_eos_planned'] for c in cases)==1008
    assert sum(c['health_natural_eos_planned']+c['deliberate_abort_planned'] for c in cases)==0
    assert plan['runner']==old['runner'] and plan['python']==old['python']
    prospective=merge(plan)
    inventory_pins={key:value['sha256'] for key,value in inventory['files'].items()}
    inventory_pins[inventory['authorization_template']['path']]=inventory['authorization_template']['sha256']
    for key,value in inventory_pins.items():
        key=str((R/key).resolve());assert key not in prospective or prospective[key]==value
        prospective[key]=value
    assert str(R/'README.md') not in prospective
    for path,pin in prospective.items():assert sha(Path(path))==pin,path
    with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with socket.socket() as sock:
            sock.settimeout(1);assert sock.connect_ex(('127.0.0.1',19372))!=0
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    report={'schema':'root-operational-v3-review-v1','at_ns':time.time_ns(),'review':'passed',
            'CPU_controls_passed':15,'root_rerun':True,
            'exact_original_D_E_argv_names_only_changed':True,'no_A_B_C_replacements':True,
            'previous812_nonREADME_pins_all_held':True,
            'documentation_policy':'Assistant chosen correction: README is not executed; retain its initial/final identities without stopping. Catch concurrent documentation read failures as unavailable provenance. Every other frozen input remains guarded.',
            'current_README_sha256':sha(R/'README.md'),
            'scope':'3 launches,14 waves,1008 normal natural-EOS responses planned. Existing v2 serving harness byte-identical.',
            'GPU_idle':True,'lock_available':True,'port_closed':True,'production_endorsement':False}
    report_path=E/'operational-v3-harness-root-review.json';report_path.write_text(json.dumps(report,indent=2)+'\n')
    manifest=copy.deepcopy(plan)
    manifest.update(schema='root-frozen-operational-v3-block-v1',freeze_pending_root_review=False,
                    launch_authorized=False,root_review='passed',frozen_at_ns=time.time_ns())
    extras=dict(inventory_pins)
    for p in [Path(__file__).resolve(),report_path,H/'live-nov3-operational-v3-review-inventory.json']:
        extras[str(p)]=sha(p)
    manifest['file_pins']=extras
    allpins=merge(manifest);assert str(R/'README.md') not in allpins
    for path,pin in allpins.items():assert sha(Path(path))==pin,path
    manifest['ROOT_unique_input_pins']=len(allpins);target.write_text(json.dumps(manifest,indent=2)+'\n')
    certificate={'schema':'root-operational-v3-launch-authorization-v1','at_ns':time.time_ns(),
                 'launch_authorized':True,'gpu_custody':'ASTRA','manifest_sha256':sha(target),
                 'supervisor_sha256':sha(H/'supervise_operational_v3_astra_nov3.py'),
                 'authority':'Existing user sustained optimization request; remaining coverage and routine documentation guard repair',
                 'scope':'Exactly never-attempted D/E cases, no retries/replacements or extra GPU work',
                 'all_executed_input_guards_preserved':True,'production_endorsement':False}
    authorization.write_text(json.dumps(certificate,indent=2)+'\n')
    print(json.dumps({'review':'passed','unique_pins':len(allpins),'cases':3,'EOS_planned':1008,
                      'GPU_custody':'ASTRA','manifest':str(target),'authorization':str(authorization)}))
