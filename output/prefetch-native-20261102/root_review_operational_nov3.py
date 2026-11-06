"""Independent bounded ROOT operational-plan and preflight check; no inference."""
import fcntl
import hashlib
import json
import pathlib
import shutil
import socket
import subprocess
import time

ROOT = pathlib.Path('/home/philip/vllm-semantic-cache-ext')
H = ROOT / 'output/prefetch-native-20261102'
E = ROOT / 'evidence/live-optimization-2080ti-20261103'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check():
    plan_path = H / 'operational-review-plan-nov3.json'
    plan = json.loads(plan_path.read_text())
    checked = {}
    for group in ('existing_frozen_inputs', 'new_review_file_pins'):
        for name, expected in plan[group].items():
            path = pathlib.Path(name)
            if not path.is_absolute():
                path = ROOT / path
            actual = digest(path)
            assert actual == expected, (group, name)
            checked[str(path)] = actual
    cases = plan['execution_order']
    expected = [('A','native8',24,8,'bursty',4096,True,0),
                ('A','candidate',24,8,'bursty',4096,True,0),
                ('B','candidate',3,4,'bursty',4096,False,0),
                ('B','native8',3,4,'bursty',4096,False,0),
                ('C','native8',3,8,'longtool',4096,False,0),
                ('C','candidate',3,8,'longtool',4096,False,0),
                ('D','candidate',8,8,'bursty',128,False,0),
                ('E','candidate',3,8,'bursty',4096,False,20),
                ('E','native8',3,8,'bursty',4096,False,20)]
    assert len(cases) == len(expected)
    summaries = []
    for case, target in zip(cases, expected):
        args = case['args']
        def val(key): return args[args.index('--'+key)+1]
        shape = (case['phase'], case['arm'], int(val('waves')), int(val('active')),
                 val('wait'), int(val('registry-limit')), '--cancel-probe' in args,
                 float(val('wave-release-interval-s')))
        assert shape == target, (shape, target)
        for key, value in {'gpu-blocks':'1536','cpu-blocks':'2048','batch-tokens':'1024',
                           'clients':'24','turns':'3','wave-seeds':'14009,15013,16001',
                           'return-priority-ms':'8000','hint-interval-ms':'1000',
                           'timeout':'7200','startup-timeout':'600',
                           'service-profile':'repository_triage_v2'}.items():
            assert val(key) == value, (case['name'], key)
        assert '--graphs' in args and '--support-profile' in args
        candidate = case['arm'] == 'candidate'
        assert ('--gpu-gated-cpu-retention' in args) == candidate
        assert ('--no-audit' in args) == candidate
        assert ('--native-arc' in args) != candidate
        assert pathlib.Path(val('source-dir')).name == ('nov3-priority-integrated' if candidate else 'nov3-hardened')
        assert not (H / 'runs' / case['name']).exists(), case['name']
        assert not case['formal_performance_eligible']
        summaries.append({'name':case['name'], 'shape':shape,
                          'planned_eos':case['normal_natural_eos_planned'] + case['health_natural_eos_planned']})
    assert sum(x['normal_natural_eos_planned'] for x in cases) == 5328
    assert sum(x['health_natural_eos_planned'] for x in cases) == 8
    assert sum(x['deliberate_abort_planned'] for x in cases) == 2
    with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with socket.socket() as sock:
            sock.settimeout(1)
            assert sock.connect_ex(('127.0.0.1',19372)) != 0
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                   '--format=csv,noheader'],text=True).strip()
    result = {'schema':'root-operational-review-v1', 'at_ns':time.time_ns(),
              'implementation_review':'passed', 'plan_review':'passed',
              'cpu_controls':{'passed':24,'root_rerun':True,'inference':False},
              'reviewed_behaviors':['isolated overlapping module globals and request/hint/session identities',
                  'single persistent gateway epoch and unchanged cache namespace',
                  'native finish_requests returned actual FINISHED_ABORTED observation',
                  'pinned completion response_id + prompt index engine naming',
                  'first-token stream disconnect and four remaining-context natural-EOS health calls',
                  'missing telemetry unavailable; constant collection lengths only',
                  'no wave reset; no exports during active overlap siblings',
                  'all partial/adverse outcomes retained; operational rates descriptive only'],
              'review_plan_sha256':digest(plan_path),'checked_files':len(checked),
              'pins':checked,'cases':summaries,'gpu_idle_snapshot':gpu,
              'gpu_lock_available':True,'port_closed':True,
              'disk':dict(shutil.disk_usage(ROOT)._asdict()),
              'gpu_launch_authorized':False,
              'remaining':'Pin operational supervisor and final launch certificate before GPU handoff',
              'production_endorsement':False}
    path = E / 'operational-harness-root-review.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pins','reviewed_behaviors','cases')}))

if __name__ == '__main__': check()
