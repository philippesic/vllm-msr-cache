"""Advance the existing overhead process to operational after a frozen boundary audit.

The independent driver waits on exact recorded processes, never retries a cell,
and launches only the unchanged operational phase. Boundary checks execute from
frozen SOURCE with the literal serving interpreter; they preserve all attempts.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'output/stage6-5080-20261109'
SOURCE=REPO/'output/stage6-5080-20261108/primary-execution-bundle-01/source'
OUT=REPO/'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
EVIDENCE=REPO/'evidence/stage6-5080-20261109'
PYTHON='/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
DRIVER='stage6-5080-supplementary-driver-20261110-01.service'
UNITS={p:f'stage6-5080-{p}-20261110-01.service' for p in ('overhead','operational')}


def state(unit):
    return dict(v.split('=',1) for v in subprocess.check_output(['systemctl','--user','show',unit,
        '--property=ActiveState,SubState,MainPID,Result,ExecMainStatus'],text=True).splitlines())


def wait_original(phase):
    owner=json.loads((BASE/f'{phase}-owner-01.private.json').read_text())
    assert owner['command'][0]==PYTHON and owner['cwd']==str(SOURCE)
    try:
        p=psutil.Process(owner['pid'])
        if abs(p.create_time()-owner['birth'])<.01:
            assert p.cmdline()==owner['command'] and p.cwd()==owner['cwd']
            s=state(UNITS[phase]);assert int(s['MainPID'])==p.pid and s['SubState']=='running'
            while True:
                try:p.wait(timeout=55);break
                except psutil.TimeoutExpired:continue
    except psutil.NoSuchProcess:
        pass
    for _ in range(25):
        s=state(UNITS[phase])
        if s.get('MainPID')=='0':break
        time.sleep(.2)
    assert s.get('SubState')=='exited' and s.get('Result')=='success' and s.get('ExecMainStatus')=='0',s
    print(phase,'original coordinator exited successfully; auditing measurements',flush=True)


def boundary(phase):
    assert Path.cwd()==SOURCE.resolve() and sys.executable==PYTHON and sys.dont_write_bytecode
    sys.path[:]=[str(SOURCE)]+[p for p in sys.path if Path(p).resolve()!=Path(__file__).resolve().parent]
    from benchmarks.serving_suite.campaign_contracts import load,digest,atomic_json,verify_tree,Ledger
    from benchmarks.serving_suite.campaign import identity_for,source_snapshot
    from benchmarks.serving_suite import campaign_completion as completion
    from benchmarks.serving_suite.verify import verify
    assert all(Path(f.__code__.co_filename).resolve().is_relative_to(SOURCE.resolve())for f in(load,identity_for,verify,completion.overhead))
    assert OUT.parent.is_mount()
    target=EVIDENCE/f'{phase}-boundary-01.json';assert not target.exists()
    s=state(UNITS[phase]);assert s['MainPID']=='0' and s['SubState']=='exited' and s['Result']=='success' and s['ExecMainStatus']=='0'
    protocol=load(OUT/'protocol.lock.json');source=source_snapshot();assert source==protocol['source']
    expected={(b['model'],b['cell'],b['repeat'],system)for b in protocol['supplementary_schedules'][phase]for system in b['systems']}
    assert len(expected)=={'overhead':38,'operational':6}[phase]
    cells={(c['model'],c['cell']):c for c in load(OUT/'resolved.json')['cells']}
    ledger=Ledger(OUT)
    def entries():return [e for e in ledger.index().values()if e['details']['identity']['phase']==phase]
    def key(e):return tuple(e['details']['identity'][k]for k in('model','cell','repeat','system'))
    with ledger.writer():
        current=entries();before=digest(current)
        assert len(current)==len(expected) and {key(e)for e in current}==expected
        assert all(e['state']in('verified','failed','interrupted')for e in current)
        rows=[]
        for e in current:
            i=e['details']['identity'];k=key(e)
            assert i==identity_for(OUT,phase,cells[k[:2]],k[2],k[3],source)
            d=OUT/e['details']['path'];files=verify_tree(d,e['details']['seal'])
            inner=verify(d/'measurement');result=load(d/'measurement/result.json');case=result['cases'][0]
            rows.append(dict(attempt=e['attempt_id'],identity={k:i[k]for k in('model','cell','repeat','system')},
                path=e['details']['path'],state=e['state'],seal=e['details']['seal'],verified_files=files,
                inner_verification=inner,evaluation=case['evaluation']['status'],cleanup_verified=case['cleanup']['verified'],
                completed_requests=case['summary']['completed'],expected_requests=case['expected_requests'],
                measurement_errors=case.get('campaign',{}).get('measurement_errors',[])))
        outcome=getattr(completion,phase)(OUT)
        assert digest(entries())==before and source_snapshot()==source
        complete=bool(outcome['complete'] and all(r['state']=='verified' and r['evaluation']=='passed' and r['cleanup_verified']
            and r['completed_requests']==r['expected_requests'] and not r['measurement_errors'] for r in rows))
        receipt=dict(schema='stage6-5080-frozen-supplementary-boundary-v1',phase=phase,recorded_unix=time.time(),
            execution_complete=True,frozen_phase_complete=complete,planned_configurations=len(expected),retained_attempts=len(rows),
            verified_files=sum(r['verified_files']for r in rows),source_digest=source['digest'],protocol_digest=digest(protocol),
            phase_ledger_digest=before,service_terminal_state=s,attempts=rows,frozen_outcome=outcome,
            scope='All original supplementary identities once, full identity/outer and inner artifacts plus frozen control or operational requirements. No retry, source/design/workload/repetition change. A passing boundary does not close final all-attempt-cap/campaign/portability/results audits.')
        atomic_json(target,receipt)
    print(json.dumps({k:v for k,v in receipt.items()if k not in('attempts','frozen_outcome')},indent=2),flush=True)
    return 0 if complete else 2


def drive():
    assert Path.cwd()==REPO and sys.executable==PYTHON and sys.dont_write_bytecode and OUT.parent.is_mount()
    log=BASE/'supplementary-driver-01.log';owner_path=BASE/'supplementary-driver-owner-01.private.json'
    assert not log.exists() and not owner_path.exists()
    from benchmarks.serving_suite.campaign_contracts import atomic_json,load
    with log.open('xb')as stream:os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
    me=psutil.Process()
    atomic_json(owner_path,dict(pid=me.pid,birth=me.create_time(),command=me.cmdline(),cwd=me.cwd(),unit=DRIVER,recorded_unix=time.time(),scope='Event-driven exact-process wait and unchanged phase handoff; never owns a measured GPU worker or retries a case.'))
    owner_path.chmod(0o600)
    terminal=load(EVIDENCE/'confirmation-attempts-complete.json');assert terminal['execution_complete']and terminal['terminal_logical_configurations']==524
    wait_original('overhead')
    subprocess.run([PYTHON,str(Path(__file__).resolve()),'--boundary','overhead'],cwd=SOURCE,check=True)
    from benchmarks.research_checks.stage6_healthcheck import coordinators,gpu_snapshot
    from harness.gpu_lock import GPULock
    assert not coordinators(OUT)and not gpu_snapshot()['processes']
    with GPULock():pass
    assert not(BASE/'operational-01.log').exists()and not(BASE/'operational-owner-01.private.json').exists()
    assert subprocess.check_output(['systemctl','--user','show',UNITS['operational'],'--property=LoadState','--value'],text=True).strip()=='not-found'
    import shutil
    assert shutil.which('rg')
    subprocess.run(['systemd-run','--user','--unit='+UNITS['operational'],'--property=Type=exec','--property=Restart=no',
        '--property=RemainAfterExit=yes','--property=KillMode=control-group','--working-directory='+str(REPO),
        '--setenv=PATH='+os.environ['PATH'],'--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=HF_HOME=/mnt/c/vllm-semantic-cache-ext-stage6-5080/huggingface',PYTHON,'-m',
        'benchmarks.research_checks.stage6_5080_launch','--phase','operational','--log',str(BASE/'operational-01.log'),
        '--receipt',str(BASE/'operational-owner-01.private.json')],check=True)
    deadline=time.monotonic()+60
    while not(BASE/'operational-owner-01.private.json').exists():
        assert time.monotonic()<deadline,'Operational launch did not create owner receipt'
        time.sleep(.2)
    print('Operational independent service launched; no release or final-completion claim',flush=True)
    wait_original('operational')
    subprocess.run([PYTHON,str(Path(__file__).resolve()),'--boundary','operational'],cwd=SOURCE,check=True)
    atomic_json(EVIDENCE/'supplementary-driver-complete-01.json',dict(schema='stage6-5080-supplementary-driver-v1',
        recorded_unix=time.time(),overhead='overhead-boundary-01.json',operational='operational-boundary-01.json',
        both_frozen_phase_boundaries_pass=True,final_campaign_complete=False,gpu_release_announced=False,
        scope='Only the38 overhead and six operational identities advanced, without retry. Final caps/campaign/portability/results/push and resource cleanup/notification remain required.'))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--boundary',choices=('overhead','operational'))
    args=parser.parse_args();raise SystemExit(boundary(args.boundary)if args.boundary else drive())
