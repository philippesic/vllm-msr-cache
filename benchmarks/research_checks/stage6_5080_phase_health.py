"""One bounded health check for the active 5080 supplementary phase; no repairs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
from benchmarks.serving_suite.campaign_contracts import Ledger, atomic_json, load
from benchmarks.research_checks.stage6_healthcheck import coordinators, gpu_snapshot, lock_owners
from harness.gpu_lock import GPU_LOCK

REPO = Path(__file__).resolve().parents[2]
BASE = REPO/'output/stage6-5080-20261109'
VOLUME = REPO/'output/stage6-5080-20261108/linux-volume-01'
OUT = VOLUME/'primary-20261109'
SOURCE = REPO/'output/stage6-5080-20261108/primary-execution-bundle-01/source'
PYTHON = '/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'


def check(phase, unit, owner_path):
    assert Path.cwd() == REPO and sys.executable == PYTHON and sys.dont_write_bytecode
    assert VOLUME.is_mount(), 'Do not write into an unmounted image path'
    protocol = load(OUT/'protocol.lock.json')
    expected = {(b['model'],b['cell'],b['repeat'],s)
                for b in protocol['supplementary_schedules'][phase] for s in b['systems']}
    entries = [e for e in Ledger(OUT).index().values() if e['details']['identity']['phase'] == phase]
    def identity(e):
        i = e['details']['identity']
        return tuple(i[k] for k in ('model','cell','repeat','system'))
    assert all(identity(e) in expected and e['details']['identity']['source'] == protocol['source']['digest'] for e in entries)
    state = dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show',unit,
        '--property=ActiveState,SubState,MainPID,Result,ExecMainStatus'],text=True).splitlines())
    owner = load(owner_path)
    assert owner['command'][0] == PYTHON and owner['cwd'] == str(SOURCE)
    issues, owned, exact_alive = [], [], False
    try:
        process = psutil.Process(owner['pid'])
        exact_alive = abs(process.create_time()-owner['birth']) < .01
        if exact_alive:
            assert process.cmdline() == owner['command'] and process.cwd() == owner['cwd']
            assert state['ActiveState'] == 'active' and state['SubState'] == 'running' and int(state['MainPID']) == process.pid
            assert process.pid in lock_owners(GPU_LOCK)
            assert load(OUT/'host.private.json')['runtime_python'] == PYTHON
            owned = [process] + process.children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    actual_coordinators = coordinators(OUT)
    assert len(actual_coordinators) <= 1
    if actual_coordinators and (not exact_alive or actual_coordinators[0].pid != owner['pid']):
        issues.append('unmatched_live_coordinator_do_not_restart')
    active = [e for e in entries if e['state'] in ('planned','running')]
    terminal = {identity(e) for e in entries if e['state'] in ('verified','failed','interrupted')}
    unit_done = state.get('MainPID') == '0' and state.get('SubState') == 'exited' and state.get('Result') == 'success' and state.get('ExecMainStatus') == '0'
    execution_complete = not exact_alive and not active and terminal == expected and unit_done
    if not exact_alive and not execution_complete:
        issues.append('coordinator_exited_before_all_planned_identities_terminal')
    failures = [dict(path=e['details']['path'],state=e['state'],identity=identity(e)) for e in entries if e['state'] in ('failed','interrupted')]
    if failures:
        issues.append('retained_phase_failures_require_review')
    gpu = gpu_snapshot()
    if exact_alive and set(gpu['processes']) - {p.pid for p in owned}:
        issues.append('gpu_client_outside_owned_tree')
    now = time.time()
    result = dict(schema='stage6-5080-supplementary-health-v1',checked_unix=now,phase=phase,unit=unit,
        coordinator_identity_checked_live=exact_alive,service=state,issues=issues,
        expected_configurations=len(expected),terminal_configurations=len(terminal),retained_attempts=len(entries),
        states=dict(Counter(e['state'] for e in entries)),execution_complete=execution_complete,
        input_attestation_before_first_attempt=exact_alive and not entries,
        active=[dict(identity=identity(e),state=e['state'],age_s=round(now-e['timestamp_unix'])) for e in active],
        retained_failures=failures,gpu=gpu,ext4_mount_verified=True,
        disk_free_gib={k:round(shutil.disk_usage(p).free/1024**3,2) for k,p in [('ext4',VOLUME),('root',Path('/')),('Windows',Path('/mnt/c'))]},
        scope='Bounded read-only campaign inspection; only this private health receipt is written. Execution completion is distinct from seal, control, operational, source/quality or campaign validity. Never restarts workers or changes a measurement.')
    destination = BASE/(phase+'-health-01')/'latest.private.json'
    atomic_json(destination,result)
    destination.chmod(0o600)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('overhead','operational'),required=True)
    parser.add_argument('--unit',required=True)
    parser.add_argument('--owner',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.phase,args.unit,args.owner),indent=2))
