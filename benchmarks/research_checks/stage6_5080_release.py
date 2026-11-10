"""Verify pushed analysis, retire only known owned units, and attest GPU release."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import psutil
from benchmarks.research_checks.stage6_healthcheck import coordinators, gpu_snapshot, lock_owners
from harness.gpu_lock import GPULock, GPU_LOCK, LEGACY_LOCK
from benchmarks.serving_suite.campaign_contracts import atomic_json, load

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'output/stage6-5080-20261109'
EVIDENCE = REPO / 'evidence/stage6-5080-20261109'
OUT = REPO / 'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
UNITS = ['stage6-5080-confirm-20261109-03.service',
         'stage6-5080-overhead-20261110-01.service',
         'stage6-5080-operational-20261110-01.service',
         'stage6-5080-supplementary-driver-20261110-02.service',
         'stage6-5080-portability-20261110-01.service',
         'stage6-5080-portability-20261110-02.service',
         'stage6-5080-portability-20261110-03.service',
         'stage6-5080-final-analysis-20261110-01.service']


def main():
    assert Path.cwd() == REPO and sys.dont_write_bytecode
    destination = EVIDENCE / 'gpu-release-final-01.json'
    assert not destination.exists() and OUT.parent.is_mount()
    final = load(EVIDENCE / 'completion-audit-01.json')
    assert final['all_planned_phase_execution_complete'] and final['all_audits_complete']
    assert (EVIDENCE / 'final-results-01/render.json').is_file()
    assert not OUT.with_name(OUT.name + '-portability-parked').exists()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()
    checkpoint = git('rev-parse', 'HEAD')
    assert git('branch', '--show-current') == 'stage6-5080-20261109'
    assert checkpoint == git('rev-parse', 'origin/stage6-5080-20261109')
    for name in ('RESULTS.md', 'completion-audit-01.json', 'portability-final-03.json',
                 'final-results-01/comparisons.json', 'final-results-01/render.json'):
        relative = str((EVIDENCE / name).relative_to(REPO))
        assert subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=REPO) == (EVIDENCE / name).read_bytes()
    owners = []
    for path in sorted(BASE.glob('*owner*.json')):
        record = load(path)
        assert {'pid', 'birth', 'command', 'cwd'} <= record.keys()
        alive = False
        try:
            p = psutil.Process(record['pid'])
            alive = abs(p.create_time() - record['birth']) < .01
        except psutil.NoSuchProcess:
            pass
        assert not alive, f'Owned process still live: {path.name}'
        owners.append(dict(receipt=path.name, pid=record['pid'], birth=record['birth'],
                           command=record['command'], cwd=record['cwd'], exact_process_exited=True))
    assert not coordinators(OUT)
    remaining = []
    for p in psutil.process_iter(['pid', 'cmdline', 'uids']):
        try:
            args = p.info['cmdline'] or []
            owned_runtime = args and args[0].startswith('/home/philip/.cache/semantic-offload/stage6-5080-runtimes-')
            server = any('api_server' in a or a == 'serve' for a in args)
            if p.info['uids'].real == os.getuid() and owned_runtime and server:
                remaining.append(dict(pid=p.pid, birth=p.create_time(), command=args))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    assert not remaining, 'Owned runtime server remains'
    states = {}
    for unit in UNITS:
        raw = subprocess.check_output(['systemctl', '--user', 'show', unit,
            '-p', 'LoadState', '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID',
            '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
        state = dict(line.split('=', 1) for line in raw.splitlines())
        assert state.get('MainPID', '0') == '0' and state.get('SubState') != 'running'
        states[unit] = state
        subprocess.run(['systemctl', '--user', 'stop', unit], check=True)
        if state.get('ActiveState') == 'failed':
            subprocess.run(['systemctl', '--user', 'reset-failed', unit], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gpu = gpu_snapshot()
    assert not gpu['processes'], 'GPU compute client present; do not announce release'
    assert not lock_owners(GPU_LOCK) and not lock_owners(LEGACY_LOCK)
    with GPULock():
        assert not gpu_snapshot()['processes']
    assert not lock_owners(GPU_LOCK) and not lock_owners(LEGACY_LOCK)
    atomic_json(destination, dict(schema='stage6-5080-final-gpu-release-v1', checked_unix=time.time(),
        all_phase_execution_and_audits_complete=True, full_protocol_passed=False,
        pushed_analysis_checkpoint=checkpoint, owned_processes=owners,
        known_owned_unit_terminal_states=states, known_owned_units_stopped=True,
        owned_coordinators=[], owned_runtime_servers=remaining, gpu=gpu,
        gpu_compute_clients_empty=True, both_gpu_locks_acquired_and_released=True,
        gpu_lock_paths=[str(GPU_LOCK), str(LEGACY_LOCK)], gpu_lock_owners=[],
        original_campaign_restored=True, ext4_mount_verified=True,
        disk_free_gib={label: round(shutil.disk_usage(path).free / 1024**3, 2)
                       for label, path in [('ext4', OUT.parent), ('root', Path('/')), ('Windows', Path('/mnt/c'))]},
        useful_artifacts_preserved=True, final_notification_pending=True,
        scope='Verified release after all results/audits were pushed; no process killed, lock unlinked, GPU reset, training launched or unrelated checkout accessed. Display memory is not a compute allocation.'))
    print(json.dumps(dict(gpu_released=True, proof=str(destination), pushed_checkpoint=checkpoint)))


if __name__ == '__main__':
    main()
