"""Advance the exact portability proof to final reporting without GPU execution."""
from pathlib import Path
import os
import subprocess
import sys
import time
import psutil
from benchmarks.serving_suite.campaign_contracts import atomic_json, load

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'output/stage6-5080-20261109'
SOURCE = REPO / 'output/stage6-5080-20261108/primary-execution-bundle-01/source'
PYTHON = '/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
UNIT = 'stage6-5080-final-analysis-20261110-01.service'


def main():
    assert Path.cwd() == REPO and sys.executable == PYTHON and sys.dont_write_bytecode
    owner = BASE / 'final-analysis-owner-01.private.json'
    log = BASE / 'final-analysis-01.log'
    done = REPO / 'evidence/stage6-5080-20261109/final-analysis-driver-01.json'
    assert not owner.exists() and not log.exists() and not done.exists()
    p = psutil.Process()
    atomic_json(owner, dict(pid=p.pid, birth=p.create_time(), command=p.cmdline(), cwd=p.cwd(),
        unit=UNIT, recorded_unix=time.time(), scope='GPU-free final reporting after exact owned portability proof exits successfully'))
    owner.chmod(0o600)
    with log.open('xb') as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
    proof = load(BASE / 'portability-owner-03.private.json')
    try:
        worker = psutil.Process(proof['pid'])
        if abs(worker.create_time() - proof['birth']) < .01:
            assert worker.cmdline() == proof['command'] and worker.cwd() == proof['cwd']
            while worker.is_running() and worker.status() != psutil.STATUS_ZOMBIE:
                try:
                    worker.wait(timeout=10)
                except psutil.TimeoutExpired:
                    pass
    except psutil.NoSuchProcess:
        pass
    for _ in range(50):
        raw = subprocess.check_output(['systemctl', '--user', 'show', proof['unit'],
            '-p', 'SubState', '-p', 'MainPID', '-p', 'Result', '-p', 'ExecMainStatus'], text=True)
        state = dict(line.split('=', 1) for line in raw.splitlines())
        if state.get('MainPID') == '0' and state.get('SubState') != 'running':
            break
        time.sleep(.2)
    assert state == dict(SubState='exited', MainPID='0', Result='success', ExecMainStatus='0'), state
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH', None)
    subprocess.run([PYTHON, str(REPO / 'benchmarks/research_checks/stage6_5080_final_findings.py')],
                   cwd=SOURCE, env=env, check=True)
    subprocess.run([PYTHON, '-m', 'benchmarks.research_checks.stage6_5080_final_document'],
                   cwd=REPO, env=env, check=True)
    atomic_json(done, dict(schema='stage6-5080-final-analysis-driver-v1', complete=True,
        recorded_unix=time.time(), proof_unit=proof['unit'], proof_terminal_state=state,
        final_audit='completion-audit-01.json', findings='final-results-01/', results='RESULTS.md',
        render_push_cleanup_notification_pending=True,
        scope='GPU-free derived reporting only; failed throughput requirements preserved'))


if __name__ == '__main__':
    main()
