"""Exec the existing frozen campaign CLI with explicit cwd/interpreter ownership."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time

import psutil

from benchmarks.serving_suite.campaign_contracts import atomic_json, load

ROOT = Path(__file__).resolve().parents[2]
VOLUME = ROOT / 'output/stage6-5080-20261108/linux-volume-01'
SOURCE = ROOT / 'output/stage6-5080-20261108/primary-execution-bundle-01/source'
PYTHON = '/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'


def launch(phase, log, receipt, *, resume=False, only=None):
    if not VOLUME.is_mount():
        raise ValueError('Existing ext4 output volume is not mounted; do not write into its mountpoint')
    output = VOLUME / 'primary-20261109'
    host, resolved = load(output / 'host.private.json'), load(output / 'resolved.json')
    if host['runtime_python'] != PYTHON or any(c['profile']['runtime_python'] != PYTHON for c in resolved['cells']):
        raise ValueError('Workers must explicitly bind the pinned serving interpreter')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PATH='/usr/lib/wsl/lib:' + os.environ['PATH'])
    env.pop('PYTHONPATH', None)
    expected = load(output / 'imported-protocol.json')['source']['digest']
    actual = subprocess.check_output([PYTHON, '-c',
        'from benchmarks.serving_suite.campaign import source_snapshot; print(source_snapshot()["digest"])'],
        cwd=SOURCE, env=env, text=True).strip()
    if actual != expected:
        raise ValueError('Exported acquisition source differs from the imported protocol')
    if log.exists() or receipt.exists():
        raise ValueError('Preserve existing launch logs/receipts; choose fresh paths')
    command = [PYTHON, '-m', 'benchmarks.serving_suite.campaign',
               'resume' if resume else 'run', '--out', str(output), '--phase', phase]
    if only:
        command += ['--only', only]
    log.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, dict(command=command, cwd=str(SOURCE), source=actual,
        pid=os.getpid(), birth=psutil.Process().create_time(), launched_unix=time.time(),
        ext4_mount_verified=True, session_scope='Exec into existing coordinator; its GPU reservation and child cleanup remain unchanged'))
    with log.open('xb') as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
    os.chdir(SOURCE)
    os.execvpe(PYTHON, command, env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'screen', 'confirm', 'overhead', 'operational'), required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--only')
    args = parser.parse_args()
    launch(args.phase, args.log.resolve(), args.receipt.resolve(), resume=args.resume, only=args.only)
