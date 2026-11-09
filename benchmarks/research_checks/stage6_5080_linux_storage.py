"""Recover the retained DrvFS sealing failure on project-owned Linux storage."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import psutil

from benchmarks.serving_suite.campaign_contracts import (
    Ledger, atomic_json, file_hash, load,
)
from benchmarks.serving_suite.campaign_verify import verify_campaign
from benchmarks.serving_suite.verify import verify
from benchmarks.research_checks.stage6_healthcheck import gpu_snapshot
from harness.gpu_lock import GPULock


def tree_identity(root):
    result = {}
    for path in root.rglob('*'):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            result[relative] = dict(symlink=str(path.readlink()))
        elif path.is_file():
            result[relative] = dict(sha256=file_hash(path))
    return result


def recover(base, source):
    base, source = Path(base).resolve(), Path(source).resolve()
    storage = load(base / 'linux-storage.private.json')
    failure = load(base / 'filesystem-repair-01.private.json')
    point = Path(storage['mountpoint'])
    mount = json.loads(subprocess.check_output(
        ['findmnt', '--json', '--target', str(point), '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS'], text=True))
    if mount != storage['mount'] or not point.is_mount():
        raise ValueError('Required project Linux filesystem is not mounted')
    alias = base / 'primary-01'
    old = Path(failure['original_canonical_campaign'])
    destination = point / 'primary-01'
    if not alias.is_symlink() or alias.resolve() != old or destination.exists():
        raise ValueError('Inspect existing campaign placement before recovery')
    for process in psutil.process_iter(['cmdline']):
        args = process.info['cmdline'] or []
        if 'benchmarks.serving_suite.campaign' in args and any(str(p) in args for p in (old, alias, destination)):
            raise RuntimeError('Campaign process still active')
    with GPULock():
        if gpu_snapshot()['processes']:
            raise RuntimeError('GPU compute clients still active')
        # Exercise the exact pinned launcher on the target filesystem first.
        stage = point / 'native-stage-probe'
        with (base / 'linux-stage-probe.log').open('x') as log:
            subprocess.run([sys.executable, 'tools/vllm_runtime.py', sys.executable,
                            '-c', 'import vllm; print(vllm.__version__)'],
                           cwd=source, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
                                               SEMANTIC_VLLM_STAGE_PARENT=str(stage),
                                               PYTHONDONTWRITEBYTECODE='1'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if any(stage.iterdir()):
            raise RuntimeError('Native launcher left staging on Linux storage')
        original = tree_identity(old)
        shutil.copytree(old, destination, symlinks=True)
        if tree_identity(destination) != original:
            raise RuntimeError('Campaign copy differs; old output remains preserved')
        campaign = load(old / 'resolved.json')['campaign_id']
        attempt = destination / failure['path']
        if file_hash(attempt / 'measurement/artifacts.json') != failure['measurement_artifacts_sha256']:
            raise RuntimeError('Original measurement manifest changed')
        verify(attempt / 'measurement')
        stages = list(attempt.glob('measurement/*/runtime-staging'))
        if len(stages) != 1:
            raise ValueError('Inspect unexpected native staging layout')
        quarantine = point / 'staging-quarantine' / failure['attempt']
        quarantine.mkdir(parents=True)
        stages[0].rename(quarantine / 'runtime-staging')
        verify(attempt / 'measurement')
        # Switch only the alias, preserving the complete original Windows tree.
        temporary = base / '.primary-01-linux-alias'
        if temporary.exists() or temporary.is_symlink():
            raise ValueError('Unexpected pending alias')
        temporary.symlink_to(destination, target_is_directory=True)
        if alias.resolve() != old:
            raise RuntimeError('Campaign alias changed concurrently')
        os.replace(temporary, alias)
        with Ledger(destination).writer():
            recovered = Ledger(destination).recover(seal_pending=True)
        event = recovered[failure['attempt']]
        if event['state'] != 'interrupted' or not event['details'].get('seal'):
            raise RuntimeError('Incomplete attempt was not preserved with an interrupted seal')
        verification = verify_campaign(destination)
        if verification != verify_campaign(alias):
            raise RuntimeError('Alias verification differs')
        if load(destination / 'resolved.json')['campaign_id'] != campaign:
            raise RuntimeError('Campaign identity changed')
        record = dict(
            schema='stage6-5080-linux-storage-recovery-v1',
            original_windows_campaign_preserved=str(old),
            canonical_campaign=str(destination), alias=str(alias),
            campaign_id=campaign, interrupted_attempt=failure['attempt'],
            original_measurement_manifest_preserved=True,
            original_full_tree_copy_verified=True,
            native_stage_probe_clean=True,
            staging_quarantine=str(quarantine),
            verification=verification,
            replacement='Entire first B01 matched screen block, using ordinary resume; no measurement credit for original unmatched ARC',
            timed_source_models_policy_capacities_order_repeats_unchanged=True,
            private_host_files_remain_in_wsl=True,
        )
        atomic_json(base / 'linux-storage-recovery.private.json', dict(record, original_tree=original))
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.base, args.source), indent=2))
