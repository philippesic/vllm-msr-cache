"""Place an inactive campaign on Windows without changing its input paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import psutil

from benchmarks.serving_suite.campaign_contracts import (
    Ledger, atomic_json, file_hash, load,
)
from benchmarks.serving_suite.campaign_verify import verify_campaign
from benchmarks.research_checks.stage6_healthcheck import gpu_snapshot
from harness.gpu_lock import GPULock


def manifest(root):
    return {str(p.relative_to(root)): file_hash(p)
            for p in root.rglob('*') if p.is_file()}


def place(out, destination, private, receipt):
    out, destination, private, receipt = map(Path, (out, destination, private, receipt))
    out, destination, private = out.absolute(), destination.absolute(), private.absolute()
    backup = out.with_name(out.name + '-before-windows-placement')
    if out.is_symlink() or any(p.exists() for p in (destination, private, receipt, backup)):
        raise ValueError('Preserve existing placements/receipts; do not overwrite')
    for p in psutil.process_iter(['cmdline']):
        args = p.info['cmdline'] or []
        if 'benchmarks.serving_suite.campaign' in args and str(out) in args:
            raise RuntimeError('Campaign process still active')
    with GPULock():
        if gpu_snapshot()['processes']:
            raise RuntimeError('GPU compute clients still active')
        before_verification = verify_campaign(out)
        before = manifest(out)
        if any(p.is_symlink() for p in out.rglob('*')):
            raise ValueError('Unexpected input symlink before placement')
        private_names = {p.name for p in out.glob('*.private.json')}
        if private_names != {'host.private.json', 'hardware.private.json'}:
            raise ValueError('Inspect unexpected private-file layout before placement')
        private.mkdir(mode=0o700, parents=True)
        for name in private_names:
            shutil.copy2(out / name, private / name)
            (private / name).chmod(0o600)
        shutil.copytree(out, destination,
                        ignore=lambda root, names: private_names if Path(root) == out else [])
        for name in private_names:
            (destination / name).symlink_to(private / name)
        if manifest(destination) != before:
            raise RuntimeError('Copy content differs; original remains intact')
        if verify_campaign(destination) != before_verification:
            raise RuntimeError('Copied campaign verification differs')
        # Prove the Windows filesystem enforces the ledger's process lock.
        script = '''
import sys
from benchmarks.serving_suite.campaign_contracts import Ledger
try:
    with Ledger(sys.argv[1]).writer(): pass
except RuntimeError:
    raise SystemExit(0)
raise SystemExit('Second writer incorrectly acquired the held lock')
'''
        with Ledger(destination).writer():
            subprocess.run([sys.executable, '-c', script, str(destination)], check=True)
        out.rename(backup)
        try:
            out.symlink_to(destination, target_is_directory=True)
            if manifest(out) != before or verify_campaign(out) != before_verification:
                raise RuntimeError('Verification through the original path failed')
            # Reads must now be recoverable without relying on the old tree.
            for name in private_names:
                assert (out / name).resolve() == private / name
                assert (private / name).stat().st_mode & 0o777 == 0o600
            record = dict(
                schema='stage6-5080-storage-placement-v1',
                original_path=str(out), canonical_path=str(destination),
                private_bindings=str(private), private_files=sorted(private_names),
                files_preserved=len(before), content_hashes_preserved=True,
                campaign_id=load(out / 'resolved.json')['campaign_id'],
                frozen_input_strings_identity_order_repeats_preserved=True,
                windows_writer_lock_contention_passed=True,
                verification=before_verification,
                old_tree_removed_after_copy_and_alias_verification=True,
            )
            atomic_json(receipt, dict(record, file_hashes=before))
        except BaseException:
            if out.is_symlink():
                out.unlink()
            backup.rename(out)
            raise
        shutil.rmtree(backup)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'destination', 'private', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(place(args.out, args.destination, args.private, args.receipt), indent=2))
