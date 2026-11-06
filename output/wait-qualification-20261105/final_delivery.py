"""Postbuild handoff receipts; does not modify the verified archive or tested source."""
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

b = Path(__file__).resolve().parent
root = b.parents[1]
sys.path.insert(0, str(root))
from harness.gpu_lock import GPULock

release = b / 'release'
archive = release / 'wait-hint-qualification-20261105.tar.xz'
def read(path):
    return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')
def size(path):
    paths = [path] if path.is_file() else path.rglob('*')
    total = files = hardlinks = 0
    for p in paths:
        s = p.lstat()
        if stat.S_ISREG(s.st_mode):
            total += s.st_size
            files += 1
            hardlinks += s.st_nlink > 1
    return dict(regular_file_bytes=total, regular_files=files, shared_hardlink_files=hardlinks)

pre = read(b / 'delivery-prepackage.json')
members = read(archive.with_suffix('.xz.verification.json'))
links = read(archive.with_suffix('.xz.links.json'))
verified = read(b / 'all-retained-verification.json')
assert all(x['passed'] for x in [pre, members, links, verified])
assert archive.stat().st_size == members['bytes'] and sha(archive) == members['sha256']
completion = read(b / 'completion-inventory.json')
assert not completion['missing_or_running'] and len(completion['cells']) == 43
matrix = read(b / 'support-matrix.json')['cells']
assert len(matrix) == 14 and all(c['execution'] == 'passed' for c in matrix)
assert verified['verified_attempts'] == 46
copy_manifest = read(b / 'isolated-reproduction/source-copy.json')
source_copy = Path(copy_manifest['copy'])
assert all(sha(source_copy / n) == h for n, h in copy_manifest['files'].items())
with GPULock():
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], capture_output=True, text=True, check=True)
    assert not q.stdout.strip(), 'GPU ownership changed: inspect before handoff'
    gpu = dict(both_locks_acquired=True, compute_processes=q.stdout.strip(), query_stderr=q.stderr)
subprocess.run(['git', 'diff', '--check'], cwd=root, check=True)

cleanup = read(b / 'storage-cleanup.json')
retained = []
for previous in cleanup['retained']:
    p = Path(previous['path'])
    assert p.exists()
    retained.append(dict(previous, **size(p)))
retained.append(dict(path=str(archive), category='verified evidence archive', owner='this task', retention_reason='Portable source and measurement evidence; member and link checks passed', next_review='After a durable copy is secured and local reproduction is no longer needed', **size(archive)))
inventory = dict(recorded_unix=time.time(), filesystem='Local WSL filesystem', retained=retained, available_bytes=shutil.disk_usage(b).free, cleanup_receipt=str(b / 'storage-cleanup.json'), cleanup_free_bytes_before=cleanup['free_bytes_before'], cleanup_free_bytes_after=cleanup['free_bytes_after'], cleanup_observed_free_bytes_delta=cleanup['observed_free_bytes_delta'], scope='Postbuild bounded logical-size inventory. Nested paths overlap and many files share hardlinks; sizes are not reclaim estimates. Available bytes also reflect archive creation and concurrent activity. No Windows volume or VHDX reclaim measurement. No additional deletions in this pass.')
write(release / 'retained-inventory.json', inventory)
handoff = [root / 'docs/STATUS.md', root / 'docs/PROJECT_HISTORY.md', root / 'evidence/README.md', b / 'changes-since-initial-snapshot.patch', b / 'changes-since-initial-snapshot.json']
evidence = sorted((root / 'evidence/wait-qualification-20261105').glob('*.md'))
result = dict(passed=True, recorded_unix=time.time(), archive=members, archive_links=dict(documents=links['documents'], resolved=links['resolved'], missing=links['missing_task_or_source_links']), selected_execution_entries=43, selected_matrix_cells=14, verified_coordinator_attempts=46, isolated_source_files_reverified=len(copy_manifest['files']), gpu=gpu, git_diff_check_passed=True, final_handoff_sha256={str(p.relative_to(root)): sha(p) for p in handoff}, evidence_sha256={str(p.relative_to(root)): sha(p) for p in evidence}, retained_inventory=str(release / 'retained-inventory.json'), scope='Selected local execution and delivery verification, not universal performance, answer-quality, independent reproduction or production qualification. Final handoff documents and task diff were updated after packaging; their current hashes and postbuild receipts are adjacent, not embedded in the immutable archive. The archive retains its prepackage task diff and the tested source copy with explicitly earlier status documents.')
write(release / 'final-delivery.json', result)
print(json.dumps(dict(passed=True, archive_bytes=members['bytes'], archive_files=members['files'], resolved_links=links['resolved'], source_files=len(copy_manifest['files']), gpu_idle=True, final_available_bytes=inventory['available_bytes'])))
