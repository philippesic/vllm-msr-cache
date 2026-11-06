"""Bounded cleanup of owned disposable test outputs; preserve research evidence."""
import fcntl, json, os, platform, shutil, time
from collections import defaultdict
from pathlib import Path
import psutil

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
targets = [H/'integration/pytest-scratch', H/'integration/baseline-pytest-scratch', H/'__pycache__']
def size(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
with open('/tmp/vllm-semantic-cache-gpu.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    excluded = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    active = []
    for proc in psutil.process_iter():
        if proc.pid in excluded:
            continue
        try:
            paths = [proc.cwd(), *(f.path for f in proc.open_files())]
            if str(H) in ' '.join(proc.cmdline()) or any(str(p).startswith(str(t)) for p in paths for t in targets):
                active.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    assert not active, active
    assert not [c for c in psutil.net_connections('tcp') if c.status == 'LISTEN' and c.laddr.port == 19372]
    # Only task-created test basetemps and import cache. Pytest's "current"
    # symlinks must resolve within the owned tree; rmtree never follows them.
    for target in targets:
        assert target.is_dir() and not target.is_symlink()
        assert target.stat().st_uid == os.getuid()
        assert all(p.lstat().st_uid == os.getuid() for p in target.rglob('*'))
        assert all(p.resolve().is_relative_to(target) for p in target.rglob('*') if p.is_symlink())
    before = shutil.disk_usage(H).free
    removed = []
    for target in targets:
        removed.append({'path': str(target), 'logical_bytes': size(target),
                        'reason': 'Owned reproducible pytest scratch or bytecode; all consumers finished, useful JUnit/log/source evidence preserved.'})
        shutil.rmtree(target)
    after = shutil.disk_usage(H).free
    categories = defaultdict(lambda: {'files': 0, 'logical_bytes': 0})
    largest = []
    for path in H.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        n = path.stat().st_size
        kind = 'compressed_raw' if path.name == 'raw.json.gz' else 'compressed_server_logs' if path.name.endswith('.log.gz') else 'sources_protocols_aggregates_and_other_evidence'
        categories[kind]['files'] += 1
        categories[kind]['logical_bytes'] += n
        largest.append({'path': str(path), 'logical_bytes': n})
    cached = Path('/home/philip/.cache/uv/archive-v0/84uHcFTAEf8_yTR5')
    result = {'host': platform.node(), 'filesystem_path': str(H), 'device': os.stat(H).st_dev,
              'completed_unix_seconds': time.time(), 'free_before_bytes': before, 'free_after_bytes': after,
              'observed_free_delta_bytes': after-before, 'removed': removed,
              'retained': [
                  {'path': str(H), 'logical_bytes': sum(v['logical_bytes'] for v in categories.values()),
                   'owner': 'This sustained prefetch investigation', 'categories': dict(categories),
                   'reason': 'Unique successful and negative compressed measurements, exact-work inventories, byte proofs, frozen source/runtime/protocol snapshots, baseline recovery and reproduction scripts. Large raw/log originals already compressed and removed; largest remaining consumers are evidence.',
                   'next_review': 'After conclusions and reproduction no longer require full evidence; do not remove pinned source/protocol dependencies or unique failed experiments without review.'},
                  {'path': str(ROOT/'evidence/cpu-wait-recency-2080ti-20261102'),
                   'logical_bytes': size(ROOT/'evidence/cpu-wait-recency-2080ti-20261102'),
                   'owner': 'Project result evidence', 'reason': 'Small portable curves, paired CSV, configuration, manifest and aggregate verifier.'},
                  {'path': str(cached), 'logical_bytes': size(cached), 'owner': 'Shared uv dependency cache',
                   'reason': 'Reusable Matplotlib plotting dependency archive. Preserve shared caches; incremental task footprint of other cached dependencies is unmeasured.',
                   'next_review': 'Under shared cache owner retention policy.'}],
              'largest_retained_files': sorted(largest, key=lambda x: x['logical_bytes'], reverse=True)[:8],
              'prior_retained_inventory': str(ROOT/'output/prefetch-efficiency-20261101/STORAGE.json'),
              'protected': ['Prior campaigns', 'Shared Hugging Face/model, Triton and uv caches', 'Serving virtualenv', 'Existing imported dirty checkout work'],
              'process_state': 'Finite controllers exited; both integrated raw receipts report no owned surviving descendants; no current task process or port19372 listener; NVIDIA compute-process query empty.',
              'limits': 'Free-byte delta is observed on WSL filesystem and may include concurrent activity. Logical sizes are not reclaimed-size estimates. No Windows backing-volume or VHDX reclamation claim.'}
    (H/'storage-retained.json').write_text(json.dumps(result, indent=2)+'\n')
    print({k: result[k] for k in ['free_before_bytes', 'free_after_bytes', 'observed_free_delta_bytes']})
    print({'retained_task_logical_bytes': result['retained'][0]['logical_bytes'], 'removed_logical_bytes': sum(r['logical_bytes'] for r in removed)})
