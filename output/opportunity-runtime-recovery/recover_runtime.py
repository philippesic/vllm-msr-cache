import hashlib
import io
import json
import pathlib
import tarfile

out = pathlib.Path(__file__).resolve().parent
records = json.loads((out / 'wait-hint-runtime-manifest.json').read_text())
prefix = 'eviction-prefetch-campaign-20260905/vllm-wait-hint-git/'
by_digest = {}
for record in records:
    relative = pathlib.PurePosixPath(record['path'].removeprefix(prefix))
    assert not relative.is_absolute() and '..' not in relative.parts
    by_digest.setdefault(record['sha256'], []).append((relative, record))
archive = pathlib.Path('/Users/pippo/.codex/backups/vllm-msr-5080-cleanup-20260916/msr-cleanup-preserve-20260916.tar.gz')
target = out / 'recovered-vllm-runtime.tar.gz'
found = set()
with tarfile.open(archive, 'r|gz') as source, tarfile.open(target, 'w:gz') as dest:
    for member in source:
        digest = member.name.rsplit('/', 1)[-1]
        if '/objects/' not in member.name or digest not in by_digest:
            continue
        payload = source.extractfile(member).read()
        assert hashlib.sha256(payload).hexdigest() == digest
        for relative, record in by_digest[digest]:
            assert len(payload) == record['size']
            item = tarfile.TarInfo('vllm-wait-hint-git/' + str(relative))
            item.size = len(payload)
            item.mode = record['mode']
            item.mtime = record['mtime_ns'] // 1_000_000_000
            dest.addfile(item, io.BytesIO(payload))
        found.add(digest)
assert found == by_digest.keys(), f'Missing {by_digest.keys() - found}'
summary = {'files': len(records), 'logical_bytes': sum(r['size'] for r in records),
           'archive_bytes': target.stat().st_size,
           'archive_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
           'all_content_digests_verified': True,
           'source': str(archive), 'archive': str(target)}
(out / 'recovery-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
