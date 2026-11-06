"""Package retained wait qualification evidence with ordinary tar/xz and hashes.

Run only after GPU measurements and the isolated reproduction have finished.
Large models, environments, Git metadata and personal recovery backups stay out.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import lzma
import tarfile
import time
from pathlib import Path


def package(base: Path, output: Path, evidence: Path):
    if output.exists():
        raise FileExistsError('Preserve the existing release attempt')
    raw_prefix = 'source/output/' + base.name + '/'
    files: dict[str, Path] = {}
    def add(source, name):
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Expected an ordinary evidence file: {source}')
        if name in files:
            raise ValueError(f'Duplicate package path: {name}')
        files[name] = source
    for root in sorted(base.iterdir()):
        if root.is_dir() and (root/'result.json').is_file() and root.name != 'isolated-reproduction':
            result = json.loads((root/'result.json').read_text())
            if result.get('status') == 'running':
                raise RuntimeError(f'Campaign is active: {root}')
            for path in sorted(root.rglob('*')):
                relative = path.relative_to(root)
                if any(part in ('runtime-staging','__pycache__','.pytest_cache','.git') for part in relative.parts):
                    continue
                if path.is_file():
                    add(path, raw_prefix+str(path.relative_to(base)))
        elif root.is_file() and not root.name.startswith('preexisting') and root.name not in ('CHECKPOINT.md', 'NEXT_BOUNDARY.md') and root.suffix in ('.json','.md','.py','.log','.txt','.csv','.html','.png','.pdf','.patch','.sha256'):
            add(root, raw_prefix+root.name)
    # Retain directly usable public inputs as well as indexed historical archives.
    public_identity = json.loads((base/'independent-source-identities.json').read_text())
    for name, expected in public_identity.items():
        path = Path(name).resolve()
        relative = path.relative_to(base.resolve())
        if relative.parts[0] not in ('more-itertools-source','peps-source'):
            raise ValueError('Unexpected public source root: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen public source changed: '+name)
        add(path, raw_prefix+str(relative))
    license_path = base/'more-itertools-source/LICENSE'
    add(license_path, raw_prefix+'more-itertools-source/LICENSE')
    for path in sorted((base/'independent-inputs').glob('*.json')):
        add(path, raw_prefix+'independent-inputs/'+path.name)
    for path in sorted(evidence.rglob('*')):
        if path.is_file():
            add(path, 'source/evidence/'+evidence.name+'/'+str(path.relative_to(evidence)))
    isolation = base/'isolated-reproduction'
    copy_receipt = isolation/'source-copy.json'
    receipt = json.loads(copy_receipt.read_text())
    for name, expected in receipt['files'].items():
        path = isolation/'source'/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Isolated source changed: {name}')
        add(path, 'source/'+name)
    for path in sorted(isolation.iterdir()):
        if path.is_file():
            add(path, raw_prefix+'isolated-reproduction/'+path.name)
    # Adjacent identical source snapshots compress well even when their gzip
    # headers differ. Keep ordinary tar/xz rather than a custom object format.
    manifest = dict(schema='wait-qualification-release-v1',created_unix=time.time(),files={},
        scope='Every retained measurement attempt, frozen inputs/source archives, raw per-run records, compact evidence, final analysis code and isolated source. No model weights, environments, Git metadata, personal recovery backups or temporary continuation notes. Machine-specific paths remain provenance; see REPRODUCING.md for relocation.')
    output.parent.mkdir(parents=True,exist_ok=True)
    readme=output.parent/'README.md'
    if readme.exists():
        raise FileExistsError('Preserve the existing release README')
    readme.write_text('''# Wait-hint qualification evidence

Start with [the outcome and claim limits](source/evidence/wait-qualification-20261105/README.md)
and [reproduction instructions](source/evidence/wait-qualification-20261105/REPRODUCING.md).
`source/` is the exact isolated project copy used for the installation exercise.
`source/output/wait-qualification-20261105/` retains every measurement attempt,
frozen selection, derived report and isolated-reproduction receipt at the same
relative paths used by evidence links. Its public-repository directories contain
only the exact hashed files used by independent task tools, not complete checkouts;
follow REPRODUCING.md to obtain the complete pinned repositories when needed.
Model weights and Python environments are excluded. Original absolute paths are
provenance and need relocation before running on another checkout. Select each
measurement's executed-source snapshot when investigating a historical attempt.
The isolated copy's status documents precede its own installation/verification;
the outcome index above records the final results without rewriting tested source.

`MANIFEST.json` gives every member's size and SHA-256. To verify an extracted
bundle, run from its root with Python 3.12:

```python
import hashlib, json
from pathlib import Path
for name, expected in json.loads(Path('MANIFEST.json').read_text())['files'].items():
    path = Path(name)
    assert path.stat().st_size == expected['bytes'], name
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256'], name
print('All manifest members verified')
```
'''.replace('source/output/wait-qualification-20261105/', raw_prefix)
       .replace('source/evidence/wait-qualification-20261105/', 'source/evidence/'+evidence.name+'/'))
    add(readme,'README.md')
    ordered = sorted(files, key=lambda name: (0 if files[name].name.endswith('.tar.gz') else 1, files[name].name if files[name].name.endswith('.tar.gz') else name, name))
    with lzma.open(output,'wb',format=lzma.FORMAT_XZ,filters=[dict(id=lzma.FILTER_LZMA2,dict_size=64*1024*1024,preset=3)]) as stream:
        with tarfile.open(fileobj=stream,mode='w|') as archive:
            for index,name in enumerate(ordered):
                source=files[name]
                manifest['files'][name]=dict(bytes=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                archive.add(source,arcname=name,recursive=False)
                if index and index%500==0:
                    print(f'Packaged {index}/{len(ordered)} files',flush=True)
            encoded=(json.dumps(manifest,indent=2)+'\n').encode()
            info=tarfile.TarInfo('MANIFEST.json');info.size=len(encoded);info.mtime=int(time.time())
            archive.addfile(info,io.BytesIO(encoded))
    # Verify every packaged member from the finished archive, not just source
    # hashes calculated before compression. Failed builds remain visible.
    verified={}
    with tarfile.open(output,'r:xz') as archive:
        for member in archive:
            if member.name=='MANIFEST.json':
                continue
            expected=manifest['files'][member.name]
            data=archive.extractfile(member)
            actual=hashlib.file_digest(data,'sha256').hexdigest()
            if member.size!=expected['bytes'] or actual!=expected['sha256']:
                raise RuntimeError('Package verification failed: '+member.name)
            verified[member.name]=actual
    if len(verified)!=len(manifest['files']):
        raise RuntimeError('Missing packaged files')
    with output.open('rb') as stream:
        archive_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
    result=dict(passed=True,archive=str(output.resolve()),bytes=output.stat().st_size,
                sha256=archive_sha256,files=len(verified))
    output.with_suffix(output.suffix+'.verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(package(args.base,args.output,args.evidence),indent=2))

if __name__=='__main__':
    main()
