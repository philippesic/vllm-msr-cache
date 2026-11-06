"""Verify prepared model/tool content and retain acquisition attestations."""
from pathlib import Path
import time

from .campaign_contracts import atomic_json, digest, file_hash, load


def stat_identity(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size,modified_ns=value.st_mtime_ns,changed_ns=value.st_ctime_ns,
                inode=value.st_ino,device=value.st_dev)


def attest(out, cells, host, phase, *, full_models=True):
    from .assets import model_identity
    out = Path(out)
    expected = load(out/'campaign.json')['asset_identities']
    identities, files = {}, {}
    for model in sorted({c['model'] for c in cells}):
        path = host['models'].get(model)
        if not path:
            raise ValueError(f'Missing prepared local model: {model}')
        identity = model_identity(path) if full_models else expected[model]['identity']
        if full_models and identity['content_digest']!=expected[model]['identity']['content_digest']:
            raise ValueError(f'Prepared model content changed: {model}')
        identities[model] = identity
        for relative in identity['files']:
            files[str(Path(path)/relative)] = stat_identity(Path(path)/relative)
    for cell in cells:
        for binding in (cell['workload_binding'],cell.get('confirmation_binding')):
            if not binding:
                continue
            for name in ('plan','corpus'):
                if digest(load(binding[name]))!=binding[name+'_digest']:
                    raise ValueError(f'Prepared {name} binding changed')
            for session in load(binding['corpus'])['sessions']:
                for path,expected_hash in session.get('input_files',{}).items():
                    if path not in files:
                        if file_hash(path)!=expected_hash:
                            raise ValueError(f'Executed public tool input changed: {path}')
                        files[path] = stat_identity(path)
    record = dict(schema='serving-campaign-input-attestation-v1',phase=phase,
                  models=identities,files=files,started_unix=time.time(),
                  model_verification='full content hashes' if full_models else
                    'phase-start full hashes plus unchanged size, inode, device, mtime and ctime guards')
    atomic_json(out/'attestations'/(str(time.time_ns())+'-'+phase+'.json'),record)
    return files


def unchanged(files):
    for path,identity in files.items():
        if stat_identity(path)!=identity:
            raise ValueError(f'Acquisition input changed since full content verification: {path}')
