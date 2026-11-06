"""Offline verification and portable campaign transfer; no model/GPU imports."""
from __future__ import annotations

import shutil
from pathlib import Path

from .campaign_contracts import (SCHEMA, Ledger, atomic_json, digest, file_hash,
                                 load, local_path, seal_tree, validate_host, verify_tree)


def verify_campaign(out):
    out = Path(out)
    campaign, resolved = load(out / 'campaign.json'), load(out / 'resolved.json')
    if digest(campaign) != resolved['campaign_id']:
        raise ValueError('Campaign specification/catalog identity changed')
    ledger = Ledger(out)
    counts, files = {}, 0
    for event in ledger.index().values():
        counts[event['state']] = counts.get(event['state'], 0) + 1
        details = event['details']
        if details.get('seal'):
            directory = local_path(out, details['path'])
            files += verify_tree(directory, details['seal'])
            if load(directory / 'identity.json') != details['identity']:
                raise ValueError('Attempt identity disagrees with ledger')
            if (directory / 'measurement/artifacts.json').exists():
                from .verify import verify
                verify(directory / 'measurement')
        elif event['state'] == 'verified':
            raise ValueError('Verified attempt lacks an artifact seal')
    locked = out / 'protocol.lock.json'
    if locked.exists():
        record = load(locked)
        if record['campaign_id'] != resolved['campaign_id'] or record['cells_digest'] != digest(resolved['cells']):
            raise ValueError('Frozen protocol campaign/cell identity mismatch')
    return dict(schema='serving-campaign-verification-v1', attempts=counts,
                verified_files=files, protocol_frozen=locked.exists(),
                scope='Integrity and aggregate consistency; acquisition/correctness/quality completion is separate')


def transform(value, bindings):
    if isinstance(value, str):
        for source, destination in sorted(bindings.items(), key=lambda item: -len(item[0])):
            value = value.replace(source, destination)
        return value
    if isinstance(value, list):
        return [transform(item, bindings) for item in value]
    if isinstance(value, dict):
        return {transform(key, bindings): transform(item, bindings) for key, item in value.items()}
    return value


def export_campaign(out, destination):
    out, destination = Path(out), Path(destination)
    verified = verify_campaign(out)
    if destination.exists():
        raise ValueError('Export destination must be new')
    locked = load(out / 'protocol.lock.json')
    resolved = load(out / 'resolved.json')
    if resolved['unresolved']:
        raise ValueError('Cannot export a runnable protocol with unresolved cells')
    # Every executed data/tool input must be declared, not inferred from a local
    # output tree. Files are materialized so symlinks never escape the bundle.
    bindings = load(out / 'portable-bindings.json')
    destination.mkdir(parents=True)
    replacements = {}
    for logical, entry in bindings['assets'].items():
        target = local_path(destination, 'assets/' + logical)
        source = Path(entry['source'])
        if not source.is_file() or file_hash(source) != entry['sha256']:
            raise ValueError(f'Portable asset changed: {logical}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        replacements[str(source)] = '${bundle}/assets/' + logical
    for source, marker in bindings.get('path_markers', {}).items():
        if marker not in ('${project}', '${asset_cache}'):
            raise ValueError('Unknown portable path marker')
        replacements[source] = marker
    portable_cells = transform(resolved['cells'], replacements)
    for cell in portable_cells:
        binding = cell['workload_binding']
        for key in ('plan', 'corpus'):
            original = next(c for c in resolved['cells'] if (c['model'], c['cell']) == (cell['model'], cell['cell']))['workload_binding'][key]
            content = transform(load(original), replacements)
            relative = f'inputs/{cell["model"]}/{cell["cell"]}/{key}.json'
            atomic_json(destination / relative, content)
            binding[key] = '${bundle}/' + relative
            binding[key + '_digest'] = digest(content)
    from .campaign import ROOT
    for relative, expected in locked['source']['files'].items():
        source = local_path(ROOT, relative)
        if file_hash(source) != expected:
            raise ValueError(f'Executed source changed before export: {relative}')
        target = local_path(destination, 'source/' + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(destination / 'campaign.json', load(out / 'campaign.json'))
    atomic_json(destination / 'portable.json', dict(schema='serving-campaign-transfer-v1',
                parent_campaign_id=resolved['campaign_id'], cells=portable_cells,
                protocol=transform(locked, replacements), verification=verified,
                raw_evidence_identity=digest(Ledger(out).events()),
                weight_reacquisition=load(out / 'campaign.json')['asset_identities']))
    atomic_json(destination / 'host.example.json', load(ROOT / 'benchmarks/serving_suite/templates/host.example.json'))
    seal = seal_tree(destination)
    return dict(destination=str(destination), seal=seal, files=verify_tree(destination, seal))


def import_campaign(bundle, host_path, out):
    bundle, out = Path(bundle).resolve(), Path(out)
    verify_tree(bundle)
    portable = load(bundle / 'portable.json')
    if portable['schema'] != 'serving-campaign-transfer-v1':
        raise ValueError('Unknown transfer schema')
    host = validate_host(load(host_path))
    if out.exists():
        raise ValueError('Import requires a new output directory')
    out.mkdir(parents=True)
    # Retain a materialized portable payload under the new campaign. This keeps
    # inputs usable after removal of the original transfer mount.
    shutil.copytree(bundle, out / 'imported')
    from .campaign import ROOT
    replacements = {'${bundle}': str((out / 'imported').resolve()),
                    '${project}': str(ROOT), '${asset_cache}': host['asset_cache']}
    cells = transform(portable['cells'], replacements)
    unresolved = ['Imported hardware/runtime correctness preflight required']
    for cell in cells:
        if cell['model'] not in host['models']:
            unresolved.append(cell['model'] + ': local pinned model binding required')
        binding = cell['workload_binding']
        # Templates are immutable. Rebinding produces new local execution files
        # and digests while preserving the portable task identities.
        for key in ('plan', 'corpus'):
            content = transform(load(binding[key]), replacements)
            path = out / 'inputs' / cell['model'] / cell['cell'] / (key + '.json')
            if key == 'plan' and cell['model'] in host['models']:
                content['model'] = host['models'][cell['model']]
            atomic_json(path, content)
            binding[key] = str(path.resolve())
            binding[key + '_digest'] = digest(content)
    campaign = load(bundle / 'campaign.json')
    hardware_identity = digest(dict(parent=portable['parent_campaign_id'], host=host['id'], portable=digest(portable)))
    # campaign_id remains the immutable logical protocol identity; acquisition_id
    # distinguishes hardware-bound executions and links the parent explicitly.
    atomic_json(out / 'campaign.json', campaign)
    atomic_json(out / 'resolved.json', dict(schema=SCHEMA, campaign_id=digest(campaign),
                acquisition_id=hardware_identity, parent_campaign_id=portable['parent_campaign_id'],
                host_id=host['id'], cells=cells, configurations_per_repeat=sum(len(c['systems']) for c in cells),
                unresolved=sorted(set(unresolved)), runnable=False,
                cost=portable['protocol']['precision']))
    atomic_json(out / 'imported-protocol.json', portable['protocol'])
    atomic_json(out / 'host.private.json', host)
    (out / 'host.private.json').chmod(0o600)
    return dict(out=str(out), acquisition_id=hardware_identity, runnable=False,
                next='Resolve local bindings and pass imported hardware/runtime preflight before confirmation')
