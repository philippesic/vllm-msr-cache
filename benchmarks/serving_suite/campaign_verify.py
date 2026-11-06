"""Offline verification and portable campaign transfer; no model/GPU imports."""
from __future__ import annotations

import shutil
import copy
import json
import subprocess
from pathlib import Path

from .campaign_contracts import (SCHEMA, Ledger, atomic_json, cell_inputs, digest, file_hash,
                                 load, local_path, seal_tree, validate_host, verify_tree)
from .campaign_contracts import read_protocol,write_protocol


def verify_campaign(out):
    out = Path(out)
    campaign, resolved = load(out / 'campaign.json'), load(out / 'resolved.json')
    if digest(campaign) != resolved['campaign_id']:
        raise ValueError('Campaign specification/catalog identity changed')
    expected_inputs = campaign.get('input_identities')
    if expected_inputs is not None:
        observed = digest(resolved['cells']) if isinstance(expected_inputs,str) else [cell_inputs(c) for c in resolved['cells']]
        if observed!=expected_inputs:
            raise ValueError('Compiled resource/workload/runtime identity changed')
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
        record = read_protocol(out)
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


def portable_inventory(out, destination):
    """Materialize executed tool inputs, including the full importable repo.

    Large public model/dataset downloads are reacquired by pinned content; the
    actual selected text, references, graphs and tools are included in the bundle.
    """
    host = load(out/'host.private.json')
    resolved = load(out/'resolved.json')
    from .campaign import ROOT
    replacements = {str(ROOT):'${project}',host['asset_cache']:'${asset_cache}',
                    host['runtime_python']:'${runtime_python}'}
    replacements.update({p:'${model:'+m+'}' for m,p in host['models'].items()})
    sources = load(Path(host['asset_cache'])/'sources.json')
    manifest = {}

    def add(source, relative, expected=None):
        source = Path(source)
        actual = file_hash(source)
        if expected and actual!=expected:
            raise ValueError(f'Portable executed input changed: {source.name}')
        target = local_path(destination,'assets/'+relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_hash(target)!=actual:
            raise ValueError('Portable asset collision')
        shutil.copyfile(source,target)
        manifest['assets/'+relative] = actual
        replacements[str(source)] = '${bundle}/assets/'+relative

    for name,entry in sources.items():
        root = Path(entry['path'])
        replacements[str(root)] = '${bundle}/assets/sources/'+name
        if name=='more-itertools':
            # pytest imports helpers beyond the selected source/test excerpts.
            files = subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
            for relative in filter(None,files):
                add(root/relative,'sources/'+name+'/'+relative)
        else:
            for relative,expected in entry['licenses'].items():
                add(root/relative,'sources/'+name+'/'+relative,expected)
    for cell in resolved['cells']:
        for phase in ('workload_binding','confirmation_binding'):
            binding = cell.get(phase)
            if not binding:
                raise ValueError('Both development and confirmation inputs are required')
            for session in load(binding['corpus'])['sessions']:
                for path,expected in session.get('input_files',{}).items():
                    source = Path(path)
                    try:
                        relative = str(source.relative_to(host['asset_cache']))
                    except ValueError:
                        raise ValueError('Executed external input is outside the prepared asset cache')
                    add(source,relative,expected)
    public_sources = {k:{a:b for a,b in v.items() if a!='path'} for k,v in sources.items()}
    return replacements,dict(included_files=manifest,public_sources=public_sources,
        model_reacquisition=load(out/'campaign.json')['asset_identities'],
        dataset_scope='Selected source text and references included; full original datasets reacquired from pinned preparation recipes',
        runtime_reacquisition='source/benchmarks/serving_suite/runtime_setup.py and exact runtime_requirements/*.txt',
        raw_evidence='Retained separately; attempt seals and archive inventory identify the full local evidence')


def export_campaign(out, destination):
    out, destination = Path(out).resolve(), Path(destination).resolve()
    verified = verify_campaign(out)
    if destination.exists():
        raise ValueError('Export destination must be new')
    locked = read_protocol(out)
    resolved = load(out/'resolved.json')
    if resolved['unresolved']:
        raise ValueError('Cannot export unresolved cells')
    destination.mkdir(parents=True)
    replacements,inventory = portable_inventory(out,destination)
    portable_cells = copy.deepcopy(resolved['cells'])
    for cell in portable_cells:
        cell.pop('runtime_identity',None)
        cell.pop('system_identities',None)
        cell['profile'].pop('allocation_receipts',None)
        cell['profile']['runtime_python'] = '${runtime_python}'
        for phase in ('workload_binding','confirmation_binding'):
            binding = cell[phase]
            for key in ('plan','corpus'):
                content = load(binding[key])
                if key=='plan':
                    # The new host must resolve/import its actual native runtime.
                    content.pop('published_runtimes',None)
                content = transform(content,replacements)
                relative = f'inputs/{phase}/{cell["model"]}/{cell["cell"]}/{key}.json'
                atomic_json(destination/relative,content)
                binding[key] = '${bundle}/'+relative
                binding[key+'_digest'] = digest(content)
    from .campaign import ROOT
    for relative,expected in locked['source']['files'].items():
        source = local_path(ROOT,relative)
        if file_hash(source)!=expected:
            raise ValueError(f'Executed source changed before export: {relative}')
        target = local_path(destination,'source/'+relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    # Validation support is part of the transfer seal, not the timed serving
    # source identity. Keep focused native checks and their shared test fixtures.
    validation = list((ROOT/'tests').glob('test_campaign*.py'))
    validation += [ROOT/'tests'/name for name in ('__init__.py','test_config.py',
        'test_wait_hint_cpu_recency.py','test_native_child_runtime.py','test_arc_optimization.py',
        'test_arc_serving_profile.py','test_step_1_5_prefetch.py','test_preservation_audit_mode.py')]
    for path in validation:
        if path.is_file():
            target = destination/'source/tests'/path.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    campaign = load(out/'campaign.json')
    # Retain the parent's exact logical identity privately in evidence; portable
    # acquisition rebinds paths and gets its own explicitly linked identity.
    portable = dict(schema='serving-campaign-transfer-v2',parent_campaign_id=resolved['campaign_id'],
        cells=portable_cells,protocol={k:locked[k] for k in ('source','schedule','supplementary_schedules','precision')},
        specification=campaign['specification'],catalogs=campaign['catalogs'],inventory=inventory,
        verification=verified,raw_evidence_identity=digest(Ledger(out).events()))
    for event in Ledger(out).index().values():
        entry = dict(attempt=event['attempt_id'],state=event['state'],seal=event['details'].get('seal'))
        inventory.setdefault('attempts',[]).append(entry)
    text = json.dumps(portable)
    if '/home/' in text or '/tmp/' in text or str(out) in text:
        raise ValueError('Host-private path remains in portable acquisition metadata')
    atomic_json(destination/'portable.json',portable)
    instructions = ROOT/'docs/STAGE6_PIPELINE.md'
    if instructions.is_file():
        shutil.copyfile(instructions,destination/'README.md')
    atomic_json(destination/'host.example.json',load(ROOT/'benchmarks/serving_suite/templates/host.example.json'))
    seal = seal_tree(destination)
    return dict(destination=str(destination),seal=seal,files=verify_tree(destination,seal),inventory=inventory)


def import_campaign(bundle, host_path, out):
    bundle, out = Path(bundle).resolve(), Path(out).resolve()
    verify_tree(bundle)
    portable = load(bundle/'portable.json')
    if portable['schema']!='serving-campaign-transfer-v2':
        raise ValueError('Unknown transfer schema')
    host = validate_host(load(host_path))
    if out.exists():
        raise ValueError('Import requires a new output directory')
    from .campaign import ROOT,source_snapshot
    if source_snapshot()['digest']!=portable['protocol']['source']['digest']:
        raise ValueError('Run import using the exported source closure')
    out.mkdir(parents=True)
    shutil.copytree(bundle,out/'imported')
    replacements = {'${bundle}':str(out/'imported'),'${project}':str(ROOT),
        '${asset_cache}':host['asset_cache'],'${runtime_python}':host['runtime_python']}
    replacements.update({'${model:'+m+'}':p for m,p in host['models'].items()})
    cells = transform(portable['cells'],replacements)
    from .assets import runtime_identity,model_identity
    environments = {}
    unresolved = []
    for cell in cells:
        model = cell['model']
        if model not in host['models'] or not Path(host['models'][model]).is_dir():
            unresolved.append(model+': local pinned model required')
        elif model not in environments:
            expected = portable['inventory']['model_reacquisition'][model]['identity']['content_digest']
            if model_identity(host['models'][model])['content_digest']!=expected:
                raise ValueError('Imported model content differs from the frozen protocol')
            environments[model] = True
        python = host['runtime_python']
        if python not in environments:
            environments[python] = runtime_identity(python)
        cell['runtime_identity'] = digest(environments[python])
        cell['system_identities'] = {}
        for system in cell['systems']:
            if system in ('arc','wait'):
                continue
            runtime = host.get('resolved_published_runtimes',{}).get(system)
            if not runtime:
                unresolved.append(system+': resolved pinned runtime required')
                continue
            external_python = runtime['python']
            if external_python not in environments:
                environments[external_python] = runtime_identity(external_python)
            cell['system_identities'][system] = dict(environment_digest=digest(environments[external_python]),
                execution_digest=digest(runtime['execution_identity']))
        for phase in ('workload_binding','confirmation_binding'):
            binding = cell[phase]
            for key in ('plan','corpus'):
                content = transform(load(binding[key]),replacements)
                if key=='plan' and content.get('published_variants'):
                    runtimes = copy.deepcopy(host.get('resolved_published_runtimes',{}))
                    for system,runtime in runtimes.items():
                        runtime['cpu_bytes'] = content['cpu_bytes']
                        runtime['kv_transfer_config'].setdefault('kv_connector_extra_config',{})['cpu_bytes_to_use'] = content['cpu_bytes']
                        if system=='continuum':
                            runtime['env']['LMCACHE_MAX_LOCAL_CPU_SIZE'] = str(content['cpu_bytes']/2**30)
                    content['published_runtimes'] = runtimes
                if '${' in json.dumps(content):
                    raise ValueError('Unresolved portable input marker')
                path = out/'inputs'/phase/model/cell['cell']/(key+'.json')
                atomic_json(path,content)
                binding[key],binding[key+'_digest'] = str(path),digest(content)
    campaign = dict(schema=SCHEMA,specification=portable['specification'],catalogs=portable['catalogs'],
        asset_identities=portable['inventory']['model_reacquisition'],
        parent_campaign_id=portable['parent_campaign_id'],transfer_identity=digest(portable),
        acquisition_host=host['id'],input_identities=digest(cells))
    campaign_id = digest(campaign)
    atomic_json(out/'campaign.json',campaign)
    atomic_json(out/'resolved.json',dict(schema=SCHEMA,campaign_id=campaign_id,
        parent_campaign_id=portable['parent_campaign_id'],host_id=host['id'],cells=cells,
        configurations_per_repeat=sum(len(c['systems']) for c in cells),
        unresolved=sorted(set(unresolved)),runnable=not unresolved,cost=portable['protocol']['precision'],
        imported_hardware_preflight_required=True))
    atomic_json(out/'imported-protocol.json',portable['protocol'])
    atomic_json(out/'host.private.json',host)
    (out/'host.private.json').chmod(0o600)
    Ledger(out)
    return dict(out=str(out),campaign_id=campaign_id,parent_campaign_id=portable['parent_campaign_id'],
        runnable=not unresolved,confirmation_ready=False,unresolved=sorted(set(unresolved)),
        next='Run preflight and all development screens, then activate-import; confirmation repeats/order remain frozen')


def activate_import(out):
    out = Path(out)
    resolved = load(out/'resolved.json')
    if (out/'protocol.lock.json').exists():
        raise ValueError('Imported protocol already activated')
    parent = load(out/'imported-protocol.json')
    from .campaign import source_snapshot
    from .campaign_gates import qualify
    capability = qualify(out)
    if resolved['unresolved'] or not capability['all_required_correctness_passed']:
        raise ValueError('Complete imported native correctness, activation and capacity screens first')
    if source_snapshot()['digest']!=parent['source']['digest']:
        raise ValueError('Imported source changed')
    record = dict(parent,campaign_id=resolved['campaign_id'],parent_campaign_id=resolved['parent_campaign_id'],
        cells_digest=digest(resolved['cells']),capability_digest=digest(capability),
        hardware_digest=digest(load(out/'hardware.private.json')),imported_hardware_preflight_passed=True)
    for phase,blocks in parent['supplementary_schedules'].items():
        atomic_json(out/(phase+'-schedule.json'),dict(blocks=blocks))
    write_protocol(out,record)
    return dict(confirmation_ready=True,repetitions_replanned=False,blocks=len(parent['schedule']))
