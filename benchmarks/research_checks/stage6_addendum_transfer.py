"""Separate addendum evidence relocation and GPU-free input/runtime import proof.

This derived tool never changes the frozen campaign or makes it executable. The
imported receipt requires new-host hardware preflight; no measured attempt is run.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from benchmarks.serving_suite.campaign_contracts import (
    Ledger, atomic_json, digest, file_hash, load, read_protocol, seal_tree, verify_tree,
)
from benchmarks.serving_suite.campaign_verify import portable_inventory, transform, verify_campaign


def export(out: Path, destination: Path) -> dict:
    if destination.exists():
        raise ValueError('Use a new addendum transfer destination')
    protocol = read_protocol(out)
    if load(out/'execution.json')['status'] not in ('complete', 'budget_exhausted'):
        raise ValueError('Finish bounded acquisition first')
    verified = verify_campaign(out)
    destination.mkdir(parents=True)
    resolved = load(out/'resolved.json')
    # Inventory's ordinary-campaign contract expects both bindings. A temporary
    # derived view aliases the same addendum input, without modifying acquisition.
    with tempfile.TemporaryDirectory(prefix='addendum-inventory-') as temporary:
        view = Path(temporary)
        for name in ('host.private.json', 'campaign.json'):
            shutil.copyfile(out/name, view/name)
        adapted = copy.deepcopy(resolved)
        for cell in adapted['cells']:
            cell['confirmation_binding'] = copy.deepcopy(cell['workload_binding'])
        atomic_json(view/'resolved.json', adapted)
        replacements, inventory = portable_inventory(view, destination)
    replacements[str(out)] = '${original_addendum}'
    cells = copy.deepcopy(resolved['cells'])
    for cell in cells:
        cell.pop('runtime_identity', None)
        cell.pop('system_identities', None)
        cell['profile'].pop('allocation_receipts', None)
        cell['profile']['runtime_python'] = '${runtime_python}'
        binding = cell['workload_binding']
        for kind in ('plan', 'corpus'):
            content = load(binding[kind])
            if kind == 'plan':
                content.pop('published_runtimes', None)
            content = transform(content, replacements)
            relative = f'inputs/{cell["cell"]}/{kind}.json'
            atomic_json(destination/relative, content)
            binding[kind] = '${bundle}/'+relative
            binding[kind+'_digest'] = digest(content)
    source = Path(protocol['snapshot'])
    for relative, expected in protocol['source']['files'].items():
        if file_hash(source/relative) != expected:
            raise ValueError('Attested addendum source changed: '+relative)
        target = destination/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, target)
    shutil.copyfile(out/'protocol.lock.json', destination/'original-protocol.lock.json')
    shutil.copytree(out, destination/'evidence')
    portable = dict(schema='stage6-addendum-transfer-v1', parent_campaign_id=protocol['campaign_id'],
        original_protocol_sha256=file_hash(out/'protocol.lock.json'), source=protocol['source'],
        schedule=protocol['schedule'], limits=protocol['limits'], cells=cells, inventory=inventory,
        verification=verified, original_analysis_digest=digest(load(out/'analysis/addendum-comparison.json')),
        hardware_preflight_required=True, execution_authorized=False,
        scope='Distinct frozen addendum source, inputs, schedule, all attempts and audits; not a new campaign or budget reset')
    atomic_json(destination/'addendum-portable.json', portable)
    seal = seal_tree(destination)
    return dict(seal=seal, files=verify_tree(destination, seal))


IMPORT_SCRIPT = r'''
import copy,json,sys
from pathlib import Path
source=Path.cwd().resolve()
from benchmarks.serving_suite.campaign import source_snapshot
from benchmarks.serving_suite.campaign_contracts import load,digest,atomic_json,file_hash,verify_tree
from benchmarks.serving_suite.campaign_verify import transform
from benchmarks.serving_suite.assets import model_identity,runtime_identity
from benchmarks.serving_suite.contracts import validate_corpus,validate_plan
from benchmarks.serving_suite.runner import plan_for_variant,command
from benchmarks.serving_suite.published import load_runtimes
bundle,host_path,runtimes,destination=map(Path,sys.argv[1:])
p=load(bundle/'addendum-portable.json');host=load(host_path)
assert source_snapshot()==p['source']
assert file_hash(bundle/'original-protocol.lock.json')==p['original_protocol_sha256']
verify_tree(bundle)
models={}
for model in {c['model'] for c in p['cells']}:
 path=Path(host['models'][model])
 assert model_identity(path)['content_digest']==p['inventory']['model_reacquisition'][model]['identity']['content_digest']
 models[model]=str(path)
python=str(runtimes/'local-env/bin/python')
runtime=runtime_identity(python)
published=load_runtimes(runtimes/'published-runtimes.json')
bindings={'${bundle}':str(bundle),'${project}':str(source),'${asset_cache}':str(bundle/'assets'),'${runtime_python}':python}
bindings.update({'${model:'+k+'}':v for k,v in models.items()})
cells=transform(p['cells'],bindings);compiled={}
for c in cells:
 binding=c['workload_binding'];plan=transform(load(binding['plan']),bindings);corpus=transform(load(binding['corpus']),bindings)
 assert '${' not in json.dumps([plan,corpus])
 validate_plan(plan);validate_corpus(corpus)
 for session in corpus['sessions']:
  for path,expected in session.get('input_files',{}).items():assert file_hash(path)==expected
 if plan.get('published_variants'):
  plan['published_runtimes']=copy.deepcopy(published)
  for name,r in plan['published_runtimes'].items():
   r['cpu_bytes']=plan['cpu_bytes'];r['kv_transfer_config'].setdefault('kv_connector_extra_config',{})['cpu_bytes_to_use']=plan['cpu_bytes']
   if name=='continuum':r['env']['LMCACHE_MAX_LOCAL_CPU_SIZE']=str(plan['cpu_bytes']/2**30)
 for kind,content in (('plan',plan),('corpus',corpus)):
  target=destination.parent/(destination.stem+'-inputs')/c['cell']/(kind+'.json')
  atomic_json(target,content);binding[kind]=str(target);binding[kind+'_digest']=digest(content)
 compiled[c['cell']]=(plan,corpus)
commands=[]
for block in p['schedule']:
 for cell,system in block['cases']:
  base='arc' if system=='O0' else 'wait' if system=='O2' else system
  plan,corpus=compiled[cell];effective=plan_for_variant(copy.deepcopy(plan),base)
  if base=='arc':effective.update(publish_hints=False,qualification_conditions=['arc','wait'])
  if system in ('O0','O2'):effective.update(publish_hints=False,load_plugin=system!='O0',native_arc_control=system=='O0',qualification_trace=False,required_activity=[],control=system)
  args,environment=command(effective,base,8123)
  compiled_command=[python if a==sys.executable else a for a in args]
  assert compiled_command[0].startswith(str(runtimes))
  if system=='O0':assert 'SemanticOffloadingConnector' not in ' '.join(compiled_command)
  if system=='O2':assert 'SemanticOffloadingConnector' in ' '.join(compiled_command)
  commands.append(dict(block=block['id'],cell=cell,system=system,command=compiled_command,environment={k:v for k,v in environment.items() if k.startswith(('VLLM_','LMCACHE_')) or k in ('PYTHONPATH','CUDA_VISIBLE_DEVICES')},sessions=len(corpus['sessions'])))
assert len(commands)==78
assert all(str(source) in json.dumps(x['command']) or x['system'] in ('continuum','upstream') for x in commands)
record=dict(schema='stage6-addendum-import-v1',parent_campaign_id=p['parent_campaign_id'],source=p['source']['digest'],original_protocol_sha256=p['original_protocol_sha256'],schedule=p['schedule'],limits=p['limits'],models=models,runtime_python=python,runtime_identity_digest=digest(runtime),published_runtime_root=str(runtimes),cells=cells,compiled_configurations=commands,hardware_preflight_required=True,execution_authorized=False,scope='GPU-free relocated source/input/runtime rebinding and command compilation; no native serving or renewed measurement authorization')
atomic_json(destination,record)
print(json.dumps(dict(source_root=str(source),compiled_configurations=len(commands),schedule_digest=digest(p['schedule']),limits_digest=digest(p['limits']),imported_receipt=str(destination),hardware_preflight_required=True,execution_authorized=False)))
'''

VERIFY_SCRIPT = r'''
import json,sys
from pathlib import Path
source=Path.cwd().resolve();bundle,original,project,cache=map(Path,sys.argv[1:5]);models=json.loads(sys.argv[5])
from benchmarks.serving_suite.campaign_portability import install_offline_guard
install_offline_guard(original,models,forbidden_roots=[project,cache],allowed_roots=[bundle])
from benchmarks.serving_suite.campaign_verify import verify_campaign
from benchmarks.serving_suite.campaign_contracts import load,digest
assert not original.exists()
negative=0
for path in (project/'pyproject.toml',cache/'sources.json'):
 try:path.read_bytes()
 except RuntimeError:negative+=1
 else:raise AssertionError('Original path readable')
r=verify_campaign(bundle/'evidence')
p=load(bundle/'addendum-portable.json')
assert digest(load(bundle/'evidence/analysis/addendum-comparison.json'))==p['original_analysis_digest']
print(json.dumps(dict(verification=r,original_path_negative_tests=negative,source_root=str(source),analysis_preserved=True)))
'''


def prove(out: Path, destination: Path, runtimes: Path) -> dict:
    out, destination, runtimes = (p.resolve() for p in (out,destination,runtimes))
    destination.mkdir(parents=True, exist_ok=False)
    receipt = export(out, destination/'initial-bundle')
    bundle = destination/'relocated-bundle'
    (destination/'initial-bundle').rename(bundle)
    verify_tree(bundle, receipt['seal'])
    host = load(out/'host.private.json')
    host_path = destination/'original-host.private.json'
    atomic_json(host_path, host);host_path.chmod(0o600)
    parked = out.with_name(out.name+'-portability-parked')
    if parked.exists():
        raise ValueError('Preserve existing parked evidence')
    with Ledger(out).writer():
        if any(e['state'] in ('running','planned') for e in Ledger(out).index().values()):
            raise ValueError('Active acquisition')
        out.rename(parked)
        try:
            raw = subprocess.check_output([sys.executable,'-c',VERIFY_SCRIPT,str(bundle),str(out),
                host['runtime_bindings']['project_root'],host['asset_cache'],json.dumps(list(host['models'].values()))],
                cwd=bundle/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),text=True)
            offline = json.loads(raw)
        finally:
            parked.rename(out)
    raw = subprocess.check_output([sys.executable,'-c',IMPORT_SCRIPT,str(bundle),str(host_path),str(runtimes),str(destination/'imported-addendum.json')],
        cwd=bundle/'source',env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),text=True)
    imported=json.loads(raw)
    result=dict(schema='stage6-addendum-portability-proof-v1',complete=True,bundle=str(bundle),bundle_seal=receipt['seal'],
        bundle_files=verify_tree(bundle,receipt['seal']),offline=offline,imported=imported,
        rebuilt_runtime=str(runtimes),original_evidence_absent_during_verification=True,
        scope='Real bundle/evidence relocation, model-forbidden integrity verification and separate rebuilt-runtime GPU-free input import/78-command compilation; no native execution or new budget')
    atomic_json(out/'analysis/portability-proof.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','destination','runtimes'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(prove(args.out,args.destination,args.runtimes),indent=2))

if __name__=='__main__':main()
