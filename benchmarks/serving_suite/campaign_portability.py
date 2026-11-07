"""Real relocation, exported-source offline verification and import dry compile."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

from .campaign_contracts import Ledger,atomic_json,digest,load,read_protocol,verify_tree
from .campaign_verify import export_campaign


def install_offline_guard(original,models,*,forbidden_roots=(),allowed_roots=()):
    original = Path(original).resolve()
    models = [Path(p).resolve() for p in models]
    forbidden = [original,*[Path(p).resolve() for p in forbidden_roots]]
    allowed = [Path(p).resolve() for p in allowed_roots]
    # Snapshot files may resolve outside the snapshot into sibling blobs/.
    model_files = {p.resolve() for root in models for p in root.rglob('*') if p.is_file()}
    def audit(event,args):
        if event=='import' and str(args[0]).split('.')[0] in ('torch','vllm','transformers','lmcache'):
            raise RuntimeError('GPU/model dependency imported during offline verification')
        if event=='open' and isinstance(args[0],(str,bytes)):
            path = Path(os.fsdecode(args[0])).resolve()
            model_access = path in model_files or any(path.is_relative_to(p) for p in models)
            original_access = any(path.is_relative_to(p) for p in forbidden) and not any(path.is_relative_to(p) for p in allowed)
            if model_access or original_access:
                raise RuntimeError('Offline verifier opened an original artifact/model path')
    sys.addaudithook(audit)


def import_relocated(destination,moved,host,*,runtimes=None):
    """Run binding/import/compilation from exported code in a fresh process."""
    destination,moved = Path(destination).resolve(),Path(moved).resolve()
    original_host = destination/'original-host.private.json'
    atomic_json(original_host,host)
    original_host.chmod(0o600)
    script = '''
import json,sys
from pathlib import Path
source=Path.cwd().resolve()
from benchmarks.serving_suite import campaign,campaign_verify
assert Path(campaign.__file__).resolve().is_relative_to(source)
assert Path(campaign_verify.__file__).resolve().is_relative_to(source)
from benchmarks.serving_suite.campaign_contracts import atomic_json,load
destination,bundle,original_host=map(Path,sys.argv[1:4])
host=load(original_host)
host_path=destination/'import-host.private.json'
if sys.argv[4]:
 from benchmarks.serving_suite.transfer_setup import prepare
 prepare(bundle,Path(sys.argv[4]),destination/'rebound-assets',host_path,host['id']+'-relocated',offline=True)
else:
 host=dict(host,id=host['id']+'-relocated',runtime_bindings=dict(host.get('runtime_bindings',{}),project_root=str(source)))
 atomic_json(host_path,host)
 host_path.chmod(0o600)
assert Path(load(host_path)['runtime_bindings']['project_root']).resolve()==source
imported=destination/'imported-campaign'
result=campaign_verify.import_campaign(bundle,host_path,imported)
dry=campaign.run_phase(imported,'preflight',dry_run=True)
blocked=False
try:
 campaign.run_phase(imported,'confirm',dry_run=True)
except ValueError as error:
 blocked='locked protocol' in str(error)
print(json.dumps(dict(import_result=result,preflight_dry_compile=dry,confirmation_blocked=blocked,source_root=str(source))))
'''
    source = moved/'source'
    raw = subprocess.check_output([sys.executable,'-c',script,str(destination),str(moved),
        str(original_host),str(Path(runtimes).resolve()) if runtimes else ''],cwd=source,
        env=dict(os.environ,PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),text=True)
    result = json.loads(raw)
    if Path(result['source_root']).resolve()!=source:
        raise ValueError('Import/dry compilation did not use relocated source')
    return result


def prove(out,destination,*,runtimes=None):
    out,destination = Path(out).resolve(),Path(destination).resolve()
    if destination.exists():
        raise ValueError('Use a new portability proof destination')
    if destination.is_relative_to(out):
        raise ValueError('Portability destination must be outside the original campaign')
    destination.mkdir(parents=True)
    exported = destination/'initial-bundle'
    receipt = export_campaign(out,exported)
    moved = destination/'relocated-bundle'
    exported.rename(moved)
    verify_tree(moved,receipt['seal'])
    relocated = destination/'relocated-evidence'
    parked = out.with_name(out.name+'-portability-parked')
    if parked.exists():
        raise ValueError('A parked original already exists; preserve it for recovery')
    # Hold the original writer inode throughout the reversible move. No active
    # acquisition can append to the evidence while the original path is hidden.
    with Ledger(out).writer():
        if any(e['state'] in ('running','planned') for e in Ledger(out).index().values()):
            raise ValueError('Finish or recover active attempts before relocation')
        shutil.copytree(out,relocated)
        host = load(out/'host.private.json')
        out.rename(parked)
        try:
            script = '''
import json,os,sys
from pathlib import Path
original=Path(sys.argv[2]).resolve()
models=[Path(p).resolve() for p in json.loads(sys.argv[3])]
source,cache=map(Path,sys.argv[4:6])
from benchmarks.serving_suite.campaign_portability import install_offline_guard
install_offline_guard(original,models,forbidden_roots=[source,cache],allowed_roots=[Path.cwd(),Path(sys.argv[1])])
guard_checks=0
for root in models:
 weight=next(root.rglob('*.safetensors'))
 for path in (root/'config.json',weight,weight.resolve()):
  try:
   with path.open('rb'): pass
  except RuntimeError as error:
   assert 'original artifact/model path' in str(error)
   guard_checks+=1
  else:
   raise AssertionError('Model/symlink access guard failed')
original_checks=0
for path in (source/'pyproject.toml',cache/'sources.json'):
 try:
  path.read_bytes()
 except RuntimeError as error:
  assert 'original artifact/model path' in str(error)
  original_checks+=1
 else:
  raise AssertionError('Original source/asset path guard failed')
from benchmarks.serving_suite import campaign_verify
assert Path(campaign_verify.__file__).resolve().is_relative_to(Path.cwd())
result=campaign_verify.verify_campaign(sys.argv[1])
from benchmarks.serving_suite.campaign_report import report
analysis=report(sys.argv[1])
from benchmarks.serving_suite.campaign_contracts import digest
assert not original.exists()
print(json.dumps(dict(verification=result,model_access_guard_negative_tests=guard_checks,original_path_guard_negative_tests=original_checks,report=dict(
 throughput_acquisition_complete=analysis['throughput_acquisition_complete'],
 comparisons=len(analysis['comparisons']),comparisons_digest=digest(analysis['comparisons']),
 quality_status=analysis['quality']['status']))))
'''
            from .campaign import ROOT
            original_project = host.get('runtime_bindings',{}).get('project_root',str(ROOT))
            raw = subprocess.check_output([sys.executable,'-c',script,str(relocated),str(out),
                json.dumps(list(host['models'].values())),original_project,host['asset_cache']],cwd=moved/'source',
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),text=True)
            verification = json.loads(raw)
        finally:
            parked.rename(out)
    rebound = import_relocated(destination,moved,host,runtimes=runtimes)
    imported = destination/'imported-campaign'
    result,dry,blocked = (rebound[k] for k in ('import_result','preflight_dry_compile','confirmation_blocked'))
    parent = read_protocol(out)
    child = load(imported/'imported-protocol.json')
    preserved = all(parent[k]==child[k] for k in ('source','schedule','supplementary_schedules','precision'))
    from .campaign_report import report
    reference = report(out)
    if verification['report']['comparisons_digest']!=digest(reference['comparisons']):
        raise ValueError('Relocated offline report disagrees with original evidence')
    if not result['runnable'] or not blocked or not preserved or result['campaign_id']==result['parent_campaign_id']:
        raise ValueError('Relocated import protocol or hardware-preflight guard failed')
    proof = dict(schema='stage6-portability-proof-v1',complete=True,
        relocated_bundle=str(moved),bundle_seal=receipt['seal'],bundle_files=verify_tree(moved,receipt['seal']),
        relocated_evidence=str(relocated),offline_verification=verification['verification'],
        offline_report=verification['report'],relocated_comparisons_match_original=True,
        original_evidence_path_absent_during_verification=True,model_files_forbidden_during_verification=True,
        model_blob_symlinks_forbidden=True,model_access_guard_negative_tests=verification['model_access_guard_negative_tests'],
        original_source_historical_outputs_and_assets_forbidden=True,
        original_path_guard_negative_tests=verification['original_path_guard_negative_tests'],
        gpu_model_imports_forbidden=True,verifier_source='relocated exported source closure',
        imported_campaign=str(imported),import_result=result,preflight_dry_compile=dry,
        import_and_dry_compile_source=rebound['source_root'],import_from_relocated_exported_source=True,
        separately_rebuilt_runtime_bindings=bool(runtimes),
        confirmation_blocked_until_hardware_preflight=blocked,parent_protocol_preserved=preserved,
        protocol_identity=digest(parent),
        scope='Offline integrity, relocation and GPU-free import compilation; no 5080 or imported native execution')
    atomic_json(out/'analysis/portability-proof.json',proof)
    return proof


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--runtimes',type=Path,help='Optional independently rebuilt runtimes; reuse pinned cached weights offline')
    args = parser.parse_args()
    print(json.dumps(prove(args.out,args.destination,runtimes=args.runtimes),indent=2))
