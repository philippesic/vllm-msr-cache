"""Real relocation, exported-source offline verification and import dry compile."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

from .campaign_contracts import Ledger,atomic_json,digest,load,read_protocol,verify_tree
from .campaign_verify import export_campaign,import_campaign


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
def audit(event,args):
 if event=='import' and str(args[0]).split('.')[0] in ('torch','vllm','transformers','lmcache'):
  raise RuntimeError('GPU/model dependency imported during offline verification')
 if event=='open' and isinstance(args[0],(str,bytes)):
  path=Path(os.fsdecode(args[0])).resolve()
  if path.is_relative_to(original) or any(path.is_relative_to(p) for p in models):
   raise RuntimeError('Offline verifier opened an original artifact/model path')
sys.addaudithook(audit)
from benchmarks.serving_suite import campaign_verify
assert Path(campaign_verify.__file__).resolve().is_relative_to(Path.cwd())
result=campaign_verify.verify_campaign(sys.argv[1])
from benchmarks.serving_suite.campaign_report import report
analysis=report(sys.argv[1])
from benchmarks.serving_suite.campaign_contracts import digest
assert not original.exists()
print(json.dumps(dict(verification=result,report=dict(
 throughput_acquisition_complete=analysis['throughput_acquisition_complete'],
 comparisons=len(analysis['comparisons']),comparisons_digest=digest(analysis['comparisons']),
 quality_status=analysis['quality']['status']))))
'''
            raw = subprocess.check_output([sys.executable,'-c',script,str(relocated),str(out),
                json.dumps(list(host['models'].values()))],cwd=moved/'source',
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),text=True)
            verification = json.loads(raw)
        finally:
            parked.rename(out)
    host_path = destination/'import-host.private.json'
    if runtimes:
        from .transfer_setup import prepare
        prepare(moved,runtimes,destination/'rebound-assets',host_path,host['id']+'-relocated',offline=True)
    else:
        local_host = dict(host,id=host['id']+'-relocated')
        atomic_json(host_path,local_host)
        host_path.chmod(0o600)
    imported = destination/'imported-campaign'
    result = import_campaign(moved,host_path,imported)
    from .campaign import run_phase
    dry = run_phase(imported,'preflight',dry_run=True)
    blocked = False
    try:
        run_phase(imported,'confirm',dry_run=True)
    except ValueError as error:
        blocked = 'locked protocol' in str(error)
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
        gpu_model_imports_forbidden=True,verifier_source='relocated exported source closure',
        imported_campaign=str(imported),import_result=result,preflight_dry_compile=dry,
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
