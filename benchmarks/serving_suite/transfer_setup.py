"""Prepare host bindings for an exported protocol; never change its workload.

Only pinned model weights are downloaded. Selected corpora, references, real
tool sources and source/patch/runtime recipes are already in the transfer bundle.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .assets import MODEL_PATTERNS,model_identity
from .campaign_contracts import atomic_json,load,validate_host,verify_tree
from .published import load_runtimes


def prepare(bundle,runtimes,cache,out,host_id,*,offline=False):
    from huggingface_hub import snapshot_download
    bundle,runtimes,cache,out = map(lambda p:Path(p).resolve(),(bundle,runtimes,cache,out))
    if out.exists():
        raise ValueError('Use a new host binding file')
    verify_tree(bundle)
    portable = load(bundle/'portable.json')
    if portable['schema']!='serving-campaign-transfer-v2':
        raise ValueError('Unknown transfer schema')
    models = {}
    for model,record in portable['inventory']['model_reacquisition'].items():
        path = snapshot_download(record['repository'],revision=record['revision'],
            allow_patterns=MODEL_PATTERNS,max_workers=2,local_files_only=offline)
        identity = model_identity(path)
        if identity['content_digest']!=record['identity']['content_digest']:
            raise ValueError('Reacquired model differs from the frozen content manifest')
        models[model] = path
    published_path = runtimes/'published-runtimes.json'
    published = load_runtimes(published_path)
    from .campaign import ROOT
    host = validate_host(dict(schema='serving-campaign-host-v1',id=host_id,device='cuda:0',
        runtime_python=str(runtimes/'local-env/bin/python'),asset_cache=str(cache),models=models,
        runtime_bindings=dict(project_root=str(ROOT)),published_runtimes=str(published_path),
        resolved_published_runtimes=published,
        limits=dict(startup_timeout_s=900,request_timeout_s=180,tool_timeout_s=180,case_timeout_s=1800)))
    cache.mkdir(parents=True,exist_ok=True)
    atomic_json(out,host)
    out.chmod(0o600)
    return dict(host=str(out),models_verified=sorted(models),offline=offline,
        scope='Prepared pinned bindings; native hardware correctness/capacity preflight remains mandatory')


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','runtimes','cache','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--id',required=True)
    parser.add_argument('--offline',action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle,args.runtimes,args.cache,args.out,args.id,offline=args.offline),indent=2))


if __name__=='__main__':
    main()
