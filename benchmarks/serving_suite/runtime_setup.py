"""Reconstruct isolated runtimes from public pins, without historical run trees."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .campaign_contracts import atomic_json, file_hash
from .published import PINS, load_runtimes

ROOT = Path(__file__).resolve().parents[2]
WHEELS = {
    'continuum':'https://files.pythonhosted.org/packages/a2/1a/365479f413e7408b314c0237d6c929569874d5c002bc7c8b5a7fbf40c7d9/vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl',
    'upstream':'https://wheels.vllm.ai/ccda9098cc5a91b6b6f73e0c0e697022382e01b4/vllm-0.30.1rc1.dev530%2Bgccda9098c-cp38-abi3-manylinux_2_28_x86_64.whl'}


def prepare(destination, *, dry_run=False):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Use a new runtime destination; preserve existing environments and logs')
    commands = []
    for name in ('local','continuum','upstream'):
        python = str(destination/(name+'-env')/'bin/python')
        requirements = ROOT/'benchmarks/serving_suite/runtime_requirements'/(name+'.txt')
        commands += [dict(step=name+' environment',argv=['uv','venv','--seed','--python','3.12',str(destination/(name+'-env'))]),
                     dict(step=name+' dependencies',argv=['uv','pip','install','--python',python,'--no-deps','-r',str(requirements)])]
        if name=='local':
            commands.append(dict(step='local extension',argv=['uv','pip','install','--python',python,'--no-deps','-e',str(ROOT)]))
            continue
        source = destination/(name+'-source')
        repository,revision = PINS[name]
        commands += [dict(step=name+' checkout',argv=['git','init','-q',str(source)]),
                     dict(step=name+' origin',argv=['git','-C',str(source),'remote','add','origin',repository]),
                     dict(step=name+' fetch',argv=['git','-C',str(source),'fetch','--depth','1','origin',revision]),
                     dict(step=name+' pin',argv=['git','-C',str(source),'checkout','--detach',revision]),
                     dict(step=name+' editable',argv=['uv','pip','install','--python',python,'--no-deps','--no-build-isolation','-e',str(source)],
                          env=dict(VLLM_TARGET_DEVICE='cuda',VLLM_USE_PRECOMPILED='1',VLLM_PRECOMPILED_WHEEL_LOCATION=WHEELS[name],
                                   SETUPTOOLS_SCM_PRETEND_VERSION='0.10.2+continuum.316a587' if name=='continuum' else '0.1.dev1+g7966f717d'))]
    if dry_run:
        return dict(destination=str(destination),commands=commands,scope='Public pinned precompiled native bases; hardware correctness remains mandatory')
    destination.mkdir(parents=True)
    atomic_json(destination/'preparation-plan.json',dict(commands=commands,wheels=WHEELS,
        requirements={p.name:file_hash(p) for p in (ROOT/'benchmarks/serving_suite/runtime_requirements').glob('*.txt')}))
    for index,step in enumerate(commands):
        print(step['step'],flush=True)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='',**step.get('env',{}))
        with (destination/f'{index:02d}-setup.log').open('xb') as log:
            subprocess.run(step['argv'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for name in ('local','continuum','upstream'):
        # A successful --no-deps install alone does not prove dependency closure.
        command = [str(destination/(name+'-env')/'bin/python'),'-c',
            'import torch,vllm,json; print(json.dumps(dict(torch=torch.__version__,vllm=vllm.__version__,architectures=torch.cuda.get_arch_list())))']
        result = subprocess.check_output(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True)
        (destination/(name+'-import-check.txt')).write_text(result)
    execution = destination/'upstream-execution'
    shutil.copytree(destination/'upstream-source',execution,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    patch = ROOT/'evidence/upstream-pin-fix-20261104/mapped-host-pointer.patch'
    subprocess.run(['patch','-s','-p1','-d',str(execution),'-i',str(patch)],check=True)
    cpu = 74*256*147456
    runtimes = dict(schema='published-agent-runtimes-v1',runtimes={
        name:dict(source=str(destination/(name+'-source')),python=str(destination/(name+'-env')/'bin/python'),
            cpu_bytes=cpu,kv_transfer_config=dict(kv_connector='LMCacheConnectorV1' if name=='continuum' else 'OffloadingConnector',kv_role='kv_both'),
            env=dict(LMCACHE_MAX_LOCAL_CPU_SIZE=str(cpu/2**30),LMCACHE_SAVE_DECODE_CACHE='true') if name=='continuum' else
                dict(VLLM_USE_V2_MODEL_RUNNER='1',VLLM_WSL2_ENABLE_PIN_MEMORY='1')) for name in PINS})
    runtimes['runtimes']['upstream'].update(execution_source=str(execution),compatibility_patch=str(patch),
        compatibility_patch_sha256=file_hash(patch),retention_ttl_s=60,label='upstream draft + host-pointer compatibility fix')
    runtimes['runtimes']['upstream']['kv_transfer_config']['kv_connector_extra_config']=dict(cpu_bytes_to_use=cpu,eviction_policy='arc',store_threshold=0)
    path = destination/'published-runtimes.json'
    atomic_json(path,runtimes)
    resolved = load_runtimes(path)
    atomic_json(destination/'published-identities.json',resolved)
    return dict(destination=str(destination),local_python=str(destination/'local-env/bin/python'),
                published_runtimes=str(path),published_identities=str(destination/'published-identities.json'),
                scope='Prepared isolated dependencies; native GPU correctness and capacity are not yet qualified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.out,dry_run=args.dry_run),indent=2))


if __name__=='__main__':
    main()
