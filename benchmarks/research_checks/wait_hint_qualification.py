"""Finite, source-frozen qualification campaigns using the maintained serving suite.

A spec declares the base plan, frozen corpus, conditions and complete order before
launch. Each case retains ordinary suite artifacts and is verified offline. This
coordinator holds both GPU reservations across all cases and idle gaps.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import signal
import sys
import time
from pathlib import Path

from harness.gpu_lock import gpu_reserved, GPULock, LEGACY_LOCK
from benchmarks.serving_suite.contracts import SCHEMA, digest, write_json
from benchmarks.serving_suite.runner import (run_case, source_identity, freeze_source)
from benchmarks.serving_suite.resources import available_gpu
from benchmarks.serving_suite.report import paired_comparison
from benchmarks.serving_suite.verify import seal, verify


def identity():
    result = source_identity()
    for name in ('wait_hint_qualification','wait_hint_runtime_probe','qualification_inputs','qualification_quality'):
        p=Path(__file__).with_name(name+'.py')
        result['files'][str(p.resolve().relative_to(Path(__file__).resolve().parents[2]))]=hashlib.sha256(p.read_bytes()).hexdigest()
    result['digest']=digest(result['files'])
    return result


def comparisons(cases, conditions):
    result = {}
    for baseline, candidate in itertools.combinations(conditions,2):
        chosen = [{**c,'arm':'arc' if c['variant']==baseline else 'candidate'}
                  for c in cases if c['variant'] in (baseline,candidate)]
        result[f'{candidate}_vs_{baseline}'] = paired_comparison(chosen)
    return result


def check_replay_row(row, expected):
    if row.get('wave') == -1:
        if row.get('session_id') != 'cancel' or not row.get('payload',{}).get('kv_transfer_params',{}).get('suite_probe_id'):
            raise ValueError('Unexpected non-workload replay row')
        return  # Abort is independently checked by the operational contract.
    if row['success'] and (row['payload']['prompt'],row['output_token_ids']) != expected[row['source_id'],row['turn']]:
        raise ValueError('Replay is not identical frozen work')


def offline(directory):
    directory = Path(directory)
    checked = verify(directory)
    result = json.loads((directory/'result.json').read_text())
    expected = comparisons(result['cases'],result['plan']['qualification_conditions'])
    if expected != result['qualification_comparisons']:
        raise ValueError('Qualification aggregate mismatch')
    if result['plan'].get('diagnostic_replay'):
        corpus = json.loads((directory/'corpus.json').read_text())
        expected = {(s['id'],i):(r['payload']['prompt'],r['tokens'])
                    for s in corpus['sessions'] for i,r in enumerate(s['replay'])}
        for case in result['cases']:
            for line in (directory/case['name']/'requests.jsonl').read_text().splitlines():
                row = json.loads(line)
                check_replay_row(row, expected)
    return checked


@gpu_reserved
def execute(spec, output):
    output.mkdir(exist_ok=False,parents=True)
    plan = copy.deepcopy(spec['plan'])
    plan.update(output=str(output.resolve()),schema=SCHEMA,
                qualification_conditions=list(spec['conditions']))
    write_json(output/'spec.json',spec)
    write_json(output/'plan.json',plan)
    write_json(output/'corpus.json',spec['corpus'])
    result = dict(schema=SCHEMA,plan=plan,cases=[],status='running',comparison=None,
                  qualification_comparisons={},started_unix=time.time())
    initial = identity()
    write_json(output/'source.json',initial)
    freeze_source(initial,output)
    # The coordinator itself is outside source_identity's normal suite set.
    coordinator = Path(__file__)
    coordinator_hash = hashlib.sha256(coordinator.read_bytes()).hexdigest()
    (output/'coordinator.py').write_bytes(coordinator.read_bytes())
    inputs=spec.get('input_files',{})
    if inputs:
        import tarfile
        with tarfile.open(output/'external-inputs.tar.gz','w:gz') as archive:
            for index,(name,expected) in enumerate(inputs.items()):
                path=Path(name)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise RuntimeError('External input changed before freeze: '+name)
                archive.add(path,arcname=f'inputs/{index:04d}/{path.name}',recursive=False)
        write_json(output/'external-input-identities.json',inputs)
    if plan.get('published_runtimes'):
        from benchmarks.serving_suite.published import freeze_sources
        freeze_sources(plan['published_runtimes'],output)
    def unchanged():
        if identity()['digest'] != initial['digest'] or hashlib.sha256(coordinator.read_bytes()).hexdigest()!=coordinator_hash:
            raise RuntimeError('Executed source changed; preserve measured cases and stop dispatch')
        for name,expected in inputs.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Executed external input changed: '+name)
    def interrupt(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    old = signal.signal(signal.SIGTERM,interrupt)
    try:
        with GPULock(path=LEGACY_LOCK):
            result['hardware'] = available_gpu()
            model_files = {}
            for p in sorted(Path(plan['model']).iterdir()):
                if p.is_file():
                    with p.open('rb') as stream:
                        model_files[p.name] = dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
            write_json(output/'model-files.json',model_files)
            for invocation in spec['order']:
                repeat,condition = invocation
                unchanged()
                available_gpu()
                case_plan = copy.deepcopy(plan)
                case_plan.update(copy.deepcopy(spec['conditions'][condition]))
                if case_plan.get('published_runtime'):
                    from benchmarks.serving_suite.published import check_source
                    check_source(case_plan['published_variant'],case_plan['published_runtime'])
                write_json(output/f'case-plan-{repeat:02d}-{condition}.json',case_plan)
                print(f'Running {repeat:02d}-{condition}',flush=True)
                case = run_case(case_plan,spec['corpus']['sessions'][:plan['sessions']],
                                'arc' if condition=='arc' else 'candidate',repeat,output,variant=condition)
                # Retain the exact effective plan for verification of variants.
                case['qualification_plan_overrides'] = spec['conditions'][condition]
                if case_plan.get('published_runtime'):
                    check_source(case_plan['published_variant'],case_plan['published_runtime'])
                result['cases'].append(case)
                result['qualification_comparisons'] = comparisons(result['cases'],list(spec['conditions']))
                write_json(output/'result.json',result)
                print(json.dumps(dict(case=case['name'],seconds=case['load_duration_s'],
                    evaluation=case['evaluation'],error=case.get('error'))),flush=True)
                if not case['cleanup']['verified']:
                    raise RuntimeError('Owned process cleanup failed')
                if case.get('error') or case['evaluation']['status']=='failed':
                    raise RuntimeError('Case failed: diagnose before continuing; no hidden retry')
            unchanged()
            result['status'] = 'passed' if all(c['evaluation']['status']=='passed' for c in result['cases']) else 'inconclusive'
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        signal.signal(signal.SIGTERM,old)
        result['wall_s'] = time.time()-result['started_unix']
        result['qualification_comparisons'] = comparisons(result['cases'],list(spec['conditions']))
        write_json(output/'result.json',result)
        seal(output)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.verify:
        print(json.dumps(offline(args.output),indent=2));return 0
    if not args.spec:
        p.error('--spec is required for execution')
    result=execute(json.loads(args.spec.read_text()),args.output)
    print(json.dumps(offline(args.output),indent=2))
    return 0 if result['status']=='passed' else 2

if __name__=='__main__':
    raise SystemExit(main())
