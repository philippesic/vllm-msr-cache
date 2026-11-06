"""Prepare fixed Stage 3 campaigns after selecting the Stage 2 substitute.

Preparation does not launch GPU work. Every generated spec retains exact corpus,
condition overrides and order for the source-frozen qualification coordinator.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from benchmarks.serving_suite.contracts import digest
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def select_substitute(base):
    measurements={}
    for name in ('stage2-published','stage2-resource'):
        result=read(base/name/'result.json')
        if result['status'] not in ('passed','inconclusive'):
            raise ValueError('Complete and diagnose Stage 2 before selecting a substitute')
        for case in result['cases']:
            condition=case['variant']
            if condition in ('wait','arc-more-cpu'):
                continue
            if case.get('error') or case['summary']['failed'] or case['summary']['completed']!=case['expected_requests']:
                raise ValueError('Do not select rivals from successful survivors')
            measurements.setdefault(condition,[]).append(case['load_duration_s'])
    means={key:math.exp(sum(map(math.log,values))/len(values)) for key,values in measurements.items()}
    return min(means,key=means.get),dict(rule='Lowest geometric mean complete batch time among equal-budget noncandidate Stage 2 systems. Quality/activation limitations remain visible.',batch_seconds=means)


def prepare(base,alternative):
    main=read(base/'stage2-published-spec.json')
    resource=read(base/'stage2-resource-spec.json')
    candidates={**main['conditions'],**resource['conditions']}
    names=list(dict.fromkeys(('arc','wait',alternative)))
    external=read(base/'independent-source-identities.json')
    if 'files' in external:
        external=external['files']
    # Normalize the retained source inventory to exact file hashes.
    if isinstance(external,list):
        external={item['path']:item['sha256'] for item in external}
    inputs={}
    for key,value in external.items():
        if isinstance(value,str) and Path(key).is_file():inputs[str(Path(key).resolve())]=value
    for path in sorted((base/'independent-inputs').glob('document-references-*.json')):
        inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    def campaign(corpus,changes=None,*,mixed=False,waves=1,extra_alternatives=()):
        cell_names=list(dict.fromkeys(names+list(extra_alternatives)))
        plan=copy.deepcopy(main['plan']);plan.update(repeats=3,waves=waves,**(changes or {}))
        plan.update(corpus_provenance=corpus.get('provenance',plan['corpus_provenance']),
                    corpus_sha256=digest(corpus))
        conditions={key:copy.deepcopy(candidates[key]) for key in cell_names}
        for key,condition in conditions.items():
            if key in ('arc','wait','lru'):condition['required_activity']=['store']
            if condition.get('published_runtime'):
                runtime=condition['published_runtime'];runtime['cpu_bytes']=plan['cpu_bytes']
                if key=='continuum':runtime['env']['LMCACHE_MAX_LOCAL_CPU_SIZE']=str(plan['cpu_bytes']/2**30)
                else:runtime['kv_transfer_config'].setdefault('kv_connector_extra_config',{})['cpu_bytes_to_use']=plan['cpu_bytes']
        corpus=copy.deepcopy(corpus)
        if mixed:
            for index,session in enumerate(corpus['sessions']):session['publish_hints']=index%2==0
        plan['corpus_sha256']=digest(corpus)
        order=[[repeat,cell_names[(position+repeat)%len(cell_names)]] for repeat in range(3) for position in range(len(cell_names))]
        files=copy.deepcopy(inputs) if corpus.get('workload_kind','').startswith('live_') else {}
        # All session-declared executed inputs are included even if inventory shape changes.
        for session in corpus['sessions']:
            for name,expected in session.get('input_files',{}).items():
                path=Path(name)
                # The original corpus retains its earlier context snapshot.
                # Current repository tools execute the source frozen now.
                files[str(path.resolve())]=(expected if path.is_absolute()
                    else hashlib.sha256(path.read_bytes()).hexdigest())
        return dict(plan=plan,corpus=corpus,conditions=conditions,order=order,input_files=files,
            input_scope_note='Corpus context and its original hashes remain frozen. Relative repository files execute at the current source freeze; independent absolute public inputs must still match their original hashes.')
    outputs={}
    for family in ('repository','documents'):
        corpus=read(base/'independent-inputs'/f'{family}-evaluation.json')
        outputs[f'stage3-{family}-spec.json']=campaign(corpus,extra_alternatives=('upstream',))
        outputs[f'stage3-{family}-cohost-spec.json']=campaign(corpus,dict(cpu_bytes=1409286144,gpu_blocks=1280),extra_alternatives=('upstream',))
        for key in (f'stage3-{family}-spec.json',f'stage3-{family}-cohost-spec.json'):
            outputs[key]['plan']['task_file']=str((base/'independent-inputs'/f'{family}-evaluation.json').resolve())
    for name,changes in {
        'ample':dict(cpu_bytes=3758096384,concurrency=4),
        'cpu-pressure':dict(cpu_bytes=2583691264),
        'concurrency':dict(concurrency=12),
        'serialized-tools':dict(tool_concurrency=1),
        'mixed':{},
    }.items():
        extras=('continuum','upstream') if name=='serialized-tools' else ('upstream',) if name in ('cpu-pressure','concurrency','mixed') else ()
        outputs[f'stage3-{name}-spec.json']=campaign(main['corpus'],changes,mixed=name=='mixed',extra_alternatives=extras)
    soak=campaign(main['corpus'],dict(arrival_interval_s=.25,case_timeout_s=1500),mixed=True,waves=6)
    # A finite soak is an exposure check, not six independent statistical repetitions.
    soak['plan']['repeats']=1;soak['order']=[[0,key] for key in names]
    outputs['stage3-soak-spec.json']=soak
    for name,spec in outputs.items():
        destination=base/name
        if destination.exists():raise FileExistsError(destination)
        destination.write_text(json.dumps(spec,indent=2)+'\n')
    return list(outputs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base',type=Path)
    args=parser.parse_args()
    selected,selection=select_substitute(args.base)
    selection['selected']=selected
    names=prepare(args.base,selected)
    selection['prepared_specs']=names
    path=args.base/'stage3-substitute-selection.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(selection,indent=2))

if __name__=='__main__':main()
