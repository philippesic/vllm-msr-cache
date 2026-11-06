"""Prepare and execute finite Stage 5 tranches using the maintained case runner.

Prior local traces are explicit development inputs. Each tranche freezes its
plan, executed code and all external inputs before the first measurement.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

from benchmarks.serving_suite.contracts import digest, write_json
from benchmarks.research_checks.wait_hint_qualification import execute, offline
from harness.gpu_lock import gpu_reserved

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'output/wait-qualification-20261105'

# Each entry changes one declared dimension, unless explicitly a combination.
VARIANTS = {
    'arc': {'publish_hints':False},
    'wait': {},
    'recent': {'advice':{'queue':'recent'}},
    'frequent': {'advice':{'queue':'frequent'}},
    'boost25': {'advice':{'boost':.25}},
    'boost50': {'advice':{'boost':.5}},
    'prefix25': {'advice':{'fraction':.25}},
    'prefix50': {'advice':{'fraction':.5}},
    'prefix128': {'advice':{'max_keys':128}},
    'suffix-order': {'advice':{'order':'suffix'}},
    'cooldown2': {'advice':{'cooldown_ms':2000}},
    'cooldown5': {'advice':{'cooldown_ms':5000}},
    'pressure75': {'advice':{'min_fill':.75}},
    'pressure95': {'advice':{'min_fill':.95}},
    'budget512': {'advice':{'session_budget':512}},
    'budget1024': {'advice':{'session_budget':1024}},
    'cadence025': {'hint_refresh_s':.25},
    'cadence2': {'hint_refresh_s':2.0},
    'cadence5': {'hint_refresh_s':5.0},
    'single': {'hint_initial_only':True},
    'queue-only': {'hint_during_wait':False},
    'tool-only': {'hint_through_queue':False},
    'window250': {'hint_declared_wait_ms':250},
    'window50': {'hint_declared_wait_ms':50},
    'grace750': {'extra':{'wait_hint_cpu_nomination_grace_ms':750}},
    'grace2250': {'extra':{'wait_hint_cpu_nomination_grace_ms':2250}},
    'gpu-gate': {'extra':{'wait_hint_cpu_retention_gpu_gate':True}},
    'gpu-bridge': {'extra':{'wait_hint_cpu_retention_gpu_aware':True}},
    'noop': {'extra':{'benchmark_wait_hint_noop':True}},
    'linear-arc': {'publish_hints':False,'extra':{'arc_linear_eviction':True,'wait_hint_cpu_retention_only':False,'prefetch_budget_fraction':0}},
    'linear-wait': {'extra':{'arc_linear_eviction':True}},
    'audit-off': {'extra':{'preservation_audit':False}},
}


def balanced_order(names, repeats, seed):
    """Balance positions in complete cycles and pair order in two cycles."""
    base=list(names)
    if not base or len(set(base)) != len(base) or repeats < 1:
        raise ValueError('Distinct conditions and positive repetitions required')
    random.Random(seed).shuffle(base)
    order=[]
    for repeat in range(repeats):
        block=base if len(base)==2 or (repeat//len(base))%2==0 else list(reversed(base))
        shift=repeat%len(base)
        order.extend([[repeat,n] for n in block[shift:]+block[:shift]])
    return order


def prepare(workload, names, repeats, seed, overrides=None):
    source=OLD/'stage1-replay-fresh-sockets/spec.json'
    if workload.startswith('live'):
        source=OLD/('stage3-documents-admission-diagnostic-spec.json' if workload=='live-documents' else 'stage3-mixed-spec.json' if workload=='live-mixed' else 'stage4-graphs/spec.json')
    original=json.loads(source.read_text())
    plan=copy.deepcopy(original['plan'])
    for k in ('published_runtimes','published_variants','published_variant','published_runtime','comparison_scope','scope_note'):
        plan.pop(k,None)
    corpus=copy.deepcopy(original['corpus'])
    plan.update(repeats=repeats,execution='eager',qualification_trace=True)
    if workload=='ample':
        plan.update(concurrency=4,gpu_blocks=2560,cpu_bytes=3758096384)
    if workload in ('tight','live-documents'):
        plan.update(concurrency=8,gpu_blocks=1280,cpu_bytes=1409286144)
    if workload=='mixed':
        for i,s in enumerate(corpus['sessions']): s['publish_hints']=i%2==0
    if workload=='uneven':
        plan['replay_wait_origin']='authored_uneven_wait_diagnostic'
        for i,s in enumerate(corpus['sessions']):
            for turn,r in enumerate(s['replay']):
                if turn: r['delay_s']=(.05,.5,2.,5.)[(i+turn)%4]
    if workload=='live-repository':
        # Remove the historical mixed-advice assignment if supplied by a source.
        for s in corpus['sessions']:s.pop('publish_hints',None)
    plan.update(overrides or {})
    corpus['provenance']='Stage 5 development reuse: '+str(corpus.get('provenance',plan.get('corpus_provenance','')))+'; prior evaluation status does not carry over.'
    if workload=='uneven':
        corpus['provenance'] += ' Tool delays replaced by declared 0.05/0.5/2/5-second pattern; these are authored diagnostic waits.'
    plan.update(corpus_sha256=digest(corpus),corpus_provenance=corpus['provenance'])
    conditions={}
    for name in names:
        opts=copy.deepcopy(VARIANTS[name]);config=copy.deepcopy(plan['candidate_config'])
        extra=config['kv_connector_extra_config'];extra['cpu_bytes_to_use']=plan['cpu_bytes']
        extra.update(opts.pop('extra',{}))
        if 'advice' in opts:extra['wait_hint_arc_advice']=opts.pop('advice')
        conditions[name]=dict(publish_hints=True,required_activity=['store'],candidate_config=config)
        conditions[name].update(opts)
    # Store-only acceptance allows legitimate zero restore/advice in neutral cells.
    # Required mechanism activity is separately inspected before any policy verdict.
    order=balanced_order(names,repeats,seed)
    files=dict(original.get('input_files',{})) if workload.startswith('live') else {}
    for name in list(files):
        path=Path(name).resolve()
        if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'output'):
            files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (source,Path(__file__)):
        files[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    spec = dict(plan=plan,corpus=corpus,conditions=conditions,order=order,input_files=files,
                stage5=dict(workload=workload,seed=seed,scope='development only',variants=names))
    preflight(spec)
    return spec


def preflight(spec):
    """Reject unsupported client settings before loading the model."""
    from semantic_offload.request_priority import gateway_priority
    from semantic_offload.wait_hint import KnownPrefixWaitHint, WAIT_HINT_SCHEMA, WAIT_HINT_PROVENANCE
    for condition in spec['conditions'].values():
        plan = {**spec['plan'], **condition}
        gateway_priority(0, 0, 0, plan.get('continuation_credit_ms', 0))
        KnownPrefixWaitHint.from_mapping(dict(schema=WAIT_HINT_SCHEMA,event_id='0'*64,
            observed_prefix_id='1'*64,owner_id='preflight',run_id='suite',cell_id='case',
            session_id='session',generation=0,prefix_boundary_tokens=16,
            observation_available_ns=0,input_publication_ns=0,
            declared_wait_ms=plan.get('hint_declared_wait_ms',750),wait_provenance=WAIT_HINT_PROVENANCE,
            namespace={k:'preflight' for k in ('model_revision','tokenizer_revision','adapter_namespace','layout_namespace')}))
        extra = plan.get('candidate_config',{}).get('kv_connector_extra_config',{})
        if extra.get('offload_prompt_only') and extra.get('wait_hint_completed_transcript'):
            raise ValueError('Prompt-only storage cannot publish completed transcripts')
        if extra.get('offload_prompt_only') and plan.get('publish_hints') and plan.get('hint_completed_transcript',True):
            raise ValueError('Prompt-only advice requires prompt-scoped client metadata')


@gpu_reserved
def run(spec_paths, output):
    specs = [json.loads(path.read_text()) for path in spec_paths]
    for spec in specs:
        preflight(spec)
    output.mkdir(parents=True,exist_ok=False)
    for index,spec in enumerate(specs):
        target=output/f'{index:02d}-{spec["stage5"]["workload"]}'
        result=execute.__wrapped__(spec,target)
        write_json(output/f'{target.name}-verification.json',offline(target))
        if result['status']=='failed':
            raise RuntimeError('Retained failed tranche: '+str(target))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    prep=s.add_parser('prepare');prep.add_argument('--workload',required=True);prep.add_argument('--variants',required=True);prep.add_argument('--repeats',type=int,default=1);prep.add_argument('--seed',type=int,default=51);prep.add_argument('--overrides',default='{}');prep.add_argument('--output',type=Path,required=True)
    runp=s.add_parser('run');runp.add_argument('--spec',type=Path,nargs='+',required=True);runp.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':
        if a.output.exists():raise FileExistsError(a.output)
        write_json(a.output,prepare(a.workload,a.variants.split(','),a.repeats,a.seed,json.loads(a.overrides)))
    else:run(a.spec,a.output)

if __name__=='__main__':main()
