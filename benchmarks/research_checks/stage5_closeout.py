"""Reconstruct the bounded Stage 5 closeout recipes; run with stage5_optimization.

These preserve development provenance and do not turn old workloads into fresh
confirmation. The initial profile specs predate this convenience generator.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from benchmarks.research_checks.stage5_optimization import prepare, balanced_order, preflight
from benchmarks.research_checks.stage5_followups import add
from benchmarks.serving_suite.contracts import write_json

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('primary-profile', 'tight-profile', 'long-batch-profile', 'prompt-storage',
          'combination', 'audit-control', 'functional')


def make(label):
    if label in ('primary-profile', 'tight-profile', 'long-batch-profile'):
        workload = {'primary-profile':'repository', 'tight-profile':'tight',
                    'long-batch-profile':'live-documents'}[label]
        long = label == 'long-batch-profile'
        spec = prepare(workload, ['arc','linear-arc'] if long else ['arc','wait','linear-wait'],
                       1, 818 if long else 815,
                       {'arc_eviction_profile':True, **({'batch_tokens':4096} if long else {})})
        spec['stage5']['question'] = ('Actual eviction body timing and exact cache-state replay. '
            'Long-batch cell uses real documents and larger supported prefill batches to chase CPU lead; '
            'profiled/natural-output times do not qualify a speed promotion.')
    elif label == 'prompt-storage':
        spec = prepare('tight',['arc','wait'],1,708)
        add(spec,'arc-prompt-only',publish_hints=False,extra={'offload_prompt_only':True,
            'wait_hint_completed_transcript':False,'wait_hint_cpu_retention_only':False,
            'prefetch_budget_fraction':0,'wait_hint_cpu_retention_gpu_gate':False})
        add(spec,'wait-prompt-scope',hint_completed_transcript=False,extra={'wait_hint_completed_transcript':False})
        add(spec,'wait-prompt-only',hint_completed_transcript=False,
            extra={'offload_prompt_only':True,'wait_hint_completed_transcript':False})
        spec['order'] = balanced_order(list(spec['conditions']),1,708)
        spec['stage5']['question'] = 'Storage ablation with matching prompt-scope/full-store control.'
    elif label == 'combination':
        spec = prepare('tight',['arc','wait'],1,916)
        add(spec,'combined',batch_tokens=512,continuation_credit_ms=0,extra={'arc_linear_eviction':True})
        spec['order'] = balanced_order(list(spec['conditions']),1,916)
        spec['stage5']['question'] = 'Bounded interaction screen; no pooling earlier isolated screens.'
    elif label == 'audit-control':
        spec = prepare('repository',['arc','wait'],1,917,{'tier':'operational','repository_workload':True})
        for c in spec['conditions'].values():
            c['candidate_config']['kv_connector_extra_config']['preservation_audit'] = False
        spec['stage5']['question'] = 'Matched audit-off ARC/wait; one workload wave plus cancellation/drain/recovery, not full operational qualification.'
    elif label == 'functional':
        source = ROOT/'output/wait-qualification-20261105/runtime-probe-long/spec.json'
        spec = json.loads(source.read_text())
        spec['conditions']['linear-grace-wait'] = copy.deepcopy(spec['conditions']['wait'])
        c = spec['conditions']['linear-grace-wait']
        c['candidate_config'] = copy.deepcopy(spec['plan']['candidate_config'])
        c['candidate_config']['kv_connector_extra_config'].update(arc_linear_eviction=True,wait_hint_cpu_nomination_grace_ms=750)
        spec['order'] = [[0,k] for k in spec['conditions']]
        spec['stage5'] = dict(workload='functional-probe',scope='Three unforced waves, actual restore versus fresh recomputation; functional only')
        spec['input_files'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        raise ValueError(label)
    spec['stage5']['variants'] = list(spec['conditions'])
    # Old external data stays pinned. Refresh authored local executed sources.
    for name in spec['input_files']:
        p = Path(name).resolve()
        if p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'output'):
            spec['input_files'][name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (Path(__file__), Path(__file__).with_name('stage5_optimization.py'),
              Path(__file__).with_name('stage5_followups.py'),
              Path(__file__).with_name('arc_serving_profile.py')):
        spec['input_files'][str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    preflight(spec)
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', nargs='+', choices=LABELS, default=list(LABELS))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for label in args.labels:
        write_json(args.output/(label+'-spec.json'),make(label))


if __name__ == '__main__':
    main()
