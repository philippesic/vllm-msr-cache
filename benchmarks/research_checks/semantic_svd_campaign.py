"""Bounded factorial live experiments using maintained serving-suite run_case.

Holds shared GPU reservation across all cases, preserves each exact plan, source
identity, model identity, raw serving artifacts, and independently sampled costs.
No engine replacement or run-local code injection.
"""
import hashlib
import argparse
import copy
import json
import signal
from pathlib import Path
from harness.gpu_lock import gpu_reserved, GPULock, LEGACY_LOCK
from benchmarks.serving_suite import runner
from benchmarks.serving_suite.resources import available_gpu
from benchmarks.serving_suite.contracts import write_json


@gpu_reserved
def execute(a):
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'hardware.json', available_gpu())
    identity = runner.source_identity()
    write_json(root/'source.json', identity)
    write_json(root/'model.json', {p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in a.model.iterdir() if p.is_file()})
    corpus = json.loads(a.corpus.read_text())
    write_json(root/'corpus.json', corpus)
    base = json.loads((Path(__file__).with_name('fixtures') / 'semantic_svd_control.json').read_text())
    e = base['kv_connector_extra_config']
    e.update(cpu_bytes_to_use=a.cpu_bytes, preservation_audit=True,
             wait_hint_cpu_retention_gpu_gate=True)
    e['admission_namespace'].update(model_revision=a.model.name, tokenizer_revision=a.model.name)
    arms = a.arms.split(',')
    cases = []
    for repeat in range(a.repeats):
        order = arms if repeat % 2 == 0 else list(reversed(arms))
        for arm in order:
            if runner.source_identity() != identity:
                raise RuntimeError('Executed source changed during campaign')
            cfg = copy.deepcopy(base)
            x = cfg['kv_connector_extra_config']
            if arm == 'arc':
                x.update(prefetch_budget_fraction=0, wait_hint_cpu_retention_only=False, wait_hint_cpu_retention_gpu_gate=False)
            elif arm != 'wait':
                x.update(instrumentation_only=False, wait_hint_cpu_retention_only=False,
                         wait_hint_cpu_retention_gpu_gate=False,
                         wait_hint_diagnostic_no_dispatch=False, wait_hint_ttl_ms=a.ttl_ms,
                         prefetch_selector='semantic', prefetch_budget_fraction=a.prefetch_budget if 'prefetch' in arm else 0,
                         capture_stride=a.stride, kv_svd_rank=a.rank if 'svd' in arm else 0,
                         kv_svd_expand_capacity=a.expand, prefetch_require_full_prefix=not a.partial_prefetch)
            out = root/f'{repeat:02d}-{arm}'
            out.mkdir()
            if 'svd' in arm and not a.no_codec_trace:
                x['kv_svd_trace'] = str(out/'codec.jsonl')
            write_json(out/'config.json',cfg)
            args = ['screen','--model',str(a.model),'--output',str(out/'plan'),
                    '--candidate-config',str(out/'config.json'),'--corpus',str(a.corpus),
                    '--sessions',str(a.sessions),'--turns','3','--concurrency',str(a.concurrency),
                    '--max-tokens',str(a.max_tokens),'--gpu-blocks',str(a.gpu_blocks),
                    '--cpu-bytes',str(a.cpu_bytes),'--max-model-len',str(a.context),
                    '--batch-tokens','1024','--publish-hints','--require-activity','store,restore', '--plan']
            # ARC has no hint namespace and never publishes advice.
            if arm == 'arc': args.remove('--publish-hints')
            if a.trace: args.append('--qualification-trace')
            runner.main(args)
            plan = json.loads((out/'plan/plan.json').read_text())
            plan['hint_delay_s'] = a.hint_delay_s
            write_json(out/'plan/plan.json', plan)
            selected = runner.select_sessions(corpus, plan, 0)
            print(f'RUN {repeat} {arm}',flush=True)
            with GPULock(LEGACY_LOCK):
                case = runner.run_case(plan, selected, 'candidate', repeat, out)
            case['variant'] = arm
            cases.append(case)
            write_json(root/'results.json',cases)
            print(json.dumps({'variant':arm,'summary':case['summary'],'activity':case.get('activity'),
                              'error':case.get('error'),'evaluation':case['evaluation']}),flush=True)
            available_gpu()
            if not case['cleanup']['verified']:
                raise RuntimeError('Cleanup unverified')
    write_json(root/'final-source.json',runner.source_identity())


def main():
    p=argparse.ArgumentParser()
    for name in ('model','corpus','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prefetch-budget',type=float,default=0.6)
    p.add_argument('--ttl-ms',type=int,default=5000)
    p.add_argument('--no-codec-trace',action='store_true')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--hint-delay-s', type=float, default=0.0)
    p.add_argument('--partial-prefetch', action='store_true')
    p.add_argument('--expand', action='store_true')
    p.add_argument('--arms',default='arc,wait,base,base-svd,base-prefetch,base-svd-prefetch')
    for name, default in [('rank',8),('stride',1),('repeats',1),('sessions',6),('concurrency',2),
                          ('max-tokens',48),('gpu-blocks',128),('cpu-bytes',67108864),('context',2048)]:
        p.add_argument('--'+name,type=int,default=default)
    a=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    execute(a)

if __name__=='__main__': main()
