"""Inventory measured compatibility cells without implying an untested cross-product."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.serving_suite.contracts import write_json


FUNCTIONAL = (
    'runtime-probe', 'runtime-probe-qwen15', 'runtime-probe-opt125',
    'runtime-probe-minimal', 'runtime-probe-abandon', 'runtime-probe-long',
    'runtime-probe-qwen15-long', 'runtime-probe-graphs',
)
PERFORMANCE = (
    'stage4-graphs', 'stage4-qwen15-performance', 'stage4-qwen15-cohost',
    'stage4-minimal-performance',
)
EXAMPLES = ('example-client', 'isolated-example-client')


def report(directory):
    directory = Path(directory)
    replacement_path = directory/'compatibility-replacements.json'
    replacements = json.loads(replacement_path.read_text()) if replacement_path.exists() else {}
    rows = []
    for name in (*FUNCTIONAL, *PERFORMANCE, *EXAMPLES):
        selected = replacements.get('replacements', {}).get(name, name)
        chain = replacements.get('attempts', {}).get(name,
            [name, selected] if selected != name else [name])
        attempts = []
        for attempt in chain:
            receipt = directory/attempt/'result.json'
            prior = json.loads(receipt.read_text()) if receipt.exists() else None
            attempts.append(dict(attempt=attempt,
                execution=(prior.get('status') or ('passed' if prior.get('passed') else 'failed')) if prior else 'pending',
                error=prior.get('error') if prior else None,
                result=str(receipt.resolve()) if prior else None))
        path = directory / selected / 'result.json'
        kind = 'functional' if name in FUNCTIONAL else 'performance' if name in PERFORMANCE else 'author integration exercise'
        row = dict(cell=name, evaluated_attempt=selected, attempts=attempts,
                   kind=kind, execution='pending')
        if path.exists():
            result = json.loads(path.read_text())
            row.update(execution=result.get('status') or ('passed' if result.get('passed') else 'failed'),
                       error=result.get('error'), result=str(path.resolve()))
            if 'cases' in result:
                plan = result['plan']
                row.update(model=plan['model'], execution_mode=plan['execution'],
                    requested_gpu_blocks=plan['gpu_blocks'], requested_cpu_bytes=plan['cpu_bytes'],
                    cases=len(result['cases']), completed_requests=sum(c['summary']['completed'] for c in result['cases']),
                    failed_requests=sum(c['summary']['failed'] for c in result['cases']),
                    operational=[dict(case=c['name'], checks=c.get('operational'),
                        cleanup=c['cleanup'], evaluation=c['evaluation']) for c in result['cases']])
            if kind == 'functional':
                checks = [dict(case=c['name'], wave=w['wave'], **check)
                    for c in result['cases']
                    for p in sorted((directory/selected/c['name']).glob('runtime-probe-*.json'))
                    for w in [json.loads(p.read_text())] for check in w['checks']]
                row['checks'] = checks
                row['all_observed_functional_checks_passed'] = bool(checks) and all(c['passed'] for c in checks)
                decision = directory/(selected+'-functional-decision.json')
                row['complete_functional_decision'] = json.loads(decision.read_text()) if decision.exists() else None
                row['scope'] = 'Observed generation and protocol checks only; consult expected-wave coverage, native restore/fresh-computation receipts and operational cleanup. Not a performance or KV-byte-equality claim.'
            elif kind == 'performance':
                summary_path = directory/(selected+'-summary.json')
                row['comparison'] = json.loads(summary_path.read_text())['comparisons'].get('wait_vs_arc') if summary_path.exists() else None
                row['scope'] = 'Paired whole-workflow measurements; useful-answer quality and functional correctness require separate evidence. No automatic product qualification.'
            else:
                row.update(client=result.get('client'), cleanup=result.get('cleanup'), scope=result.get('scope'))
            if 'graphs' in name:
                receipt = directory/(selected+'-graph-receipts.json')
                row['graph_path_receipt'] = json.loads(receipt.read_text()) if receipt.exists() else None
            allocation = directory/(selected+'-allocation-receipts.json')
            row['allocation_path_receipt'] = json.loads(allocation.read_text()) if allocation.exists() else None
        rows.append(row)
    return dict(schema='wait-hint-measured-matrix-v1', cells=rows,
        untested=['other GPU', 'native Linux host', 'BF16 on supported hardware',
                  'distributed/tensor-parallel serving', 'hybrid/sliding-window/multiple KV groups'],
        scope='Inventory of selected actual cells, not their cross-product or an automatic support decision. Newer-runtime API compatibility, isolated installation and artifact verification have separate receipts. Missing and failed attempts remain visible.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, report(args.directory))


if __name__ == '__main__':
    main()
