"""Collect retained live-serving attempts without hiding failed operating points.

Run after GPU measurements: python -m benchmarks.research_checks.wait_opportunity_report
CAMPAIGN --output compact-results.json. Source runs are read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def returning_transcript_floor(rows, block_size=16):
    """New prompt work after the longest exact, previously completed prefix.

    The last predicted output token has not itself run through the model. Round
    its preceding known prefix down to a reusable full block, as the producer does.
    This is a corpus-specific bound, not a measurement of physical GPU execution.
    """
    by_key = {(r['session_id'], r['turn']): r for r in rows}
    total = 0
    for row in rows:
        if not row['turn']:
            continue
        previous = by_key.get((row['session_id'], row['turn']-1))
        if previous is None or not previous.get('success') or not row.get('success'):
            return None
        known = previous['payload']['prompt'] + previous['output_token_ids'][:-1]
        length = 0
        for left, right in zip(known, row['payload']['prompt']):
            if left != right:
                break
            length += 1
        total += len(row['payload']['prompt']) - length // block_size * block_size
    return total


def case_metrics(directory, case):
    a=case.get('agent',{});s=case.get('summary',{});cache=a.get('cache',{})
    t=case.get('telemetry',{});h=a.get('hints',{});q=a.get('quality',{})
    samples=[]
    path=directory/'resources.jsonl'
    if path.exists():
        samples=[json.loads(line) for line in path.read_text().splitlines() if line]
    def difference(key):
        values=[r[key] for r in samples if r.get(key) is not None]
        return values[-1]-values[0] if values else None
    rows=[]
    path=directory/'requests.jsonl'
    if path.exists():
        rows=[json.loads(line) for line in path.read_text().splitlines() if line]
    complete=s.get('completed')==case.get('expected_requests') and not s.get('failed')
    floor=returning_transcript_floor(rows) if complete and rows else None
    actual=cache.get('returning_locally_executed_prefill_tokens')
    sources=cache.get('first_admission_sources',{})
    return dict(case=case['name'],variant=case.get('variant',case['arm']),
        repeat=case['repeat'],status=case['evaluation']['status'],
        errors=case['evaluation'].get('errors',[]),completed=s.get('completed'),
        duration_s=case.get('load_duration_s'),requests_per_s=s.get('completed_requests_per_s'),
        workflows_per_s=a.get('sessions',{}).get('per_s'),output_tokens=s.get('output_tokens'),
        output_tokens_per_s=s.get('output_tokens_per_s'),offered_ftl_mean_s=s.get('end_user_ttft_mean_s'),
        offered_ftl_p95_s=s.get('end_user_ttft_p95_s'),
        return_ftl_mean_s=a.get('latency',{}).get('return_ttft_s',{}).get('mean'),
        tool_ready_ftl_mean_s=a.get('latency',{}).get('tool_to_first_token_s',{}).get('mean'),
        server_queue_mean_s=a.get('latency',{}).get('server_queue_s',{}).get('mean'),
        first_computed_tokens=sources.get('num_computed_tokens'),
        first_cpu_restored_tokens=sources.get('num_external_cached_tokens'),
        cumulative_original_prompt_compute=cache.get('locally_executed_prefill_tokens'),
        returning_transcript_floor_tokens=floor,
        returning_compute_above_floor=actual-floor if actual is not None and floor is not None else None,
        physical_load_bytes=cache.get('physical_demand_load_bytes'),preemptions=cache.get('preemptions'),
        cpu_eviction_affected_returns=a.get('opportunity',{}).get('returning_requests_with_prior_cpu_eviction'),
        ready_cpu_key_refreshes=h.get('ready_cpu_keys_refreshed'),gpu_gate_skips=h.get('gpu_gate_skipped_calls'),
        hint_rtt_mean_s=h.get('rtt_s',{}).get('mean'),hint_errors=h.get('errors'),
        score_update_batches=case.get('activity',{}).get('scoring'),
        prefetch_jobs=a.get('prefetch',{}).get('submitted_jobs'),
        gpu_peak_bytes=t.get('gpu_peak_bytes'),server_rss_peak_bytes=t.get('host_rss_peak_bytes'),
        client_tool_rss_peak_bytes=t.get('client_tool_rss_peak_bytes'),
        server_cpu_s=t.get('process_cpu_elapsed_s'),client_tool_cpu_s=t.get('client_tool_cpu_elapsed_s'),
        gpu_util_mean_pct=a.get('telemetry',{}).get('gpu_utilization_pct',{}).get('mean'),
        gpu_energy_j=a.get('telemetry',{}).get('sampled_gpu_energy_j'),
        tool_execution_total_s=a.get('tools',{}).get('total_wall_s'),
        tool_queue_mean_s=a.get('tools',{}).get('queue_s',{}).get('mean'),
        host_swap_in_bytes=difference('host_swap_in_bytes'),host_swap_out_bytes=difference('host_swap_out_bytes'),
        test_status_correct=q.get('passed'),test_status_checks=q.get('checks'),
        strict_format_failures=q.get('strict_format_failures'),
        contradictory_test_claims=q.get('contradictory_claims'),
        unparseable_test_status=q.get('unparseable_status'))


def collect(campaign):
    runs=[]
    for path in sorted(Path(campaign).glob('*/result.json')):
        result=json.loads(path.read_text());plan=result['plan']
        cases=[case_metrics(path.parent/c['name'],c) for c in result['cases']]
        runs.append(dict(run=path.parent.name,artifact=str(path),status=result['status'],
            conditions={k:plan.get(k) for k in ('sessions','concurrency','client_concurrency','tool_concurrency',
                'gpu_blocks','cpu_bytes','continuation_credit_ms','hint_timing','hint_through_queue',
                'compact_events','stable_tool_output','prompt_source_accounting_version')},
            cases=cases,comparisons=result.get('comparisons') or {'candidate_vs_arc':result.get('comparison')}))
    return dict(schema='wait-opportunity-comparison-v1',runs=runs,
        scope='Separate operating points; never pool selecting screens, failures or different conditions into confirmation intervals.',
        floor_scope='Exact completed-prefix transcript bound. Above-floor work includes cache misses and any preemption recomputation; it is not full physical GPU FLOPs.',
        quality_scope='Reported test outcomes only; broader reasoning accuracy ungraded.')


def aggregate(cases):
    """Summarize complete cases only; callers must declare the chosen campaign."""
    if not cases or any(c['status']!='passed' for c in cases):
        raise ValueError('Aggregate requires complete, passed cases')
    average_keys=['duration_s','offered_ftl_mean_s','offered_ftl_p95_s','return_ftl_mean_s',
        'tool_ready_ftl_mean_s','first_computed_tokens','first_cpu_restored_tokens',
        'returning_compute_above_floor','physical_load_bytes','gpu_peak_bytes',
        'server_rss_peak_bytes','server_cpu_s','client_tool_cpu_s','gpu_util_mean_pct','gpu_energy_j']
    result={key:statistics.mean(c[key] for c in cases) if all(c[key] is not None for c in cases) else None
            for key in average_keys}
    result.update(cases=len(cases),requests=sum(c['completed'] for c in cases),
        output_tokens=sum(c['output_tokens'] for c in cases),
        output_tokens_per_s=sum(c['output_tokens'] for c in cases)/sum(c['duration_s'] for c in cases),
        workflows_per_s=statistics.mean(c['workflows_per_s'] for c in cases),
        test_status_correct=sum(c['test_status_correct'] for c in cases),
        test_status_checks=sum(c['test_status_checks'] for c in cases),
        preemptions=sum(c['preemptions'] for c in cases),
        score_update_batches=sum(c['score_update_batches'] for c in cases),
        ready_cpu_key_refreshes=sum(c['ready_cpu_key_refreshes'] for c in cases))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=collect(args.campaign)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    rows=[dict(run=run['run'],**c) for run in result['runs'] for c in run['cases']]
    if rows:
        with args.output.with_suffix('.csv').open('w') as output:
            writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(runs=len(result['runs']),cases=len(rows),output=str(args.output))))


if __name__=='__main__':
    main()
