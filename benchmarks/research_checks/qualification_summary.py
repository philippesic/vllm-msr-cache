"""Compact, failure-aware summaries of retained wait-hint qualification artifacts."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from benchmarks.serving_suite.contracts import write_json
from benchmarks.serving_suite.report import percentile


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def summarize(directory):
    directory=Path(directory);result=json.loads((directory/'result.json').read_text())
    summary=dict(schema='wait-hint-qualification-summary-v1',measurement_status=result['status'],
        error=result.get('error'),diagnostic_replay=bool(result['plan'].get('diagnostic_replay')),
        cases=[],comparisons={},scope='Conditional on frozen workloads/configuration. A measurement pass is not automatically a policy win or product qualification.')
    for case in result['cases']:
        agent=case.get('agent',{});resources=jsonl(directory/case['name']/'resources.jsonl') if (directory/case['name']/'resources.jsonl').exists() else []
        rows=jsonl(directory/case['name']/'requests.jsonl')
        def delta(key):
            return resources[-1][key]-resources[0][key] if resources and all(key in r for r in (resources[0],resources[-1])) else None
        cache=agent.get('cache',{})
        summary['cases'].append(dict(name=case['name'],condition=case['variant'],repeat=case['repeat'],
            evaluation=case['evaluation'],error=case.get('error'),batch_s=case['load_duration_s'],
            expected_requests=case['expected_requests'],completed=case['summary']['completed'],failed=case['summary']['failed'],
            output_tokens=case['summary']['output_tokens'],finish_reasons=case['summary']['finish_reasons'],
            offered_ftl_mean_s=case['summary']['end_user_ttft_mean_s'],
            return_ftl=agent.get('latency',{}).get('tool_to_first_token_s'),
            first_admission_sources=cache.get('first_admission_sources'),
            native_scheduled_prefill_tokens=cache.get('locally_executed_prefill_tokens'),
            terminal_episode_loaded_tokens=cache.get('loaded_tokens'),
            authoritative_native_receipts=cache.get('authoritative_receipts'),
            accounting_scope=cache.get('accounting_scope'),
            demand_load_bytes=cache.get('physical_demand_load_bytes'),preemptions=cache.get('preemptions'),
            load_attribution_gap_bytes=cache.get('load_attribution_gap_bytes'),
            transfer_attribution_scope=cache.get('transfer_attribution_scope'),
            hint_publications=agent.get('hints',{}).get('publications'),
            ready_cpu_keys_refreshed=agent.get('hints',{}).get('ready_cpu_keys_refreshed'),
            native_cache_competition=agent.get('opportunity',{}).get('cache_competition_observed'),
            quality=agent.get('quality'),
            server_cpu_s=case.get('telemetry',{}).get('process_cpu_elapsed_s'),
            client_tool_cpu_s=case.get('telemetry',{}).get('client_tool_cpu_elapsed_s'),
            swap_in_bytes=delta('host_swap_in_bytes'),swap_out_bytes=delta('host_swap_out_bytes'),
            gpu_temperature_range_c=[min(r['gpu_temperature_c'] for r in resources),max(r['gpu_temperature_c'] for r in resources)] if resources and all('gpu_temperature_c' in r for r in resources) else None,
            groups={group:dict(requests=len(selected),failed=sum(not r['success'] for r in selected),
                offered_ftl_mean_s=statistics.mean(r['ttft_s']+r.get('client_queue_s',0) for r in selected if r.get('ttft_s') is not None),
                return_ftl_p95_s=percentile([(r['first_token_ns']-r['tool_completed_ns'])/1e9 for r in selected if r.get('tool_completed_ns') and r.get('first_token_ns')],.95))
                for group in sorted({r['hint_group'] for r in rows if 'hint_group' in r})
                for selected in [[r for r in rows if r.get('hint_group')==group]]}))
    for name,comparison in result.get('qualification_comparisons',{}).items():
        ratio=comparison['geometric_workflow_throughput_ratio'];bounds=comparison['workflow_throughput_ratio_95ci']
        summary['comparisons'][name]=dict(paired_runs=comparison['eligible_pairs'],
            candidate_batch_time_reduction_pct=100*(1-1/ratio) if ratio else None,
            candidate_batch_time_reduction_95ci_pct=[100*(1-1/x) for x in bounds] if bounds else None,
            workflow_rate_ratio=ratio,workflow_rate_ratio_95ci=bounds,
            end_user_ftl_delta_s=comparison['end_user_ttft_mean_delta_s'],
            identical_prompt_and_output_ids=all(p.get('same_prompt_tokens') and p.get('same_output_tokens') for p in comparison['pairs']) if comparison['pairs'] else None,
            all_pairs_eligible=all(p.get('eligible') for p in comparison['pairs']),
            interpretation='Positive time reduction favors the condition before _vs_; negative favors the condition after _vs_.')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();write_json(args.output,summarize(args.directory))

if __name__=='__main__':main()
