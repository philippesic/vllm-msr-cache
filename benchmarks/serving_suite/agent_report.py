"""Recomputable diagnostics for real repository-tool serving workloads."""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from .contracts import write_json
from .report import percentile


def wait_hint_config(plan, enabled):
    path = Path(__file__).parents[1]/'research_checks/fixtures/semantic_svd_control.json'
    cfg = json.loads(path.read_text())
    e = cfg['kv_connector_extra_config']
    # This runtime checks the nonzero enable budget before the CPU-only branch.
    # CPU-only + diagnostic_no_dispatch prevent any speculative GPU allocation.
    e.update(cpu_bytes_to_use=plan['cpu_bytes'], prefetch_budget_fraction=.6 if enabled else 0,
             instrumentation_only=True, preservation_audit=True, benchmark_telemetry=True,
             wait_hint_cpu_retention_only=enabled, wait_hint_cpu_retention_gpu_gate=enabled)
    e['admission_namespace'].update(model_revision=Path(plan['model']).name, tokenizer_revision=Path(plan['model']).name)
    return cfg


def distribution(values):
    values = [v for v in values if v is not None]
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                p50=percentile(values, .5), p95=percentile(values,.95), max=max(values) if values else None)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def native_events(text, marker):
    return [json.JSONDecoder().raw_decode(part)[0] for part in text.split(marker)[1:]]


def diagnose(directory, case, plan, write=True):
    directory = Path(directory)
    rows = jsonl(directory/'requests.jsonl')
    events = [r for p in sorted(directory.glob('workflow-*.jsonl')) for r in jsonl(p)]
    tools = [r for r in events if r['event']=='tool']
    windows = [r for r in events if r['event']=='hint_window']
    log = (directory/'server.log').read_text(errors='replace')
    preservation = native_events(log, 'SEMANTIC_PRESERVATION_EVENT ')
    matched = [e for e in preservation if e.get('external_request_id') in {r['request_id'] for r in rows}]
    by_id = {e['external_request_id']:e for e in matched}
    hints = case['hints']
    scheduler = jsonl(directory/'kv-qualification.jsonl') if (directory/'kv-qualification.jsonl').exists() else []
    steps = [r for r in scheduler if r['event']=='step']
    samples = jsonl(directory/'resources.jsonl')
    errors = []
    if len(by_id) != len(rows) or len(matched) != len(rows):
        errors.append('missing_or_duplicate_native_request_receipts')
    if len(tools) != len(rows)-len(case['source_ids'])*plan['waves']:
        errors.append('incomplete_real_tool_coverage')
    if any(not t['success'] for t in tools):
        errors.append('tool_execution_failure')
    if not steps:
        errors.append('no_native_pressure_samples')
    if not samples or any('gauges' not in r for r in samples):
        errors.append('missing_timeseries_metrics')
    starts = {r['session_id']:r['ns'] for r in events if r['event']=='session_start'}
    ends = {r['session_id']:r['ns'] for r in events if r['event']=='session_end'}
    sessions = [(ends[s]-starts[s])/1e9 for s in ends]
    if len(ends) != len(case['source_ids'])*plan['waves']:
        errors.append('incomplete_sessions')
    lead = []
    timely_windows = 0
    post_tool_hints = 0
    for w in windows:
        valid = [r for r in w['receipts'] if not r.get('error') and r.get('response') and not r.get('cancelled_before_ack')]
        available = [(w['tool_ended_ns']-r['ended_ns'])/1e9 for r in valid]
        lead += available
        timely_windows += any(v>0 for v in available)
        post_tool_hints += sum(v<0 for v in available)
    comparisons = []
    for row in rows:
        e = by_id.get(row['request_id'], {})
        comparisons.append(dict(request_id=row['request_id'], source_id=row['source_id'], turn=row['turn'],
            prompt_tokens=len(row['payload']['prompt']), output_tokens=len(row.get('output_token_ids',[])),
            success=row['success'], finish_reason=row.get('finish_reason'), ttft_s=row.get('ttft_s'),
            client_queue_s=row.get('client_queue_s'), latency_s=row.get('latency_s'),
            tool_to_first_token_s=(row['first_token_ns']-row['tool_completed_ns'])/1e9 if row.get('first_token_ns') and row.get('tool_completed_ns') else None,
            locally_executed_prefill_tokens=e.get('actual_prompt_tokens_locally_executed'),
            loaded_tokens=e.get('loaded_tokens'), reused_tokens=e.get('reused_tokens'),
            required_tokens=e.get('required_tokens'), recomputed_tokens=e.get('recomputed_tokens'),
            prior_cpu_evicted_keys=len(e['prior_evicted_keys']) if 'prior_evicted_keys' in e else None,
            load_worker_seconds=e.get('load_worker_seconds'), store_worker_seconds=e.get('store_worker_seconds'),
            server_queue_s=e.get('server_timing',{}).get('queue_delay_ns',0)/1e9 if e.get('server_timing',{}).get('ordering_valid') else None,
            load_critical_path_s=e.get('load_critical_path_s'),
            native_outcome=e.get('outcome'), native_authoritative=e.get('authoritative')))
    # Only checks a concrete claim about observed tests. This is not code-review accuracy.
    test_by_session = {t['session_id']:t for t in tools if t['kind']=='test'}
    quality = []
    legacy_quality = 'agent' in case and case['agent'].get('quality',{}).get('scorer_version') is None
    for row in rows:
        if row['turn'] != 3:
            continue
        text = row.get('generated_text','')
        claims = re.findall(r'test_status\s*=\s*(PASS|FAIL)', text, re.I) if legacy_quality else re.findall(r'test[_ ]status\s*[:=]\s*(PASS|FAIL)', text.replace('*',''), re.I)
        test = test_by_session.get(row['session_id'])
        expected = 'PASS' if test and test.get('exit_code')==0 else 'FAIL'
        checked = dict(source_id=row['source_id'], test_status_claims=claims,
                       observed_test_status=expected, passed=len(claims)==1 and claims[0].upper()==expected if legacy_quality else bool(claims) and all(c.upper()==expected for c in claims))
        if not legacy_quality:
            checked.update(strict_format_satisfied=len(re.findall(r'test_status\s*=\s*(PASS|FAIL)',text,re.I))==1,
                           contradictory_claim=any(c.upper()!=expected for c in claims), status_unparseable=not claims)
        quality.append(checked)
    if len(quality) != len(ends):
        errors.append('missing_final_quality_observations')
    peak = lambda key: max((r[key] for r in steps), default=None)
    evicted = sum(len(e.get('prior_evicted_keys',[])) for e in matched)
    returning_evicted = sum((r['prior_cpu_evicted_keys'] or 0)>0 for r in comparisons if r['turn']>0)
    refreshed = [h for h in hints if h.get('response',{}).get('reason')=='cpu_prefix_recency_refreshed']
    useful_refreshed = {h['hint']['session_id'] for h in refreshed if h['response'].get('ready_cpu_prefix_keys',0)>0}
    sessions_by_request = {r['request_id']:r['session_id'] for r in rows}
    returned_refreshed = [r for r in comparisons if r['turn']>0 and sessions_by_request[r['request_id']] in useful_refreshed]
    wait_events = native_events(log, 'SEMANTIC_WAIT_HINT_EVENT ')
    def total(field, selected=comparisons):
        values = [r.get(field) for r in selected]
        return sum(values) if len(by_id)==len(rows) and values and all(v is not None for v in values) else None
    gauges = {}
    for key in sorted({k for r in samples for k in r.get('gauges',{})}):
        gauges[key] = distribution([r.get('gauges',{}).get(key) for r in samples])
    duration = case['load_duration_s']
    result = dict(schema='real-agent-diagnostics-v1', measurement_errors=errors,
        scope='Live application-managed repository search/test/evidence workflow. Authored tasks, real source/tools/model/history. No sleeps for tool latency, replayed outputs, or model-selected shell commands.',
        sessions=dict(completed=len(ends), per_s=len(ends)/duration if duration else None, duration_s=distribution(sessions)),
        tools=dict(executed=len(tools), failed=sum(not t['success'] for t in tools),
            by_kind={kind:distribution([t['duration_s'] for t in tools if t['kind']==kind]) for kind in sorted({t['kind'] for t in tools})},
            total_wall_s=sum(t['duration_s'] for t in tools), total_output_bytes=sum(t['output_bytes'] for t in tools),
            excerpts_truncated=sum(t['prompt_output_truncated'] for t in tools)),
        hints=dict(publications=len(hints), errors=sum(bool(h.get('error')) for h in hints),
            cancelled_before_ack=sum(bool(h.get('cancelled_before_ack')) for h in hints),
            acknowledgement_reasons=dict(Counter(h.get('response',{}).get('reason','unavailable') for h in hints)),
            ready_cpu_keys_refreshed=sum(h['response'].get('ready_cpu_prefix_keys',0) for h in refreshed),
            distinct_sessions_refreshed=len(useful_refreshed), refresh_calls=len(refreshed),
            rtt_s=distribution([h.get('latency_s') for h in hints]),
            acknowledged_before_tool_completion=distribution([v for v in lead if v>=0]),
            tool_windows=len(windows), windows_with_advance_ack=timely_windows, after_completion=post_tool_hints,
            refreshed_session_return_load_tokens=sum(r['loaded_tokens'] or 0 for r in returned_refreshed),
            attribution='Refresh and subsequent demand restore are separate observations, not proof that advice caused the hit. Paired arms and eviction/recompute counts establish comparative work.'),
        opportunity=dict(cache_competition_observed=returning_evicted>0,
            prior_cpu_evicted_key_observations=evicted, returning_requests_with_prior_cpu_eviction=returning_evicted,
            scheduler_steps=len(steps), peak_running=peak('running'), peak_waiting=peak('waiting_count'),
            min_reclaimable_gpu_blocks=min((r['reclaimable'] for r in steps), default=None),
            total_gpu_blocks=peak('total_blocks'),
            caution='Reclaimable includes idle cached blocks. CPU evictions of returning prefix keys demonstrate actual competing reuse; high device allocation alone does not.'),
        latency=dict(initial_ttft_s=distribution([r['ttft_s'] for r in comparisons if r['turn']==0]),
            return_ttft_s=distribution([r['ttft_s'] for r in comparisons if r['turn']>0]),
            tool_to_first_token_s=distribution([r['tool_to_first_token_s'] for r in comparisons if r['turn']>0]),
            server_queue_s=distribution([r['server_queue_s'] for r in comparisons]),
            native_load_critical_path_s=distribution([r['load_critical_path_s'] for r in comparisons])),
        cache=dict(native_request_receipts=len(by_id), authoritative_receipts=sum(e.get('authoritative') is True for e in matched),
            locally_executed_prefill_tokens=total('locally_executed_prefill_tokens'),
            returning_locally_executed_prefill_tokens=total('locally_executed_prefill_tokens',[r for r in comparisons if r['turn']>0]),
            loaded_tokens=total('loaded_tokens'),
            raw_counter_deltas=case['metrics']),
        prefetch=dict(submitted_jobs=sum(e['event']=='staging_submitted' for e in wait_events),
            ready_jobs=sum(e['event']=='staging_ready' for e in wait_events),
            consumed_sources=sum(e['event']=='staging_consumed' for e in wait_events),
            submitted_logical_bytes=sum(e.get('submitted_bytes',0) for e in wait_events if e['event']=='staging_submitted'),
            completed_physical_bytes=sum(e.get('transfer_bytes',0) for e in wait_events if e['event']=='staging_ready'),
            consumed_logical_bytes=sum(e.get('consumed_bytes',0) for e in wait_events if e['event']=='staging_consumed'),
            demand_reclaimed_blocks=sum(e.get('reclaimed_ready_blocks',0) for e in wait_events if e['event']=='demand_reclaim'),
            nomination_cpu_ns=sum(e.get('nomination_cpu_ns',0) for e in wait_events if e['event']=='nomination_decision'),
            event_counts=dict(Counter(e['event'] for e in wait_events)),
            scope='Native staging receipts; aggregate demand load bytes alone are never called prefetch.'),
        telemetry=dict(gauges=gauges, metrics_scrape_s=distribution([r.get('metrics_scrape_s') for r in samples]),
            gpu_utilization_pct=distribution([r.get('gpu_utilization_pct') for r in samples]),
            gpu_power_w=distribution([r.get('gpu_power_w') for r in samples]),
            sampled_gpu_energy_j=sum((b['monotonic_s']-a['monotonic_s'])*(a['gpu_power_w']+b['gpu_power_w'])/2 for a,b in zip(samples,samples[1:]) if a.get('gpu_power_w') is not None and b.get('gpu_power_w') is not None),
            observation_overhead='Live HTTP scraping, native scheduler trace and detailed preservation events enabled equally in both arms; costs charged to measured serving. No uninstrumented-overhead estimate.'),
        quality=dict(assessment='Final claimed test outcome versus real subprocess exit status only; reasoning accuracy and task resolution ungraded.',
            checks=len(quality), passed=sum(q['passed'] for q in quality), failed=sum(not q['passed'] for q in quality), details=quality),
        request_diagnostics=comparisons,
        limitations=['Authored local maintenance tasks; no production arrival distribution.',
                    'One paired run is a harness acceptance test, not a repeatable performance claim.',
                    'Live outputs and tool durations can diverge between arms; token/prompt fingerprints are retained.',
                    'CPU retention does not perform speculative prefetch. Candidate configs must request and prove it separately.',
                    'Sampled gauges can lag native cache state; scheduler/key events provide finer evidence.'])
    if not legacy_quality:
        q = result['quality']
        q.update(scorer_version='normalized-status-v2',
                 strict_format_failures=sum(not x['strict_format_satisfied'] for x in quality),
                 contradictory_claims=sum(x['contradictory_claim'] for x in quality),
                 unparseable_status=sum(x['status_unparseable'] for x in quality))
        if 'tool_concurrency' in plan:
            result['tools']['queue_s'] = distribution([t.get('queue_s') for t in tools])
            result['tools']['configured_workers'] = plan['tool_concurrency'] or 'all_ready'
        def compare(measured, endpoint):
            return dict(native=measured, endpoint=endpoint,
                        agrees=measured==endpoint if measured is not None and endpoint is not None else None)
        result['consistency'] = {
            'computed_prefill':compare(result['cache']['locally_executed_prefill_tokens'],case['metrics'].get('vllm:request_prefill_kv_computed_tokens_sum')),
            'loaded_tokens':compare(result['cache']['loaded_tokens'],case['metrics'].get('vllm:external_prefix_cache_hits_total')),
            'prompt_tokens':compare(sum(len(r['payload']['prompt']) for r in rows),case['metrics'].get('vllm:prefix_cache_queries_total')),
            'request_count':compare(len(rows),case['metrics'].get('vllm:request_prefill_kv_computed_tokens_count')),
        }
        if plan.get('prompt_source_accounting_version') in (2,3):
            sources = [e.get('first_admission_prompt_source') for e in matched]
            complete = len(sources)==len(rows) and all(isinstance(s,dict) for s in sources)
            source_total = lambda name: sum(s[name] for s in sources) if complete else None
            result['cache']['first_admission_sources'] = {
                name:source_total(name) for name in ('num_prompt_tokens','num_computed_tokens',
                    'num_cached_tokens','num_local_cached_tokens','num_external_cached_tokens')}
            result['cache']['physical_demand_load_bytes'] = sum(e.get('load_bytes',0) for e in matched)
            result['cache']['preemptions'] = sum(e.get('preemption_count',0) for e in matched)
            result['cache']['accounting_scope'] = ('Endpoint prefill and external-hit counters describe first-admission sources. '
                'Native locally-executed tokens accumulate scheduled original-prompt work across preemptions; '
                'loaded_tokens describes terminal-episode reuse. Physical demand load bytes count all completed transfers. '
                'These quantities are reported separately, never equated across definitions.')
            result['consistency'].update(
                computed_prefill=compare(source_total('num_computed_tokens'),case['metrics'].get('vllm:request_prefill_kv_computed_tokens_sum')),
                loaded_tokens=compare(source_total('num_external_cached_tokens'),case['metrics'].get('vllm:external_prefix_cache_hits_total')),
                demand_load_bytes=compare(result['cache']['physical_demand_load_bytes'],case['metrics'].get('vllm:kv_offload_load_bytes_total')),
                preemptions=compare(result['cache']['preemptions'],case['metrics'].get('vllm:num_preemptions_total')))
            if plan.get('prompt_source_accounting_version') == 3:
                attributed = result['cache'].pop('physical_demand_load_bytes')
                endpoint = case['metrics'].get('vllm:kv_offload_load_bytes_total')
                result['cache']['physical_demand_load_bytes'] = endpoint
                result['cache']['request_attributed_load_bytes'] = attributed
                result['cache']['load_attribution_gap_bytes'] = endpoint-attributed if endpoint is not None else None
                result['cache']['transfer_attribution_scope'] = ('Engine-wide completed transfer bytes are authoritative for traffic. '
                    'Per-request receipts do not exhaust engine-wide transfers under preemption; the signed gap is retained, not called zero or a counter identity.')
                result['consistency'].pop('demand_load_bytes')
        if any(c['agrees'] is False for c in result['consistency'].values()):
            errors.append('native_endpoint_accounting_mismatch')
        if any(c['agrees'] is None for c in result['consistency'].values()):
            errors.append('native_endpoint_accounting_unavailable')
    if plan.get('client_concurrency') is not None:
        result['latency'].update(
            offered_to_first_token_s=distribution([r['ttft_s']+r['client_queue_s'] for r in comparisons if r['ttft_s'] is not None]),
            client_queue_s=distribution([r['client_queue_s'] for r in comparisons]),
            all_http_ttft_s=distribution([r['ttft_s'] for r in comparisons]))
        result['opportunity']['by_turn'] = {
            str(turn):dict(requests=len(selected), prompt_tokens=sum(r['prompt_tokens'] for r in selected),
                computed_tokens=total('locally_executed_prefill_tokens', selected),
                restored_tokens=total('loaded_tokens', selected),
                prior_evicted_requests=sum(bool(r['prior_cpu_evicted_keys']) for r in selected))
            for turn in sorted({r['turn'] for r in comparisons})
            for selected in [[r for r in comparisons if r['turn']==turn]]}
        result['hints']['gpu_gate_skipped_calls'] = sum(h.get('response',{}).get('cpu_retention_skipped_full_gpu') is True for h in hints)
        result['hints']['queue_publications'] = sum(e['publications'] for e in events if e['event']=='queue_hint_window')
        result['hints']['pre_tool_publication_s'] = distribution([e['duration_s'] for e in events if e['event']=='pre_tool_hint'])
    if plan.get('diagnostic_replay'):
        result['tools'].update(executed=0,replayed_delays=len(tools),
            replayed_wait_wall_s=result['tools']['total_wall_s'],total_wall_s=0,
            scope='Recorded delays only; no live tool processes or output bytes produced.')
        result['quality']['assessment']='Frozen historical test-status reporting only; no fresh tool execution or live application quality evidence.'
        result['limitations'].insert(0,'Diagnostic fixed-token replay with recorded tool delays, not live application performance. All prompt and continuation token IDs must match the selected trace.')
    if write:
        write_json(directory/'agent.json', result)
        if comparisons:
            with (directory/'request-diagnostics.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
                writer.writeheader()
                writer.writerows(comparisons)
    return result
