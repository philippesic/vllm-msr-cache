"""Read-only comparison of completed2080Ti/5080 campaigns and5080 critical paths."""
from pathlib import Path
import hashlib
import json
import math
import statistics

REPO = Path(__file__).resolve().parents[2]
OLD = REPO / 'evidence/stage6-2080ti-20261106/results/comparisons.json'
NEW = REPO / 'evidence/stage6-5080-20261109/final-results-01/comparisons.json'
OUT = REPO / 'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
BUNDLE = REPO / 'output/stage6-5080-20261108/primary-execution-bundle-01'
DEST = REPO / 'evidence/stage6-5080-20261109/cross-gpu-01'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def old_row(row):
    r = {k: row[k] for k in ('model', 'cell', 'process_pairs', 'throughput_ratio', 'paired_95_interval', 'per_run_ratios')}
    r.update(numerator=row['candidate'], denominator=row['baseline'])
    for source, target in [('candidate', 'numerator_summary'), ('baseline', 'denominator_summary')]:
        work, seconds = row[source + '_work'], row[source + '_seconds']
        r[target] = dict(serving_seconds=seconds, completed_requests=work['requests'],
            failed_requests=work['failures'], prompt_tokens=work['prompt_tokens'],
            output_tokens=work['output_tokens'], length_caps=work['caps'], attempts=work['artifacts'])
    if r['denominator'] == 'wait':
        r['numerator'], r['denominator'] = r['denominator'], r['numerator']
        r['numerator_summary'], r['denominator_summary'] = r['denominator_summary'], r['numerator_summary']
        r['throughput_ratio'] = 1 / r['throughput_ratio']
        r['paired_95_interval'] = [1 / r['paired_95_interval'][1], 1 / r['paired_95_interval'][0]]
        r['per_run_ratios'] = [1 / x for x in r['per_run_ratios']]
    return r


def key(row):
    return tuple(row[k] for k in ('model', 'cell', 'numerator', 'denominator'))


def main():
    assert OUT.is_dir() and OUT.parent.is_mount() and not DEST.exists()
    old, new = load(OLD), load(NEW)
    ledger_before = sha(OUT / 'ledger.jsonl')
    report_path = OUT / 'analysis/current/results.json'
    report_before = sha(report_path)
    report = load(report_path)
    old_index = {key(r): r for r in map(old_row, old['primary_comparisons'])}
    new_index = {key(r): r for r in new['comparisons']}
    assert len(old_index) == len(new_index) == 42 and old_index.keys() == new_index.keys()
    comparisons, arms = [], {}
    for identity, a in old_index.items():
        b = new_index[identity]
        for r in (a, b):
            assert len(r['per_run_ratios']) == r['process_pairs']
            assert math.isclose(math.exp(statistics.mean(map(math.log, r['per_run_ratios']))), r['throughput_ratio'], rel_tol=1e-12)
        comparisons.append(dict(model=identity[0], cell=identity[1], numerator=identity[2], denominator=identity[3],
            **{host: {k: r[k] for k in ('process_pairs', 'throughput_ratio', 'paired_95_interval', 'per_run_ratios')}
               for host, r in [('rtx2080ti', a), ('rtx5080', b)]},
            relative_effect_ratio_5080_over_2080=b['throughput_ratio'] / a['throughput_ratio'],
            ratio_of_relative_effects_scope='Descriptive cross-host comparison, not a randomized GPU-only causal effect'))
        for host, r in [('rtx2080ti', a), ('rtx5080', b)]:
            for role in ('numerator', 'denominator'):
                system, s = r[role], r[role + '_summary']
                value = dict(model=identity[0], cell=identity[1], system=system, processes=r['process_pairs'],
                    serving_seconds=s['serving_seconds'], requests=s['completed_requests'],
                    requests_per_process=s['completed_requests'] / r['process_pairs'],
                    requests_per_serving_second=s['completed_requests'] / sum(s['serving_seconds']['values']),
                    output_tokens=s['output_tokens'], caps=s['length_caps'])
                arm_key = (host, identity[0], identity[1], system)
                if arm_key in arms:
                    assert arms[arm_key] == value
                arms[arm_key] = value
    events = {}
    for line in (OUT / 'ledger.jsonl').read_text().splitlines():
        event = json.loads(line)
        events[event['attempt_id']] = event
    cases, checked_files = {}, {}

    def case_for(point):
        attempt = point['attempt']
        if attempt not in cases:
            event = events[attempt]
            folder = OUT / event['details']['path']
            seal = load(folder / 'seal.json')
            assert canonical_digest(seal) == event['details']['seal']
            path = folder / 'measurement/result.json'
            assert sha(path) == seal['files']['measurement/result.json']
            cases[attempt] = (folder, load(path)['cases'][0], seal)
            checked_files[str(path.relative_to(OUT))] = sha(path)
        return cases[attempt]

    activity = []
    for row in report['comparisons']:
        if row['candidate'] != 'wait' or row['baseline'] != 'arc':
            continue
        for role, system in [('baseline', 'arc'), ('candidate', 'wait')]:
            cc = [case_for(p[role])[1] for p in row['points']]
            native = [c['campaign']['native_cache'] for c in cc]
            old_activity = old['secondary_actual_pressure'].get(row['model'] + ':' + row['cell'] + ':' + system)
            activity.append(dict(model=row['model'], cell=row['cell'], system=system, processes=len(cc),
                old_recorded_secondary=old_activity,
                old_restored_tokens_per_process=old_activity['returning_loaded_tokens'] / old_activity['processes'] if old_activity else None,
                old_eviction_affected_return_fraction=old_activity['eviction_affected_returns'] / old_activity['returning_requests'] if old_activity else None,
                new_restored_tokens_per_process=statistics.mean(n['returning_loaded_tokens'] for n in native),
                new_recomputed_returning_prompt_tokens_per_process=statistics.mean(n['returning_computed_prompt_tokens'] for n in native),
                new_eviction_affected_return_fraction=sum(n['returning_requests_with_prior_cpu_eviction'] for n in native) / sum(n['returning_requests'] for n in native) if sum(n['returning_requests'] for n in native) else None,
                new_hint_events_per_process=statistics.mean(len(c['hints']) for c in cc),
                new_load_bytes_per_process=statistics.mean(c['metrics'].get('vllm:kv_offload_load_bytes_total', 0) for c in cc),
                new_store_bytes_per_process=statistics.mean(c['metrics'].get('vllm:kv_offload_store_bytes_total', 0) for c in cc),
                new_prefill_request_time_sum_mean=statistics.mean(c['metrics'].get('vllm:request_prefill_time_seconds_sum', 0) for c in cc),
                new_decode_request_time_sum_mean=statistics.mean(c['metrics'].get('vllm:request_decode_time_seconds_sum', 0) for c in cc),
                component_time_scope='Sums of per-request times with concurrent requests; never add/subtract as a wall-time decomposition'))
    mixed = next(r for r in report['comparisons'] if (r['model'], r['cell'], r['candidate'], r['baseline']) ==
                 ('qwen3_8b_awq', 'B11', 'wait', 'arc'))
    tool_cases, paired_tools = [], []
    for point in mixed['points']:
        pair = {}
        for role, system in [('baseline', 'arc'), ('candidate', 'wait')]:
            arm = point[role]
            folder, case, seal = case_for(arm)
            path = folder / 'measurement' / case['name'] / 'workflow-0.jsonl'
            relative = str(path.relative_to(folder))
            assert sha(path) == seal['files'][relative]
            checked_files[str(path.relative_to(OUT))] = sha(path)
            timeline = [json.loads(line) for line in path.read_text().splitlines()]
            start = timeline[0]['ns']
            tool = next(e for e in timeline if e['event'] == 'tool' and
                        'tests/test_more.py::TestConcurrentTee' in e.get('argv', []))
            end = next(e for e in timeline if e['event'] == 'session_end' and e['session_id'] == tool['session_id'])
            output = Path(tool['output_artifact'])
            assert sha(output) == seal['files'][str(output.relative_to(folder))]
            checked_files[str(output.relative_to(OUT))] = sha(output)
            assert tool['success'] and '1 passed' in output.read_text()
            event = events[arm['attempt']]
            n = case['campaign']['native_cache']
            record = dict(repeat=event['details']['identity']['repeat'], system=system, attempt=arm['attempt'],
                case_seconds=case['load_duration_s'], output_tokens=case['summary']['output_tokens'],
                thread_test_seconds=tool['duration_s'],
                thread_test_start_s=(tool['started_ns'] - start) / 1e9,
                thread_test_end_s=(tool['ended_ns'] - start) / 1e9,
                thread_test_session_end_s=(end['ns'] - start) / 1e9,
                thread_test_session_finishes_last=end['ns'] == max(e['ns'] for e in timeline if e['event'] == 'session_end'),
                tool_output=output.read_text().strip(),
                tool_total_cpu_seconds=case['telemetry']['client_tool_cpu_elapsed_s'],
                restored_returning_tokens=n['returning_loaded_tokens'], recomputed_returning_prompt_tokens=n['returning_computed_prompt_tokens'],
                hints=len(case['hints']),
                summed_request_prefill_seconds=case['metrics']['vllm:request_prefill_time_seconds_sum'],
                summed_request_decode_seconds=case['metrics']['vllm:request_decode_time_seconds_sum'],
                maximum_swap_in_bytes=case['telemetry']['host_swap_in_delta_bytes'],
                maximum_swap_out_bytes=case['telemetry']['host_swap_out_delta_bytes'],
                artifact_path=str(folder.relative_to(OUT)), artifact_seal=event['details']['seal'],
                workflow_sha256=sha(path), tool_output_sha256=sha(output))
            tool_cases.append(record)
            pair[system] = record
        paired_tools.append(dict(repeat=pair['arc']['repeat'],
            total_seconds_saved_by_candidate=pair['arc']['case_seconds'] - pair['wait']['case_seconds'],
            thread_test_seconds_difference=pair['arc']['thread_test_seconds'] - pair['wait']['thread_test_seconds']))
    old_mixed = old_index[('qwen3_8b_awq', 'B11', 'wait', 'arc')]
    sd_old = statistics.stdev(map(math.log, old_mixed['per_run_ratios']))
    sd_new = statistics.stdev(map(math.log, mixed['estimate']['per_run_ratios']))
    protocol, portable = load(OUT / 'protocol.lock.json'), load(BUNDLE / 'portable.json')
    original_source = portable['protocol']['parent_acquisition_source']
    current_source = protocol['source']
    source_differences = sorted(name for name in original_source['files'].keys() | current_source['files'].keys()
                                if original_source['files'].get(name) != current_source['files'].get(name))
    allowed = {'benchmarks/research_checks/stage6_healthcheck.py', 'benchmarks/serving_suite/campaign.py',
               'benchmarks/serving_suite/campaign_verify.py', 'benchmarks/serving_suite/campaign_portability.py',
               'benchmarks/serving_suite/campaign_report.py', 'benchmarks/serving_suite/campaign_validity.py'}
    assert set(source_differences) == allowed
    assert all(protocol[k] == portable['protocol'][k] for k in ('schedule', 'supplementary_schedules', 'precision'))
    profiles_match = []
    old_cells = {(c['model'], c['cell']): c for c in portable['cells']}
    for cell in load(OUT / 'resolved.json')['cells']:
        a, b = dict(old_cells[(cell['model'], cell['cell'])]['profile']), dict(cell['profile'])
        for profile in (a, b):
            profile.pop('runtime_python', None)
            profile.pop('allocation_receipts', None)
        assert a == b
        profiles_match.append(dict(model=cell['model'], cell=cell['cell']))
    tests = OUT / 'imported/assets/sources/more-itertools/tests/test_more.py'
    source = tests.read_text()
    fragment = source.split('class TestConcurrentTee(TestCase):', 1)[1].split('\n\ndef grow_to_window', 1)[0]
    assert 'limit = 10**5' in fragment and 'num_threads = 100' in fragment
    assert sha(OUT / 'ledger.jsonl') == ledger_before and sha(report_path) == report_before
    payload = dict(schema='stage6-cross-gpu-analysis-v1', comparisons=comparisons,
        arms=[dict(host=k[0], **v) for k, v in sorted(arms.items())], native_activity=activity,
        mixed_tool_cases=tool_cases, mixed_tool_pair_differences=paired_tools,
        mixed_variation=dict(old_log_ratio_sd=sd_old, new_log_ratio_sd=sd_new, log_sd_multiplier=sd_new / sd_old,
            old_wins=sum(r > 1 for r in old_mixed['per_run_ratios']), new_wins=sum(r > 1 for r in mixed['estimate']['per_run_ratios']),
            eight_planned_pairs_present_on_both_hosts=True,
            old_effect_inside_5080_interval=mixed['estimate']['paired_95_interval'][0] <= old_mixed['throughput_ratio'] <= mixed['estimate']['paired_95_interval'][1],
            tool_vs_total_saved_seconds_correlation=statistics.correlation(
                [p['thread_test_seconds_difference'] for p in paired_tools], [p['total_seconds_saved_by_candidate'] for p in paired_tools]),
            correlation_scope='Descriptive eight-point association, not causal regression or a corrected throughput estimate'),
        provenance=dict(original_2080_source=original_source['digest'], imported_5080_source=current_source['digest'],
            differing_files=source_differences, serving_policy_and_workload_modules_unchanged=True,
            schedule_order_repeat_precision_equal=True, all_22_resource_profiles_equal=profiles_match,
            cache_capacity_not_increased_for_5080=True,
            lmcache_rebuild_only_changes_the_continuum_runtime=True),
        thread_test=dict(name='more_itertools.TestConcurrentTee.test_concurrent_consumers',
            source_sha256=sha(tests), producer_elements=100000, consumer_threads=100,
            source_excerpt='class TestConcurrentTee(TestCase):' + fragment,
            all_16_5080_invocations_passed=True),
        limitations=['2080 full raw request/tool timelines are not copied to this machine; no2080 machine contacted.',
            'Old source/cap/transfer records and all42 aggregate/per-run comparison inputs are available locally.',
            'Different hosts and natural generated outputs; not a GPU-only randomized experiment.',
            'No new GPU measurement, profiler, campaign rerun, excluded-run replacement or adjusted throughput.',
            'Faster execution makes tool delays proportionally larger; exact causal split between hardware, scheduling, host CPU and I/O remains unmeasured.'],
        source_hashes={str(p.relative_to(REPO)): sha(p) for p in (OLD, NEW, report_path, BUNDLE / 'portable.json')},
        checked_5080_files=checked_files, original_ledger_and_reports_unchanged=True)
    DEST.mkdir()
    (DEST / 'comparison.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(comparisons=len(comparisons), unique_arms=len(arms), cases_verified=len(cases),
        mixed_variation=payload['mixed_variation'], resource_profiles=len(profiles_match),
        selected_files_verified=len(checked_files)), indent=2))


if __name__ == '__main__':
    main()
