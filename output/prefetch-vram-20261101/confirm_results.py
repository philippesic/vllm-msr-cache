"""Analyze only a finite confirmation plan; stdlib only, no server/GPU imports.

    python output/prefetch-vram-20261101/confirm_results.py --require-complete

Cases require phase, seed, policy, role (or reference), and pair_id for the
primary arc_full/candidate arms. Workload defaults to the plan's workload.
Screens and other unplanned runs are never included. Missing evidence remains
missing: preemption does not waive terminal or physical-identity proofs.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

H = Path(__file__).resolve().parent
ROLES = ('arc_full', 'candidate', 'arc_small', 'original')
PROOFS = ('terminal_observed', 'terminal_authoritative', 'ordered_keys_complete',
          'physical_identity_complete', 'block_table_complete')
COUNTERS = ('prefix_mean_evaluations', 'prefix_mean_evaluated_candidates', 'prefix_mean_centroid_builds', 'prefix_mean_centroid_hits', 'prefix_mean_applied_observations', 'prefix_mean_applied_candidates', 'prefix_mean_applied_key_count', 'worker_capture_calls', 'worker_capture_rows', 'applied_score_keys',
            'score_apply_calls', 'scored_candidate_count', 'constrained_decisions',
            'staging_jobs', 'staged_bytes', 'ready_bytes', 'native_adoptions',
            'prefetch_consumed_blocks', 'prefetch_consumed_bytes',
            'prefetch_unused_completed_bytes', 'prefetch_waste_bytes',
            'demand_h2d_bytes', 'd2h_bytes', 'splice_bytes', 'return_demand_h2d_bytes',
            'return_splice_bytes', 'preemptions', 'prefetch_allocation_failures',
            'transfer_failed_jobs', 'speculative_transfer_failed_jobs')
MEMORY = ('cuda_allocated_before_bytes', 'cuda_reserved_before_bytes',
          'cuda_allocated_after_bytes', 'cuda_reserved_after_bytes',
          'cuda_load_peak_allocated_bytes', 'cuda_load_peak_reserved_bytes',
          'nvml_load_peak_bytes_sampled', 'nvml_before_bytes_sampled')


def pin(path):
    return {'path': str(path.resolve()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def helper():
    spec = importlib.util.spec_from_file_location('confirmation_analyze', H / 'analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # analyze() normally updates each run's derived artifacts. Confirmation must
    # preserve existing evidence and write only its own outputs.
    module.write = lambda *args, **kwargs: None
    module.csvwrite = lambda *args, **kwargs: None
    return module


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def total(values):
    values = list(values)
    return sum(values) if values and all(finite(v) for v in values) else None


def pct(value, baseline):
    return (value / baseline - 1) * 100 if finite(value) and finite(baseline) and baseline else None


def expected_name(case, plan):
    return case.get('run_name') or '-'.join(str(case[k] if k in case else plan.get(k, 'competing_long_dram'))
                                           for k in ('phase', 'workload', 'seed', 'policy'))


def strict_integrity(raw, summary, rows):
    bad = {p: [r['index'] for r in rows if r.get(p) is not True] for p in PROOFS}
    bad['event_integrity'] = [r['index'] for r in rows if r.get('event_integrity') != 'complete']
    failed = [r['index'] for r in rows if r.get('success') is not True]
    ids = [r['payload']['request_id'] for r in raw['requests']]
    indices = [r['index'] for r in raw['requests']]
    terminal_ids = [t.get('external_request_id') for t in raw.get('terminals', [])]
    reasons = []
    if raw.get('status') != 'complete': reasons.append('run_not_complete')
    if raw.get('error'): reasons.append('run_error')
    if not rows: reasons.append('no_requests')
    if failed: reasons.append('request_failures')
    if any(bad.values()): reasons.append('missing_or_failed_terminal_proof')
    if len(ids) != len(set(ids)) or len(indices) != len(set(indices)):
        reasons.append('duplicate_request_identity')
    if any(terminal_ids.count(rid) != 1 for rid in ids): reasons.append('terminal_not_unique')
    for field in ('transfer_failed_jobs', 'speculative_transfer_failed_jobs', 'prefetch_allocation_failures'):
        if summary.get(field) != 0: reasons.append(field + '_nonzero_or_missing')
    if raw.get('owned_descendants_remaining') != []: reasons.append('descendants_remaining_or_missing')
    if not finite(summary['resources'].get('cpu_s_sampled')) or summary['resources'].get('cpu_process_identity_stable') is False:
        reasons.append('missing_or_unstable_cpu_evidence')
    if not finite(summary.get('load_gpu_peak_allocated_bytes')) or not finite(summary['resources'].get('device_memory_used_peak_bytes_sampled')):
        reasons.append('missing_actual_memory_evidence')
    if not finite(summary['all']['elapsed_s']) or summary['all']['elapsed_s'] <= 0:
        reasons.append('invalid_elapsed')
    if any(not finite(r.get('release_ttft_ms')) for r in rows): reasons.append('missing_ttft')
    if any('output_token_ids' not in r or not r['output_token_ids'] for r in raw['requests']):
        reasons.append('missing_output_tokens')
    return {'passed': not reasons, 'reasons': reasons, 'failed_request_indices': failed,
            'bad_proof_indices': bad, 'owned_descendants_remaining': raw.get('owned_descendants_remaining')}


def run_record(case, path, data, analyze):
    raw, s, rows, _ = data
    ret = [r for r in rows if 'return' in r['role']]
    a, ra = s['all'], analyze.aggregate(ret, s['all']['elapsed_s'])
    before = (raw.get('before_load_export', {}).get('workers') or [{}])[0]
    after = (raw.get('postload_export', {}).get('workers') or [{}])[0]
    events = raw.get('wait_hint_events', [])
    decisions = [e for e in events if e.get('event') == 'nomination_decision']
    adoptions = [r['native_adoption'] for r in ret if r.get('native_adoption')]
    cpu = s['resources'].get('cpu_s_sampled')
    extra = (raw.get('config') or {}).get('kv_connector_extra_config') or {}
    record = {'name': s['name'], 'role': case.get('role', case.get('reference')),
              'pair_id': case.get('pair_id'), 'seed': s['seed'], 'phase': s['phase'],
              'workload': s['workload'], 'policy': s['policy'], 'status': s['status'],
              'elapsed_s': a['elapsed_s'], 'output_tokens': a['output_tokens'],
              'output_tokens_s': a['output_tokens_s'], 'cpu_s_sampled': cpu,
              'cpu_s_per_1000_output_tokens': cpu / a['output_tokens'] * 1000 if finite(cpu) and a['output_tokens'] else None,
              'ttft_mean_ms': a['release_ttft_ms']['mean'], 'ttft_p95_ms': a['release_ttft_ms']['p95'],
              'return_ttft_mean_ms': ra['release_ttft_ms']['mean'], 'return_ttft_p95_ms': ra['release_ttft_ms']['p95'],
              'mean_itl_ms': a['mean_itl_ms'], 'return_mean_itl_ms': ra['mean_itl_ms'],
              'itl_intervals': a['itl_intervals'], 'return_itl_intervals': ra['itl_intervals'],
              'requests': a['requests'], 'failures': a['failures'],
              'prefetch_selector': extra.get('prefetch_selector'),
              'configured_eviction_policy': extra.get('eviction_policy'),
              'geometry': raw.get('geometry'), 'manager_type': s['manager_type'], 'policy_type': s['policy_type'],
              'candidate_mode': 'ARC cache admission/eviction with semantic prefetch ordering' if case.get('role', case.get('reference')) == 'candidate' else None,
              'native_adoptions': len(adoptions),
              'all_adoptions_same_physical_ids': all(x.get('same_physical_blocks') is True for x in adoptions) if adoptions else None,
              'applied_score_keys': s['manager_counts'].get('relevance_apply_keys', 0),
              'score_apply_calls': s['manager_counts'].get('relevance_apply_calls', 0),
              'scoring_metric': 'prefix_mean_raw_dot' if extra.get('prefix_mean_prefetch_scoring') else 'per_block_rank_ema',
              'scored_candidate_count': sum(e.get('scored_candidate_count', 0) for e in decisions),
              'constrained_decisions': sum(bool(e.get('shared_budget_constrained')) for e in decisions),
              'staging_jobs': sum(e.get('event') == 'staging_submitted' for e in events),
              'staged_bytes': sum(e.get('submitted_bytes', 0) for e in events if e.get('event') == 'staging_submitted'),
              'ready_bytes': sum(e.get('transfer_bytes', 0) for e in events if e.get('event') == 'staging_ready'),
              'return_demand_h2d_bytes': total(r['demand_h2d_bytes'] for r in ret),
              'return_splice_bytes': total(r['splice_bytes'] for r in ret),
              'cuda_load_peak_allocated_bytes': s['load_gpu_peak_allocated_bytes'],
              'cuda_load_peak_reserved_bytes': s['load_gpu_peak_reserved_bytes'],
              'nvml_load_peak_bytes_sampled': s['resources'].get('device_memory_used_peak_bytes_sampled'),
              'nvml_before_bytes_sampled': next((r['gpu']['memory_used_bytes'] for r in reversed(raw.get('resource_samples', [])) if r['at_ns'] <= raw['load_start_ns'] and 'gpu' in r), None),
              'resources': s['resources'], 'raw_evidence': pin(path),
              'identity': {k: raw.get(k) for k in ('experiment', 'config', 'configuration', 'runtime_resolution', 'server_command')},
              'planned_case': case, 'integrity': strict_integrity(raw, s, rows)}
    for time, worker in [('before', before), ('after', after)]:
        for kind in ('allocated', 'reserved'):
            record[f'cuda_{kind}_{time}_bytes'] = worker.get(f'gpu_{kind}_bytes')
    for field in ('gpu_blocks', 'gpu_kv_bytes', 'cpu_blocks', 'cpu_kv_bytes', 'block_bytes'):
        record[field] = (raw.get('geometry') or {}).get(field)
    for field in COUNTERS:
        if field not in record: record[field] = s.get(field)
    semantic_proof = {
        'arc_cache_policy': 'arc' in str(s.get('policy_type', '')).lower() and extra.get('eviction_policy') == 'arc',
        'semantic_ordering_configured': extra.get('prefetch_selector') == 'semantic',
        'semantic_decision_observed': any(e.get('selector') == 'semantic' and e.get('scored_candidate_count', 0) >= 2 and e.get('shared_budget_constrained') is True for e in decisions),
        'capture_exercised': record['worker_capture_calls'] > 0,
        'score_application_exercised': record['applied_score_keys'] > 0 or record['prefix_mean_applied_candidates'] > 0,
        'candidate_scoring_exercised': record['scored_candidate_count'] > 0,
        'constrained_selection_exercised': record['constrained_decisions'] > 0,
        'prefetch_consumed': (record['prefetch_consumed_bytes'] or 0) > 0,
        'native_adoption_exercised': bool(adoptions) and record['all_adoptions_same_physical_ids'] is True,
        'no_gpu_splice': record['splice_bytes'] == 0,
    }
    # Numerical scoring and a budget-constrained ranking decision answer
    # different questions. An all-fit workload can exercise the former without
    # supplying evidence for the latter; retain the original strict conjunction.
    exercised_scoring = all(semantic_proof[k] for k in (
        'capture_exercised', 'score_application_exercised', 'candidate_scoring_exercised'))
    constrained_selection = all(semantic_proof[k] for k in (
        'arc_cache_policy', 'semantic_ordering_configured',
        'semantic_decision_observed', 'constrained_selection_exercised'))
    native_prefetch = all(semantic_proof[k] for k in (
        'prefetch_consumed', 'native_adoption_exercised', 'no_gpu_splice'))
    record['semantic_functionality'] = {
        'checks': semantic_proof, 'exercised_scoring': exercised_scoring,
        'constrained_semantic_selection': constrained_selection,
        'native_prefetch_exercised': native_prefetch,
        'passed': all(semantic_proof.values())}
    # Flat evidence columns keep the concise per-run CSV equally explicit.
    record.update(exercised_scoring=exercised_scoring,
                  constrained_semantic_selection=constrained_selection,
                  native_prefetch_exercised=native_prefetch,
                  strict_semantic_functionality_passed=all(semantic_proof.values()))
    return record


def exact_comparison(baseline, current, analyze):
    # Request IDs may differ across independent servers; join by planned index,
    # then look up each side's own terminal ID (analyze.compare assumes equal IDs).
    ar, _, _, _ = baseline['data']; cr, _, _, _ = current['data']
    aa = {r['index']: r for r in ar['requests']}; cc = {r['index']: r for r in cr['requests']}
    at = {t.get('external_request_id'): t for t in ar.get('terminals', [])}
    ct = {t.get('external_request_id'): t for t in cr.get('terminals', [])}
    result = {'baseline': baseline['record']['name'], 'candidate': current['record']['name'],
              'baseline_role': baseline['record']['role'], 'seed': cr['seed'],
              'request_sets_equal': set(aa) == set(cc), 'matched_requests': len(set(aa) & set(cc)),
              'prompt_mismatch_indices': [], 'output_mismatch_indices': [],
              'native_key_mismatch_indices': [], 'missing_native_key_indices': [],
              'missing_token_evidence_indices': []}
    def key(t):
        try:
            return [(p['group_idx'], p['chunk_index'], p['key']) for p in t['ordered_key_positions']]
        except (KeyError, TypeError):
            return None
    for i in sorted(set(aa) & set(cc)):
        a, c = aa[i], cc[i]
        if not isinstance(a['payload'].get('prompt'), list) or not isinstance(c['payload'].get('prompt'), list) or 'output_token_ids' not in a or 'output_token_ids' not in c:
            result['missing_token_evidence_indices'].append(i)
        if a['payload'].get('prompt') != c['payload'].get('prompt'): result['prompt_mismatch_indices'].append(i)
        if a.get('output_token_ids') != c.get('output_token_ids'): result['output_mismatch_indices'].append(i)
        aterm, cterm = at.get(a['payload']['request_id'], {}), ct.get(c['payload']['request_id'], {})
        if any(t.get('ordered_key_positions_complete') is not True or not isinstance(t.get('ordered_key_positions'), list) or not t['ordered_key_positions'] for t in (aterm, cterm)):
            result['missing_native_key_indices'].append(i)
        elif key(aterm) is None or key(cterm) is None:
            result['missing_native_key_indices'].append(i)
        elif key(aterm) != key(cterm): result['native_key_mismatch_indices'].append(i)
    result['terminal_and_physical_proofs_passed'] = baseline['record']['integrity']['passed'] and current['record']['integrity']['passed']
    result['passed'] = result['request_sets_equal'] and result['matched_requests'] > 0 and result['terminal_and_physical_proofs_passed'] and not any(v for k, v in result.items() if k.endswith('_indices'))
    return result


def pooled(items, analyze):
    rows = [r for item in items for r in item['data'][2]]
    records = [item['record'] for item in items]
    elapsed = sum(r['elapsed_s'] for r in records)
    a = analyze.aggregate(rows, elapsed)
    ret = analyze.aggregate([r for r in rows if 'return' in r['role']], elapsed)
    cpu = total(r['cpu_s_sampled'] for r in records)
    result = {'runs': len(items), 'elapsed_s': elapsed, 'output_tokens': a['output_tokens'],
              'output_tokens_s': a['output_tokens_s'], 'cpu_s_sampled_total': cpu,
              'cpu_s_sampled_mean_per_run': cpu / len(items) if cpu is not None else None,
              'cpu_s_per_1000_output_tokens': cpu / a['output_tokens'] * 1000 if cpu is not None and a['output_tokens'] else None,
              'ttft_mean_ms': a['release_ttft_ms']['mean'], 'ttft_p95_ms': a['release_ttft_ms']['p95'],
              'return_ttft_mean_ms': ret['release_ttft_ms']['mean'], 'return_ttft_p95_ms': ret['release_ttft_ms']['p95'],
              'mean_itl_ms': a['mean_itl_ms'], 'return_mean_itl_ms': ret['mean_itl_ms'],
              'itl_intervals': a['itl_intervals'], 'return_itl_intervals': ret['itl_intervals'],
              'requests': a['requests'], 'failures': a['failures'],
              'all_integrity_passed': all(r['integrity']['passed'] for r in records),
              'configured_geometries': list({json.dumps(r['geometry'], sort_keys=True): r['geometry'] for r in records}.values())}
    for field in ('gpu_blocks', 'gpu_kv_bytes', 'cpu_blocks', 'cpu_kv_bytes'):
        values = {r[field] for r in records}
        result[field] = next(iter(values)) if len(values) == 1 else None
    for field in COUNTERS: result[field] = total(r[field] for r in records)
    for field in MEMORY:
        values = [r[field] for r in records if finite(r[field])]
        result[field + '_range'] = [min(values), max(values)] if values else None
        result[field + '_max'] = max(values) if values else None
    return result


def pair_metrics(arc, candidate):
    a, c = arc['record'], candidate['record']
    result = {'pair_id': c['pair_id'], 'seed': c['seed'], 'arc_run': a['name'], 'candidate_run': c['name']}
    for field in ('output_tokens_s', 'cpu_s_sampled', 'cpu_s_per_1000_output_tokens', 'ttft_mean_ms', 'ttft_p95_ms', 'return_ttft_mean_ms', 'return_ttft_p95_ms', 'mean_itl_ms', 'return_mean_itl_ms', *MEMORY):
        result['arc_' + field] = a[field]; result['candidate_' + field] = c[field]
        result[field + '_delta_pct'] = pct(c[field], a[field])
        result[field + '_delta'] = c[field] - a[field] if finite(c[field]) and finite(a[field]) else None
    ag, cg = a.get('geometry') or {}, c.get('geometry') or {}
    result['equal_cpu_cache_bytes'] = ag.get('cpu_kv_bytes') is not None and ag.get('cpu_kv_bytes') == cg.get('cpu_kv_bytes')
    result['configured_gpu_kv_saving_bytes'] = ag['gpu_kv_bytes'] - cg['gpu_kv_bytes'] if all(finite(g.get('gpu_kv_bytes')) for g in (ag, cg)) else None
    result['throughput_at_least_99pct_arc'] = result['output_tokens_s_delta_pct'] >= -1 if result['output_tokens_s_delta_pct'] is not None else False
    result['integrity_passed'] = a['integrity']['passed'] and c['integrity']['passed']
    return result


def csvwrite(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k, value in row.items() if not isinstance(value, (list, dict))))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k in fields} for row in rows)


def build(plan_path):
    plan = json.loads(plan_path.read_text())
    analyze = helper()
    cases = plan.get('cases', [])
    errors, missing, items, seen = [], [], [], set()
    if not cases: errors.append('plan_has_no_cases')
    for index, case in enumerate(cases):
        role = case.get('role', case.get('reference'))
        if role not in ROLES:
            errors.append({'case_index': index, 'error': 'unknown_role', 'role': role}); continue
        if any(k not in case for k in ('phase', 'seed', 'policy')):
            errors.append({'case_index': index, 'error': 'missing_phase_seed_or_policy'}); continue
        name = expected_name(case, plan)
        if name in seen:
            errors.append({'case_index': index, 'error': 'duplicate_run', 'name': name}); continue
        seen.add(name)
        relative = case.get('raw_path', f'runs/{name}/raw.json.gz')
        path = Path(relative); path = path if path.is_absolute() else plan_path.parent / path
        if not path.exists():
            missing.append({'name': name, 'role': role, 'pair_id': case.get('pair_id'), 'path': str(path)}); continue
        try:
            data = analyze.analyze(path)
            record = run_record(case, path, data, analyze)
            expected = {'name': name, 'phase': case['phase'], 'seed': case['seed'], 'policy': case['policy'], 'workload': case.get('workload', plan.get('workload', 'competing_long_dram'))}
            mismatch = {k: {'expected': v, 'actual': data[0].get(k)} for k, v in expected.items() if data[0].get(k) != v}
            if mismatch:
                record['integrity']['passed'] = False
                record['integrity']['reasons'].append('planned_identity_mismatch')
                record['identity_mismatch'] = mismatch
            items.append({'record': record, 'data': data})
        except Exception as exc:
            errors.append({'name': name, 'path': str(path), 'error': repr(exc)})
    by_pair = collections.defaultdict(lambda: collections.defaultdict(list))
    expected_pairs = collections.defaultdict(collections.Counter)
    for case in cases:
        role = case.get('role', case.get('reference'))
        if role in ('arc_full', 'candidate'):
            if case.get('pair_id') is None:
                errors.append({'case': case, 'error': 'primary_case_missing_pair_id'}); continue
            expected_pairs[str(case['pair_id'])][role] += 1
    for item in items:
        r = item['record']
        if r['role'] in ('arc_full', 'candidate'): by_pair[str(r['pair_id'])][r['role']].append(item)
    pairs = []
    for pair_id, counts in expected_pairs.items():
        if counts != {'arc_full': 1, 'candidate': 1}:
            errors.append({'pair_id': pair_id, 'error': 'plan_pair_not_one_arc_full_and_candidate', 'counts': dict(counts)})
            continue
        entries = by_pair[pair_id]
        if len(entries['arc_full']) != 1 or len(entries['candidate']) != 1: continue
        arc, candidate = entries['arc_full'][0], entries['candidate'][0]
        if (arc['record']['seed'], arc['record']['workload']) != (candidate['record']['seed'], candidate['record']['workload']):
            errors.append({'pair_id': pair_id, 'error': 'pair_seed_or_workload_mismatch'}); continue
        pair = pair_metrics(arc, candidate)
        pair['exact_correctness_passed'] = exact_comparison(arc, candidate, analyze)['passed']
        pairs.append(pair)
    matched_pairs = []
    expected_matched = [case for case in cases if case.get('role',case.get('reference'))=='arc_small' and case.get('pair_id') is not None]
    for case in expected_matched:
        controls=[i for i in items if i['record']['role']=='arc_small' and i['record']['pair_id']==case['pair_id']]
        candidates=[i for i in items if i['record']['role']=='candidate' and i['record']['pair_id']==case['pair_id']]
        if len(controls)==1 and len(candidates)==1:
            pair=pair_metrics(controls[0],candidates[0])
            pair['equal_gpu_cache_bytes']=controls[0]['record']['gpu_kv_bytes']==candidates[0]['record']['gpu_kv_bytes']
            pair['exact_correctness_passed']=exact_comparison(controls[0],candidates[0],analyze)['passed']
            matched_pairs.append(pair)
        elif controls and candidates:
            errors.append({'pair_id':case['pair_id'],'error':'matched_pair_not_unique'})
    exact = []
    for candidate in items:
        if candidate['record']['role'] != 'candidate': continue
        for arc in items:
            if arc['record']['role'] not in ('arc_full', 'arc_small', 'original'): continue
            if (arc['record']['seed'], arc['record']['workload']) != (candidate['record']['seed'], candidate['record']['workload']): continue
            try: exact.append(exact_comparison(arc, candidate, analyze))
            except Exception as exc: errors.append({'candidate': candidate['record']['name'], 'baseline': arc['record']['name'], 'error': repr(exc)})
    groups = {role: pooled([i for i in items if i['record']['role'] == role], analyze) for role in ROLES if any(i['record']['role'] == role for i in items)}
    complete = len(items) == len(cases) and not missing and not errors and len(pairs) == len(expected_pairs) and bool(expected_pairs) and len(matched_pairs)==len(expected_matched)
    deltas = [p['output_tokens_s_delta_pct'] for p in pairs if finite(p['output_tokens_s_delta_pct'])]
    references = {role: {field + '_delta_pct': pct(groups['candidate'][field], values[field]) for field in ('output_tokens_s', 'cpu_s_sampled_mean_per_run', 'cpu_s_per_1000_output_tokens', 'ttft_mean_ms', 'ttft_p95_ms', 'return_ttft_mean_ms', 'return_ttft_p95_ms', 'mean_itl_ms', 'return_mean_itl_ms')} for role, values in groups.items() if role != 'candidate' and 'candidate' in groups}
    delta_ranges = {}
    for field in {k for pair in pairs for k in pair if k.endswith('_delta_pct')}:
        values = [p[field] for p in pairs if finite(p[field])]
        delta_ranges[field] = {'n': len(values), 'min': min(values) if values else None, 'max': max(values) if values else None}
    candidate_items = [i for i in items if i['record']['role'] == 'candidate']
    arc_exact = [c for c in exact if c['baseline_role'] in ('arc_full', 'arc_small')]
    correctness_passed = complete and all(i['record']['integrity']['passed'] for i in items) and bool(arc_exact) and all(c['passed'] for c in arc_exact)
    scoring_exercised = complete and bool(candidate_items) and all(i['record']['exercised_scoring'] for i in candidate_items)
    mode_verified = complete and bool(candidate_items) and all(
        i['record']['semantic_functionality']['checks']['arc_cache_policy'] and
        i['record']['semantic_functionality']['checks']['semantic_ordering_configured']
        for i in candidate_items)
    return {'schema_version': 2, 'plan_pin': pin(plan_path), 'analyzer_pin': pin(Path(__file__)),
            'helper_pin': pin(H / 'analyze.py'), 'plan': plan,
            'status': 'complete' if complete else 'partial_or_invalid',
            'complete': complete, 'planned_runs': len(cases), 'available_runs': len(items),
            'expected_primary_pairs': len(expected_pairs), 'available_primary_pairs': len(pairs),
            'missing_runs': missing, 'errors': errors, 'runs': [i['record'] for i in items],
            'pooled_by_role': groups, 'pooled_candidate_deltas_vs_reference': references,
            'primary_pairs': pairs, 'matched_budget_pairs': matched_pairs, 'primary_pair_delta_ranges': delta_ranges, 'exact_comparisons': exact,
            'criteria': {'all_matched_budget_pairs_present': complete and bool(expected_matched) and len(matched_pairs)==len(expected_matched),
                         'all_matched_budget_pairs_exact_and_equal_budgets': complete and bool(matched_pairs) and all(p['integrity_passed'] and p['exact_correctness_passed'] and p['equal_cpu_cache_bytes'] and p['equal_gpu_cache_bytes'] for p in matched_pairs),
                         'all_matched_budget_pairs_throughput_at_least_99pct_arc': complete and bool(matched_pairs) and all(p['throughput_at_least_99pct_arc'] for p in matched_pairs),
'all_planned_integrity_passed': complete and all(i['record']['integrity']['passed'] for i in items),
                         'all_matched_seed_arc_exact_correctness_passed': complete and bool(arc_exact) and all(c['passed'] for c in arc_exact),
                         'all_candidate_numerical_scoring_exercised': scoring_exercised,
                         'all_candidate_arc_cache_semantic_ordering_verified': mode_verified,
                         'confirmation_eligible_correctness_and_scoring': correctness_passed and scoring_exercised and mode_verified,
                         'all_candidate_native_prefetch_exercised': complete and bool(candidate_items) and all(i['record']['native_prefetch_exercised'] for i in candidate_items),
                         'any_candidate_constrained_semantic_selection_exercised': complete and any(i['record']['constrained_semantic_selection'] for i in candidate_items),
                         'all_candidate_constrained_semantic_selection_exercised': complete and bool(candidate_items) and all(i['record']['constrained_semantic_selection'] for i in candidate_items),
                         'candidate_runs_with_constrained_semantic_selection': sum(i['record']['constrained_semantic_selection'] for i in candidate_items),
                         'all_candidate_semantic_functionality_exercised': complete and bool(candidate_items) and all(i['record']['semantic_functionality']['passed'] for i in candidate_items),
                         'all_primary_pairs_exact_correctness_passed': complete and all(p['exact_correctness_passed'] for p in pairs),
                         'pooled_primary_throughput_at_least_99pct_arc': complete and references.get('arc_full', {}).get('output_tokens_s_delta_pct') is not None and references['arc_full']['output_tokens_s_delta_pct'] >= -1,
                         'all_primary_pairs_equal_cpu_cache_bytes': complete and all(p['equal_cpu_cache_bytes'] for p in pairs),
                         'all_primary_pairs_throughput_at_least_99pct_arc': complete and all(p['throughput_at_least_99pct_arc'] for p in pairs),
                         'pair_throughput_delta_pct_min': min(deltas) if deltas else None,
                         'pair_throughput_delta_pct_max': max(deltas) if deltas else None,
                         'latency_materiality': 'reported, no automatic materiality threshold assumed'},
            'definitions': {'throughput': 'sum successful output tokens / sum load elapsed seconds; no averaging of run rates',
                            'ttft': 'pooled request release-to-first-token latency; return subset role contains return',
                            'itl': 'sum token receipt spans / sum inter-token intervals',
                            'cpu': 'sampled owned process-tree CPU seconds; window overhang retained in run resources',
                            'memory': 'CUDA process allocator peaks and sampled whole-device NVML peaks; configured KV bytes are separate from actual allocation',
                            'candidate': 'ARC cache policy with semantic ordering; numerical evidence is real block EMA application or separate validated raw-prefix scalar application. Prefix scalars never fabricate blockscore counts.',
                            'exercised_scoring': 'positive capture calls, applied score keys, and numerically scored prefetch candidates; does not imply ranking changed allocation',
                            'constrained_semantic_selection': 'observed ARC cache and semantic selector with an actual semantic decision involving at least two scored candidates under a constrained shared budget; separate from numerical scoring and not itself proof of ranking quality',
                            'strict_semantic_functionality_passed': 'unchanged conjunction of every per-run semantic check, including constrained selection and native prefetch',
                            'confirmation_eligibility': 'complete planned correctness plus numerical scoring and verified ARC-cache/semantic-selector mode; throughput, latency, native-prefetch benefit, and constrained ranking quality remain separate criteria',
                            'threshold': 'at least 99% of matched ARC throughput, with each planned primary pair checked'},
            'limitations': ['Finite generated workload; descriptive repeated paired results, no statistical equivalence claim.',
                            'Paired controls use pair_id; extra controls and original references are seed-matched, not additional primary repetitions.',
                            'No terminal or physical-identity proof is waived for preempted requests.',
                            'All source/configuration/harness pins are retained in per-run identity and preserved raw evidence.',
                            'Missing or invalid planned runs prevent affirmative confirmation criteria; screens are excluded.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=H / 'confirmation.json')
    parser.add_argument('--output', type=Path, default=H / 'confirmation-results.json')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = build(args.plan.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    stem = args.output.with_suffix('')
    csvwrite(stem.with_name(stem.name + '-runs.csv'), report['runs'])
    csvwrite(stem.with_name(stem.name + '-pairs.csv'), report['primary_pairs'])
    csvwrite(stem.with_name(stem.name + '-matched-pairs.csv'), report['matched_budget_pairs'])
    csvwrite(stem.with_name(stem.name + '-pooled.csv'), [{'role': k, **v} for k, v in report['pooled_by_role'].items()])
    print(json.dumps({k: report[k] for k in ('status', 'planned_runs', 'available_runs', 'expected_primary_pairs', 'available_primary_pairs', 'missing_runs', 'errors', 'pooled_candidate_deltas_vs_reference', 'criteria')}, indent=2))
    return 2 if args.require_complete and not report['complete'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
