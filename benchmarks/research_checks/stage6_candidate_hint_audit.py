"""Audit retained candidate hint delivery/effects without executing serving code."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import statistics

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
DEST = REPO / 'evidence/stage6-5080-20261109/candidate-audit-01'
SOURCE = REPO / 'output/stage6-5080-20261108/primary-execution-bundle-01/source'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    assert OUT.is_dir() and OUT.parent.is_mount()
    target = DEST / 'hint-flow.json'
    assert not target.exists()
    ledger = sha(OUT / 'ledger.jsonl')
    report_path = OUT / 'analysis/current/results.json'
    report_sha = sha(report_path)
    report = load(report_path)
    selected = {p[role]['attempt'] for r in report['comparisons'] for p in r['points']
                for role, system in [('candidate', r['candidate']), ('baseline', r['baseline'])] if system == 'wait'}
    assert len(selected) == 207
    events = {}
    for line in (OUT / 'ledger.jsonl').read_text().splitlines():
        event = json.loads(line)
        events[event['attempt_id']] = event
    records, skipped, source_checks = [], [], {}
    for event in events.values():
        identity = event['details']['identity']
        system, phase = identity.get('system'), identity.get('phase')
        if system != 'wait' and not (phase == 'overhead' and system in ('O1', 'O2')):
            continue
        folder = OUT / event['details']['path']
        path = folder / 'measurement/result.json'
        if not path.exists():
            skipped.append(dict(attempt=event['attempt_id'], state=event['state'], identity=identity,
                                reason='No completed measurement result; preserved setup/interrupted attempt'))
            continue
        seal = load(folder / 'seal.json')
        assert digest(seal) == event['details']['seal']
        assert sha(path) == seal['files']['measurement/result.json']
        cc = load(path)
        if not cc.get('cases'):
            skipped.append(dict(attempt=event['attempt_id'], state=event['state'], reason='Empty cases'))
            continue
        c = cc['cases'][0]
        options = c.get('configuration', {}).get('kv_connector_extra_config', {})
        hints = c.get('hints', [])
        reasons, statuses, http = Counter(), Counter(), Counter()
        positive, zero, touches, replay, inconsistent, late_positive, nonzero_gpu, errors = 0, 0, 0, 0, 0, 0, 0, 0
        latencies, sessions, positive_sessions, ids, flags = [], set(), set(), set(), set()
        for h in hints:
            raw, response = h.get('hint', {}), h.get('response', {})
            reason = response.get('reason', 'missing_response')
            reasons[reason] += 1
            statuses[response.get('status', 'missing')] += 1
            http[str(h.get('http_status', 'missing'))] += 1
            errors += bool(h.get('error'))
            sid, eid = raw.get('session_id'), raw.get('event_id')
            sessions.add(sid)
            if eid in ids:
                flags.add('repeated_event_id')
            ids.add(eid)
            replay += bool(response.get('idempotent_replay'))
            if response.get('event_id') not in (None, eid):
                inconsistent += 1
            if reason == 'cpu_prefix_recency_refreshed':
                ready = response.get('ready_cpu_prefix_keys', 0)
                assert type(ready) is int and ready >= 0
                positive += ready > 0
                zero += ready == 0
                touches += ready
                if ready > 0:
                    positive_sessions.add(sid)
                    late_positive += response['observed_at_ns'] >= response['nomination_deadline_ns']
            nonzero_gpu += bool(response.get('reserved_gpu_blocks', 0) or response.get('speculative_transfer_bytes', 0))
            if type(h.get('latency_s')) in (int, float):
                latencies.append(h['latency_s'])
            elif type(h.get('ended_ns')) is int:
                latencies.append((h['ended_ns'] - h['started_ns']) / 1e9)
        native = c.get('campaign', {}).get('native_cache') or {}
        snapshots = c.get('campaign', {}).get('registry_snapshots', [])
        records.append(dict(attempt=event['attempt_id'], phase=phase, model=identity['model'], cell=identity['cell'],
            repeat=identity['repeat'], system=system, state=event['state'], selected_confirmation=event['attempt_id'] in selected,
            case_evaluation=c.get('evaluation'), artifact_path=event['details']['path'], seal=event['details']['seal'], result_sha256=sha(path),
            requests=c.get('summary', {}).get('completed'), hint_events=len(hints), hint_sessions=len(sessions),
            positive_refresh_events=positive, zero_ready_refresh_events=zero, positive_hint_sessions=len(positive_sessions),
            repeated_key_touches=touches, unique_event_ids=len(ids), idempotent_replay_responses=replay,
            response_reasons=dict(reasons), response_statuses=dict(statuses), http_statuses=dict(http),
            delivery_errors=errors, inconsistent_event_acknowledgments=inconsistent, late_positive_refreshes=late_positive,
            hints_with_gpu_reservation_or_speculative_bytes=nonzero_gpu,
            hint_latency_s=dict(count=len(latencies), mean=statistics.mean(latencies), maximum=max(latencies)) if latencies else None,
            native_cache=native,
            activity=c.get('activity'),
            config={k: options.get(k, default) for k, default in [('wait_hint_cpu_retention_only', False),
                ('wait_hint_cpu_retention_gpu_gate', False), ('wait_hint_cpu_retention_gpu_aware', False),
                ('wait_hint_arc_advice', None), ('preservation_audit', True),
                ('prefetch_budget_fraction', None), ('observed_session_registry_size', 4096)]},
            registry_session_limit=sorted({s.get('session_limit') for s in snapshots}, key=str),
            registry_max_sessions=max((s['sessions'] for s in snapshots), default=None),
            operational_advice_checks=c.get('campaign', {}).get('operational_advice_checks'), flags=sorted(flags)))
    grouped = {}
    for r in records:
        if not r['selected_confirmation']:
            continue
        key = r['model'] + ':' + r['cell']
        g = grouped.setdefault(key, dict(cases=0, hints=0, positive_refreshes=0, zero_ready_refreshes=0,
            repeated_key_touches=0, delivery_errors=0, no_cpu_restore_cases=0, hint_sessions=0, positive_hint_sessions=0,
            reasons=Counter(), http=Counter()))
        g['cases'] += 1
        for source, field in [('hint_events', 'hints'), ('positive_refresh_events', 'positive_refreshes'),
            ('zero_ready_refresh_events', 'zero_ready_refreshes'), ('repeated_key_touches', 'repeated_key_touches'),
            ('delivery_errors', 'delivery_errors'), ('hint_sessions', 'hint_sessions'), ('positive_hint_sessions', 'positive_hint_sessions')]:
            g[field] += r[source]
        g['no_cpu_restore_cases'] += r['native_cache'].get('loaded_tokens') == 0
        g['reasons'].update(r['response_reasons'])
        g['http'].update(r['http_statuses'])
    relevant = ['semantic_offload/connector.py', 'semantic_offload/manager.py', 'semantic_offload/wait_hint.py',
        'semantic_offload/spec.py', 'benchmarks/serving_suite/campaign_workload.py',
        'benchmarks/serving_suite/campaign_validity.py', 'benchmarks/serving_suite/campaign_completion.py']
    for name in relevant:
        source_checks[name] = dict(current=sha(REPO / name), frozen=sha(SOURCE / name))
        if name == 'benchmarks/serving_suite/campaign_workload.py':
            def ordinary(text):
                return text.split('owner = rid\n', 1)[1].split("elif publish and plan.get('operational_advice_probes')", 1)[0]
            assert ordinary((REPO / name).read_text()) == ordinary((SOURCE / name).read_text())
            source_checks[name]['ordinary_hint_branch_equal'] = True
            source_checks[name]['authority'] = 'Frozen source; mutable module additionally contains historical addendum arrival/advice branches'
        else:
            assert source_checks[name]['current'] == source_checks[name]['frozen']
    assert sha(OUT / 'ledger.jsonl') == ledger and sha(report_path) == report_sha
    totals = dict(measured_cases=len(records), selected_confirmation_cases=sum(r['selected_confirmation'] for r in records),
        hint_events=sum(r['hint_events'] for r in records),
        selected_hint_events=sum(r['hint_events'] for r in records if r['selected_confirmation']),
        selected_positive_refresh_events=sum(r['positive_refresh_events'] for r in records if r['selected_confirmation']),
        selected_zero_ready_events=sum(r['zero_ready_refresh_events'] for r in records if r['selected_confirmation']),
        selected_delivery_errors=sum(r['delivery_errors'] for r in records if r['selected_confirmation']),
        inconsistent_event_acks=sum(r['inconsistent_event_acknowledgments'] for r in records),
        late_positive_refreshes=sum(r['late_positive_refreshes'] for r in records),
        hints_with_gpu_side_effects=sum(r['hints_with_gpu_reservation_or_speculative_bytes'] for r in records))
    data = dict(schema='stage6-candidate-hint-flow-audit-v1', totals=totals, confirmation_groups=grouped,
        cases=records, incomplete_setup_attempts=skipped, source_checks=source_checks,
        original_ledger_sha256=ledger, original_report_sha256=report_sha, originals_unchanged=True,
        scope='Retained local5080 measurements. Positive key touches are repeated recency operations, not unique retained blocks or useful recomputation avoided. No GPU or CPU tool workload executed. Historical2080 evidence is assessed separately.')
    target.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(totals, indent=2))
    for name, g in grouped.items():
        if g['hints']:
            print(name, g['cases'], 'hints', g['hints'], 'positive', g['positive_refreshes'], 'zero', g['zero_ready_refreshes'],
                  'sessions', g['positive_hint_sessions'], '/', g['hint_sessions'], 'reasons', dict(g['reasons']))


if __name__ == '__main__':
    main()
