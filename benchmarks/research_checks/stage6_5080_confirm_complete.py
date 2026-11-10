"""Reconcile terminal confirmation without retrying invalid activity cases.

Run this absolute script from the frozen acquisition source with the pinned
interpreter and PYTHONDONTWRITEBYTECODE=1. --check-ready is read-only and exits
2 while acquisition is pending. The full audit refuses to overwrite evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'output/stage6-5080-20261108/primary-execution-bundle-01/source'
OUT = REPO / 'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
EVIDENCE = REPO / 'evidence/stage6-5080-20261109'
PYTHON = '/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
UNIT = 'stage6-5080-confirm-20261109-03.service'


def logical(identity):
    return tuple(identity[k] for k in ('model', 'cell', 'repeat', 'system'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-ready', action='store_true')
    args = parser.parse_args()
    assert Path.cwd() == SOURCE.resolve() and sys.executable == PYTHON
    assert sys.dont_write_bytecode and OUT.parent.is_mount()
    # Prevent the neighboring research_checks/http.py from shadowing stdlib.
    directory = Path(__file__).resolve().parent
    sys.path[:] = [str(SOURCE)] + [p for p in sys.path if Path(p).resolve() != directory]
    from benchmarks.serving_suite import campaign_contracts as contracts
    from benchmarks.serving_suite.campaign import identity_for, source_snapshot
    from benchmarks.serving_suite.campaign_report import completed_blocks
    from benchmarks.serving_suite.campaign_validity import measurement_validity
    from benchmarks.serving_suite.campaign_workload import request_keys
    from benchmarks.serving_suite.verify import verify
    assert Path(contracts.__file__).resolve().is_relative_to(SOURCE.resolve())
    load, digest = contracts.load, contracts.digest
    ledger = contracts.Ledger(OUT)
    protocol = contracts.read_protocol(OUT)
    planned = {(b['model'], b['cell'], b['repeat'], s)
               for b in protocol['schedule'] for s in b['systems']}
    required = {(b['model'], b['cell'], b['repeat']) for b in protocol['schedule']}
    assert len(planned) == 524 and len(required) == 210

    def entries():
        return [e for e in ledger.index().values() if e['details']['identity']['phase'] == 'confirm']

    current = entries()
    identities = {logical(e['details']['identity']) for e in current}
    assert identities <= planned, 'Unexpected confirmation identity'
    active = [e['attempt_id'] for e in current if e['state'] in ('planned', 'running')]
    raw = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
        '--property=ActiveState,SubState,MainPID,Result,ExecMainStatus'], text=True)
    service = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
    ready = (identities == planned and not active and service.get('MainPID') == '0'
             and service.get('SubState') == 'exited' and service.get('Result') == 'success'
             and service.get('ExecMainStatus') == '0')
    readiness = dict(ready=ready, planned_configurations=524,
        observed_logical_configurations=len(identities), pending_attempts=active,
        retained_attempts=len(current), service=service)
    print(json.dumps(readiness), flush=True)
    if args.check_ready or not ready:
        return 0 if ready else 2

    destination = EVIDENCE / 'confirmation-attempts-complete.json'
    assert not destination.exists(), 'Preserve the existing terminal inventory'
    source = source_snapshot()
    assert source == protocol['source']
    cells = {(c['model'], c['cell']):c for c in load(OUT / 'resolved.json')['cells']}
    expected_identities = {key:identity_for(OUT, 'confirm', cells[key[:2]], key[2], key[3], source)
                           for key in planned}
    anchor = load(EVIDENCE / 'confirmation-qwen8b-attempts-complete.json')
    anchor_ids = {r['attempt'] for r in anchor['attempts']}
    known_invalid = {r['path'] for r in anchor['attempts'] if not r['measurement_validity']['valid']}
    activity_paths = set()
    for p in EVIDENCE.glob('confirmation-ministral-b06-inactive-restore-*.json'):
        activity_paths.update(r['path'] for r in load(p)['attempts'] if r['identity']['system'] == 'wait')
    assert len(activity_paths) == 2
    known_invalid.update(activity_paths)
    missing_logical = {('qwen3_8b_awq', 'B06', 0, 'wait'),
                       ('ministral3_3b', 'B06', 0, 'wait'),
                       ('ministral3_3b', 'B06', 3, 'wait')}
    with ledger.writer():
        current = entries()
        snapshot = digest(current)
        assert all(e['state'] in ('verified', 'failed', 'interrupted') for e in current)
        assert {e['attempt_id'] for e in current if e['details']['identity']['model'] == 'qwen3_8b_awq'} == anchor_ids
        assert len(current) == len(anchor_ids) + 160
        rows, valid_identities, invalid_paths = [], set(), set()
        for event in current:
            details, identity = event['details'], event['details']['identity']
            key = logical(identity)
            assert identity == expected_identities[key]
            attempt = contracts.local_path(OUT, details['path'])
            assert load(attempt / 'identity.json') == identity
            sealed_files = contracts.verify_tree(attempt, details['seal'])
            validity = measurement_validity(attempt, event)
            measured = attempt / 'measurement'
            result = load(measured / 'result.json') if (measured / 'result.json').is_file() else None
            case = result['cases'][0] if result else None
            inner = verify(measured) if (measured / 'artifacts.json').is_file() else None
            if validity['valid']:
                valid_identities.add(key)
            else:
                invalid_paths.add(details['path'])
                assert details['path'] in known_invalid, 'Investigate a new invalid attempt'
            coverage = False
            if validity['valid'] or key in missing_logical:
                assert case and inner and case['cleanup']['verified'] and not case['summary']['failed']
                assert case['summary']['completed'] == case['expected_requests']
                corpus = load(measured / 'corpus.json')
                waves = result['plan']['waves']
                expected = {tuple(k) for w in range(waves) for k in request_keys(corpus['sessions'], w)}
                requests = [json.loads(line) for line in (measured / case['name'] / 'requests.jsonl').read_text().splitlines()]
                actual = [(r['wave'], r['source_id'], r['turn']) for r in requests]
                assert len(actual) == len(set(actual)) == len(expected) and set(actual) == expected
                assert all(r['success'] and r['terminal_complete'] and not r.get('cancel_probe') for r in requests)
                workflow = case['campaign']
                assert not workflow['measurement_errors']
                assert workflow['completed_units'] == workflow['offered_units'] == len(corpus['sessions']) * waves
                assert workflow['tools'] == waves * sum(n['kind'] == 'tool' for s in corpus['sessions'] for n in s['graph']['nodes'])
                coverage = True
            rows.append(dict(attempt=event['attempt_id'], case_id=event['case_id'],
                identity={k:identity[k] for k in ('model', 'cell', 'repeat', 'system')},
                path=details['path'], block_id=details['block_id'], state=event['state'],
                seal=details['seal'], verified_files=sealed_files, inner_verification=inner,
                measurement_validity=validity, full_declared_graph_tools_units_verified=coverage,
                completed_requests=case['summary']['completed'] if case else None,
                length_caps=case['summary']['finish_reasons'].get('length', 0) if case else None,
                cleanup_verified=case['cleanup']['verified'] if case else None))
        assert invalid_paths == known_invalid and planned - valid_identities == missing_logical
        for model, count in [('qwen3_8b_awq', 32), ('ministral3_3b', 12)]:
            b06 = [e for e in current if (e['details']['identity']['model'], e['details']['identity']['cell']) == (model, 'B06')]
            assert len(b06) == len({logical(e['details']['identity']) for e in b06}) == count
        selected = completed_blocks(OUT, 'confirm', source=source['digest'])
        missing_blocks = required - set(selected)
        assert missing_blocks == {k[:3] for k in missing_logical}
        selected_ids = {r['event']['attempt_id'] for b in selected.values() for r in b['systems'].values()}
        for row in rows:
            row['selected_for_matched_comparison'] = row['attempt'] in selected_ids
        assert digest(entries()) == snapshot and source_snapshot() == source
        receipt = dict(schema='stage6-5080-terminal-confirmation-inventory-v1', recorded_unix=time.time(),
            execution_complete=True, acquisition_complete=False, phase='confirm',
            source_digest=source['digest'], protocol_digest=digest(protocol),
            planned_configurations=524, terminal_logical_configurations=524,
            valid_logical_configurations=len(valid_identities), retained_attempts=len(rows),
            valid_raw_attempts=sum(r['measurement_validity']['valid'] for r in rows),
            selected_matched_attempts=len(selected_ids), states=dict(Counter(r['state'] for r in rows)),
            verified_files=sum(r['verified_files'] for r in rows), required_matched_blocks=210,
            complete_matched_blocks=len(selected), missing_blocks=[dict(model=m, cell=c, repeat=r) for m,c,r in sorted(missing_blocks)],
            activity_retry=False, inventory_stable=True, ledger_confirmation_digest=snapshot,
            service_terminal_state=service, helper_sha256=contracts.file_hash(Path(__file__)), attempts=rows,
            scope='All524 frozen confirmation configurations executed; all retained outer/inner seals and valid or activity-invalid request graphs/tools/units/cleanup verified. Three known activity failures remain excluded, with no retry, imputation, relabeling or repeat reduction. Historical setup replacements remain retained and selected only through the frozen whole-block selector. Overhead/operational and final audits still required.')
        contracts.atomic_json(destination, receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k != 'attempts'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
