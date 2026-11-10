"""Reconcile the finished anchor model without touching live acquisition."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'output/stage6-5080-20261108/primary-execution-bundle-01/source'
OUT = REPO / 'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
EVIDENCE = REPO / 'evidence/stage6-5080-20261109'
PYTHON = '/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
MODEL = 'qwen3_8b_awq'


def main():
    assert Path.cwd() == SOURCE.resolve() and sys.executable == PYTHON
    assert sys.dont_write_bytecode, 'Set PYTHONDONTWRITEBYTECODE=1 before running this audit'
    assert OUT.parent.is_mount()
    destination = EVIDENCE / 'confirmation-qwen8b-attempts-complete.json'
    assert not destination.exists(), 'Preserve the existing inventory receipt'
    # Absolute script execution must not let research_checks/http.py shadow the
    # standard library's http package when the frozen runner imports metrics.
    script_directory = Path(__file__).resolve().parent
    sys.path[:] = [str(SOURCE)] + [p for p in sys.path
                                 if Path(p).resolve() != script_directory]
    from benchmarks.serving_suite import campaign_contracts as contracts
    from benchmarks.serving_suite.campaign import source_snapshot
    from benchmarks.serving_suite.campaign_report import completed_blocks
    from benchmarks.serving_suite.campaign_validity import measurement_validity
    assert Path(contracts.__file__).resolve().is_relative_to(SOURCE.resolve())
    load, digest = contracts.load, contracts.digest
    ledger = contracts.Ledger(OUT)
    protocol = load(OUT / 'protocol.lock.json')
    assert source_snapshot()['digest'] == protocol['source']['digest']

    def anchor_entries():
        return [e for e in ledger.index().values()
                if e['details']['identity']['phase'] == 'confirm'
                and e['details']['identity']['model'] == MODEL]

    def logical(identity):
        return tuple(identity[k] for k in ('model', 'cell', 'repeat', 'system'))

    entries = anchor_entries()
    snapshot = digest(entries)
    planned = {(b['model'], b['cell'], b['repeat'], system)
               for b in protocol['schedule'] if b['model'] == MODEL
               for system in b['systems']}
    assert len(planned) == 364
    assert {logical(e['details']['identity']) for e in entries} == planned
    assert all(e['state'] in ('verified', 'failed', 'interrupted') for e in entries)
    repair = load(EVIDENCE / 'confirmation-host-interruption-02.json')
    path_repair = load(EVIDENCE / 'confirmation-tool-path-repair-03.json')
    activity_path = ('attempts/adf22cac52dee5eadff54273061da531bcb8073e28cbea8e950774c9124042a5/'
                     '9fd044a0861b4dc7b797f92d29a264fa')
    known_invalid = {activity_path, repair['interrupted_path']}
    known_invalid.update(e['path'] for e in path_repair['affected_attempts'])
    rows, valid_identities, invalid_paths = [], set(), set()
    for event in entries:
        details = event['details']
        identity = details['identity']
        assert identity['source'] == protocol['source']['digest']
        directory = OUT / details['path']
        verified = contracts.verify_tree(directory, details['seal'])
        validity = measurement_validity(directory, event)
        if validity['valid']:
            valid_identities.add(logical(identity))
        else:
            invalid_paths.add(details['path'])
            assert details['path'] in known_invalid, 'Investigate a new invalid attempt'
        result = directory / 'measurement/result.json'
        case = load(result)['cases'][0] if result.is_file() else None
        if validity['valid']:
            assert case and case['cleanup']['verified'] and not case['summary']['failed']
            assert case['summary']['completed'] == case['expected_requests']
        rows.append(dict(attempt=event['attempt_id'], case_id=event['case_id'],
            identity={k:identity[k] for k in ('model', 'cell', 'repeat', 'system')},
            block_id=details['block_id'], state=event['state'], path=details['path'],
            seal=details['seal'], verified_files=verified, measurement_validity=validity,
            completed_requests=case['summary']['completed'] if case else None,
            cleanup_verified=case['cleanup']['verified'] if case else None))
    assert invalid_paths == known_invalid
    assert planned - valid_identities == {(MODEL, 'B06', 0, 'wait')}
    b06 = [e for e in entries if e['details']['identity']['cell'] == 'B06']
    assert len(b06) == 32 and len({logical(e['details']['identity']) for e in b06}) == 32
    selected = {k:v for k,v in completed_blocks(OUT, 'confirm').items() if k[0] == MODEL}
    required = {(b['model'], b['cell'], b['repeat'])
                for b in protocol['schedule'] if b['model'] == MODEL}
    assert required - set(selected) == {(MODEL, 'B06', 0)}
    assert sum(k[1] == 'B12' for k in selected) == 12
    assert digest(anchor_entries()) == snapshot, 'Anchor inventory changed during verification'
    receipt = dict(schema='stage6-5080-terminal-model-inventory-v1', recorded_unix=time.time(),
        model=MODEL, execution_complete=True, acquisition_complete=False,
        source_digest=protocol['source']['digest'], schedule_digest=digest(protocol['schedule']),
        planned_configurations=len(planned), terminal_logical_configurations=len(planned),
        valid_logical_configurations=len(valid_identities), retained_attempts=len(rows),
        valid_raw_attempts=sum(r['measurement_validity']['valid'] for r in rows),
        selected_matched_attempts=sum(len(v['systems']) for v in selected.values()),
        states=dict(Counter(r['state'] for r in rows)),
        verified_files=sum(r['verified_files'] for r in rows),
        required_matched_blocks=len(required), complete_matched_blocks=len(selected),
        missing_blocks=[dict(model=m, cell=c, repeat=r) for m,c,r in sorted(required-set(selected))],
        b06_attempts=len(b06), b06_activity_retry=False, anchor_inventory_stable=True,
        anchor_inventory_digest=snapshot, attempts=rows,
        scope='All frozen Qwen3-8B confirmation identities executed and every retained outer seal verified. '
              'The first B06 candidate activity failure remains excluded; no favorable-activity retry or '
              'repeat reduction. Whole-block setup replacements use the frozen selector; valid unmatched '
              'original arms receive no matched timing credit. Secondary models and later phases continue. '
              'This is terminal execution for one model, not a complete campaign or performance claim.')
    contracts.atomic_json(destination, receipt)
    print({k:v for k,v in receipt.items() if k not in ('attempts',)})


if __name__ == '__main__':
    main()
