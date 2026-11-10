"""Close analysis after portability; retain all genuine failed activity gates."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'output/stage6-5080-20261108/primary-execution-bundle-01/source'
VOLUME = REPO / 'output/stage6-5080-20261108/linux-volume-01'
OUT = VOLUME / 'primary-20261109'
EVIDENCE = REPO / 'evidence/stage6-5080-20261109'


def main():
    assert Path.cwd() == SOURCE.resolve() and VOLUME.is_mount()
    assert sys.dont_write_bytecode
    assert sys.executable == '/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
    sys.path[:] = [str(SOURCE)] + [p for p in sys.path
                                if Path(p).resolve() != Path(__file__).resolve().parent]
    from benchmarks.serving_suite import campaign_report as frozen
    from benchmarks.serving_suite.campaign import source_snapshot
    from benchmarks.serving_suite.campaign_contracts import Ledger, atomic_json, digest, file_hash, load, verify_tree
    assert Path(frozen.__file__).resolve().is_relative_to(SOURCE.resolve())
    target = EVIDENCE / 'final-results-01'
    receipt = EVIDENCE / 'completion-audit-01.json'
    assert not target.exists() and not receipt.exists()
    assert not OUT.with_name(OUT.name + '-portability-parked').exists()
    integrity = load(EVIDENCE / 'post-acquisition-audit-01.json')
    protocol = load(OUT / 'protocol.lock.json')
    ledger = Ledger(OUT)
    with ledger.writer():
        before = digest(ledger.events())
        assert before == integrity['stable_ledger_digest']
        assert len(ledger.index()) == 641
        assert source_snapshot() == protocol['source']
        proof = load(OUT / 'analysis/portability-proof.json')
        assert proof['complete'] and proof['parent_protocol_preserved']
        assert proof['relocated_comparisons_match_original']
        assert proof['offline_verification'] == integrity['verification']
        assert proof['parent_acquisition_source_preserved']
        assert proof['source_revision_limited_to_derived_analysis']
        assert proof['confirmation_blocked_until_hardware_preflight']
        assert proof['gpu_model_imports_forbidden']
        bundle_files = verify_tree(Path(proof['relocated_bundle']), proof['bundle_seal'])
        assert bundle_files == proof['bundle_files']
        report = frozen.report(OUT)
        completion = report['completion']
        assert not report['complete'] and not report['throughput_acquisition_complete']
        assert not completion['complete'] and not completion['throughput']
        assert completion['quality_audit_complete']
        assert all(completion[k]['complete'] for k in ('overhead', 'operational', 'recovery', 'portability'))
        missing = {(b['model'], b['cell'], b['repeat']) for b in report['missing_blocks']}
        assert missing == {('qwen3_8b_awq', 'B06', 0), ('ministral3_3b', 'B06', 0), ('ministral3_3b', 'B06', 3)}
        assert len(completion['overhead']['controls']) == 38
        assert len(completion['operational']['cases']) == 6
        assert digest(report['comparisons']) == proof['offline_report']['comparisons_digest']
        helper = REPO / 'benchmarks/research_checks/stage6_primary_findings.py'
        spec = importlib.util.spec_from_file_location('final_presentation', helper)
        presentation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(presentation)
        data = presentation.summarize(report, retain_incomplete=True)
        previous = load(EVIDENCE / 'primary-confirmation-results-01/comparisons.json')
        assert len(data['comparisons']) == 42 and data['comparisons'] == previous['comparisons']
        data['incompleteness'] = (
            'All planned phases and audits have finished. The full protocol failed its throughput '
            'requirements: three original B06 activity-invalid blocks remain excluded. These '
            'conditional estimates retain actual pair counts and missing identities, with no '
            'activity retry, imputation, repeat reduction or forced completion.')
        data['presentation_subset'] = dict(final_execution_and_audits=True, protocol_passed=False,
            full_planned_comparisons=42, displayed_comparisons=42,
            scope='All terminal conditional comparisons; no aggregate across different workloads.')
        target.mkdir()
        atomic_json(target / 'comparisons.json', data)
        (target / 'tables.md').write_text(presentation.markdown(data))
        encoded = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
        (target / 'comparisons.html').write_text(presentation.HTML.replace('__DATA__', encoded))
        atomic_json(EVIDENCE / 'portability-final-03.json', proof)
        bindings = ['post-acquisition-audit-01.json', 'confirmation-attempts-complete.json',
            'overhead-boundary-01.json', 'operational-boundary-01.json',
            'confirm-quality-published-19.json', 'confirm-review-validation-19.json',
            'caps-all-review-index-final-01.json', 'cap-review-validation-final-01.json',
            'portability-final-03.json', 'portability-copy-repair-03.json', 'volume-growth-01.json']
        assert digest(ledger.events()) == before and source_snapshot() == protocol['source']
        atomic_json(receipt, dict(schema='stage6-5080-final-analysis-closeout-v1', recorded_unix=time.time(),
            all_planned_phase_execution_complete=True, all_audits_complete=True,
            full_protocol_passed=False, reason='Three genuine B06 activity-invalid blocks, preserved without replacement',
            frozen_report_complete=report['complete'], frozen_throughput_complete=report['throughput_acquisition_complete'],
            completion_flags={k: completion[k] if k in ('complete', 'throughput', 'quality_audit_complete')
                              else completion[k]['complete'] for k in completion if k != 'scope'},
            missing_blocks=report['missing_blocks'], retained_attempts=641, verified_files=67120,
            stable_ledger_digest=before, frozen_source=protocol['source']['digest'], frozen_source_files=2192,
            comparison_count=42, all_42_rows_exactly_reproduce_terminal_measurements=True,
            all_32_anchor_rows_previously_reproduced_exactly=True,
            report=str(OUT / 'analysis/current/results.json'), report_digest=digest(report),
            report_code_sha256=file_hash(Path(frozen.__file__)), presentation_sha256=file_hash(helper),
            findings_digest=digest(data), findings_files={str(p.relative_to(target)):file_hash(p)
                                                       for p in sorted(target.iterdir())},
            audits={name:file_hash(EVIDENCE / name) for name in bindings},
            original_campaign_restored=True, source_and_attempt_ledger_unchanged=True,
            final_render_push_cleanup_and_notification_pending=True,
            scope='Analysis and transfer closure only; no change to measured artifacts, design, estimator, validity, source judgments or exclusions. No quality equivalence or production qualification.'))
        print(json.dumps(dict(comparisons=42, missing=report['missing_blocks'],
                              completion=load(receipt)['completion_flags']), indent=2))


if __name__ == '__main__':
    main()
