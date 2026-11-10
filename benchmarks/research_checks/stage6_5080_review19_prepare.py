"""Prepare one isolated confirmation review after terminal confirmation execution."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'output/stage6-5080-20261108/primary-execution-bundle-01/source'
PYTHON = '/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'


def main():
    assert Path.cwd() == SOURCE.resolve() and sys.executable == PYTHON and sys.dont_write_bytecode
    directory = Path(__file__).resolve().parent
    sys.path[:] = [str(SOURCE)] + [p for p in sys.path if Path(p).resolve() != directory]
    from unittest.mock import patch
    import hashlib
    import json
    from benchmarks.serving_suite import campaign_quality as q
    from benchmarks.serving_suite.campaign_contracts import load, digest, atomic_json, local_path
    from benchmarks.serving_suite.campaign_report import completed_blocks
    out = REPO / 'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
    base = REPO / 'output/stage6-5080-20261109'
    evidence = REPO / 'evidence/stage6-5080-20261109'
    assert out.parent.is_mount()
    whole = load(evidence / 'confirmation-attempts-complete.json')
    assert whole['execution_complete'] and whole['terminal_logical_configurations'] == 524
    assert whole['complete_matched_blocks'] == 207 and len(whole['missing_blocks']) == 3
    terminal = load(evidence / 'confirmation-ministral-attempts-complete.json')
    assert terminal['execution_complete'] and terminal['terminal_logical_configurations'] == 60
    assert terminal['complete_matched_blocks'] == 28 and terminal['required_matched_blocks'] == 30
    workspace, destination = base / 'quality-confirm-workspace-19', base / 'quality-confirm-checkpoint-19'
    assert not workspace.exists() and not destination.exists()
    live = out / 'analysis/quality'

    def inventory(p):
        return {str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest()
                for f in p.rglob('*') if f.is_file()} if p.exists() else {}

    before = inventory(live)
    selected = completed_blocks(out, 'confirm')
    assert len(selected) == 207
    assert len([k for k in selected if k[0] == 'ministral3_3b']) == 28
    with patch.object(q, 'completed_blocks', lambda ignored, phase:selected if phase == 'confirm' else None), \
         patch.object(q, 'local_path', lambda ignored, relative:local_path(out, relative)):
        summary = q.prepare_review(workspace, 'confirm')
    (workspace / 'analysis/quality').rename(destination)
    (workspace / 'analysis').rmdir()
    workspace.rmdir()
    assert inventory(live) == before
    indices = [load(evidence / 'screen-review-index-07.json')]
    indices += [load(evidence / f'confirm-review-index-{i:02}.json') for i in range(1, 19)]
    fields = indices[0]['signature_fields']
    known = {c['signature'] for index in indices for c in index['contexts']}
    cap_index = load(evidence / 'caps-all-review-index-08.json')
    cap_bindings = load(evidence / 'caps-instruction-bindings-08.json')
    assert cap_bindings['executed_system_instruction_digest'] == indices[0]['executed_system_instruction_digest']
    assert cap_bindings['executed_instruction_source_sha256'] == indices[0]['executed_instruction_source_sha256']
    cap_packet = load(Path(cap_index['raw_packet']))
    assert digest(cap_packet['items']) == cap_index['packet_digest']
    known.update(digest({k:x[k] for k in fields}) for x in cap_packet['items'])
    packet = load(destination / 'blind-packet.json')
    seen, novels = set(), []
    for x in packet['items']:
        signature = digest({k:x[k] for k in fields})
        if signature not in known and signature not in seen:
            novels.append(x)
        seen.add(signature)
    atomic_json(destination / 'novel-contexts.json', novels)
    assert Path(q.__file__).resolve().is_relative_to(SOURCE.resolve())
    scope = dict(phase='confirm', acquisition_complete=False, status='terminal confirmation execution; three genuine invalid blocks excluded', confirmation_execution_complete=True,
        objective_outputs=summary['objective_outputs'], selected_items=summary['blind_review_items'],
        matched_blocks=len(selected), distinct_contexts=len(seen), novel_full_contexts=len(novels),
        live_quality_directory_unchanged=True, quality_module_sha256=hashlib.sha256(Path(q.__file__).read_bytes()).hexdigest(),
        source_io_routing='Frozen completed_blocks snapshot and original sealed attempt paths; derived writes materialized only in this new isolated snapshot.',
        exact_reuse_inputs='screen07, union confirm01-through18 and fully reviewed all-attempt cap08; archived SYSTEM/source equality checked',
        cap08_packet_digest=cap_index['packet_digest'],
        scope='All60 Ministral configurations executed once;28/30 pairs selected and B06 repeats0/3 excluded. All129 valid Qwen3-8B blocks preserve selection. All50 Qwen3-4B blocks included. All524 confirmation identities are terminal; three original activity-invalid blocks remain excluded without retry or imputation. Overhead/operational acquisition is separate and may continue. New contexts unread/unjudged until explicit review. No protocol, measurement, objective or live-quality changes.')
    atomic_json(destination / 'acquisition-scope.json', scope)
    print(json.dumps(dict(scope=scope, summary={k:summary[k] for k in ('objective_outputs', 'blind_review_items', 'automatic_failures', 'capped_intermediate_outputs')},
        novel_kinds=[x['reference'].get('kind', 'investigation') for x in novels]), indent=2))


if __name__ == '__main__':
    main()
