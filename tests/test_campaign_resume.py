"""Resume, tamper and writer-contention checks using real filesystem artifacts."""
import pytest

from benchmarks.serving_suite.campaign_contracts import Ledger, atomic_json, digest, load, seal_tree


def finish(ledger, identity):
    attempt, directory = ledger.begin(identity)
    row = ledger.index()[attempt]
    atomic_json(directory / 'result.json', {'status': 'passed'})
    seal = seal_tree(directory)
    ledger.event(row['case_id'], attempt, 'verified', **row['details'], seal=seal)
    return attempt, directory


def test_resume_requires_exact_identity_and_seal(tmp_path):
    ledger = Ledger(tmp_path)
    identity = {'model': 'one', 'repeat': 1, 'source': 'a'}
    attempt, directory = finish(ledger, identity)
    assert ledger.reusable(identity)['attempt_id'] == attempt
    assert ledger.reusable({**identity, 'source': 'b'}) is None
    atomic_json(directory / 'result.json', {'status': 'failed'})
    with pytest.raises(ValueError, match='Changed artifact'):
        ledger.reusable(identity)


def test_interrupted_case_is_new_whole_attempt(tmp_path):
    ledger = Ledger(tmp_path)
    identity = {'model': 'one', 'repeat': 1}
    old, directory = ledger.begin(identity)
    ledger.recover()
    assert ledger.index()[old]['state'] == 'interrupted'
    assert ledger.reusable(identity) is None
    new, fresh = ledger.begin(identity)
    assert old != new and directory != fresh
    with pytest.raises(ValueError, match='immutable'):
        ledger.event(digest(identity), old, 'running')


def test_index_is_rebuildable_and_event_chain_detects_mutation(tmp_path):
    ledger = Ledger(tmp_path)
    finish(ledger, {'cell': 'B06'})
    (tmp_path / 'index.json').write_text('corrupt index')
    assert len(ledger.index()) == 1
    path = tmp_path / 'ledger.jsonl'
    path.write_text(path.read_text().replace('B06', 'B09'))
    with pytest.raises(ValueError, match='chain'):
        ledger.index()


def test_torn_tail_retained_before_recovery(tmp_path):
    ledger = Ledger(tmp_path)
    finish(ledger, {'cell': 'B06'})
    with (tmp_path / 'ledger.jsonl').open('ab') as stream:
        stream.write(b'{"torn":')
    with pytest.raises(ValueError, match='Torn'):
        ledger.index()
    assert len(ledger.recover()) == 1
    assert next(tmp_path.glob('torn-ledger-*.bin')).read_bytes() == b'{"torn":'


def test_competing_writer_cannot_recover_live_attempt(tmp_path):
    first, second = Ledger(tmp_path), Ledger(tmp_path)
    with first.writer():
        with pytest.raises(RuntimeError, match='active writer'):
            with second.writer():
                second.recover()
