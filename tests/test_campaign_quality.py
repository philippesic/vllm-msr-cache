import pytest

from benchmarks.serving_suite.campaign_contracts import atomic_json,digest
from benchmarks.serving_suite.campaign_quality import RUBRIC,finish_review,objective


def test_qa_requires_both_format_and_reference():
    reference = dict(kind='answer_json',answers=['The Rhine'])
    assert not objective('{"answer":"Rhine"}',reference)['automatic_failure']
    assert objective('The Rhine',reference)['automatic_failure']
    assert objective('{"answer":"Seine"}',reference)['automatic_failure']
    assert objective('{"answer":"Rhine","extra":1}',reference)['automatic_failure']
    assert objective('{"answer":"Seine","answer":"Rhine"}',reference)['automatic_failure']


def test_investigation_does_not_call_pass_echo_useful():
    result = objective('groupby test_status=PASS',dict(kind='investigation',symbol='groupby'))
    assert not result['automatic_failure']
    assert result['source_review_required']
    assert 'useful' not in result
    formatted = objective('groupby **Test Status**: PASS',dict(kind='investigation',symbol='groupby'))
    assert formatted['actual_validation_status_stated']
    assert not formatted['requested_status_format']
    assert formatted['automatic_failure']


def test_objective_groups_retain_bad_outputs_and_actual_denominators():
    from benchmarks.serving_suite.campaign_quality import aggregate_objective
    ref = dict(kind='answer_json',answers=['Rhine'])
    items = [dict(model='m',cell='B02',system='wait',attempt='a',
        objective=objective(text,ref,capped=cap)) for text,cap in
        [('{"answer":"Rhine"}',False),('{"answer":"Seine"}',True)]]
    group = aggregate_objective(items,dict(a=10))['m:B02:wait']
    assert group['final_outputs']==2 and group['automatic_failures']==1 and group['length_capped']==1
    assert group['observed_checks']['reference_exact']==dict(passed=1,observed=2)
    assert group['processes']['a']['mechanical_passes_per_s']==.1
    assert 'not source-audited usefulness' in group['scope']


def test_audit_requires_every_blind_item_and_source_notes(tmp_path):
    destination = tmp_path/'analysis/quality'
    atomic_json(destination/'blind-packet.json',dict(items=[dict(review_id='one'),dict(review_id='two')]))
    ratings = dict(schema='stage6-source-audit-v1',reviewer='test',rubric_digest=digest(RUBRIC),ratings=[
        dict(review_id='one',source_supported=True,instruction_followed=True,useful=True,unsupported_claims=[],notes='source lines support claim')])
    path = tmp_path/'ratings.json'
    atomic_json(path,ratings)
    with pytest.raises(ValueError,match='complete sample'):
        finish_review(tmp_path,path)
