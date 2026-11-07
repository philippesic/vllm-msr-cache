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


def test_capped_finals_all_require_review_beyond_sample_limit(tmp_path,monkeypatch):
    import json
    from benchmarks.serving_suite import campaign_quality as quality
    measurement = tmp_path/'attempt/measurement'
    sessions = [dict(id=str(i),graph=dict(outputs=['final']),quality_reference=dict(kind='summary'),
        context='Full source',turns=['Summarize the source']) for i in range(40)]
    atomic_json(measurement/'corpus.json',dict(sessions=sessions))
    rows = [dict(source_id=str(i),node_id='final',generated_text='A nonempty capped summary',
        finish_reason='length',payload=dict(request_id=str(i)),hint_group='unhinted') for i in range(40)]
    (measurement/'case').mkdir()
    (measurement/'case/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    data = dict(event=dict(attempt_id='attempt',details=dict(path='attempt')),
        case=dict(name='case',load_duration_s=10))
    monkeypatch.setattr(quality,'completed_blocks',lambda out,phase:{('model','B03',0):dict(systems=dict(arc=data))})
    result = quality.prepare_review(tmp_path)
    assert result['objective_outputs']==40 and result['automatic_failures']==0
    assert result['blind_review_items']==40


def test_capped_intermediate_review_uses_only_available_evidence(tmp_path,monkeypatch):
    import json
    from benchmarks.serving_suite import campaign_quality as quality
    measurement = tmp_path/'attempt/measurement'
    nodes = [dict(id='r0',kind='request',depends_on=[],prompt='Plan'),
        dict(id='t0',kind='tool',depends_on=['r0']),
        dict(id='r1',kind='request',depends_on=['t0'],prompt='Explain search evidence'),
        dict(id='t1',kind='tool',depends_on=['r1']),
        dict(id='r2',kind='request',depends_on=['t1'],prompt='Report validated result')]
    session = dict(id='one',graph=dict(outputs=['r2'],nodes=nodes),
        quality_reference=dict(kind='investigation',symbol='subject'),context='Source',turns=['Plan','Explain','Report'])
    atomic_json(measurement/'corpus.json',dict(sessions=[session]))
    rows = [dict(source_id='one',node_id=node,generated_text=text,finish_reason=reason,
        payload=dict(request_id=node),hint_group='hinted') for node,text,reason in
        [('r0','Earlier plan','stop'),('r1','Truncated explanation','length'),('r2','subject test_status=PASS','stop')]]
    (measurement/'case/tools').mkdir(parents=True)
    (measurement/'case/tools/0-0-t0.txt').write_text('Search evidence')
    (measurement/'case/tools/0-0-t1.txt').write_text('Future validation')
    (measurement/'case/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    data = dict(event=dict(attempt_id='attempt',details=dict(path='attempt')),
        case=dict(name='case',load_duration_s=10))
    monkeypatch.setattr(quality,'completed_blocks',lambda out,phase:{('model','B06',0):dict(systems=dict(arc=data))})
    result = quality.prepare_review(tmp_path)
    assert result['objective_outputs']==1 and result['capped_intermediate_outputs']==1
    assert result['blind_review_items']==2 and result['automatic_failures']==0
    packet = json.loads((tmp_path/'analysis/quality/blind-packet.json').read_text())
    item = next(i for i in packet['items'] if i['response_role']=='capped_intermediate')
    assert item['current_instruction']=='Explain search evidence'
    assert item['actual_dependencies']==['Earlier plan']
    assert item['tools']==[dict(output='Search evidence')]
    assert item['reference']['kind']=='intermediate' and item['objective']['length_capped']
    objective_data = json.loads((tmp_path/'analysis/quality/objective.json').read_text())
    assert len(objective_data['outputs'])==len(objective_data['capped_intermediate_outputs'])==1
