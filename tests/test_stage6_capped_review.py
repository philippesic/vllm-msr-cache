"""Archive coverage, blinding and tamper checks for the retained cap audit."""
import json
import hashlib
import io
import tarfile

import pytest

from benchmarks.serving_suite.campaign_capped_review import prepare,finish
from benchmarks.serving_suite.campaign_contracts import Ledger,atomic_json,digest,load,seal_tree
from benchmarks.serving_suite.campaign_quality import RUBRIC


def capped_attempt(root,phase,state):
    ledger = Ledger(root/phase)
    attempt,directory = ledger.begin(dict(phase=phase,model='hidden-model',cell='B06',system='hidden-policy'))
    measurement = directory/'measurement'
    session = dict(id='source',context='Retained source',turns=['Plan','Report'],
        quality_reference=dict(symbol='source'),graph=dict(outputs=['r1'],nodes=[
            dict(id='r0',prompt='Plan',depends_on=[]),dict(id='t0',depends_on=['r0']),
            dict(id='r1',prompt='Report',depends_on=['t0'])]))
    atomic_json(measurement/'corpus.json',dict(sessions=[session]))
    code = b'SYSTEM = "Keep intermediate answers under 70 words."\n'
    name = 'benchmarks/serving_suite/campaign_workload.py'
    atomic_json(measurement/'source.json',dict(files={name:hashlib.sha256(code).hexdigest()}))
    with tarfile.open(measurement/'executed-source.tar.gz','w:gz') as archive:
        member = tarfile.TarInfo(name)
        member.size = len(code)
        archive.addfile(member,io.BytesIO(code))
    if state!='interrupted':
        atomic_json(measurement/'result.json',dict(cases=[dict(name='case')]))
    (measurement/'case/tools').mkdir(parents=True)
    (measurement/'case/tools/0-0-t0.txt').write_text('Actual tool evidence')
    rows = [dict(source_id='source',node_id=node,wave=0,finish_reason='length',
        generated_text='Capped '+node,payload=dict(request_id=node,prompt=[1,2,3])) for node in ('r0','r1')]
    rows += [dict(rows[0],wave=-1,cancel_probe=True),dict(finish_reason='length',generated_text='Warmup')]
    (measurement/'case/requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows)+
        ('{"interrupted_fragment":' if state=='interrupted' else ''))
    event = ledger.index()[attempt]
    ledger.event(event['case_id'],attempt,state,**event['details'],seal=seal_tree(directory))
    return measurement


def test_all_phases_failed_interrupted_and_capped_intermediates_are_blind_and_sealed(tmp_path):
    raw = tmp_path/'raw'
    first = None
    for phase,state in [('screen','interrupted'),('pilot','failed'),('confirm','verified'),
                        ('overhead','verified'),('operational','verified')]:
        measurement = capped_attempt(raw,phase,state)
        first = first or measurement
    out = tmp_path/'audit'
    result = prepare(raw,out)
    assert result['complete_inventory'] and result['capped_responses']==10
    assert result['unique_contexts']==2
    assert set(result['phases'])=={'screen','pilot','confirm','overhead','operational'}
    assert result['roles']==dict(capped_intermediate=5,final=5)
    assert len(result['partial_stream_fragments'])==1
    packet = load(out/'blind-packet.json')
    for item in packet['items']:
        assert not {'model','system','phase','attempt','request_id'} & item.keys()
        assert item['executed_system_instruction_source']=='SYSTEM = "Keep intermediate answers under 70 words."'
        if item['response_role']=='capped_intermediate':
            assert item['actual_dependencies']==item['tools']==[]
        else:
            assert item['actual_dependencies']==['Capped r0']
            assert item['tools']==[dict(output='Actual tool evidence')]
    ratings = tmp_path/'incomplete.json'
    atomic_json(ratings,dict(schema='stage6-source-audit-v1',reviewer='fixture',rubric_digest=digest(RUBRIC),ratings=[]))
    with pytest.raises(ValueError,match='complete sealed packet'):
        finish(out,ratings)
    (first/'case/tools/0-0-t0.txt').write_text('tampered')
    with pytest.raises(ValueError,match='Changed artifact'):
        prepare(raw,tmp_path/'tampered-audit')


def test_new_acquisition_invalidates_an_earlier_complete_cap_inventory(tmp_path):
    raw = tmp_path/'raw'
    capped_attempt(raw,'confirm','verified')
    out = tmp_path/'audit'
    assert prepare(raw,out)['complete_inventory']
    Ledger(raw/'later').begin(dict(phase='operational'))
    with pytest.raises(ValueError,match='Acquisition changed'):
        finish(out,tmp_path/'not-needed.json')


def test_active_acquisition_prevents_final_cap_audit(tmp_path):
    raw = tmp_path/'raw'
    Ledger(raw/'active').begin(dict(phase='confirm'))
    out = tmp_path/'audit'
    result = prepare(raw,out)
    assert not result['complete_inventory'] and result['pending']
    with pytest.raises(ValueError,match='Complete acquisition'):
        finish(out,tmp_path/'not-needed.json')
