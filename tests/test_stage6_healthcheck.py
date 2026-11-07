"""Coarse agent checks must catch stopped/incomplete acquisition without owning it."""
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.research_checks import stage6_healthcheck as health
from benchmarks.serving_suite.campaign_contracts import Ledger,atomic_json,seal_tree


def test_progress_uses_stream_activity_and_unchanged_deadline():
    a=health.assess_progress({},dict(ledger_sequence=1,request_bytes=0,stream_bytes=5),0,300)
    assert not health.assess_progress(a,dict(ledger_sequence=1,request_bytes=0,stream_bytes=6),400,300)['stalled']
    assert health.assess_progress(a,dict(ledger_sequence=1,request_bytes=0,stream_bytes=5),400,300)['stalled']


def test_missing_or_parked_campaign_is_never_recreated(tmp_path):
    missing=tmp_path/'missing'
    result=subprocess.run([sys.executable,'-m','benchmarks.research_checks.stage6_healthcheck',
        '--out',str(missing),'--state-dir',str(tmp_path/'state')],capture_output=True,text=True)
    assert result.returncode==0
    assert json.loads(result.stdout)['issues']==['campaign_path_missing_do_not_recreate']
    assert not missing.exists()


def make_campaign(tmp_path,monkeypatch,blocks=None):
    out=tmp_path/'campaign';ledger=Ledger(out)
    (out/'ledger.jsonl').touch()
    atomic_json(out/'protocol.lock.json',dict(source=dict(digest='frozen'),schedule=[
        dict(model='fixture',cell='B02',repeat=0,systems=['arc','wait'])]))
    atomic_json(out/'resolved.json',dict(cells=[]))
    monkeypatch.setattr(health,'coordinators',lambda p:[])
    monkeypatch.setattr(health,'gpu_snapshot',lambda:dict(processes=[],utilization=0))
    for system,block in blocks or []:
        identity=dict(model='fixture',cell='B02',repeat=0,system=system,source='frozen',phase='confirm')
        attempt,d=ledger.begin(identity,block_id=block)
        case=dict(name='case',expected_requests=1,summary=dict(completed=1,failed=0),cleanup=dict(verified=True),
            evaluation=dict(status='passed'),activity=dict(restore=1),campaign=dict(completed_units=1,offered_units=1,
                tools=0,native_cache=dict(complete=True,receipts=1)))
        atomic_json(d/'measurement/result.json',dict(cases=[case]))
        atomic_json(d/'measurement/corpus.json',dict(sessions=[dict(graph=dict(nodes=[dict(kind='request')]))]))
        row=ledger.index()[attempt]
        ledger.event(row['case_id'],attempt,'verified',**row['details'],seal=seal_tree(d))
    return out


def test_stopped_coordinator_wakes_recovery(tmp_path,monkeypatch):
    out=make_campaign(tmp_path,monkeypatch)
    result=health.inspect(out,tmp_path/'state')
    assert result['status']=='attention' and 'coordinator_stopped_before_completion' in result['issues']


def test_complete_whole_block_wakes_next_phase(tmp_path,monkeypatch):
    out=make_campaign(tmp_path,monkeypatch,[('arc','one'),('wait','one')])
    result=health.inspect(out,tmp_path/'state')
    assert result['status']=='acquisition_complete' and result['complete_matched_blocks']==1


def test_fragments_from_different_blocks_cannot_finish_acquisition(tmp_path,monkeypatch):
    out=make_campaign(tmp_path,monkeypatch,[('arc','one'),('wait','two')])
    result=health.inspect(out,tmp_path/'state')
    assert result['valid_configurations']==2 and result['complete_matched_blocks']==0
    assert result['status']=='attention'


def test_completed_but_empty_model_work_is_not_healthy(tmp_path,monkeypatch):
    from benchmarks.serving_suite.campaign_contracts import load
    out=make_campaign(tmp_path,monkeypatch,[('arc','one'),('wait','one')])
    event=next(iter(Ledger(out).index().values()))
    path=out/event['details']['path']/'measurement/result.json'
    data=load(path);data['cases'][0]['summary']['output_tokens']=0;atomic_json(path,data)
    result=health.inspect(out,tmp_path/'state')
    assert result['status']=='attention'
    assert any(isinstance(i,dict) and i.get('kind')=='no_model_output_for_completed_workload' for i in result['issues'])


def test_recovery_rejects_unattested_analysis_before_launch(tmp_path):
    out=tmp_path/'out'
    atomic_json(out/'protocol.lock.json',dict(source=dict(digest='original',files={})))
    receipt=tmp_path/'receipt.json'
    atomic_json(receipt,dict(analysis_source=dict(files={'benchmarks/serving_suite/campaign_validity.py':'incorrect'})))
    result=subprocess.run([sys.executable,'-m','benchmarks.research_checks.stage6_resume',
        '--out',str(out),'--analysis-receipt',str(receipt),'--check'],capture_output=True,text=True)
    assert result.returncode!=0 and 'Derived predicate changed' in result.stderr
