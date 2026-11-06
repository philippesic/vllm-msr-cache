import copy,json
from pathlib import Path
import pytest
from frontier import classify
OLD=Path(__file__).resolve().parents[1]/'quantized-realworld-20261101'

def sample():
 r={'status':'complete','owned_descendants_remaining':[],'planned_requests':2,'requests':[{'index':0,'success':True},{'index':1,'success':True}],'submission_events':[{'index':0},{'index':1}],'load_start_ns':1}
 s={'valid':True,'proofs':{'requests':2,'passed':2,'failures':[],'all_passed':True},'all':{'elapsed_s':3}}
 return r,s

def test_complete():
 r,s=sample();assert classify(r,s)['complete_load_eligible']

def test_traffic_cap_retains_completed_proofs_and_unfinished_not_fake_passes():
 r,s=sample();r.update(status='failed',error='TimeoutError()',planned_requests=4,submission_events=[{'index':0},{'index':1},{'index':2}]);s['valid']=False
 d=classify(r,s);assert d['outcome']=='traffic_time_capped' and not d['complete_load_eligible'];assert d['completed_strict_proofs']==2 and d['unfinished_submitted_indices']==[2] and d['unreleased_requests']==1

@pytest.mark.parametrize('bad',['proof_gap','missing_proof','fake_complete','duplicate_submit','duplicate_response','unexpected_failure','live_process'])
def test_fail_closed(bad):
 r,s=sample()
 if bad=='proof_gap':s['proofs']['passed']=1
 elif bad=='missing_proof':s['proofs']={}
 elif bad=='fake_complete':r['planned_requests']=3
 elif bad=='duplicate_submit':r['submission_events'].append({'index':0})
 elif bad=='duplicate_response':r['requests'][1]['index']=0
 elif bad=='unexpected_failure':r.update(status='failed',error='ValueError(incorrectKV)')
 elif bad=='live_process':r['owned_descendants_remaining']=[123]
 with pytest.raises(RuntimeError):classify(r,s)

def test_actual_old_invalid_proof_is_rejected_even_on_timeout():
 s=json.loads((OLD/'runs/confirm-agent-a8-c24-candidate-s1103/summary.json').read_text());r={'status':'failed','error':'TimeoutError()','requests':[{'index':i,'success':True} for i in range(72)],'submission_events':[{'index':i} for i in range(72)]}
 with pytest.raises(RuntimeError,match='strict proof'):classify(r,s)

def test_empty_startup_cap_is_censored_not_passed_performance():
 r={'status':'failed','error':'TimeoutError(startup)','requests':[],'submission_events':[],'owned_descendants_remaining':[]};s={'proofs':{'requests':0,'passed':0,'failures':[]}}
 d=classify(r,s);assert d['outcome']=='startup_time_capped' and d['completed_strict_proofs']==0 and not d['complete_load_eligible']
