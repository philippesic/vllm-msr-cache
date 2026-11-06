import copy
import pytest
from qualify_transfer_bytes import validate
def case():
 jobs=[dict(job_id=i,direction=d,success=True,cancelled=False,contributors=1,expected_contributors=1,
            bytes=917504,source_block_ids=[[0,1]],destination_block_ids=[[0,2]]) for i,d in [(1,'store'),(2,'load')]]
 raw=dict(status='complete',performance_eligible=False,planned_requests=72,geometry={'block_bytes':917504},
          requests=[{'payload':{'request_id':f'r{i}'}} for i in range(72)],
          terminals=[{'external_request_id':f'r{i}','transfer_job_statuses':copy.deepcopy(jobs)} for i in range(72)])
 summary=dict(valid=True,proofs={'all_passed':True,'passed':72},all={'requests':72},
              offline_staging_utility={'global_explicit_successful_completion_bytes_observed':0})
 audits=[dict(job_id=i,direction=d,source_physical_ids=[1],destination_physical_ids=[2],total_bytes=917504,
              components=[dict(tensor_idx=k,page_size_bytes=32768,bytes=32768,equal=True,invalid_byte_rejected=True) for k in range(28)]) for i,d in [(1,'D2H'),(2,'H2D')]]
 return raw,summary,audits
def test_actual_all_component_coverage_required_for_every_serving_job():
 assert validate(*case())['serving_bytes_checked']==1835008
@pytest.mark.parametrize('bad',['missing_job','missing_component','bad_byte','unexecuted_wrong_control','wrong_direction','physical','duplicate','wrong_bytes'])
def test_known_wrong_byte_qualifications_are_rejected(bad):
 raw,s,a=case()
 if bad=='missing_job':a.pop()
 if bad=='missing_component':a[0]['components'].pop()
 if bad=='bad_byte':a[0]['components'][0]['equal']=False
 if bad=='unexecuted_wrong_control':a[0]['components'][0]['invalid_byte_rejected']=False
 if bad=='wrong_direction':a[0]['direction']='H2D'
 if bad=='physical':a[0]['destination_physical_ids']=[3]
 if bad=='duplicate':a.append(copy.deepcopy(a[0]))
 if bad=='wrong_bytes':a[0]['total_bytes']-=1
 with pytest.raises(AssertionError):validate(raw,s,a)
