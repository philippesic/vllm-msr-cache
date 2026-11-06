"""Direct native Request mapping and failure-diagnostic negative controls."""
from types import SimpleNamespace
import pytest
import operational_abort_v2_nov3 as observer
import operational_failure_capture_v2_nov3 as capture
from operational_case_receipt_v2_astra_nov3 import verify_abort_http
from test_operational_v2_supervisor_nov3 import valid_probe


def test_real_native_request_preserves_direct_external_mapping(monkeypatch):
 from vllm import SamplingParams
 from vllm.v1.request import Request,RequestStatus
 params={'external_request_id':'actual-external-id'}
 req=Request('internal-with-random-suffix',[1,2],SamplingParams(max_tokens=10,extra_args={'kv_transfer_params':params}),None)
 assert req.kv_transfer_params is params
 req.status=RequestStatus.FINISHED_ABORTED
 out=[req]
 class NativeBoundary:
  def finish_requests(self,*args,**kwargs):return out
 observer.ABORTS.clear();monkeypatch.setattr(observer,'ABORT_COUNT',0)
 observer.install(NativeBoundary)
 assert NativeBoundary().finish_requests('external',finished_status=RequestStatus.FINISHED_ABORTED) is out
 record=observer.snapshot()['records'][0]
 assert record['external_mapping_available'] and record['external_request_id']=='actual-external-id'
 assert req.kv_transfer_params is params and params=={'external_request_id':'actual-external-id'}


@pytest.mark.parametrize('defect',['missing_mapping','wrong_type','duplicate','additional_foreign','clock','origin','credit','arithmetic'])
def test_v2_direct_mapping_and_clock_rejects(defect):
 p=valid_probe();record=p['settled_after_health_export']['scheduler']['operational_abort_transitions']['records'][0]
 if defect=='missing_mapping':record.pop('external_request_id');record['external_mapping_available']=False
 elif defect=='wrong_type':record['external_request_id']=5
 elif defect=='duplicate':p['settled_after_health_export']['scheduler']['operational_abort_transitions']['records']*=2
 elif defect=='additional_foreign':p['settled_after_health_export']['scheduler']['operational_abort_transitions']['records'].append({'external_mapping_available':True,'external_request_id':'other','request_id':'x','status':'FINISHED_ABORTED'})
 elif defect=='clock':p['health_requests'][0]['priority_clock_ns']=-1
 elif defect=='origin':p['health_requests'][0]['priority_origin_ns']=100
 elif defect=='credit':p['health_requests'][0]['priority_credit_ms']=2000
 elif defect=='arithmetic':p['health_requests'][0]['payload']['priority']+=1
 assert not verify_abort_http(p,origin_ns=0)['qualified']


def test_internal_id_format_irrelevant_for_acceptance():
 p=valid_probe();p['settled_after_health_export']['scheduler']['operational_abort_transitions']['records'][0]['request_id']='NOT-A-PREFIX'
 assert verify_abort_http(p,origin_ns=0)['qualified']


@pytest.mark.parametrize('export_failure,metrics_failure',[(False,False),(True,False),(False,True),(True,True)])
def test_failure_capture_preserves_original_failure_and_attempts_both(monkeypatch,export_failure,metrics_failure):
 calls=[]
 def control(handle,op):
  calls.append(op)
  if export_failure:raise RuntimeError('export failure')
  return {'retained':'export'}
 class Response:
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def read(self):return b'metric 1\n'
 def urlopen(url,timeout):
  calls.append('metrics')
  if metrics_failure:raise RuntimeError('metrics failure')
  return Response()
 monkeypatch.setattr(capture,'control',control);monkeypatch.setattr(capture.urllib.request,'urlopen',urlopen)
 result={'status':'failed','error':'ORIGINAL','error_traceback':'ORIGINAL TRACE'}
 capture.capture(SimpleNamespace(metrics_url=lambda:'fake'),result)
 assert calls==['export','metrics']
 assert result['status']=='failed' and result['error']=='ORIGINAL' and result['error_traceback']=='ORIGINAL TRACE'
 assert ('postload_export' in result) is (not export_failure)
 assert ('metrics_final' in result) is (not metrics_failure)


def test_failure_capture_does_not_replace_existing_evidence(monkeypatch):
 monkeypatch.setattr(capture,'control',lambda *_:pytest.fail('unexpected export'))
 monkeypatch.setattr(capture.urllib.request,'urlopen',lambda *_:pytest.fail('unexpected HTTP'))
 result={'status':'failed','error':'original','postload_export':{'existing':1},'metrics_final':'existing'}
 capture.capture(None,result)
 assert result['postload_export']=={'existing':1} and result['metrics_final']=='existing'
