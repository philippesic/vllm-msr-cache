"""CPU-only controls for supervisor launch gates and evidence classification."""
import copy,hashlib,json
from pathlib import Path
import pytest
import supervise_operational_astra_nov3 as s
from operational_case_receipt_astra_nov3 import verify_abort_http


def test_pin_merge_conflict_is_rejected(tmp_path):
    path=str(tmp_path/'one')
    with pytest.raises(ValueError,match='Conflicting'):
        s.merged_pins({'existing_frozen_inputs':{path:'a'},'file_pins':{path:'b'}})


def test_pin_merge_deduplicates_identical_paths(tmp_path):
    path=str(tmp_path/'one')
    assert s.merged_pins({'existing_frozen_inputs':{path:'a'},'file_pins':{path:'a'}})=={path:'a'}


@pytest.mark.parametrize('auth',[{}, {'launch_authorized':True}, {'launch_authorized':False,'gpu_custody':'ASTRA'}])
def test_no_execution_certificate_cannot_launch(tmp_path,auth):
    with pytest.raises(ValueError,match='ROOT launch'):
        s.validate_authorization(tmp_path/'nonexistent',{},auth,{})


def test_certificate_is_manifest_bound(tmp_path):
    path=tmp_path/'manifest';path.write_text('{}')
    with pytest.raises(ValueError,match='manifest mismatch'):
        s.validate_authorization(path,{}, {'launch_authorized':True,'gpu_custody':'ASTRA','manifest_sha256':'wrong'}, {})


def test_certificate_is_supervisor_bound(tmp_path):
    path=tmp_path/'manifest';path.write_text('{}')
    with pytest.raises(ValueError,match='supervisor mismatch'):
        s.validate_authorization(path,{}, {'launch_authorized':True,'gpu_custody':'ASTRA','manifest_sha256':s.sha(path),'supervisor_sha256':'wrong'}, {})


def test_explicit_execute_is_required_before_file_read():
    with pytest.raises(SystemExit) as exc:
        s.main(['--manifest','does-not-exist','--authorization','does-not-exist'])
    assert exc.value.code==2


def test_verified_compression_retains_content_and_refuses_overwrite(tmp_path):
    path=tmp_path/'one.log';payload=b'valuable evidence\n'*100;path.write_bytes(payload)
    receipt=s.compress(path)
    assert receipt['uncompressed_sha256']==hashlib.sha256(payload).hexdigest()
    assert not path.exists()
    path.write_bytes(b'new')
    with pytest.raises(FileExistsError):s.compress(path)
    assert path.read_bytes()==b'new'


def valid_probe():
    abort_id='abort';response_id='cmpl-'+abort_id
    payload=lambda rid:{'request_id':rid,'prompt':[1,2,3],'max_tokens':7677}
    abort={'payload':payload(abort_id),'response_id':response_id,'terminal_before_close':False,'disconnect_ns':123,'http_status':200}
    health=[{'payload':payload('health'+str(i)),'http_status':200,'natural_eos':True,'terminal_complete':True,'finish_reasons':['stop']} for i in range(4)]
    events=[]
    for kind,row in [('intentional_disconnect',abort)]+[('health_generation',x) for x in health]:
        for label in ['request_start','connection_create_start','connection_create_end','request_headers_received']:
            events.append({'request':{'kind':kind,'request_id':row['payload']['request_id']},'event':label,'status':200})
    return {'qualified':True,'http_client_configuration':{'force_close':True,'automatic_POST_retry':False,'application_retries':0,'connection_trace':True},'abort_request':abort,'health_requests':health,'http_connection_events':events,'settled_after_health_export':{'scheduler':{'operational_abort_transitions':{'records':[{'request_id':response_id+'-0','status':'FINISHED_ABORTED'}]}}}}


def test_abort_receipt_qualified_exact_identity_and_five_posts():
    result=verify_abort_http(valid_probe())
    assert result['qualified'] and result['HTTP_attempts']==5 and result['natural_eos_health']==4


@pytest.mark.parametrize('defect',['wrong_backend_id','already_terminal','reuse','missing_trace','health_length','arbitrary_output_cap','duplicate_abort','retry'])
def test_abort_receipt_negative_controls(defect):
    p=valid_probe()
    if defect=='wrong_backend_id':p['settled_after_health_export']['scheduler']['operational_abort_transitions']['records'][0]['request_id']='foreign'
    elif defect=='already_terminal':p['abort_request']['terminal_before_close']=True
    elif defect=='reuse':p['http_connection_events'].insert(1,{'event':'connection_reuse','request':{'kind':'intentional_disconnect','request_id':'abort'}})
    elif defect=='missing_trace':p['http_connection_events'].pop()
    elif defect=='health_length':p['health_requests'][0]['finish_reasons']=['length']
    elif defect=='arbitrary_output_cap':p['health_requests'][0]['payload']['max_tokens']=32
    elif defect=='duplicate_abort':p['settled_after_health_export']['scheduler']['operational_abort_transitions']['records']*=2
    elif defect=='retry':p['http_client_configuration']['application_retries']=1
    assert not verify_abort_http(p)['qualified']

@pytest.mark.parametrize('change',[None,'argv','missing_supervisor_pin'])
def test_exact_reviewed_freeze_gate_without_launch(tmp_path,change):
    manifest=json.loads(s.REVIEW.read_text())
    required=[Path(s.__file__),s.H/'operational_case_receipt_astra_nov3.py',s.REVIEW]
    pins={str(p.resolve()):s.sha(p) for p in required}
    if change=='argv':manifest['execution_order'][0]['args'].append('--unreviewed-option')
    if change=='missing_supervisor_pin':pins.pop(str(Path(s.__file__).resolve()))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    certificate={'launch_authorized':True,'gpu_custody':'ASTRA','manifest_sha256':s.sha(path),'supervisor_sha256':s.sha(s.__file__)}
    if change:
        with pytest.raises(ValueError):s.validate_authorization(path,manifest,certificate,pins)
    else:s.validate_authorization(path,manifest,certificate,pins)
