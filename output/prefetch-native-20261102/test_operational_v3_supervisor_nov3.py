"""CPU-only v3 launch/continuation/documentation controls; no GPU calls."""
import copy,json
from pathlib import Path
import pytest
import supervise_operational_v3_astra_nov3 as s

def test_prior_inputs_preserved_except_documentation():
 old=json.loads((s.ROOT/'evidence/live-optimization-2080ti-20261103/operational-v2-frozen.json').read_text())
 oldpins={}
 for field in ('existing_frozen_inputs','new_review_file_pins','file_pins'):
  for k,v in old.get(field,{}).items():oldpins[str((s.ROOT/k).resolve())]=v
 readme=str((s.ROOT/'README.md').resolve());oldpins.pop(readme)
 review=json.loads(s.REVIEW.read_text())
 assert review['existing_frozen_inputs']==oldpins and len(oldpins)==812
 assert readme not in s.merged_pins(review)
 assert not s.pin_mismatches(s.merged_pins(review))

def test_exact_remaining_cases_only_names_change():
 old=json.loads((s.ROOT/'evidence/live-optimization-2080ti-20261103/operational-v2-frozen.json').read_text())
 before=[x for x in old['execution_order'] if x['phase'] in ('D','E')]
 new=json.loads(s.REVIEW.read_text())['execution_order']
 assert len(new)==3 and sum(x['normal_natural_eos_planned'] for x in new)==1008
 for a,b in zip(before,new):
  args=a['args'].copy();args[args.index('--name')+1]=b['name'];assert b['args']==args
  assert a['phase']==b['phase'] and a['arm']==b['arm']
  assert b['deliberate_abort_planned']==b['health_natural_eos_planned']==0
  assert not (s.H/'runs'/b['name']).exists()

def test_document_changes_and_missing_are_provenance_only(tmp_path,monkeypatch):
 monkeypatch.setattr(s,'ROOT',tmp_path);p=tmp_path/'README.md';p.write_text('first')
 a=s.document_provenance();p.write_text('second');b=s.document_provenance()
 assert a['sha256']!=b['sha256']
 p.unlink();assert s.document_provenance()['sha256'] is None

def test_document_cannot_reenter_live_pins():
 with pytest.raises(ValueError,match='provenance-only'):s.merged_pins({'file_pins':{str(s.ROOT/'README.md'):'a'}})

def test_document_replacement_read_race_is_unavailable_provenance(tmp_path,monkeypatch):
 monkeypatch.setattr(s,'ROOT',tmp_path);(tmp_path/'README.md').write_text('concurrent')
 def replaced(path):raise FileNotFoundError('concurrent replacement')
 monkeypatch.setattr(s,'sha',replaced)
 result=s.document_provenance()
 assert result['sha256'] is None and 'concurrent replacement' in result['read_error']

def test_execution_pin_drift_and_missing_fail(tmp_path):
 p=tmp_path/'runtime.py';p.write_text('first');pins={str(p):s.sha(p)}
 assert not s.pin_mismatches(pins)
 p.write_text('changed');assert s.pin_mismatches(pins)==[str(p)]
 p.unlink();assert s.pin_mismatches(pins)==[str(p)]

def test_conflicting_pins_fail(tmp_path):
 p=str(tmp_path/'source.py')
 with pytest.raises(ValueError,match='Conflicting'):s.merged_pins({'existing_frozen_inputs':{p:'a'},'file_pins':{p:'b'}})

@pytest.mark.parametrize('defect',[None,'argv','count','supervisor_pin','no_authority','manifest_binding','supervisor_binding'])
def test_review_and_certificate_gates_without_execution(tmp_path,defect):
 manifest=json.loads(s.REVIEW.read_text());pins=s.merged_pins(manifest)
 pins.update({str(p.resolve()):s.sha(p) for p in [Path(s.__file__),s.H/'operational_case_receipt_v2_astra_nov3.py',s.REVIEW]})
 if defect=='argv':manifest['execution_order'][0]['args'].append('--unsafe')
 if defect=='count':manifest['execution_order'].pop()
 if defect=='supervisor_pin':pins.pop(str(Path(s.__file__).resolve()))
 path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
 auth={'launch_authorized':True,'gpu_custody':'ASTRA','manifest_sha256':s.sha(path),'supervisor_sha256':s.sha(s.__file__)}
 if defect=='no_authority':auth['launch_authorized']=False
 if defect=='manifest_binding':auth['manifest_sha256']='wrong'
 if defect=='supervisor_binding':auth['supervisor_sha256']='wrong'
 if defect:
  with pytest.raises(ValueError):s.validate_authorization(path,manifest,auth,pins)
 else:s.validate_authorization(path,manifest,auth,pins)

def test_no_implicit_execution():
 with pytest.raises(SystemExit):s.main(['--manifest','absent','--authorization','absent'])
