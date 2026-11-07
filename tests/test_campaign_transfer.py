import shutil
import subprocess
import sys
import copy
from pathlib import Path

import pytest

from benchmarks.serving_suite import assets,campaign
from benchmarks.serving_suite.campaign_contracts import Ledger,atomic_json,digest,file_hash,load,write_protocol
from benchmarks.serving_suite.campaign_verify import export_campaign,import_campaign


@pytest.fixture
def transfer(tmp_path,monkeypatch):
    project = tmp_path/'project'
    project.mkdir()
    (project/'module.py').write_text('value=1\n')
    source = dict(files={'module.py':file_hash(project/'module.py')})
    source['digest'] = digest(source['files'])
    monkeypatch.setattr(campaign,'ROOT',project)
    monkeypatch.setattr(campaign,'source_snapshot',lambda:source)
    monkeypatch.setattr(assets,'runtime_identity',lambda python:dict(runtime='fixture'))
    cache = tmp_path/'cache'
    repo = cache/'sources/more-itertools'
    repo.mkdir(parents=True)
    (repo/'selected.py').write_text('from helper import VALUE\n')
    (repo/'helper.py').write_text('VALUE=1\n')
    (repo/'LICENSE').write_text('fixture license\n')
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    atomic_json(cache/'sources.json',{'more-itertools':dict(path=str(repo),repository='https://example.test/repo',revision='pin',licenses={'LICENSE':file_hash(repo/'LICENSE')})})
    model = cache/'model'
    model.mkdir()
    atomic_json(model/'config.json',dict(hidden_size=8,num_attention_heads=2,num_hidden_layers=2))
    (model/'model.safetensors').write_bytes(b'fixture only, never loaded')
    model_record = dict(repository='example/model',revision='pinned',status='prepared',identity=assets.model_identity(model))
    host = dict(schema='serving-campaign-host-v1',id='first',device='cuda:0',runtime_python=sys.executable,
        asset_cache=str(cache),models={'fixture':str(model)},runtime_bindings={'project_root':str(project)},
        limits={k:5 for k in ('startup_timeout_s','request_timeout_s','tool_timeout_s','case_timeout_s')})
    atomic_json(project/'benchmarks/serving_suite/templates/host.example.json',host)
    out = tmp_path/'original'
    out.mkdir()
    cell = dict(model='fixture',cell='B06',systems=['arc','wait'],artifact_identity=model_record['identity']['content_digest'],
        profile=dict(gpu_blocks=8,cpu_bytes=4096,max_model_len=128,status='development',runtime_python=sys.executable))
    for field,partition in [('workload_binding','development'),('confirmation_binding','confirmation')]:
        directory = out/partition
        plan = dict(model=str(model),cpu_bytes=4096)
        corpus = dict(sessions=[dict(id=partition,repository=str(repo),input_files={str(repo/'selected.py'):file_hash(repo/'selected.py')},
            tools=[dict(cwd=str(repo),argv=['rg','VALUE',str(repo/'selected.py')])])])
        plan['operational_corpus'] = copy.deepcopy(corpus)
        plan['operational_corpus']['sessions'].append(dict(id=partition+'-operational-only',
            repository=str(repo),input_files={str(repo/'helper.py'):file_hash(repo/'helper.py')},
            tools=[dict(cwd=str(repo),argv=['rg','VALUE',str(repo/'helper.py')])]))
        atomic_json(directory/'plan.json',plan)
        atomic_json(directory/'corpus.json',corpus)
        cell[field] = dict(plan=str(directory/'plan.json'),corpus=str(directory/'corpus.json'),
            plan_digest=digest(plan),corpus_digest=digest(corpus),partition=partition)
    logical = dict(specification=dict(id='fixture'),catalogs={},asset_identities={'fixture':model_record})
    atomic_json(out/'campaign.json',logical)
    atomic_json(out/'resolved.json',dict(campaign_id=digest(logical),cells=[cell],unresolved=[]))
    atomic_json(out/'host.private.json',host)
    write_protocol(out,dict(campaign_id=digest(logical),cells_digest=digest([cell]),source=source,
        schedule=[dict(model='fixture',cell='B06',repeat=0,systems=['arc','wait'])],
        supplementary_schedules=dict(overhead=[],operational=[]),precision=dict(repeats={'fixture:B06':6})))
    Ledger(out)
    return out,host,tmp_path


def test_transfer_relocates_both_cohorts_and_full_repo(transfer):
    original,host,root = transfer
    bundle = root/'bundle'
    export_campaign(original,bundle)
    moved = root/'moved'
    shutil.move(bundle,moved)
    host = dict(host,id='second')
    host_path = root/'new-host.json'
    atomic_json(host_path,host)
    imported = root/'imported'
    result = import_campaign(moved,host_path,imported)
    assert result['runnable'] and not result['confirmation_ready']
    assert result['campaign_id']!=result['parent_campaign_id']
    shutil.rmtree(moved)
    shutil.rmtree(original)
    resolved = load(imported/'resolved.json')
    for field,partition in [('workload_binding','development'),('confirmation_binding','confirmation')]:
        corpus = load(resolved['cells'][0][field]['corpus'])
        session = corpus['sessions'][0]
        assert session['id']==partition
        assert (Path(session['repository'])/'helper.py').read_text()=='VALUE=1\n'
        assert str(root/'original') not in str(corpus)
        plan = load(resolved['cells'][0][field]['plan'])
        operational = plan['operational_corpus']['sessions']
        assert len(operational)==2 and operational[0]==session
        extra = operational[1]
        assert extra['id']==partition+'-operational-only'
        assert Path(extra['tools'][0]['argv'][-1]).read_text()=='VALUE=1\n'
        assert str(root/'original') not in str(plan)
    assert load(imported/'imported-protocol.json')['precision']['repeats']=={'fixture:B06':6}
    assert campaign.run_phase(imported,'preflight',dry_run=True)['blocks']
    with pytest.raises(ValueError,match='locked protocol'):
        campaign.run_phase(imported,'confirm',dry_run=True)
    with pytest.raises(ValueError,match='Imported repetitions'):
        campaign.freeze(imported)


def test_transfer_refuses_missing_or_changed_input(transfer):
    original,host,root = transfer
    (Path(host['asset_cache'])/'sources/more-itertools/selected.py').write_text('tampered\n')
    with pytest.raises(ValueError,match='input changed'):
        export_campaign(original,root/'bad-export')


def test_import_checks_bundle_and_executed_source(transfer,monkeypatch):
    original,host,root = transfer
    bundle = root/'bundle'
    export_campaign(original,bundle)
    host_path = root/'host.json'
    atomic_json(host_path,host)
    monkeypatch.setattr(campaign,'source_snapshot',lambda:dict(digest='different'))
    with pytest.raises(ValueError,match='source closure'):
        import_campaign(bundle,host_path,root/'bad-import')
    (bundle/'assets/sources/more-itertools/helper.py').unlink()
    with pytest.raises(ValueError,match='unsealed'):
        import_campaign(bundle,host_path,root/'missing-import')


def test_operational_development_is_separate_and_preserves_parent(transfer):
    from benchmarks.serving_suite.campaign_development import prepare_operation
    from benchmarks.serving_suite.campaign_verify import verify_campaign
    parent,host,root = transfer
    destination = root/'operation-development'
    with pytest.raises(ValueError,match='before freeze'):
        prepare_operation(parent,destination,'fixture')
    (parent/'protocol.lock.json').unlink()
    before = load(parent/'resolved.json')
    binding = before['cells'][0]['workload_binding']
    corpus = load(binding['corpus'])
    corpus['sessions'][0]['graph'] = dict(nodes=[dict(kind='request'),dict(kind='tool')])
    full = copy.deepcopy(corpus)
    full['sessions'].append(dict(copy.deepcopy(full['sessions'][0]),id='second'))
    plan = load(binding['plan'])
    plan['operational_corpus'] = full
    for key,value in [('plan',plan),('corpus',corpus)]:
        atomic_json(binding[key],value)
        binding[key+'_digest'] = digest(value)
    atomic_json(parent/'resolved.json',before)
    result = prepare_operation(parent,destination,'fixture',waves=2)
    assert result['diagnostic_only']
    assert load(parent/'resolved.json')==before
    child = load(destination/'resolved.json')
    assert child['campaign_id']!=before['campaign_id']
    assert child['cells'][0]['confirmation_binding'] is None
    plan = load(child['cells'][0]['workload_binding']['plan'])
    assert plan['tier']=='operational' and plan['waves']==2
    assert plan['sessions']==2 and plan['expected_requests_per_wave']==2
    assert load(child['cells'][0]['workload_binding']['corpus'])==full
    assert verify_campaign(destination)['attempts']=={}
    with pytest.raises(ValueError,match='locked protocol'):
        campaign.run_phase(destination,'operational',dry_run=True)


def test_transfer_host_uses_frozen_deadlines_and_rejects_escaping_plan(tmp_path):
    from benchmarks.serving_suite.transfer_setup import frozen_limits
    base = dict(startup_timeout_s=900,request_timeout_s=900,tool_timeout_s=180,case_timeout_s=3600)
    atomic_json(tmp_path/'development.json',base)
    atomic_json(tmp_path/'confirmation.json',dict(base,case_timeout_s=5400))
    cell = dict(workload_binding=dict(plan='${bundle}/development.json'),
        confirmation_binding=dict(plan='${bundle}/confirmation.json'))
    assert frozen_limits(tmp_path,dict(cells=[cell]))==dict(base,case_timeout_s=5400)
    cell['confirmation_binding']['plan']='${bundle}/../escape.json'
    with pytest.raises(ValueError,match='path|Path'):
        frozen_limits(tmp_path,dict(cells=[cell]))


def test_roomy_capacity_probe_keeps_tasks_and_parent_immutable(transfer):
    from benchmarks.serving_suite.campaign_development import prepare_roomy
    parent,host,root = transfer
    (parent/'protocol.lock.json').unlink()
    original = load(parent/'resolved.json')
    original['cells'][0]['cell']='B07'
    binding = original['cells'][0]['workload_binding']
    plan = load(binding['plan'])
    plan['candidate_config']={'kv_connector_extra_config':{'cpu_bytes_to_use':4096}}
    atomic_json(binding['plan'],plan)
    binding['plan_digest']=digest(plan)
    atomic_json(parent/'resolved.json',original)
    result = prepare_roomy(parent,root/'capacity-probe','fixture',2048,startup_timeout_s=180)
    assert result['diagnostic_only']
    assert load(parent/'resolved.json')==original
    child = load(root/'capacity-probe/resolved.json')['cells'][0]
    assert child['systems']==['arc'] and child['confirmation_binding'] is None
    assert load(child['workload_binding']['corpus'])==load(binding['corpus'])
    effective = load(child['workload_binding']['plan'])
    assert effective['cpu_bytes']==2048 and effective['startup_timeout_s']==180
    assert effective['candidate_config']['kv_connector_extra_config']['cpu_bytes_to_use']==2048
    assert load(binding['plan'])['cpu_bytes']==4096
    dry = campaign.run_phase(root/'capacity-probe','screen',dry_run=True)
    assert dry['blocks'][0]['systems']==['arc']
    with pytest.raises(ValueError,match='only reduces'):
        prepare_roomy(parent,root/'bad-capacity','fixture',4096)


def test_envelope_fallback_reduces_identical_complete_workflows(transfer):
    from benchmarks.serving_suite.campaign_development import prepare_envelope
    parent,host,root = transfer
    (parent/'protocol.lock.json').unlink()
    resolved = load(parent/'resolved.json')
    original = resolved['cells'][0]
    corpus = load(original['workload_binding']['corpus'])
    session = corpus['sessions'][0]
    session['graph'] = dict(outputs=['final'],nodes=[dict(id='first',kind='request'),
        dict(id='tool',kind='tool'),dict(id='final',kind='request')])
    corpus['sessions']=[dict(session,id='session-'+str(i)) for i in range(4)]
    cells=[]
    for name,cpu in [('B06',4096),('B07',8192)]:
        cell=copy.deepcopy(original)
        cell['cell']=name
        plan=dict(model='fixture',cpu_bytes=cpu,candidate_config={'kv_connector_extra_config':{'cpu_bytes_to_use':cpu}})
        for key,value in [('plan',plan),('corpus',corpus)]:
            path=parent/name/(key+'.json')
            atomic_json(path,value)
            cell['workload_binding'][key]=str(path)
            cell['workload_binding'][key+'_digest']=digest(value)
        cells.append(cell)
    resolved['cells']=cells
    atomic_json(parent/'resolved.json',resolved)
    result=prepare_envelope(parent,root/'envelope','fixture',sessions=2,roomy_cpu_bytes=6144)
    assert result['diagnostic_only'] and load(parent/'resolved.json')==resolved
    selected=load(root/'envelope/resolved.json')['cells']
    corpora=[load(c['workload_binding']['corpus']) for c in selected]
    assert corpora[0]==corpora[1] and len(corpora[0]['sessions'])==2
    assert all(len(s['graph']['nodes'])==3 for s in corpora[0]['sessions'])
    for cell in selected:
        plan=load(cell['workload_binding']['plan'])
        assert plan['sessions']==2 and plan['expected_requests_per_wave']==4 and plan['expected_tools_per_wave']==2
    dry=campaign.run_phase(root/'envelope','screen',dry_run=True)
    assert [b['cell'] for b in dry['blocks']]==['B06','B07']
    assert all(b['systems']==['arc'] for b in dry['blocks'])


def test_offline_guard_rejects_snapshot_and_direct_blob_access(tmp_path):
    model = tmp_path/'snapshots/pinned'
    model.mkdir(parents=True)
    blobs = tmp_path/'blobs'
    blobs.mkdir()
    weight = blobs/'weight'
    weight.write_bytes(b'must not be read')
    (model/'model.safetensors').symlink_to(weight)
    atomic_json(model/'config.json',{})
    original = tmp_path/'original'
    original.mkdir()
    (original/'result.json').write_text('{}')
    historical = tmp_path/'historical'
    historical.mkdir()
    (historical/'old-input.json').write_text('{}')
    allowed = historical/'relocated/result.json'
    allowed.parent.mkdir()
    allowed.write_text('{}')
    script = '''
import sys
from pathlib import Path
from benchmarks.serving_suite.campaign_portability import install_offline_guard
original,model,blob,allowed,historical = map(Path,sys.argv[1:])
install_offline_guard(original,[model],forbidden_roots=[historical],allowed_roots=[allowed.parent])
for path in (model/'config.json',model/'model.safetensors',blob,original/'result.json',historical/'old-input.json'):
 try:
  path.read_bytes()
 except RuntimeError as error:
  assert 'original artifact/model path' in str(error)
 else:
  raise AssertionError('Forbidden file was read: '+str(path))
assert allowed.read_text()=='{}'
try:
 import torch
except RuntimeError as error:
 assert 'GPU/model dependency' in str(error)
else:
 raise AssertionError('GPU import was allowed')
'''
    subprocess.run([sys.executable,'-c',script,str(original),str(model),str(weight),str(allowed),str(historical)],check=True)
