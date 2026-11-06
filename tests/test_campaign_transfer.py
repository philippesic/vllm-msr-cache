import shutil
import subprocess
import sys
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
    result = prepare_operation(parent,destination,'fixture',waves=2)
    assert result['diagnostic_only']
    assert load(parent/'resolved.json')==before
    child = load(destination/'resolved.json')
    assert child['campaign_id']!=before['campaign_id']
    assert child['cells'][0]['confirmation_binding'] is None
    plan = load(child['cells'][0]['workload_binding']['plan'])
    assert plan['tier']=='operational' and plan['waves']==2
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
