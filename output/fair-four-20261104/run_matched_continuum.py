"""Predeclared supported decode-publication correction, one Continuum pass."""
import hashlib,json,shutil,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path('/home/philip/vllm-semantic-cache-ext');sys.path.insert(0,str(ROOT))
from harness.gpu_lock import GPULock
from benchmarks.serving_suite.contracts import SCHEMA,write_json
from benchmarks.serving_suite.published import check_source,freeze_sources
from benchmarks.serving_suite.report import render_report
from benchmarks.serving_suite.runner import available_gpu,source_identity,freeze_source,plan_for_variant,run_case,select_sessions
from benchmarks.serving_suite.verify import verify,seal

def file_hash(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
 with GPULock():
  import psutil
  old=ROOT/'output/fair-four-20261104/comparison-final';out=ROOT/'output/fair-four-20261104/continuum-decode-matched';out.mkdir()
  before=verify(old);original=json.loads((old/'result.json').read_text());plan=json.loads((old/'plan.json').read_text());plan['output']=str(out)
  runtime=plan['published_runtimes']['continuum'];runtime['env']['LMCACHE_SAVE_DECODE_CACHE']='true'
  plan.update(measurement_variant='continuum',measurement_scope='One predeclared replacement row: supported save_decode_cache=true matches native decode publication; chosen irrespective timing, previous false row diagnostic only.')
  declaration=dict(declared_unix=time.time(),scope=plan['measurement_scope'],changed_runtime_setting={'LMCACHE_SAVE_DECODE_CACHE':{'before':False,'after':True}},preserved_settings='Same selected corpus, model weights/tokenizer, workload implementation, prompt/parser, scheduler, 256-token chunks, budgets, admission, tools, EOS stopping, warmup and source-observation semantics; added read-only decode-publication observation.',original_verified=before,original_manifest_sha256=file_hash(old/'artifacts.json'),coordinator_sha256=file_hash(__file__),command=[sys.executable,str(Path(__file__).resolve())],hardware_before=available_gpu(),host_memory_before=psutil.virtual_memory()._asdict(),swap_before=psutil.swap_memory()._asdict())
  write_json(out/'replacement-declaration.json',declaration);shutil.copyfile(__file__,out/'replacement-coordinator.py')
  for name in ['corpus.json','model-files.json','wheel-assets.json']:
   shutil.copyfile(old/name,out/name)
  write_json(out/'plan.json',plan)
  initial=source_identity();write_json(out/'source.json',initial);freeze_source(initial,out)
  check_source('continuum',runtime);freeze_sources({'continuum':runtime},out)
  dependency=Path(subprocess.check_output([runtime['python'],'-c','import lmcache;print(lmcache.__file__)'],cwd=ROOT,text=True).splitlines()[-1]).parent
  dependency_files={str(p.relative_to(dependency)):file_hash(p) for p in dependency.rglob('*') if p.is_file() and p.suffix in ('.py','.so')}
  write_json(out/'lmcache-source.json',dict(path=str(dependency),sha256=dependency_files))
  with tarfile.open(out/'lmcache-executed-python.tar.gz','w:gz',dereference=True) as archive:
   for name in dependency_files:
    if name.endswith('.py'):archive.add(dependency/name,arcname='lmcache/'+name,recursive=False)
  assert source_identity()['digest']==initial['digest']
  case_plan=plan_for_variant(plan,'continuum');selected=select_sessions(json.loads((out/'corpus.json').read_text()),plan,0)
  original_ids=next(c['source_ids'] for c in original['cases'] if c['variant']=='continuum');assert [s['id'] for s in selected]==original_ids
  started=time.monotonic();case=run_case(case_plan,selected,'candidate',0,out,variant='continuum')
  assert source_identity()['digest']==initial['digest'];check_source('continuum',runtime)
  assert dependency_files=={str(p.relative_to(dependency)):file_hash(p) for p in dependency.rglob('*') if p.is_file() and p.suffix in ('.py','.so')}
  records=[json.loads(l) for l in (out/'00-continuum/runtime-observation.jsonl').read_text().splitlines()]
  decode=[r for r in records if r['event']=='lmcache_decode_publication' and r['request_id'].startswith('sc|suite|')]
  if not decode:
   case['evaluation']['status']='inconclusive';case['evaluation']['errors'].append('No actual decode-publication activation observed')
  result=dict(schema=SCHEMA,plan=plan,cases=[case],status=case['evaluation']['status'],hardware=declaration['hardware_before'],environment=original['environment'],wall_s=time.monotonic()-started,not_started=[],replacement_declaration='replacement-declaration.json',decode_publications=len(decode),scope=plan['measurement_scope'])
  write_json(out/'result.json',result);(out/'REPORT.md').write_text(render_report(result))
  write_json(out/'replacement-completion.json',dict(previous_four_case_manifest_unchanged=file_hash(old/'artifacts.json')==declaration['original_manifest_sha256'],previous_four_case_verification=verify(old),source_unchanged=True,lmcache_source_unchanged=True,decode_publication_events=len(decode),hardware_after=available_gpu(),cleanup=case['cleanup']))
  seal(out);print(json.dumps(verify(out),indent=2),flush=True);print(json.dumps(dict(evaluation=case['evaluation'],batch_s=case['load_duration_s'],summary=case['summary'],decode_publications=len(decode))),flush=True)
if __name__=='__main__':main()
