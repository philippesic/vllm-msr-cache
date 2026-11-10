"""Consolidate the fully read, isolated confirmation source review18 once."""
from pathlib import Path
import sys
repo_root=Path(__file__).resolve().parents[2]
frozen_source=repo_root/'output/stage6-5080-20261108/primary-execution-bundle-01/source'
assert Path.cwd()==frozen_source.resolve()and sys.dont_write_bytecode
assert sys.executable=='/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
sys.path[:]=[str(frozen_source)]+[v for v in sys.path if Path(v).resolve()!=Path(__file__).resolve().parent]

from pathlib import Path
import json,copy,hashlib,shutil
from benchmarks.serving_suite.campaign_contracts import load,digest,atomic_json
from benchmarks.serving_suite.campaign_quality import RUBRIC,finish_review
from benchmarks.serving_suite.campaign_capped_review import instruction_source
assert all(Path(f.__code__.co_filename).resolve().is_relative_to(frozen_source.resolve())for f in(load,finish_review,instruction_source))
repo=Path('/home/philip/vllm-semantic-cache-ext');e=repo/'evidence/stage6-5080-20261109';b=repo/'output/stage6-5080-20261109/quality-confirm-checkpoint-18';out=repo/'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
assert out.parent.is_mount()
for name in ('confirm-source-judgments-18.json','confirm-review-index-18.json','confirm-quality-checkpoint-18.json'):
 assert not(e/name).exists(),'Preserve earlier completed records'
assert not(b.parent/'quality-confirm-audit-checkpoint-18').exists()
live=out/'analysis/quality'
def inventory(p):
 return {str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest()for f in p.rglob('*')if f.is_file()}if p.exists()else{}
live_before=inventory(live)
packet=load(b/'blind-packet.json');prior=load(e/'confirm-review-index-17.json');fields=prior['signature_fields'];pd=digest(packet['items']);known={};provenance={}
for stem,no in [('screen','07')]+[('confirm',f'{i:02}')for i in range(1,18)]:
 idx=load(e/f'{stem}-review-index-{no}.json');by={r['review_id']:r for r in load(e/f'{stem}-source-judgments-{no}.json')['ratings']}
 assert idx['executed_system_instruction_digest']==prior['executed_system_instruction_digest']and idx['executed_instruction_source_sha256']==prior['executed_instruction_source_sha256']
 for ctx in idx['contexts']:
  r=by[ctx['canonical_review_id']]
  assert all(by[v][k]==r[k]for v in ctx['review_ids']for k in ('source_supported','instruction_followed','useful','unsupported_claims'))
  if ctx['signature']in known:assert all(known[ctx['signature']][k]==r[k]for k in ('source_supported','instruction_followed','useful'))
  known[ctx['signature']]=r;provenance[ctx['signature']]=f'{stem}-review-index-{no}.json'

ci=load(e/'caps-all-review-index-08.json');cb=load(e/'caps-instruction-bindings-08.json');cp=load(Path(ci['raw_packet']))
assert ci['quality_signature_fields']==fields and digest(cp['items'])==ci['packet_digest']
assert cb['executed_system_instruction_digest']==prior['executed_system_instruction_digest']and cb['executed_instruction_source_sha256']==prior['executed_instruction_source_sha256']
cr={r['review_id']:r for r in load(e/'caps-all-source-judgments-08.json')['ratings']}
caps={}
for x in cp['items']:
 sig=digest({k:x[k]for k in fields});r=cr[x['review_id']]
 if sig in caps:assert all(caps[sig][k]==r[k]for k in ('source_supported','instruction_followed','useful'))
 caps[sig]=r
 if sig in known:assert all(known[sig][k]==r[k]for k in ('source_supported','instruction_followed','useful'))
 else:known[sig]=r;provenance[sig]='caps-all-review-index-08.json'

novels=load(b/'novel-contexts.json');new={};indices=set()
for path in sorted(e.glob('confirm-investigation-judgments-18-part-*.json')):
 manual=load(path);assert manual['packet_digest']==pd and manual['signature_fields']==fields
 for r in manual['ratings']:
  i=r['index'];x=novels[i];sig=digest({k:x[k]for k in fields});assert sig==r['signature']and x['review_id']==r['review_id']and i not in indices and sig not in known and sig not in new
  indices.add(i);new[sig]=r;provenance[sig]=path.name
assert len(new)==len(novels)==55 and indices==set(range(len(novels)));known.update(new)
components=load(e/'confirm-review-components-18.json');assert components['packet_digest']==pd and len(components['contexts'])==len(novels)
old_components={no:load(b.parent/f'quality-confirm-checkpoint-{no}/novel-contexts.json')for no in ('02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17')};old_components['18']=novels;old_components['cap08']=cp['items']
for bind in components['contexts']:
 i=bind['index'];x=novels[i];assert x['review_id']==bind['review_id']and set(bind['components'])==set(fields)-{'answer'}
 for key,part in bind['components'].items():
  assert digest(x[key])==part['digest'];old=part['prior_explicitly_read']
  if old:
   if old['snapshot']=='18':assert old['index']<i
   assert x[key]==old_components[old['snapshot']][old['index']][key]
 for part in bind['dependency_components']:
  d=part['dependency'];assert digest(x['actual_dependencies'][d])==part['digest'];old=part['prior_explicitly_read']
  if old:
   if old['snapshot']=='18':assert old['index']<i or(old['index']==i and old['dependency']<d)
   assert x['actual_dependencies'][d]==old_components[old['snapshot']][old['index']]['actual_dependencies'][old['dependency']]
protocol=load(out/'protocol.lock.json');sha=prior['executed_instruction_source_sha256'];system=prior['executed_system_instruction_digest'];assert protocol['source']['files']['benchmarks/serving_suite/campaign_workload.py']==sha
labels=load(b/'private-labels.json');attempts={v['attempt']for v in labels.values()};paths={p.parent.name:p.parent for p in (out/'attempts').glob('*/*/identity.json')};assert len(attempts)==470
for a in sorted(attempts):
 d=paths[a];identity=load(d/'identity.json');assert identity['phase']=='confirm'and identity['source']==protocol['source']['digest'];snippet,actual=instruction_source(d/'measurement');assert actual==sha and digest(snippet)==system
cap_matches=0
for sig,r in new.items():
 if sig in caps:
  assert all(r[k]==caps[sig][k]for k in ('source_supported','instruction_followed','useful'));cap_matches+=1
rows=[];groups={}
for x in packet['items']:
 sig=digest({k:x[k]for k in fields});r=known[sig];v={k:copy.deepcopy(r[k])for k in ('source_supported','instruction_followed','useful','unsupported_claims','notes')};v['review_id']=x['review_id'];v['notes']+=' Exact complete-context and archived-instruction binding: '+provenance[sig]+'.';rows.append(v);groups.setdefault(sig,[]).append(x['review_id'])
s=load(b/'summary.json');assert pd==s['packet_digest']and len(rows)==s['blind_review_items']==6667 and len(groups)==834 and len(groups)==len({digest({k:x[k]for k in fields})for x in packet['items']})
ratings=dict(schema='stage6-source-audit-v1',reviewer='Codex explicitly read every new blinded source/test/instruction/dependency/tool/answer context; exact full-context and archived-instruction reuse for prior judgments',rubric_digest=digest(RUBRIC),ratings=rows)
findings=dict(source_unsupported=sum(not r['source_supported']for r in rows),instruction_failures=sum(not r['instruction_followed']for r in rows),not_useful=sum(not r['useful']for r in rows))
prior_sources={digest(x['source_context'])for no in ('02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17')for x in old_components[no]};prior_sources.update(digest(x['source_context'])for x in cp['items']);sources={digest(x['source_context'])for x in novels}
idx=dict(schema='stage6-5080-partial-confirm-source-review-v1',phase='confirm',acquisition_complete=False,packet_digest=pd,packet_file_sha256=hashlib.sha256((b/'blind-packet.json').read_bytes()).hexdigest(),signature_fields=fields,executed_system_instruction_digest=system,executed_instruction_source_sha256=sha,contexts=[dict(signature=sig,canonical_review_id=ids[0],review_ids=ids)for sig,ids in groups.items()],reviewed_outputs=len(rows),distinct_full_contexts=len(groups),newly_judged_contexts=len(novels),explicitly_read_contexts=len(novels),raw_packet=str((b/'blind-packet.json').resolve()),review_findings=findings,objective_outputs=s['objective_outputs'],automatic_final_failures=s['automatic_failures'],capped_intermediate_outputs=s['capped_intermediate_outputs'],archived_instruction_sources_verified=len(attempts),prior_prose_correction=prior['prior_prose_correction'],review_component_binding='confirm-review-components-18.json',distinct_supplied_sources_exactly_bound_to_prior_explicit_readings=len(sources&prior_sources),distinct_supplied_sources_fully_read_without_prior_component_binding=len(sources-prior_sources),novel_group_findings={k:sum(not r[k]for r in new.values())for k in ('source_supported','instruction_followed','useful')},exact_previously_judged_cap_contexts_rechecked=cap_matches,scope='Static partial confirmation snapshot includes all60 terminal Ministral configurations,28/30 valid pairs and both B06 exclusions; all129 valid Qwen3-8B blocks and 27 current Qwen3-4B blocks selected by the frozen rules. All55 novel full contexts (four QA/51 investigation finals) versus union screen07/confirm01-through17 and completed cap08 explicitly read/judged. Complete fields and individual preceding dependencies bind prior explicitly read confirm02-through17/cap08 or earlier18; every unmatched component and answer separately read. No automatic novel judgment. All cap08 reuse checks the same archived full workload/SYSTEM identity. Archived source/SYSTEM checked for every current attempt and current IDs rebound. Whole-block setup recovery, unmatched arms and three genuine activity exclusions retain original treatment. Original objective/timing/live quality unchanged. Selected duplicate-inclusive findings are not population rates, quality equivalence or corrected throughput. Final confirmation still requires terminal acquisition and a fresh final review.')
audit=b.parent/'quality-confirm-audit-checkpoint-18';assert not audit.exists()
for name in ('confirm-source-judgments-18.json','confirm-review-index-18.json','confirm-quality-checkpoint-18.json'):assert not(e/name).exists()
atomic_json(e/'confirm-source-judgments-18.json',ratings);atomic_json(e/'confirm-review-index-18.json',idx)
dest=audit/'analysis/quality';dest.mkdir(parents=True)
for name in ('blind-packet.json','summary.json','private-labels.json','objective.json','rubric.json'):shutil.copy2(b/name,dest/name)
result=finish_review(audit,e/'confirm-source-judgments-18.json');assert result['status']=='audited'and result['reviewed']==len(rows) and result['phase']=='confirm'
qsha=hashlib.sha256((Path.cwd()/'benchmarks/serving_suite/campaign_quality.py').read_bytes()).hexdigest();assert qsha==protocol['source']['files']['benchmarks/serving_suite/campaign_quality.py']
atomic_json(e/'confirm-quality-checkpoint-18.json',dict(schema='stage6-5080-partial-quality-validator-v1',phase='confirm',acquisition_complete=False,status=result['status'],reviewed=result['reviewed'],packet_digest=pd,ratings_digest=digest(ratings),isolated_workspace=str(audit.resolve()),quality_module_sha256=qsha,archived_instruction_sources_verified=len(attempts),scope='Frozen finish validator on isolated materialized partial snapshot; original acquisition/live analysis-quality unchanged. Not a final confirmation audit.',review_findings=findings))
assert inventory(live)==live_before
print(json.dumps(dict(contexts=len(groups),novels=len(new),reviewed=len(rows),findings=findings,archived_instruction_sources_verified=len(attempts),supplied_sources_bound_to_prior=len(sources&prior_sources),new_unmatched_source_components=len(sources-prior_sources),cap_matches=cap_matches,validator=result['status']),indent=2))
