import json,hashlib,copy
from pathlib import Path
b=Path(__file__).parent;p=b/'independent-manual-review.json';a=json.loads(p.read_text());known={}
for c in a['cases']:
 if c['campaign'].startswith('stage3-documents'):
  for d in c['details']:known.setdefault((d['source_id'],d['text_sha256']),(d,c['campaign']+'/'+c['case']))
root=b/'stage3-documents-cohost';assert json.loads((root/'corpus.json').read_text())==json.loads((b/'stage3-documents/corpus.json').read_text())
notes={
'pep-0020':('supported','supported_metadata_boundary','Same obvious-way aphorism and absent Python-Version metadata with ASCII dash variation; does not establish unsupported runtime versions.'),
'pep-0380':('supported_with_qualification','supported','Formal delegation semantics and GeneratorExit nonreraising restriction supported. All situations wording is overbroad unless read with its own stated exception; PEP says as far as reasonably practicable.'),
'pep-0508':('supported','supported','Name/extras/version/marker summary and extra error outside defined context are supported. Metadata checks do not establish every content criterion.'),
'pep-0518':('supported','supported_with_qualification','Same rejected semantics-version rationale with spelling variation; incompatible semantics may use different key/table names, not only table renaming.'),
'pep-0567':('supported','contradicted','contextvars/Context rule supported, but claim that the mechanism does not affect the core interpreter and is limited to user-defined modules is false. PEP supplies interpreter/C-API integration and asyncio changes. Metadata validation cannot confirm implementation/content correctness.')}
assert not any(c['campaign']==root.name and c['case']=='00-wait' for c in a['cases']);details=[]
for line in (root/'00-wait/requests.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r.get('turn')!=3:continue
 sid=r['source_id'];sha=hashlib.sha256(r['generated_text'].encode()).hexdigest();key=sid,sha
 if key in known:
  prior,origin=known[key];d=copy.deepcopy(prior);d['same_answer_as']=origin;d['review_method']='Exact source_id/text SHA reuse on identical frozen document corpus.'
 else:
  f,l,n=notes[sid];d=dict(source_id=sid,requested_subject_present=True,requested_metadata_correct=True,observed_test_outcome_correct=True,substantiated_behavior=f,limitation=l,source_audit=n,word_count=len(r['generated_text'].split()),text_sha256=sha)
 details.append(d)
assert len(details)==16
a['cases'].append(dict(campaign=root.name,case='00-wait',details=details));p.write_text(json.dumps(a,indent=2)+'\n');print('Recorded16 cohost wait finals:11 exact reuses/5 new checks.')
r=json.loads((root/'result.json').read_text());out=[]
for c in r['cases']:
 v=c['agent']['cache'];out.append(dict(name=c['name'],batch_s=c['load_duration_s'],output_tokens=c['summary']['output_tokens'],**{k:v.get(k) for k in ['first_admission_sources','locally_executed_prefill_tokens','physical_demand_load_bytes','preemptions','load_attribution_gap_bytes']}))
(b/'document-cohost-first-pair-diagnosis.json').write_text(json.dumps(dict(cases=out,scope='First pair only, not final estimate. Native preemption makes first-admission counters, scheduled computation, terminal-episode restore attribution and physical transfers non-equivalent. Preserve signed attribution gaps, not a fabricated counter identity.'),indent=2)+'\n')
