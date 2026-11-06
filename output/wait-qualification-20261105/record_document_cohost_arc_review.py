import json,hashlib,copy
from pathlib import Path
b=Path(__file__).parent;p=b/'independent-manual-review.json';a=json.loads(p.read_text());known={}
for c in a['cases']:
 if c['campaign']=='stage3-documents':
  for d in c['details']:known.setdefault((d['source_id'],d['text_sha256']),(d,c['campaign']+'/'+c['case']))
root=b/'stage3-documents-cohost';assert json.loads((root/'corpus.json').read_text())==json.loads((b/'stage3-documents/corpus.json').read_text())
notes={
'pep-0343':('supported','supported','Same entered-with cleanup and truthy __exit__ exception-suppression semantics, with text-format variation; not an observed implementation failure.'),
'pep-0517':('supported','partly_incorrect','Build-backend/source-tree format supported. Read-only sources MAY occur; support is SHOULD and backends MAY opt out237ff, not MUST or an invariably read-only source tree. Metadata validation does not establish content/implementation correctness.'),
'pep-0508':('supported','supported','Name/extras/versions/markers supported. extra raises an error outside defined special handling, explicitly241ff. Actual checks validate metadata, not every criterion of the dependency grammar.'),
'pep-0526':('partly_incorrect','supported','Instance variable annotation is supported but displayed __init__ parameter annotations do not demonstrate it. Bare local annotation does not initialize; supplied UnboundLocalError example is consistent with source.')}
assert not any(c['campaign']==root.name and c['case']=='00-arc' for c in a['cases']);details=[]
for line in (root/'00-arc/requests.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r.get('turn')!=3:continue
 sid=r['source_id'];sha=hashlib.sha256(r['generated_text'].encode()).hexdigest();key=sid,sha
 if key in known:
  prior,origin=known[key];d=copy.deepcopy(prior);d['same_answer_as']=origin;d['review_method']='Exact source_id/text SHA reuse on identical frozen document corpus.'
 else:
  f,l,n=notes[sid];d=dict(source_id=sid,requested_subject_present=True,requested_metadata_correct=True,observed_test_outcome_correct=True,substantiated_behavior=f,limitation=l,source_audit=n,word_count=len(r['generated_text'].split()),text_sha256=sha)
 details.append(d)
assert len(details)==16
a['cases'].append(dict(campaign=root.name,case='00-arc',details=details));p.write_text(json.dumps(a,indent=2)+'\n');print('Recorded16 cohost ARC finals;12 exact reuses and4 new source checks.')
r=json.loads((root/'result.json').read_text());c=r['cases'][0];v=c['agent']['cache'];print({k:v.get(k) for k in ['first_admission_sources','locally_executed_prefill_tokens','physical_demand_load_bytes','preemptions','load_attribution_gap_bytes']})
