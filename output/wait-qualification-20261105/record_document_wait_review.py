import json,hashlib,copy
from pathlib import Path
b=Path(__file__).parent;p=b/'independent-manual-review.json';a=json.loads(p.read_text());known={}
for c in a['cases']:
 if c['campaign']=='stage3-documents':
  for d in c['details']:known[(d['source_id'],d['text_sha256'])]=(d,c['case'])
notes={
'pep-0420':('supported','weak','PEP202 explicitly requires dynamic path recomputation before portions, permitting caching when unchanged. Dependency-complexity concern remains speculative, as in ARC.'),
'pep-0440':('supported','contradicted','Public-version grammar supported. Claimed lack of automated-tool/implementation guidance contradicted by explicit automated-tool requirements, reference implementation and parsing regex Appendix B. Actual validation checks only metadata; it does not prove full content compliance.'),
'pep-0517':('supported_with_qualification','weak','flit.api:main means import flit.api then access main, not import a separate flit.api.main module. No mandated backend is true; inconsistent behavior speculation is not a demonstrated limitation.'),
'pep-0567':('supported','supported','No direct current-Context reference and copy_context shallow-copy alternative are explicit113ff.'),
'pep-0508':('supported','supported_historical_document_scope','Dependency components and historical wheel-extra specification boundary match frozen PEP241ff; no claim about later packaging standards.'),
'pep-0526':('partly_incorrect','supported','New variable annotations/instance fields supported, but supplied __init__ parameter annotations are not an example of annotating an instance variable. Annotation without assignment does not initialize a local, so access can raise UnboundLocalError, as source explains.'),
'pep-0518':('supported','contradicted','Build-system dependency declaration supported. Frontends install declared dependencies; asserting that users must manually manage them and configure multiple tools is unsupported and conflicts with the automation objective.')}
assert not any(c['campaign']=='stage3-documents' and c['case']=='00-wait' for c in a['cases']);details=[]
for line in (b/'stage3-documents/00-wait/requests.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r.get('turn')!=3:continue
 sid=r['source_id'];sha=hashlib.sha256(r['generated_text'].encode()).hexdigest();key=sid,sha
 if key in known:
  prior,origin=known[key];d=copy.deepcopy(prior);d['same_answer_as']='stage3-documents/'+origin;d['review_method']='Exact source_id/text SHA reuse on identical frozen PEP corpus and actual passing tools.'
 else:
  f,l,n=notes[sid];d=dict(source_id=sid,requested_subject_present=True,requested_metadata_correct=True,observed_test_outcome_correct=True,substantiated_behavior=f,limitation=l,source_audit=n,word_count=len(r['generated_text'].split()),text_sha256=sha)
 details.append(d)
assert len(details)==16
a['cases'].append(dict(campaign='stage3-documents',case='00-wait',details=details));p.write_text(json.dumps(a,indent=2)+'\n')
p=b/'manual-quality-adjudications.json';a=json.loads(p.read_text())
for case,sids in [('00-arc',['pep-0484','pep-0526']),('00-wait',['pep-0484'])]:
 rows={r['source_id']:r for l in (b/'stage3-documents'/case/'requests.jsonl').read_text().splitlines() if (r:=json.loads(l)).get('turn')==3}
 for sid in sids:
  assert not any(d['campaign']=='stage3-documents' and d['case']==case and d['source_id']==sid for d in a)
  a.append(dict(campaign='stage3-documents',case=case,source_id=sid,frozen_scorer_result='unparseable_status',manual_observed_status='PASS',manual_correct=True,text_sha256=hashlib.sha256(rows[sid]['generated_text'].encode()).hexdigest(),reason='Explicitly reviewed PASS phrase/quoted JSON matches actual successful metadata validation. Frozen normalized-status-v2 formatting grade retained; not a factual-content or implementation-success adjudication.'))
p.write_text(json.dumps(a,indent=2)+'\n');print('Recorded16 wait finals (9 exact reuses/7 new audits), plus3 separate scorer-format adjudications.')
