import json,hashlib,copy
from pathlib import Path
b=Path(__file__).parent;p=b/'independent-manual-review.json';a=json.loads(p.read_text());known={}
for c in a['cases']:
 if c['campaign'].startswith('stage3-documents'):
  for d in c['details']:known.setdefault((d['source_id'],d['text_sha256']),(d,c['campaign']+'/'+c['case']))
root=b/'stage3-documents-cohost';assert json.loads((root/'corpus.json').read_text())==json.loads((b/'stage3-documents/corpus.json').read_text())
notes={
'pep-0020':('supported','supported_metadata_boundary','Obvious-way aphorism and absent Python-Version metadata supported; not a runtime compatibility claim.'),
'pep-0380':('supported','supported','Delegation/refactoring and GeneratorExit nonreraising exception supported by source. Exceeds 70-word limit.'),
'pep-0343':('supported','supported_with_qualification','Entered-with cleanup and truthy __exit__ suppression supported. Suppression is specified behavior; calling it a silent-failure defect would need application context. Exceeds length constraint.'),
'pep-0440':('partly_incorrect','supported_with_qualification','Public identifier grammar incorrectly makes epoch mandatory by omitting its optional brackets: source has [N!]. SemVer/date versioning not mandated, but inconsistency is speculative. Reprinted tool command does not rerun it.'),
'pep-0257':('supported_with_qualification','supported_metadata_boundary','First-statement string is docstring; triple quotes are recommendation and one-line form where appropriate. Active/Informational does not certify tools/runtime implementation. Long echoed answer is generated text, not executed tool.'),
'pep-0492':('supported','supported','async def always coroutine even without await; await outside async def is SyntaxError in frozen historical PEP. Command in final text not executed.'),
'pep-0498':('supported','supported_with_qualification','Shared format mini-language and expression-backslash restriction supported in frozen PEP498. Historical restriction does not establish current Python after PEP701.'),
'pep-0517':('supported','partly_incorrect','Build-backend requirement/legacy fallback supported. Read-only source MAY occur; backends SHOULD be prepared and MAY opt out, so must is overstatement237ff.'),
'pep-0484':('supported','supported','Implicit self/cls types109ff and no runtime checking75ff supported; duplicated echoed answer substantially exceeds70 words.'),
'pep-0518':('supported','unsupported','Build dependencies in pyproject supported. Claim pip may not recognize alternative build systems and specification not yet fully supported is not established by this source or metadata-only test; no tool-version evidence.')}
assert not any(c['campaign']==root.name and c['case']=='00-upstream' for c in a['cases']);details=[]
for line in (root/'00-upstream/requests.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r.get('turn')!=3:continue
 sid=r['source_id'];sha=hashlib.sha256(r['generated_text'].encode()).hexdigest();key=sid,sha
 if key in known:
  prior,origin=known[key];d=copy.deepcopy(prior);d['same_answer_as']=origin;d['review_method']='Exact source_id/text SHA reuse on identical frozen document corpus.'
 else:
  f,l,n=notes[sid];d=dict(source_id=sid,requested_subject_present=True,requested_metadata_correct=True,observed_test_outcome_correct=True,substantiated_behavior=f,limitation=l,source_audit=n,word_count=len(r['generated_text'].split()),text_sha256=sha)
 details.append(d)
assert len(details)==16
a['cases'].append(dict(campaign=root.name,case='00-upstream',details=details));p.write_text(json.dumps(a,indent=2)+'\n');print('Recorded16 cohost upstream finals;6 exact reuses/10 new checks.')
