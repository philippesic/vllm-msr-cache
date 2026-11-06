import json,hashlib,copy
from pathlib import Path
b=Path(__file__).parent;p=b/'independent-manual-review.json';r=json.loads(p.read_text());known={}
for c in r['cases']:
 for d in c['details']:known.setdefault((d['source_id'],d['text_sha256']),(d,c['campaign']+'/'+c['case']))
root=b/'stage3-repository-cohost'
assert json.loads((root/'corpus.json').read_text())==json.loads((b/'stage3-repository/corpus.json').read_text())
notes={
 '00-wait':{
 'bucket':('supported','weak','Validator excludes invalid items without caching them. Generic unmeasured large-dataset concern; no concrete missing-case proof or supported performance boundary.'),
 'chunked':('supported','supported_narrow_scope','test_odd covers a short final chunk. The selected ChunkedTests class has no direct empty-input or n=0 case; narrow coverage statement only, not an observed defect.')},
 '00-upstream':{
 'chunked':('supported','supported_narrow_scope','Actual tests cover the listed input/strict cases. Selected class has no direct empty case; very-large iterable claim is generic and does not prove a defect.'),
 'split_when':('supported','weak','Predicate splitting is supported. Blanket incomplete edge coverage and speculative unexpected behavior are not concrete source-established limitations; maxsplit tests exist.'),
 'map_reduce':('supported','supported','Source groups values into lists before reduction, so materialized storage cost is real.'),
 'ichunked':('supported','contradicted','Negative n raises ValueError. test_odd explicitly covers nondivisible input; ichunked has no strict parameter.'),
 'seekable':('supported','contradicted','Seeking beyond the end is allowed. test_elements uses map(str,count()) without maxlen, contradicting absent unbounded infinite-iterator test.'),
 'set_partitions':('supported_with_qualification','incorrect','k<1 raises ValueError; blanket group-size wording is imprecise. Claimed n/strict divisibility parameters belong to a different API, not set_partitions.'),
 'peekable':('supported','contradicted','Indexing does not advance. Both named tests are implemented, including bounded reads from count() and prepend after exhaustion.')}
}
for case,note in notes.items():
 assert not any(c['campaign']==root.name and c['case']==case for c in r['cases'])
 details=[]
 for line in (root/case/'requests.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row.get('turn')!=3:continue
  sid=row['source_id'];sha=hashlib.sha256(row['generated_text'].encode()).hexdigest();key=sid,sha
  if key in known:
   prior,origin=known[key];d=copy.deepcopy(prior);d['same_answer_as']=origin;d['review_method']='Exact source_id/text_sha256 reuse; identical frozen corpus.'
  else:
   fact,limit,text=note[sid.removeprefix('itertools-')];d=dict(source_id=sid,requested_subject_present=True,observed_test_outcome_correct=True,substantiated_behavior=fact,limitation=limit,source_audit=text,word_count=len(row['generated_text'].split()),text_sha256=sha)
  details.append(d)
 assert len(details)==16
 r['cases'].append(dict(campaign=root.name,case=case,details=details))
p.write_text(json.dumps(r,indent=2)+'\n');print('Recorded32 first-repetition finals;9 new source audits,23 exact reuses.')
