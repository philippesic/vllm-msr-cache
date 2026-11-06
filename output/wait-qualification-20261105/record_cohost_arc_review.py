import json,hashlib,copy
from pathlib import Path
b=Path(__file__).parent;p=b/'independent-manual-review.json';r=json.loads(p.read_text());known={}
for c in r['cases']:
 for d in c['details']:known.setdefault((d['source_id'],d['text_sha256']),(d,c['campaign']+'/'+c['case']))
root=b/'stage3-repository-cohost'
assert json.loads((root/'corpus.json').read_text())==json.loads((b/'stage3-repository/corpus.json').read_text())
notes={
'windowed':('supported','contradicted','Same absent-large-step claim; WindowedTests.test_step explicitly covers step8 on length7.'),
'bucket':('partly_incorrect','weak','Generic validator behavior is supported, but test_validator checks leading digit0<key<10, not an odd-number filter as asserted. No performance benchmark proves the broad size claim.'),
'split_when':('supported','weak','Empty input is tested; generic finite predicate-coverage limitation as in earlier answers.'),
'distinct_permutations':('supported','supported_with_qualification','Unsortable input is tested. Docstring explicitly warns about equal/nonidentical-element identity surprises; do not infer an input-order guarantee.'),
'split_at':('incorrect','supported_as_input_constraint','Empty input yields a group containing the empty buffer ([[]]), not an empty list, consistent with test_basic str.split expectation. Requiring iterable input is an API constraint, not an observed defect.'),
'numeric_range':('supported','partly_unsupported','Float use is supported and precision warnings documented; lack of explicit handling is not established and inexact-step tests exist.'),
'seekable':('incorrect','contradicted','test_past_end shows seeking20 yields exhaustion, then an explicit seek0 resets. Seeking past end does not itself reset to beginning as asserted.')}
rows=[json.loads(x) for x in (root/'00-arc/requests.jsonl').read_text().splitlines()];details=[]
for row in rows:
 if row.get('turn')!=3:continue
 sid=row['source_id'];sha=hashlib.sha256(row['generated_text'].encode()).hexdigest();key=sid,sha
 if key in known:
  prior,origin=known[key];d=copy.deepcopy(prior);d['same_answer_as']=origin;d['review_method']='Exact source_id/text_sha256 reuse; identical frozen corpus.'
 else:
  fact,limit,note=notes[sid.removeprefix('itertools-')];d=dict(source_id=sid,requested_subject_present=True,observed_test_outcome_correct=True,substantiated_behavior=fact,limitation=limit,source_audit=note,word_count=len(row['generated_text'].split()),text_sha256=sha)
 details.append(d)
assert len(details)==16
r['cases'].append(dict(campaign='stage3-repository-cohost',case='00-arc',details=details));p.write_text(json.dumps(r,indent=2)+'\n');print('Recorded cohost first-ARC review:9 exact reuses and7 explicitly checked new answers.')
