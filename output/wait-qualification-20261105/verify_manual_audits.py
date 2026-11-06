"""Integrity check for author review records; does not prove factual judgments."""
import hashlib,json
from pathlib import Path
b=Path(__file__).resolve().parent;path=b/'independent-manual-review.json';a=json.loads(path.read_text());bycase={};corpora={};count=0;reuse=0
for case in a['cases']:
 key=case['campaign'],case['case']
 assert key not in bycase,'Duplicate reviewed case'
 root=b/case['campaign'];corpora.setdefault(case['campaign'],json.loads((root/'corpus.json').read_text()))
 rows=[json.loads(l) for l in (root/case['case']/'requests.jsonl').read_text().splitlines()]
 finals={r['source_id']:r for r in rows if r.get('turn')==3};details={d['source_id']:d for d in case['details']}
 assert len(finals)==len(details)==16,'Incomplete/duplicate first repetition review'
 assert set(finals)==set(details)
 for sid,d in details.items():
  row=finals[sid];assert hashlib.sha256(row['generated_text'].encode()).hexdigest()==d['text_sha256']
  assert len(row['generated_text'].split())==d['word_count'];assert row['success']
  count+=1
 bycase[key]=details
for case in a['cases']:
 for d in case['details']:
  if not d.get('same_answer_as'):continue
  origin=tuple(d['same_answer_as'].split('/'));prior=bycase[origin][d['source_id']]
  assert prior['text_sha256']==d['text_sha256'],'Nonexact answer reused as exact'
  assert corpora[case['campaign']]==corpora[origin[0]],'Different corpus reused without separate source validation'
  reuse+=1
result=dict(schema='wait-qualification-manual-review-integrity-v1',passed=True,cases=len(a['cases']),finals=count,exact_reuses=reuse,review_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope='Exact response/hash/word-count/corpus and reuse consistency only; author factual judgments remain separate from mechanical verification.')
(b/'manual-review-integrity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
