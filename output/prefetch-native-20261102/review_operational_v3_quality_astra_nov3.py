"""Astra's fixed-sample grounded review; no automated accuracy claim."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=H/'live-nov3-operational-v3-quality-samples.json';x=json.loads(p.read_text())
old=json.loads((H/'live-nov3-operational-v2-quality-review.json').read_text());os=json.loads((H/'live-nov3-operational-v2-quality-samples.json').read_text())
known={(q['source_id'],q['turn'],q['text_sha256']):q for q in old['reviews']}
oldsource={(q['source_id'],q['turn'],q['generated_text_sha256']):q for q in os['selected']}
new={5:'Latest grep contains MinMaxBase/Min, not Max definition; lack of verified zero-argument behavior correctly limited.',6:'Reasonably separates initial reported fixture regression from empty exit0 latest observation.',7:'Precision issue is in initial report; latest function/module excerpt supplies location, so asking for exact files is weak responsiveness.',
10:'Issue requests adding joblib to version reporting OR template; output turns proposal into both confirmed absences. No independent current implementation evidence.',
11:'Defect: reverses/misstates database evidence. Source says problem disappears if data already exists in defaultDB, not that failure shows data present there and absent in otherDB. No source proof of nondefault data absence.',
12:'Faithful reported shared attrs mutation; no fix claimed.',13:'Faithful reported traceback/version comparison.',
14:'Defect: latest observation is None; earlier directory listing cannot prove dependency absence from repository. Confuses missing version reporting with absent dependency.',
17:'Same database-condition defect as11; latest grep only shows natural-key definitions/calls, not actual database contents.',
18:'Correctly limits class-header observation; no runtime verification claimed.',25:'Min inheritance supported, Max definition not shown by latest grep; behavior uncertainty appropriate.',
27:'Reported precision issue and unresolved implementation location reasonable; no invented fix or command execution.',
28:'Code fence/multipleparagraph violation. Slicing call is outside function at shown callsite; ambiguous wording conflates it with function body. Initial traceback identifies actual array-comparison failure.',
35:'Initial SQL report supported; grep class headers insufficient to confirm implementation details or linked-PR fix. No executed fix claim.',
37:'Signature default is not proof of runtime configuration; correctly says latest excerpt does not resolve behavior.',
47:'Defect: attributes ValueError to slicing operation rather than array truth comparison inside num_mock_patch_args from initial traceback. Fencedcode/multipleparagraph violation.',
49:'Faithful reported SQL behavior.',54:'Reported SQL problem and explicitly limited class-header grep are grounded.'}
reviews=[];seen=set();reused=0
for i,q in enumerate(x['selected']):
 key=(q['source_id'],q['turn'],q['generated_text_sha256'])
 if key in seen:continue
 seen.add(key)
 if key in known:
  prior=oldsource[key]
  assert q['initial_issue']==prior['initial_issue'] and q['latest_observation']==prior['latest_observation']
  note=known[key]['assessment'];origin='Exact source/turn/text and issue/latest-evidence match to prior grounded review';reused+=1
 else:assert i in new,i;note=new[i];origin='New distinct text read against issue/latest evidence'
 reviews.append({'selection_index':i,'source_id':q['source_id'],'turn':q['turn'],'text_sha256':key[2],'assessment':note,'review_origin':origin})
assert len(reviews)==35 and reused==17
analysis=json.loads((H/'live-nov3-operational-v3-independent-analysis.json').read_text())
out={'schema':'astra-operational-v3-quality-review-v1','selection_sha256':sha(p),'selected_observations':len(x['selected']),'distinct_texts_reviewed':len(reviews),'prior_exact_source_text_review_reused':reused,'new_distinct_texts_read':18,'distinct_issues':len({q['source_id'] for q in x['selected']}),'reviews':reviews,
 'all_normal_format_counts':{n:{k:r['format'][k] for k in ('responses','starts_status','code_fences','multiple_blank_blocks','degeneracy_flags')} for n,r in analysis['cases'].items()},
 'conclusion':'No conspicuous looping or service claim of executing edits/commands in selected texts. Important factual/source-reading errors and occasional code-format violations remain in both-arm evidence. Supports only human-reviewed drafts; no autonomous correctness, task-success, quality-equivalence or productivity claim. No outcome excluded or used to tune serving.'}
(H/'live-nov3-operational-v3-quality-review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'selected':len(x['selected']),'distinct':len(reviews),'new':18,'reused_exact':reused}))
