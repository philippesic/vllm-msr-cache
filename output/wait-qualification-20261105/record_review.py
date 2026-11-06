"""Record explicit author notes or exact prior audits, never infer factual grades."""
import copy,hashlib,json,sys
from pathlib import Path
b=Path(__file__).parent;name,case,notes_path=sys.argv[1:4];root=b/name;notes=json.loads(Path(notes_path).read_text());p=b/'independent-manual-review.json';audit=json.loads(p.read_text());corpus=json.loads((root/'corpus.json').read_text());known={}
assert not any(c['campaign']==name and c['case']==case for c in audit['cases'])
for c in audit['cases']:
 if json.loads((b/c['campaign']/'corpus.json').read_text())!=corpus:continue
 for d in c['details']:known.setdefault((d['source_id'],d['text_sha256']),(d,c['campaign']+'/'+c['case']))
rows=[json.loads(l) for l in (root/case/'requests.jsonl').read_text().splitlines()];finals=[r for r in rows if r.get('turn')==3];assert len(finals)==16;details=[];used=set();reuse=0
for r in finals:
 key=r['source_id'],hashlib.sha256(r['generated_text'].encode()).hexdigest()
 if key in known:
  prior,origin=known[key];d=copy.deepcopy(prior);d.update(same_answer_as=origin,review_method='Exact source_id/text SHA reuse on identical frozen corpus; substantive original grades unchanged.');reuse+=1
 else:
  d=copy.deepcopy(notes[key[0]]);used.add(key[0]);assert all(k in d for k in ['requested_subject_present','observed_test_outcome_correct','substantiated_behavior','limitation','source_audit']);d.update(source_id=key[0],text_sha256=key[1],word_count=len(r['generated_text'].split()))
 details.append(d)
assert used==set(notes),'Every supplied note must correspond to a new final, and vice versa'
audit['cases'].append(dict(campaign=name,case=case,details=details));p.write_text(json.dumps(audit,indent=2)+'\n');print('Recorded',name,case,'16finals;',reuse,'exact reuses;',16-reuse,'explicit author checks.')
