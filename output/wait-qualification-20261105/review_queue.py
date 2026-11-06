"""List only unaudited final texts; never assigns factual grades."""
import json,hashlib,sys
from pathlib import Path
b=Path(__file__).parent;name,case=sys.argv[1:3];root=b/name;corpus=json.loads((root/'corpus.json').read_text());audit=json.loads((b/'independent-manual-review.json').read_text());known={}
for c in audit['cases']:
 if json.loads((b/c['campaign']/'corpus.json').read_text())!=corpus:continue
 for d in c['details']:known.setdefault((d['source_id'],d['text_sha256']),(c['campaign'],c['case']))
new=[];reused=[]
for l in (root/case/'requests.jsonl').read_text().splitlines():
 r=json.loads(l)
 if r.get('turn')!=3:continue
 key=r['source_id'],hashlib.sha256(r['generated_text'].encode()).hexdigest()
 if key in known:reused.append(dict(source_id=key[0],sha256=key[1],origin=known[key]))
 else:new.append(dict(source_id=key[0],sha256=key[1],words=len(r['generated_text'].split()),text=r['generated_text']))
print(json.dumps(dict(campaign=name,case=case,exact_reuses=reused,need_manual_source_audit=new),indent=2))
