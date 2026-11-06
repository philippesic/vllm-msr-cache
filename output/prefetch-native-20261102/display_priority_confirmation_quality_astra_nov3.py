import json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
q=json.loads((H/'live-nov3-priority-confirmation-quality-samples.json').read_text());by={}
for r in q['runs']:
 for x in r.get('samples',[]):by.setdefault(x['source_id'],{})[(x['turn'],x['generated_text'])]=x
ids=sorted(by)
for i in range(int(sys.argv[1]),min(int(sys.argv[2]),len(ids))):
 sid=ids[i];xs=by[sid];print('\nSOURCE',i,sid)
 longest=max(xs.values(),key=lambda x:len(x['source_evidence_turns']))
 print('INITIAL',longest['source_evidence_turns'][0]['user_text'])
 for turn in sorted({t for t,txt in xs if t}):
  x=next(x for (t,txt),x in xs.items() if t==turn);print('LATEST turn',turn,x['source_evidence_turns'][turn]['user_text'])
 for (turn,txt),x in xs.items():print('GENERATED turn',turn,txt)
