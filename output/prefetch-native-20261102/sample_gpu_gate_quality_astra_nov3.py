"""Predeclared slots0/12 turns0/2 plus longest3 in each GPU-gated-CPU-policy workflow."""
import gzip,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
m=json.loads((H/'gpu-gate-frozen-plan-nov3.json').read_text());samples=[]
corpus=json.loads((H/'fresh-agent-nov3/session-corpus.json').read_text());byid={s['id']:s for s in corpus['sessions']}
for item in m['execution_order']:
 if item['arm']!='gate':continue
 raw=json.load(gzip.open(H/'runs'/item['name']/'raw.json.gz','rt'));q=raw['requests'];first=sorted([r for r in q if r['turn']==0],key=lambda r:r['index']);chosen={r['index']:r for r in q if r['session_id'] in [first[i]['session_id'] for i in [0,12]] and r['turn'] in [0,2]}
 for r in sorted(q,key=lambda x:len(x['output_token_ids']),reverse=True)[:3]:chosen[r['index']]=r
 for r in chosen.values():
  source=byid[r['session_id']]
  samples.append({'run':item['name'],'partition':item['partition'],'index':r['index'],'session_id':r['session_id'],'turn':r['turn'],'output_tokens':len(r['output_token_ids']),'generated_text':r['generated_text'],'initial_issue':source['turns'][0]['user_text'],'latest_observation':source['turns'][r['turn']]['user_text'] if r['turn'] else None})
(H/'live-nov3-gpu-gate-quality-samples.json').write_text(json.dumps(samples,indent=2)+'\n')
print(json.dumps({'samples':len(samples),'unique_issue_turns':len({(x['session_id'],x['turn']) for x in samples}),'selection':'Gate arm slots0/12 turns0/2 plus longest3 per run; deduplicated; no output exclusion.'}))
