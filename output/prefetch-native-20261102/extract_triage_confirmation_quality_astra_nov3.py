"""Preserve predeclared samples without model calls or output-based selection changes."""
import gzip,json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def main():
 m=json.loads((ROOT/'evidence/live-optimization-2080ti-20261103/triage-confirmation-frozen.json').read_text());corpus=json.loads((H/'fresh-agent-nov3/session-corpus.json').read_text());sessions={s['id']:s for s in corpus['sessions']};out=[]
 for pair in m['pairs'][:3]:
  for arm in ['native','candidate']:
   name=pair['runs'][arm];p=H/'runs'/name/'raw.json.gz'
   if not p.exists():raise RuntimeError('Predeclared quality sample not yet complete: '+name)
   r=json.load(gzip.open(p,'rt'));q=r['requests'];fixed={slot*4+turn for slot in [0,6,12,18] for turn in [0,2]};longest=sorted(q,key=lambda x:len(x.get('output_token_ids',[])),reverse=True)[:3];chosen=fixed|{x['index'] for x in longest};samples=[]
   for row in sorted(q,key=lambda x:x['index']):
    if row['index'] not in chosen:continue
    source=sessions[row['session_id']];prior=sorted([x for x in q if x['session_id']==row['session_id'] and x['turn']<row['turn']],key=lambda x:x['turn'])
    samples.append({'index':row['index'],'turn':row['turn'],'session_id':row['session_id'],'selection':['predeclared_slot_turn']*(row['index'] in fixed)+['longest_three']*(row in longest),'output_tokens':len(row['output_token_ids']),'generated_text':row.get('generated_text'),'source_id':source['source_id'],'source_evidence_turns':[{k:v for k,v in t.items() if k!='user_token_ids'} for t in source['turns'][:row['turn']+1]],'actual_prior_generated_texts':[x['generated_text'] for x in prior]})
   out.append({'name':name,'partition':pair['partition'],'arm':arm,'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'samples':samples})
 result={'schema':'astra-predeclared-triage-confirmation-quality-samples-v1','scope':'First pair in each of three partitions, both arms; source-grounded post-block review only, no tuning or automatic quality exclusion.','runs':out}
 (H/'live-nov3-triage-confirmation-quality-samples.json').write_text(json.dumps(result,indent=2)+'\n');print('Preserved',sum(len(x['samples']) for x in out),'deduplicated samples')
if __name__=='__main__':main()
