"""Predeclared first-four-slot samples plus every new-credit-arm format inspection flag."""
import gzip,hashlib,json,re
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def main():
 m=json.loads((ROOT/'evidence/live-optimization-2080ti-20261103/priority-confirmation-frozen.json').read_text());out=[];corpora={}
 for item in m['execution_order']:
  p=H/'runs'/item['name']/'raw.json.gz'
  if not p.exists():out.append({'name':item['name'],'missing_raw':True});continue
  r=json.load(gzip.open(p,'rt'));cp=r['corpus_identity']['path']
  if cp not in corpora:corpora[cp]={s['id']:s for s in json.loads(Path(cp).read_text())['sessions']}
  samples=[];fixed={slot*4+turn for slot in [0,6,12,18] for turn in [0,2]} if item['pair']<3 else set();longest={x['index'] for x in sorted(r.get('requests',[]),key=lambda x:len(x.get('output_token_ids',[])),reverse=True)[:3]} if item['pair']<3 else set()
  for row in sorted(r.get('requests',[]),key=lambda x:x['index']):
   text=row.get('generated_text','');flags=[]
   if not text.startswith('Status:'):flags.append('missing_Status_prefix')
   if '\n' in text:flags.append('multiline_needs_paragraph_check')
   if '```' in text:flags.append('code_fence')
   if re.search(r'\bI (?:have |had )?(?:modified|changed|ran|executed|updated|applied|fixed)\b',text,re.I):flags.append('possible_self_claimed_action')
   extra=bool(flags)
   if row['index'] not in fixed and row['index'] not in longest and not extra:continue
   source=corpora[cp][row['session_id']];prior=sorted([x for x in r['requests'] if x['session_id']==row['session_id'] and x['turn']<row['turn']],key=lambda x:x['turn'])
   samples.append({'index':row['index'],'turn':row['turn'],'session_id':row['session_id'],'source_id':source['source_id'],'selection':['fixed_slots_0_6_12_18_turn0and2']*(row['index'] in fixed)+['longest3_first_triple']*(row['index'] in longest)+['format_flag']*extra,'format_flags':flags,'output_tokens':len(row.get('output_token_ids',[])),'generated_text':text,'source_evidence_turns':[{k:v for k,v in t.items() if k!='user_token_ids'} for t in source['turns'][:row['turn']+1]],'actual_prior_generated_texts':[x.get('generated_text') for x in prior]})
  out.append({'name':item['name'],'partition':item['partition'],'arm':item['arm'],'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'samples':samples})
 result={'schema':'astra-gated-once-positive-quality-samples-v1','scope':'Firsttripleperpartition/all3arms slots0/6/12/18turn0and2 pluslongest3; allformatflagsall36runs. Noqualityequivalence claim. No outcome-based run exclusions or quality-equivalence claim.','runs':out}
 (H/'live-nov3-priority-confirmation-quality-samples.json').write_text(json.dumps(result,indent=2)+'\n');print('Preserved',sum(len(x.get('samples',[])) for x in out),'sample observations')
if __name__=='__main__':main()
