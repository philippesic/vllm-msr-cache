"""A disclosed short-observation stratum, selected without serving outcomes."""
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from corpus import safe_text,content,messages_for_turn,validate_history,SEEDS
H=Path(__file__).resolve().parent
D=H.parent/'quantized-realworld-20261101'
def sha(x):return hashlib.sha256(x).hexdigest()
def main():
 original=json.loads((H/'prepared/session-corpus.json').read_text())
 tok=AutoTokenizer.from_pretrained(original['tokenizer'],local_files_only=True)
 used={s['source_id'] for s in original['sessions'] if s['cohort']=='agent'}
 prefix='External recorded repository/tool observation from a fixed historical trace. This was not executed from your previous answer. Analyse it in the context of the issue:\n\n'
 inventory=[];eligible=[]
 for record in json.loads((D/'data/swe-source-manifest.json').read_text())['files']:
  path=D/record['local_path'];body=path.read_bytes();assert sha(body)==record['sha256']
  raw=json.loads(body);turns=[];seen=set()
  initial=next(((i,m) for i,m in enumerate(raw['messages']) if m.get('role')=='user'),None)
  if not initial:continue
  i,m=initial;text=safe_text(content(m.get('content')));ids=tok.encode(text,add_special_tokens=False);kept=ids[:3072]
  turns.append(dict(user_text=tok.decode(kept,skip_special_tokens=False),user_token_ids=kept,original_user_tokens=len(ids),retained_user_tokens=len(kept),truncated=len(ids)>len(kept),content_kind='recorded_repository_issue',source_message_index=i,observed_wait_ms=None))
  for i,m in enumerate(raw['messages']):
   if m.get('role')!='tool':continue
   text=safe_text(content(m.get('content')))
   if not text or text in seen or '<|im_' in text:continue
   ids=tok.encode(prefix+text,add_special_tokens=False)
   if len(ids)>256:continue
   seen.add(text)
   turns.append(dict(user_text=prefix+text,user_token_ids=ids,original_user_tokens=len(ids),retained_user_tokens=len(ids),truncated=False,content_kind='external_recorded_tool_observation',source_message_index=i,observed_wait_ms=None))
  inventory.append(dict(source_id=raw['instance_id'],short_distinct=len(turns)-1,in_original_corpus=raw['instance_id'] in used))
  if len(turns)<6:continue
  eligible.append(dict(id='swe-short-'+raw['instance_id'],source_id=raw['instance_id'],cohort='agent',turns=turns[:6],metadata=dict(source_path=record['path'],source_sha256=record['sha256'],historical_assistant_responses_excluded=True,live_tool_execution=False,limitation='Fixed historical observations may differ from actions this model would choose. This is serving replay, not agent correctness.')))
 eligible.sort(key=lambda s:(s['source_id'] in used,sha(s['id'].encode())))
 # Twelve disjoint cases: prioritize retained source files outside the old corpus.
 selected=eligible[:12];assert len(selected)==12
 for i,s in enumerate(selected):
  s['seed_partition']=SEEDS[i%3]
  assert len({t['source_message_index'] for t in s['turns']})==6
  for turn in range(6):validate_history(s,turn,['ACTUAL']*turn,messages_for_turn(s,turn,['ACTUAL']*turn))
 result=dict(schema='quantized-realworld-short-observations-v1',tokenizer=original['tokenizer'],sessions=selected,seeds=SEEDS,summary=dict(eligible=len(eligible),fresh_eligible=sum(s['source_id'] not in used for s in eligible),selected_fresh=sum(s['source_id'] not in used for s in selected),sessions=12,turns=6),provenance=dict(selection='Before any serving result: first five distinct nonempty recorded observations <=256 tokens including disclosure; first issue <=3072 tokens; prioritize source IDs absent from original prepared corpus, then SHA256(session ID), first12, round-robin3 partitions.',limitations='Length-stratified selected workload; not production prevalence. Skipped long intervening historical observations. No source assistant answers, repeated observations, padding, live tool execution, or future timer input to policy.',source_manifest_sha256=sha((D/'data/swe-source-manifest.json').read_bytes())))
 for name,value in [('short-tool-inventory.json',inventory),('prepared/short-tool-corpus.json',result)]: (H/name).write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps(result['summary']))
if __name__=='__main__':main()
