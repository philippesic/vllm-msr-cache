"""Token-budget fresh issue evidence without executing tools or model inference."""
from pathlib import Path
import json,hashlib,copy,collections
H=Path(__file__).resolve().parent;D=H/'fresh-agent-nov3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 from transformers import AutoTokenizer
 prior=json.loads((H/'prepared/session-corpus.json').read_text());tok=AutoTokenizer.from_pretrained(prior['tokenizer'],local_files_only=True);acq=json.loads((D/'acquisition.json').read_text());records=acq['records'];sessions=[];rejected=[];seen=set()
 old=json.loads((H/'prepared/triage-corpus-v2.json').read_text());excluded_ids={s['source_id'] for s in old['sessions']};excluded_first={tuple(s['turns'][0]['user_token_ids']) for s in old['sessions']}
 for record in records:
  if record['reject'] is not None:continue
  p=H/record['file'];assert sha(p)==record['sha256'];r=json.loads(p.read_text());budget=6000;turns=[]
  for i,t in enumerate(r['turns']):
   ids=tok.encode(t['user_text'],add_special_tokens=False);kept=ids[:min(3072 if i==0 else 2048,budget)];budget-=len(kept)
   if not kept:break
   turn=copy.deepcopy(t);turn.update(user_text=tok.decode(kept,skip_special_tokens=False),user_token_ids=kept,original_user_tokens=len(ids),retained_user_tokens=len(kept),truncated=len(kept)<len(ids),truncation='prefix token window' if len(kept)<len(ids) else None);turns.append(turn)
  if len(turns)<3:rejected.append({'source_id':r['source_id'],'reason':'fewer_than_three_retained_turns'});continue
  key=tuple(turns[0]['user_token_ids'])
  if r['source_id'] in excluded_ids or key in excluded_first or key in seen:rejected.append({'source_id':r['source_id'],'reason':'prior_or_duplicate_retained_issue'});continue
  seen.add(key);sessions.append({'id':'swe-'+r['source_id'],'cohort':'agent','source_id':r['source_id'],'turns':turns,'metadata':{'dataset':'tarsur385/swebench-verified-trajectories','revision':acq['revision'],'source_path':r['path'],'source_sha256':r['response_sha256'],'normalization_profile':'repository_issue_evidence_v2','historical_assistant_responses_excluded':True,'live_tool_execution':False,'max_source_input_tokens':6000,'normalization':'Complete original pr_description extracted before token budget; historical task instructions discarded uniformly. Issue and observations quoted as evidence.'}})
 eligible=len(sessions);assert eligible>=72,eligible;sessions=sessions[:72]
 for i,s in enumerate(sessions):s['seed_partition']=[7703,8803,9901][i%3]
 result={'schema':'quantized-realworld-sessions-v1','normalization_profile':'repository_issue_evidence_v2','tokenizer':prior['tokenizer'],'tokenizer_files':prior['tokenizer_files'],'seeds':[7703,8803,9901],'sessions':sessions,'summary':{'agent':{'sessions':72,'truncated_fragments':sum(t['truncated'] for s in sessions for t in s['turns'])}},'provenance':{'selection':'Predeclared fresh-agent-nov3/plan.json and path-selection.json; first72 eligible by SHA256(path) order, excludes all originally acquired first96 paths and previous issue IDs. No model outcomes used.','source_plan':'plan.json','historical_assistant_responses_included':False,'tools_executed':False,'timing':'Private synthetic waits, no future observation/wait schedule exposed to policy.','limits':'New issues from same SWE repository domain; no unseen-repository generalization or task-success labels.'}}
 target=D/'session-corpus.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
 assert not({s['source_id'] for s in sessions}&excluded_ids)
 validation={'schema':'fresh-agent-input-validation-v1','eligible_after_token_budget_and_dedup':eligible,'selected':72,'partition_counts':dict(collections.Counter(s['seed_partition'] for s in sessions)),'source_ids_disjoint':True,'retained_initial_inputs_disjoint':True,'historical_assistant_responses_excluded':True,'model_generation_performed':False,'rejected':rejected,'files':{str(p.relative_to(H)):sha(p) for p in [Path(__file__),D/'plan.json',D/'path-selection.json',D/'acquisition.json',target,*sorted(D.glob('normalized-*.json'))]}}
 (D/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps({k:v for k,v in validation.items() if k!='files'}))
if __name__=='__main__':main()
