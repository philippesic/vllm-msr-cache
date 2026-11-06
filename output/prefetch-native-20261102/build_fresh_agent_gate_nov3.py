"""Budget and deduplicate fixed fresh v2 inputs; tokenizer only, no generation."""
import collections,copy,hashlib,json
from pathlib import Path
from fetch_fresh_agent_gate_nov3 import H,D,REV,SEEDS,load_prior,verify,pin,write

def main():
 plan=json.loads((D/'plan.json').read_text());verify(plan);acq=json.loads((D/'acquisition.json').read_text());assert acq['plan_sha256']==pin(D/'plan.json') and acq['path_selection_sha256']==pin(D/'path-selection.json');assert acq['revision']==REV and len(acq['records'])==96;assert not (D/'session-corpus.json').exists()
 from transformers import AutoTokenizer
 import transformers,tokenizers
 tok=AutoTokenizer.from_pretrained(plan['tokenizer'],local_files_only=True);prior=load_prior();excluded_ids=prior['ids'];excluded_first={tuple(tok.encode(x,add_special_tokens=False)[:3072]) for x in prior['issues']}
 for p in [H/'prepared/session-corpus.json',H/'prepared/triage-corpus-v2.json',H/'fresh-agent-nov3/session-corpus.json']:
  old=json.loads(p.read_text())
  for s in old['sessions']:
   t=s['turns'][0]
   if 'user_token_ids' in t:excluded_first.add(tuple(t['user_token_ids']))
 seen=set();sessions=[];rejected=[];budget_policy=plan['token_budgets']
 for record in acq['records']:
  if record['reject'] is not None:continue
  p=H/record['file'];assert pin(p)==record['sha256'];r=json.loads(p.read_text());budget=budget_policy['cumulative'];turns=[]
  for i,t in enumerate(r['turns']):
   ids=tok.encode(t['user_text'],add_special_tokens=False);kept=ids[:min(budget_policy['initial'] if i==0 else budget_policy['later'],budget)];budget-=len(kept)
   if not kept:break
   x=copy.deepcopy(t);x.update(user_text=tok.decode(kept,skip_special_tokens=False),user_token_ids=kept,original_user_tokens=len(ids),retained_user_tokens=len(kept),truncated=len(kept)<len(ids),truncation='prefix token window' if len(kept)<len(ids) else None);turns.append(x)
  if len(turns)<budget_policy['minimum_retained_turns']:rejected.append({'source_id':r['source_id'],'reason':'fewer_than_three_retained_turns'});continue
  key=tuple(turns[0]['user_token_ids'])
  if r['source_id'] in excluded_ids or key in excluded_first or key in seen:rejected.append({'source_id':r['source_id'],'reason':'prior_or_duplicate_retained_issue'});continue
  seen.add(key);sessions.append({'id':'swe-'+r['source_id'],'cohort':'agent','source_id':r['source_id'],'turns':turns,'metadata':{'dataset':plan['dataset'],'revision':REV,'source_path':r['path'],'source_sha256':r['response_sha256'],'normalization_profile':'repository_issue_evidence_v2','historical_assistant_responses_excluded':True,'live_tool_execution':False,'max_source_input_tokens':6000,'normalization':'Complete original pr_description extracted before token budget; historical task instructions discarded uniformly. Issue plus first3 observations only, quoted as evidence.'}})
 eligible=len(sessions);write(D/'token-eligibility.json',{'eligible_count':eligible,'eligible_source_ids_in_order':[s['source_id'] for s in sessions],'rejected':rejected,'predeclared_required':72,'model_generation_performed':False})
 assert eligible>=72,'Fewer than72 token-eligible issues; bounded input blocker, do not extend'
 selected=sessions[:72]
 for i,s in enumerate(selected):s['seed_partition']=SEEDS[i%3]
 result={'schema':'quantized-realworld-sessions-v1','normalization_profile':'repository_issue_evidence_v2','tokenizer':plan['tokenizer'],'tokenizer_files':plan['tokenizer_files'],'seeds':SEEDS,'sessions':selected,'summary':{'agent':{'sessions':72,'truncated_fragments':sum(t['truncated'] for s in selected for t in s['turns'])}},'provenance':{'selection':'Predeclared fresh-agent-gate-nov3/plan.json and path-selection.json; first72 eligible in fixed SHA256(path) order excluding all192previously acquired SWE paths/IDs, normalized issue and retained first-input duplicates.','source_plan':'plan.json','historical_assistant_responses_included':False,'tools_executed':False,'timing':'Private synthetic waits; no future observation or wait schedule exposed to policy.','limits':'Same public SWE repository domain; exact-content disjointness is not semantic-near-duplicate or unseen-repository assurance. No model outcome-based selection.'}}
 verify(plan);target=D/'session-corpus.json';write(target,result)
 selected_ids={s['source_id'] for s in selected};selected_paths={s['metadata']['source_path'] for s in selected};selected_first={tuple(s['turns'][0]['user_token_ids']) for s in selected}
 assert len(selected_ids)==len(selected_paths)==len(selected_first)==72;assert selected_ids.isdisjoint(excluded_ids) and selected_paths.isdisjoint(prior['paths']) and selected_first.isdisjoint(excluded_first)
 all_files=[Path(__file__).resolve(),H/'fetch_fresh_agent_gate_nov3.py',D/'plan.json',D/'path-selection.json',D/'dataset-api.json',D/'acquisition.json',D/'token-eligibility.json',target,*sorted(D.glob('normalized-*.json'))]
 validation={'schema':'fresh-GPU-gate-input-validation-v1','eligible_after_token_budget_and_disjointness':eligible,'selected':72,'partition_counts':dict(collections.Counter(s['seed_partition'] for s in selected)),'original96_and_prior_fresh96_paths_excluded':True,'all_prior_acquired_issue_IDs_excluded':True,'normalized_initial_issue_duplicates_excluded':True,'retained_initial_token_inputs_disjoint':True,'retained_first_input_exclusion_count':len(excluded_first),'historical_assistant_responses_excluded':True,'raw_personal_metadata_retained':False,'raw_trajectory_responses_retained':False,'model_generation_performed':False,'tools_executed':False,'rejected':rejected,'tokenizer_versions':{'transformers':transformers.__version__,'tokenizers':tokenizers.__version__},'tokenizer_identity_verified':True,'source_fragment_token_totals_by_partition':{str(seed):{'sessions':sum(s['seed_partition']==seed for s in selected),'initial_tokens':sum(len(s['turns'][0]['user_token_ids']) for s in selected if s['seed_partition']==seed),'all_retained_source_tokens':sum(len(t['user_token_ids']) for s in selected if s['seed_partition']==seed for t in s['turns'])} for seed in SEEDS},'files':{str(p.relative_to(H)):pin(p) for p in all_files}}
 write(D/'validation.json',validation);print(json.dumps({k:v for k,v in validation.items() if k!='files'},indent=2))
if __name__=='__main__':main()
