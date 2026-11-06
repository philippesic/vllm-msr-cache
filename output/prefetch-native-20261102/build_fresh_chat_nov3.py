"""Prepare predeclared disjoint public sessions; never calls a model."""
from pathlib import Path
import json,hashlib,collections
H=Path(__file__).resolve().parent;D=H/'fresh-chat-nov3'
def digest(b):return hashlib.sha256(b).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')
def main():
 from transformers import AutoTokenizer
 prior=json.loads((H/'prepared/session-corpus.json').read_text());tok=AutoTokenizer.from_pretrained(prior['tokenizer'],local_files_only=True)
 excluded_ids={s['source_id'] for s in prior['sessions']};excluded_first={tuple(tok.encode(s['turns'][0]['user_text'],add_special_tokens=False)[:3072]) for s in prior['sessions'] if s['cohort']=='chat'}
 rows=[]
 for p in sorted(D.glob('normalized-*.json')):rows.extend(json.loads(p.read_text())['rows'])
 rows.sort(key=lambda r:digest(('wildchat-'+r['source_id']).encode()));seen_first=set();seen_ids=set();sessions=[];reject=collections.Counter()
 for r in rows:
  if r['language']!='English':reject['not_english']+=1;continue
  if r['turn']<3 or len(r['users'])<3:reject['fewer_than_three_users']+=1;continue
  if r.get('toxic') or r.get('moderation_flagged'):reject['flagged']+=1;continue
  if r['source_id'] in excluded_ids or r['source_id'] in seen_ids:reject['prior_or_duplicate_source']+=1;continue
  if any(not u['user_text'] or '<|im_' in u['user_text'] for u in r['users'][:4]):reject['empty_or_template_delimiter']+=1;continue
  budget=6000;turns=[]
  for i,u in enumerate(r['users'][:4]):
   original=tok.encode(u['user_text'],add_special_tokens=False);kept=original[:min(3072 if i==0 else 2048,budget)];budget-=len(kept)
   if not kept:break
   turns.append({'user_text':tok.decode(kept,skip_special_tokens=False),'user_token_ids':kept,'original_user_tokens':len(original),'retained_user_tokens':len(kept),'truncated':len(kept)<len(original),'truncation':'prefix token window' if len(kept)<len(original) else None,'content_kind':'real_user_message','source_message_index':u['source_message_index'],'observed_wait_ms':None})
  if len(turns)<3:reject['fewer_than_three_retained_turns']+=1;continue
  first=tuple(turns[0]['user_token_ids'])
  if first in excluded_first or first in seen_first:reject['duplicate_initial_retained_user']+=1;continue
  seen_first.add(first);seen_ids.add(r['source_id'])
  sessions.append({'id':'wildchat-'+r['source_id'],'cohort':'chat','source_id':r['source_id'],'source_row':r['row_idx'],'turns':turns,'metadata':{'dataset':'allenai/WildChat-1M','revision':'7d6490e462285cf85d91eabea0f9a954fbddcd1f','historical_assistant_responses_excluded':True,'max_source_input_tokens':6000,'truncation_policy':'First fragment max3072, later2048, cumulative6000. Actual generated replies and chat-template overhead added by driver.'}})
 assert len(sessions)>=72,len(sessions)
 eligible=len(sessions);sessions=sessions[:72]
 for i,s in enumerate(sessions):s['seed_partition']=[4409,5501,6607][i%3]
 out={'schema':'quantized-realworld-sessions-v1','tokenizer':prior['tokenizer'],'tokenizer_files':prior['tokenizer_files'],'seeds':[4409,5501,6607],'sessions':sessions,'summary':{'chat':{'sessions':72,'truncated_fragments':sum(t['truncated'] for s in sessions for t in s['turns'])}},'filter_counts':dict(reject),'provenance':{'selection':'Predeclared eight API windows; SHA256(session ID) order, first72 usable English multi-turn inputs after original controls and exclusion of all prior source IDs/retained initial inputs. No model outcomes used.','source_plan':'plan.json','historical_assistant_responses_included':False,'timing':'No observed human wait; any waits are explicitly synthetic private schedules.','limits':'Public API windows, not a representative population sample; no task-success labels.'}}
 target=D/'session-corpus.json';assert not target.exists();write(target,out)
 assert not({s['source_id'] for s in sessions}&excluded_ids)
 manifest={'schema':'fresh-chat-input-validation-v1','selected':72,'eligible':eligible,'partition_counts':dict(collections.Counter(s['seed_partition'] for s in sessions)),'source_ids_disjoint_from_prior_corpus':True,'initial_retained_user_inputs_disjoint_from_prior':True,'historical_assistant_text_excluded':True,'model_generation_performed':False,'files':{str(p.relative_to(H)):digest(p.read_bytes()) for p in [Path(__file__),D/'plan.json',target,*sorted(D.glob('normalized-*.json'))]}}
 write(D/'validation.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))
if __name__=='__main__':main()
