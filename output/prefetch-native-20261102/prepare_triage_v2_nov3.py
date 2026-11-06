"""Uniform issue-body extraction for discovery; never edits prior corpora or calls a model."""
from pathlib import Path
import json,re,hashlib,collections,copy
H=Path(__file__).resolve().parent
PREFIX='External recorded repository/tool observation from a fixed historical trace. This was not executed from your previous answer. Analyse it in the context of the issue:\n\n'
def issue_body(text,allow_truncated=False):
 start=text.find('<pr_description>');end=text.find('</pr_description>',start)
 if start<0 or (end<0 and not allow_truncated):raise ValueError('Missing complete historical issue wrapper')
 body=text[start+len('<pr_description>'):end if end>=0 else None].strip()
 body=body.removeprefix('Consider the following PR description:').strip()
 if not body:raise ValueError('Empty extracted issue')
 return body,end<0

def main():
 from transformers import AutoTokenizer
 original=H/'prepared/session-corpus.json';r=json.loads(original.read_text());tok=AutoTokenizer.from_pretrained(r['tokenizer'],local_files_only=True);out=copy.deepcopy(r);out['sessions']=[];truncated_issues=0
 for old in r['sessions']:
  if old['cohort']!='agent':continue
  s=copy.deepcopy(old);body,partial=issue_body(old['turns'][0]['user_text'],old['turns'][0].get('truncated',False));truncated_issues+=partial;budget=6000;turns=[]
  for i,t in enumerate(old['turns']):
   if i==0:text='Source repository issue (quoted evidence):\n<repository_issue>\n'+body+'\n</repository_issue>'
   else:text='Recorded repository observation (quoted evidence; not executed by this service):\n<recorded_observation>\n'+t['user_text'].removeprefix(PREFIX)+'\n</recorded_observation>'
   ids=tok.encode(text,add_special_tokens=False);kept=ids[:min(3072 if i==0 else 2048,budget)];budget-=len(kept)
   if not kept:break
   new=copy.deepcopy(t);new.update(user_text=tok.decode(kept,skip_special_tokens=False),user_token_ids=kept,original_user_tokens=len(ids),retained_user_tokens=len(kept),truncated=len(kept)<len(ids),truncation='prefix token window' if len(kept)<len(ids) else None);turns.append(new)
  assert len(turns)>=3;s['turns']=turns;s['metadata']['normalization_profile']='repository_issue_evidence_v2';s['metadata']['issue_inherited_truncated_source']=partial;out['sessions'].append(s)
 out['normalization_profile']='repository_issue_evidence_v2';out['summary']={'agent':{'sessions':len(out['sessions']),'inherited_partial_issues':truncated_issues}};out['provenance']={'source':'Existing prepared issue/observation token windows; do not pretend removed source text was recovered.','normalization':'Uniform extraction of pr_description body, remove historical task instructions after closing marker; wrap issue and observations as evidence, then same source token budgets. Inherited truncated issue uses retained prefix and is explicitly marked. No sources dropped.','profile':'New discovery revision; old failed triage evidence retained unchanged. No model output used to select individual sources.','historical_assistant_responses_included':False,'original_corpus_sha256':hashlib.sha256(original.read_bytes()).hexdigest()}
 target=H/'prepared/triage-corpus-v2.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary']))
if __name__=='__main__':main()
