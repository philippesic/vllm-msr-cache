"""Predeclared fresh recorded-observation inputs; no historical model replies retained."""
from pathlib import Path
import json,hashlib,urllib.request,re,datetime
from prepare_triage_v2_nov3 import issue_body
H=Path(__file__).resolve().parent;D=H/'fresh-agent-nov3';REV='773748a7c1222e8a642a7059821498e14293562a';REPO='tarsur385/swebench-verified-trajectories'
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')
def get(url,bound):
 with urllib.request.urlopen(url,timeout=60) as r:b=r.read(bound+1)
 if len(b)>bound:raise ValueError('Predeclared response byte bound exceeded')
 return b
def text(v):
 if isinstance(v,str):return v
 if isinstance(v,list):return '\n'.join(str(x.get('text','')) for x in v if isinstance(x,dict))
 return ''
def normalize(s):
 s=re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b','[EMAIL REDACTED]',s)
 return re.sub(r'(?<!\w)(?:\+\d[\d ()-]{10,}\d)(?!\w)','[PHONE REDACTED]',s).strip()
def main():
 D.mkdir(exist_ok=True);plan={'revision':REV,'dataset':REPO,'universe':'swebench_verified_raw/gpt-5-mini/*.traj.json at pinned revision','order':'SHA256(path), same as original source acquisition','exclusion':'Exclude first96 paths previously acquired by original corpus workflow, all prior source paths/issue IDs, and duplicate normalized initial issues. New candidate universe is next96 remaining paths.','acquisition_limit':96,'max_metadata_bytes':3000000,'max_trajectory_response_bytes':2000000,'selection':'First72 eligible in path hash order after controls; no model outcomes used. If fewer than72 eligible, report input blocker rather than adaptively extend.','normalize':'Extract complete pr_description issue body, uniformly excluding historical task-instruction wrapper; reject missing/unclosed wrapper. Retain only issue evidence and first3 nonempty recorded tool observations after it. No historical assistant replies. Regex email/phone redaction, not complete PII assurance. No tools executed.','token_budget':'Prepare with prior first3072/later2048/cumulative6000 source-token budget. Three-turn runs exercise issue + first2 tool observations; third may remain available for declared later work.','partitions':[7703,8803,9901],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 p=D/'plan.json'
 if p.exists():assert {k:v for k,v in json.loads(p.read_text()).items() if k!='created_utc'}=={k:v for k,v in plan.items() if k!='created_utc'}
 else:write(p,plan)
 api=D/'dataset-api.json'
 if not api.exists():api.write_bytes(get('https://huggingface.co/api/datasets/'+REPO+'/revision/'+REV,3000000))
 metadata=json.loads(api.read_text());assert metadata['sha']==REV
 paths=[x['rfilename'] for x in metadata['siblings'] if x['rfilename'].startswith('swebench_verified_raw/gpt-5-mini/') and x['rfilename'].endswith('.traj.json')];paths.sort(key=lambda s:sha(s.encode()))
 old=json.loads((H/'prepared/session-corpus.json').read_text());old_ids={s['source_id'] for s in old['sessions'] if s['cohort']=='agent'};old_paths={s['metadata']['source_path'] for s in old['sessions'] if s['cohort']=='agent'};old_first={sha(('Source repository issue (quoted evidence):\n<repository_issue>\n'+issue_body(normalize(s['turns'][0]['user_text']),True)[0]+'\n</repository_issue>').encode()) for s in old['sessions'] if s['cohort']=='agent'}
 candidates=[p for p in paths[96:] if p not in old_paths and Path(p).name.removesuffix('.traj.json') not in old_ids][:96];assert len(candidates)==96,len(candidates)
 write(D/'path-selection.json',{'api_sha256':sha(api.read_bytes()),'universe_count':len(paths),'prior_first96_excluded':paths[:96],'candidate_paths':candidates,'candidate_paths_sha256':sha(json.dumps(candidates).encode()),'prior_source_ids':sorted(old_ids)})
 records=[];seen_ids=set();seen_first=set(old_first)
 for i,path in enumerate(candidates):
  out=D/('normalized-'+sha(path.encode())[:20]+'.json')
  if out.exists():record=json.loads(out.read_text())
  else:
   record={'path':path,'url':'https://huggingface.co/datasets/'+REPO+'/resolve/'+REV+'/'+path}
   try:
    body=get(record['url'],2000000);raw=json.loads(body);messages=raw['messages'];record.update(response_sha256=sha(body),response_bytes=len(body),source_id=raw['instance_id'])
    initial=next(((j,m) for j,m in enumerate(messages) if m.get('role')=='user'),None)
    if initial is None:record['reject']='no_initial_user'
    else:
     j,issue=initial;tools=[(k,m) for k,m in enumerate(messages) if k>j and m.get('role')=='tool' and text(m.get('content')).strip()]
     if len(tools)<2:record['reject']='fewer_than_two_observations'
     else:
      issue_text,_=issue_body(normalize(text(issue['content'])),False);issue_text='Source repository issue (quoted evidence):\n<repository_issue>\n'+issue_text+'\n</repository_issue>';turns=[{'user_text':issue_text,'content_kind':'recorded_repository_issue','source_message_index':j,'observed_wait_ms':None}]
      for k,m in tools[:3]:turns.append({'user_text':'Recorded repository observation (quoted evidence; not executed by this service):\n<recorded_observation>\n'+normalize(text(m['content']))+'\n</recorded_observation>','content_kind':'external_recorded_tool_observation','source_message_index':k,'observed_wait_ms':None})
      if any(not t['user_text'] or '<|im_' in t['user_text'] for t in turns):record['reject']='empty_or_template_delimiter'
      else:record['turns']=turns
   except Exception as e:record['reject']='fetch_or_parse_error';record['error']=repr(e)
   write(out,record)
  sid=record.get('source_id');first=sha(record['turns'][0]['user_text'].encode()) if 'turns' in record else None
  if not record.get('reject'):
   if sid in old_ids or sid in seen_ids:record['reject']='prior_or_duplicate_issue';record.pop('turns',None);write(out,record)
   elif first in seen_first:record['reject']='duplicate_normalized_issue';record.pop('turns',None);write(out,record)
   else:seen_ids.add(sid);seen_first.add(first)
  records.append({'file':str(out.relative_to(H)),'sha256':sha(out.read_bytes()),'source_id':sid,'reject':record.get('reject')});print(json.dumps({'input':i+1,'eligible':sum(x['reject'] is None for x in records),'reject':record.get('reject')}),flush=True)
 write(D/'acquisition.json',{'revision':REV,'records':records,'eligible':sum(x['reject'] is None for x in records),'historical_assistant_responses_retained':False,'model_generation_performed':False})
 assert sum(x['reject'] is None for x in records)>=72,'Insufficient eligible fresh issues; acquisition bounded as planned'
if __name__=='__main__':main()
