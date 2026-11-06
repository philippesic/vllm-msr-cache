"""Fixed next96 pinned public trajectories; retain only normalized issue/tool evidence."""
import argparse,concurrent.futures,datetime,hashlib,json,re,urllib.request
from pathlib import Path
from prepare_triage_v2_nov3 import issue_body
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'fresh-agent-priority-nov3'
REV='773748a7c1222e8a642a7059821498e14293562a';REPO='tarsur385/swebench-verified-trajectories';SEEDS=[14009,15013,16001]
def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p):return sha(p.read_bytes())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def text(v):
 if isinstance(v,str):return v
 if isinstance(v,list):return '\n'.join(str(x.get('text','')) for x in v if isinstance(x,dict))
 return ''
def normalize(s):
 s=re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b','[EMAIL REDACTED]',s)
 return re.sub(r'(?<!\w)(?:\+\d[\d ()-]{10,}\d)(?!\w)','[PHONE REDACTED]',s).strip()
def normalized_issue(content):
 body,_=issue_body(normalize(text(content)),False)
 return 'Source repository issue (quoted evidence):\n<repository_issue>\n'+body+'\n</repository_issue>'
def rel(p):return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def load_prior():
 files={};ids=set();paths=set();issues=[]
 def read(p):files[rel(p)]=pin(p);return json.loads(p.read_text())
 original_root=H.parent/'quantized-realworld-20261101';original=read(original_root/'data/swe-source-manifest.json');assert original['revision']==REV and len(original['files'])==96
 for row in original['files']:
  p=original_root/row['local_path'];assert pin(p)==row['sha256'];r=read(p);ids.add(r['instance_id']);paths.add(row['path']);initial=next(m for m in r['messages'] if m.get('role')=='user');issues.append(normalized_issue(initial['content']))
 acq=read(H/'fresh-agent-nov3/acquisition.json');selection=read(H/'fresh-agent-nov3/path-selection.json');assert acq['revision']==REV and len(acq['records'])==96 and len(selection['candidate_paths'])==96
 paths.update(selection['candidate_paths'])
 for row in acq['records']:
  p=H/row['file'];assert pin(p)==row['sha256'];r=read(p)
  if r.get('source_id'):ids.add(r['source_id'])
  if r.get('turns'):issues.append(r['turns'][0]['user_text'])
 latest=read(H/'fresh-agent-gate-nov3/acquisition.json');latest_sel=read(H/'fresh-agent-gate-nov3/path-selection.json');assert latest['revision']==REV and len(latest['records'])==96 and len(latest_sel['candidate_paths'])==96
 paths.update(latest_sel['candidate_paths'])
 for row in latest['records']:
  p=H/row['file'];assert pin(p)==row['sha256'];r=read(p)
  if r.get('source_id'):ids.add(r['source_id'])
  if r.get('turns'):issues.append(r['turns'][0]['user_text'])
 for p in [H/'prepared/session-corpus.json',H/'prepared/triage-corpus-v2.json',H/'fresh-agent-nov3/session-corpus.json',H/'fresh-agent-gate-nov3/session-corpus.json']:
  r=read(p)
  for s in r['sessions']:
   ids.add(s['source_id'])
   if s.get('cohort')=='agent' and s.get('metadata',{}).get('source_path'):paths.add(s['metadata']['source_path'])
   if r.get('normalization_profile')=='repository_issue_evidence_v2':issues.append(s['turns'][0]['user_text'])
 return {'ids':ids,'paths':paths,'issues':issues,'files':files,'original_paths':[r['path'] for r in original['files']],'prior_fresh_paths':selection['candidate_paths'],'latest_fresh_paths':latest_sel['candidate_paths'],'catalog_expected_sha256':selection['api_sha256']}
def verify(plan):
 bad=[p for p,v in plan['file_pins'].items() if not (ROOT/p).is_file() or pin(ROOT/p)!=v]
 if bad:raise RuntimeError('Pinned input/script drift: '+repr(bad))
def prepare():
 prior=load_prior();api=H/'fresh-agent-nov3/dataset-api.json';assert pin(api)==prior['catalog_expected_sha256'];meta=json.loads(api.read_text());assert meta['sha']==REV
 paths=sorted([x['rfilename'] for x in meta['siblings'] if x['rfilename'].startswith('swebench_verified_raw/gpt-5-mini/') and x['rfilename'].endswith('.traj.json')],key=lambda x:sha(x.encode()))
 assert paths[:96]==prior['original_paths'];assert set(prior['original_paths']).isdisjoint(prior['prior_fresh_paths']);assert len(set(prior['original_paths'])|set(prior['prior_fresh_paths'])|set(prior['latest_fresh_paths']))==288
 candidates=[p for p in paths if p not in prior['paths'] and Path(p).name.removesuffix('.traj.json') not in prior['ids']][:96];assert len(candidates)==96
 files={**prior['files'],rel(api):pin(api)}
 for p in [Path(__file__).resolve(),H/'build_fresh_agent_priority_nov3.py',H/'prepare_triage_v2_nov3.py',H/'verify_fresh_agent_priority_astra_nov3.py']:files[rel(p)]=pin(p)
 old=json.loads((H/'prepared/session-corpus.json').read_text());tok=Path(old['tokenizer']);selection=json.loads((H/'model-selection.json').read_text());assert tok.name==selection['revision'];files[rel(H/'model-selection.json')]=pin(H/'model-selection.json')
 for name,v in old['tokenizer_files'].items():assert pin(tok/name)==v;files[str(tok/name)]=v
 plan={'schema':'fresh-return-priority-input-plan-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':REPO,'revision':REV,'universe':'swebench_verified_raw/gpt-5-mini/*.traj.json at exact pinned revision','order':'SHA256(path), ascending; first96 remaining after exclusions','candidate_count':96,'selected_count':72,'partitions':SEEDS,'exclusion':'All original96 acquired paths plus ALL priorfresh96 and latestgate96 candidate paths (including unused24each); all known prior issue IDs, normalized initial issue hashes, and retained first-token input duplicates.','max_trajectory_response_bytes':2000000,'workers':4,'automatic_retries':0,'selection':'First72 eligible in fixed candidate order after source/normalization/token-budget/disjointness checks. Fewer than72 eligible is a blocker; no extension/replacement/outcome selection.','normalization':'Complete pr_description body before token budget, same v2 issue wrapper and first3 nonempty tool observations after initial user. No historical assistant messages or raw metadata retained. Same email/phone regex redaction; not full PII assurance.','token_budgets':{'initial':3072,'later':2048,'cumulative':6000,'minimum_retained_turns':3},'tokenizer':str(tok),'tokenizer_files':old['tokenizer_files'],'file_pins':files,'no_model_generation':True,'no_tools_executed':True,'limits':['Same public SWE repository domain; not unseen-repository or semantic-near-duplicate assurance.','Only model-generated LIVE replies will enter future histories; no historical assistant text retained.','Third recorded observation may be retained but3turn confirmation uses issue+first2 observations.']}
 D.mkdir(exist_ok=False);write(D/'plan.json',plan);(D/'dataset-api.json').write_bytes(api.read_bytes());write(D/'path-selection.json',{'revision':REV,'api_sha256':pin(api),'universe_count':len(paths),'original96_excluded':prior['original_paths'],'prior_fresh96_excluded':prior['prior_fresh_paths'],'latest_gate96_excluded':prior['latest_fresh_paths'],'all_excluded_paths':sorted(prior['paths']),'all_prior_source_ids':sorted(prior['ids']),'prior_normalized_issue_hashes':sorted({sha(x.encode()) for x in prior['issues']}),'candidate_paths':candidates,'candidate_paths_sha256':sha(json.dumps(candidates).encode())})
 print(json.dumps({'prepared_before_network':True,'candidates':len(candidates),'excluded_paths':len(prior['paths']),'prior_IDs':len(prior['ids']),'input_script_pins':len(files),'plan_sha256':pin(D/'plan.json')}),flush=True)
def fetch():
 plan=json.loads((D/'plan.json').read_text());verify(plan);sel=json.loads((D/'path-selection.json').read_text());assert pin(D/'dataset-api.json')==sel['api_sha256'];assert len(sel['candidate_paths'])==96;assert not (D/'acquisition.json').exists();assert not list(D.glob('normalized-*.json')),'No implicit rerun/resume of acquisition'
 prior_ids=set(sel['all_prior_source_ids']);seen_ids=set();seen_first=set(sel['prior_normalized_issue_hashes']);records=[]
 def one(path):
  rec={'path':path,'url':f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{path}'}
  try:
   with urllib.request.urlopen(rec['url'],timeout=60) as response:
    body=response.read(plan['max_trajectory_response_bytes']+1);status=response.status;revision=response.headers.get('X-Repo-Commit')
   if len(body)>plan['max_trajectory_response_bytes']:raise ValueError('Predeclared response byte bound exceeded')
   if revision is not None and revision!=REV:raise ValueError('Source revision response header drift')
   raw=json.loads(body);rec.update(response_sha256=sha(body),response_bytes=len(body),http_status=status,revision_header=revision,source_id=raw['instance_id']);messages=raw['messages'];initial=next(((i,m) for i,m in enumerate(messages) if m.get('role')=='user'),None)
   if initial is None:rec['reject']='no_initial_user'
   else:
    i,user=initial;obs=[(j,m) for j,m in enumerate(messages) if j>i and m.get('role')=='tool' and text(m.get('content')).strip()]
    if len(obs)<2:rec['reject']='fewer_than_two_observations'
    else:
     turns=[{'user_text':normalized_issue(user['content']),'content_kind':'recorded_repository_issue','source_message_index':i,'observed_wait_ms':None}]
     for j,m in obs[:3]:turns.append({'user_text':'Recorded repository observation (quoted evidence; not executed by this service):\n<recorded_observation>\n'+normalize(text(m['content']))+'\n</recorded_observation>','content_kind':'external_recorded_tool_observation','source_message_index':j,'observed_wait_ms':None})
     if any(not t['user_text'] or '<|im_' in t['user_text'] for t in turns):rec['reject']='empty_or_template_delimiter'
     else:rec['turns']=turns
  except Exception as exc:rec['reject']='fetch_or_parse_error';rec['error']=repr(exc)
  return rec
 with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:
  for i,rec in enumerate(pool.map(one,sel['candidate_paths'])):
   sid=rec.get('source_id');first=sha(rec['turns'][0]['user_text'].encode()) if rec.get('turns') else None
   if not rec.get('reject'):
    if sid in prior_ids or sid in seen_ids:rec['reject']='prior_or_duplicate_issue_ID'
    elif first in seen_first:rec['reject']='prior_or_duplicate_normalized_issue'
    else:seen_ids.add(sid);seen_first.add(first)
   if rec.get('reject'):rec.pop('turns',None)
   target=D/('normalized-'+sha(rec['path'].encode())[:20]+'.json');write(target,rec);records.append({'file':str(target.relative_to(H)),'sha256':pin(target),'source_id':sid,'reject':rec.get('reject')});print(json.dumps({'input':i+1,'eligible':sum(x['reject'] is None for x in records),'reject':rec.get('reject')}),flush=True)
 verify(plan);result={'revision':REV,'records':records,'eligible':sum(x['reject'] is None for x in records),'historical_assistant_responses_retained':False,'model_generation_performed':False,'plan_sha256':pin(D/'plan.json'),'path_selection_sha256':pin(D/'path-selection.json'),'response_bytes_total':sum(json.loads((H/x['file']).read_text()).get('response_bytes',0) for x in records),'raw_responses_retained':False};write(D/'acquisition.json',result)
 assert result['eligible']>=72,'Fewer than72 eligible; fixed acquisition exhausted, no adaptive extension'
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','fetch']);args=ap.parse_args();prepare() if args.action=='prepare' else fetch()
