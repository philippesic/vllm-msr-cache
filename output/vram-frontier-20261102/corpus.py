"""Pinned public conversation/recorded-tool content for serving replay.
Never runs tools, model inference, or private requests. See corpus-notes.md.
"""
import argparse,concurrent.futures,datetime,hashlib,json,random,re,time,urllib.parse,urllib.request
from pathlib import Path
H=Path(__file__).resolve().parent;D=H/'data';P=H/'prepared'
WILD='7d6490e462285cf85d91eabea0f9a954fbddcd1f';SWE='773748a7c1222e8a642a7059821498e14293562a'
OFFSETS=[0,100,1000,10000,100000,200000,400000,600000]
SEEDS=[1103,2207,3301]
def sha(b):return hashlib.sha256(b).hexdigest()
def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def get(url,limit=3_000_000):
 with urllib.request.urlopen(url,timeout=60) as response:
  body=response.read(limit+1);headers=dict(response.headers)
 if len(body)>limit:raise ValueError('bounded download exceeded limit')
 return body,headers
def safe_text(text):
 if not isinstance(text,str):return ''
 # Drop directly recognizable contact identifiers; no assertion of complete PII removal.
 text=re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b','[EMAIL REDACTED]',text)
 text=re.sub(r'(?<!\w)(?:\+\d[\d ()-]{10,}\d)(?!\w)','[PHONE REDACTED]',text)
 return text.strip()
def timestamp_ms(value):
 if not value:return None
 try:return int(datetime.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()*1000)
 except(ValueError,TypeError):return None
def fetch():
 D.mkdir(parents=True,exist_ok=True)
 if not (D/'wildchat-normalized.json').exists():
  all_rows=[];downloads=[];removed=[]
  for offset in OFFSETS:
   params={'dataset':'allenai/WildChat-1M','config':'default','split':'train','offset':offset,'length':100,'revision':WILD}
   url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(params)
   raw=D/f'wildchat-rows-{offset}.json'
   if raw.exists():
    body=raw.read_bytes();record=next(x for x in json.loads((D/'wildchat-downloads.json').read_text()) if x['offset']==offset);headers=record['headers'];url=record['url']
   else:body,headers=get(url)
   assert headers.get('x-revision')==WILD,'rows endpoint revision drift; use retained normalized input'
   rows=json.loads(body)['rows']
   for item in rows:
    row=item['row'];messages=[]
    for m in row['conversation']:
     messages.append({'role':m['role'],'content':safe_text(m['content']),'timestamp':m.get('timestamp'),'toxic':m.get('toxic')})
    flagged=any(bool(m.get('flagged')) for m in (row.get('openai_moderation') or []) if isinstance(m,dict))
    all_rows.append({'row_idx':item['row_idx'],'conversation_hash':row['conversation_hash'],'language':row['language'],'turn':row['turn'],'toxic':row.get('toxic'),'moderation_flagged':flagged,'conversation':messages})
   downloads.append({'url':url,'dataset_revision_header':headers['x-revision'],'raw_sha256':sha(body),'raw_bytes':len(body),'offset':offset,'rows':len(rows),'network_identifiers_retained':False})
   if raw.exists():removed.append({'path':str(raw),'bytes':raw.stat().st_size,'reason':'Discarded original API response with unused network/location/header metadata; normalized source and response hash retained.'});raw.unlink()
  save(D/'wildchat-normalized.json',{'dataset':'allenai/WildChat-1M','revision':WILD,'license':'ODC-BY','rows':all_rows})
  save(D/'wildchat-source-manifest.json',{'downloads':downloads,'removed_disposable_raw':removed,'normalization':'Keep source ID/role/content/timestamp/moderation flags; discard IP, location, browser headers; redact email/phone regexes. Historical assistant text source-only, never used as generated responses.'})
 for name,repo,revision,filename in [('wildchat-license','allenai/WildChat-1M',WILD,'LICENSE.md'),('wildchat-card','allenai/WildChat-1M',WILD,'README.md'),('swe-card','tarsur385/swebench-verified-trajectories',SWE,'README.md')]:
  dest=D/(name+'.md')
  if not dest.exists():body,_=get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}');dest.write_bytes(body)
 api=D/'swe-api.json'
 if not api.exists():body,_=get('https://huggingface.co/api/datasets/tarsur385/swebench-verified-trajectories/revision/'+SWE);api.write_bytes(body)
 metadata=json.loads(api.read_text());assert metadata['sha']==SWE
 paths=[x['rfilename'] for x in metadata['siblings'] if x['rfilename'].startswith('swebench_verified_raw/gpt-5-mini/') and x['rfilename'].endswith('.traj.json')]
 paths.sort(key=lambda p:sha(p.encode()));paths=paths[:96]
 folder=D/'swe';folder.mkdir(exist_ok=True)
 def one(path):
  dest=folder/(Path(path).name)
  if not dest.exists():body,_=get('https://huggingface.co/datasets/tarsur385/swebench-verified-trajectories/resolve/'+SWE+'/'+path,2_000_000);dest.write_bytes(body)
  else:body=dest.read_bytes()
  return {'path':path,'local_path':str(dest.relative_to(H)),'sha256':sha(body),'bytes':len(body)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,paths))
 save(D/'swe-source-manifest.json',{'dataset':'tarsur385/swebench-verified-trajectories','revision':SWE,'license':'MIT','source_model':'gpt-5-mini','selection':'First96 paths in SHA256(path) order, independent of outcomes/content/length.','files':records})
 print(json.dumps({'wildchat_rows':len(json.loads((D/'wildchat-normalized.json').read_text())['rows']),'swe_files':len(records)}))
def content(value):
 if isinstance(value,str):return value
 if isinstance(value,list):return '\n'.join(str(v.get('text','')) for v in value if isinstance(v,dict))
 return ''
def build_raw_sessions():
 rows=json.loads((D/'wildchat-normalized.json').read_text())['rows'];chat=[];seen=set();filters={}
 def reject(reason):filters[reason]=filters.get(reason,0)+1
 for row in rows:
  if row['language']!='English':reject('not_english');continue
  if row['turn']<3:reject('fewer_than_three_user_turns');continue
  if row.get('toxic') or row.get('moderation_flagged'):reject('toxic_or_flagged');continue
  messages=row['conversation'];users=[(i,m) for i,m in enumerate(messages) if m['role']=='user'][:4]
  if len(users)<3 or any(not m['content'] for _,m in users):reject('empty_or_insufficient_users');continue
  if any('<|im_' in m['content'] for _,m in users):reject('template_delimiter');continue
  identity=sha(users[0][1]['content'].encode())
  if identity in seen:reject('duplicate_initial_user');continue
  seen.add(identity);turns=[];previous_assistant_time=None
  for i,m in users:
   after=next((n for n in messages[i+1:] if n['role']=='assistant'),{})
   stamp=timestamp_ms(after.get('timestamp'));interval=stamp-previous_assistant_time if stamp is not None and previous_assistant_time is not None else None
   turns.append({'user_text':m['content'],'content_kind':'real_user_message','source_message_index':i,'observed_wait_ms':None,'source_assistant_timestamp':after.get('timestamp'),'source_interresponse_interval_ms':interval})
   previous_assistant_time=stamp
  chat.append({'id':'wildchat-'+row['conversation_hash'],'cohort':'chat','source_id':row['conversation_hash'],'source_row':row['row_idx'],'turns':turns,'metadata':{'dataset':'allenai/WildChat-1M','revision':WILD,'timing':'User timestamps missing. Inter-response intervals include original generation plus human think time; not observed wait durations.','historical_assistant_responses_excluded':True}})
 agents=[]
 for record in json.loads((D/'swe-source-manifest.json').read_text())['files']:
  raw=json.loads((H/record['local_path']).read_text());messages=raw['messages'];initial=next(((i,m) for i,m in enumerate(messages) if m.get('role')=='user'),None)
  tools=[(i,m) for i,m in enumerate(messages) if m.get('role')=='tool' and content(m.get('content')).strip()]
  if initial is None or len(tools)<2:reject('swe_fewer_than_two_observations');continue
  i,m=initial;turns=[{'user_text':safe_text(content(m['content'])),'content_kind':'recorded_repository_issue','source_message_index':i,'observed_wait_ms':None}]
  for i,m in tools[:3]:
   prefix='External recorded repository/tool observation from a fixed historical trace. This was not executed from your previous answer. Analyse it in the context of the issue:\n\n'
   turns.append({'user_text':prefix+safe_text(content(m['content'])),'content_kind':'external_recorded_tool_observation','source_message_index':i,'observed_wait_ms':None})
  agents.append({'id':'swe-'+raw['instance_id'],'cohort':'agent','source_id':raw['instance_id'],'turns':turns,'metadata':{'dataset':'tarsur385/swebench-verified-trajectories','revision':SWE,'source_path':record['path'],'source_sha256':record['sha256'],'timing':'No observed tool durations; all replay waits explicitly simulated.','historical_assistant_responses_excluded':True,'live_tool_execution':False,'limitation':'Recorded external observations can differ from actions the new model would choose; not an agent correctness benchmark.'}})
 return chat,agents,filters

def messages_for_turn(session,index,generated):
 assert isinstance(index,int) and 0<=index<len(session['turns'])
 assert len(generated)==index and all(isinstance(s,str) for s in generated)
 out=[]
 for i in range(index+1):
  if i:out.append({'role':'assistant','content':generated[i-1]})
  out.append({'role':'user','content':session['turns'][i]['user_text']})
 return out

def validate_history(session,index,generated,messages):
 expected=messages_for_turn(session,index,generated)
 if messages!=expected:raise ValueError('History contains altered, future or non-generated messages')
 return True

def stats(values):
 values=sorted(values)
 return {'n':len(values),'min':values[0] if values else None,'median':values[len(values)//2] if values else None,'p95':values[min(len(values)-1,int(.95*len(values)))] if values else None,'max':values[-1] if values else None}

def prepare(tokenizer_path,output):
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
 chat,agents,filters=build_raw_sessions();selected=[];counts={};summaries={}
 for cohort,sessions in [('chat',chat),('agent',agents)]:
  sessions.sort(key=lambda s:sha(s['id'].encode()));sessions=sessions[:72]
  assert len(sessions)>=48,(cohort,len(sessions))
  for session in sessions:
   budget=6000;original_total=kept_total=0;fragments=[]
   for i,turn in enumerate(session['turns']):
    original=tokenizer.encode(turn['user_text'],add_special_tokens=False);cap=min(3072 if i==0 else 2048,budget);kept=original[:cap];budget-=len(kept);original_total+=len(original);kept_total+=len(kept)
    if not kept:break
    out=dict(turn);out.update(user_text=tokenizer.decode(kept,skip_special_tokens=False),user_token_ids=kept,original_user_tokens=len(original),retained_user_tokens=len(kept),truncated=len(kept)<len(original),truncation='prefix token window' if len(kept)<len(original) else None);fragments.append(out)
   if len(fragments)<3:reject='fewer_than_three_retained_turns';filters[reject]=filters.get(reject,0)+1;continue
   session['turns']=fragments;session['metadata'].update(original_user_tokens=original_total,retained_user_tokens=kept_total,max_source_input_tokens=6000,truncation_policy='First fragment max3072, later2048, cumulative6000, no padding or duplication. Actual model replies/chat-template overhead added by driver.')
   selected.append(session)
  usable=[s for s in selected if s['cohort']==cohort];counts[cohort]=len(usable)
  for i,s in enumerate(usable):s['seed_partition']=SEEDS[i%len(SEEDS)]
  summaries[cohort]={'sessions':len(usable),'turns':stats([len(s['turns']) for s in usable]),'first_user_tokens':stats([s['turns'][0]['retained_user_tokens'] for s in usable]),'sum_user_tokens':stats([sum(t['retained_user_tokens'] for t in s['turns']) for s in usable]),'truncated_fragments':sum(t['truncated'] for s in usable for t in s['turns'])}
  assert len(usable)>=48,(cohort,len(usable))
 result={'schema':'quantized-realworld-sessions-v1','tokenizer':str(Path(tokenizer_path).resolve()),'tokenizer_files':{p.name:sha(p.read_bytes()) for p in Path(tokenizer_path).glob('*token*') if p.is_file()},'seeds':SEEDS,'sessions':selected,'summary':summaries,'filter_counts':filters,'provenance':{'chat':'Observed public human conversations; only user turns supplied to replay.','agent':'Recorded software-agent issue/tool observations; fixed external inputs, not live tools.','waits':'No real think/tool wait is available. Driver must explicitly label synthetic arrival/wait profiles; no future timer exposed to hint producer.','prepared_inputs_include_historical_answers':False,'selection':'Deterministic hash order; disjoint source IDs across seed partitions. Limited nonrandom API row windows, English multi-turn subset, not production population weights.'}}
 save(output,result);print(json.dumps({'output':str(output),'summary':summaries,'filter_counts':filters}))

def check(path):
 d=json.loads(path.read_text());ids=[s['id'] for s in d['sessions']];assert len(ids)==len(set(ids));controls=[]
 for s in d['sessions']:
  assert len(s['turns'])>=3 and s['seed_partition'] in SEEDS
  assert all(t['observed_wait_ms'] is None and t['source_message_index']>=0 for t in s['turns'])
  for i in range(len(s['turns'])):
   generated=['ACTUAL_GENERATION_'+str(j) for j in range(i)];history=messages_for_turn(s,i,generated);assert validate_history(s,i,generated,history)
  good=messages_for_turn(s,0,[])
  wrong=good+[{'role':'user','content':s['turns'][1]['user_text']}]
  try:validate_history(s,0,[],wrong)
  except ValueError:controls.append('future_turn_rejected')
  else:raise AssertionError('future control accepted')
  wrong=messages_for_turn(s,1,['ACTUAL']);wrong[1]={'role':'assistant','content':'SOURCE_HISTORICAL_ANSWER'}
  try:validate_history(s,1,['ACTUAL'],wrong)
  except ValueError:controls.append('historical_answer_rejected')
  else:raise AssertionError('historical-answer control accepted')
 manifest={str(p.relative_to(H)):sha(p.read_bytes()) for p in [Path(__file__),path,*D.glob('*.json'),*D.glob('*.md'),*sorted((D/'swe').glob('*.json'))] if p.is_file()}
 receipt={'passed':True,'sessions':len(ids),'cohorts':d['summary'],'source_ids_unique':True,'seed_partitions_disjoint':True,'future_turn_and_historical_answer_controls_rejected':len(controls),'no_model_or_gpu_execution':True,'manifest':manifest,'runtime_consumer_boundary':'Driver must call/obey messages_for_turn using ACTUAL completed generated output, not source answers; CLI tests alone do not prove runtime use.'}
 save(P/'corpus-validation.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='manifest'}))

def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True);sub.add_parser('fetch');a=sub.add_parser('prepare');a.add_argument('--tokenizer',type=Path,required=True);a.add_argument('--output',type=Path,default=P/'session-corpus.json');a=sub.add_parser('check');a.add_argument('--input',type=Path,default=P/'session-corpus.json');args=ap.parse_args()
 if args.command=='fetch':fetch()
 elif args.command=='prepare':prepare(args.tokenizer,args.output)
 else:check(args.input)
if __name__=='__main__':main()
