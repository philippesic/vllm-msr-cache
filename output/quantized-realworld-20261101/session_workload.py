"""Public sessions with actual generated histories and causal private waits."""
import asyncio,hashlib,json,random,time
from pathlib import Path
from benchmarks.eviction_prefetch_campaign import consume_campaign_sse
from corpus import messages_for_turn,validate_history
H=Path(__file__).resolve().parent
RUN='publictrial';CELL='frozen'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rid(i):return f'sc|{RUN}|{CELL}|fixed|sample-{i}'
def observed_prefix(prompt,output):
 past=list(prompt)+list(output[:-1]);return past[:len(past)//16*16]
def validate_hint_contract(namespace):
 from semantic_offload.wait_hint import KnownPrefixWaitHint
 now=time.monotonic_ns();raw={'schema':'semantic-known-prefix-wait-hint-v1','event_id':digest('preflight'),'owner_id':rid(1),'run_id':RUN,'cell_id':CELL,'session_id':'preflight','generation':0,'observed_prefix_id':digest({'namespace':namespace,'token_ids':[1]*16}),'prefix_boundary_tokens':16,'observation_available_ns':now,'input_publication_ns':now,'declared_wait_ms':750,'wait_provenance':'driver_declared_tool_wait_budget','namespace':namespace}
 KnownPrefixWaitHint.from_mapping(raw)
 invalid=dict(raw,declared_wait_ms=1000)
 try:KnownPrefixWaitHint.from_mapping(invalid)
 except ValueError:return {'valid_hint_accepted':True,'invalid1000ms_rejected':True,'constant_estimate_ms':750,'provenance':raw['wait_provenance']}
 raise AssertionError('invalid hint oracle not rejected')

async def producer(client,url,namespace,state,index,sid,records):
 """Input excludes all future text and the private wait duration/deadline."""
 sequence=0
 while state['pending']:
  now=time.monotonic_ns();event=digest({'run':RUN,'sid':sid,'index':index,'sequence':sequence});sequence+=1
  observed=state['observed']
  hint={'schema':'semantic-known-prefix-wait-hint-v1','event_id':event,'owner_id':rid(index),'run_id':RUN,'cell_id':CELL,'session_id':sid,'generation':0,'observed_prefix_id':digest({'namespace':namespace,'token_ids':observed}),'prefix_boundary_tokens':len(observed),'observation_available_ns':now,'input_publication_ns':now,'declared_wait_ms':750,'wait_provenance':'driver_declared_tool_wait_budget','namespace':namespace}
  row={'hint':hint,'started_ns':now,'source_completed_at_ns':state['completed_ns']}
  try:
   async with client.post(url+'/plugins/semantic-offload/wait-hints',json=hint) as response:row.update(http_status=response.status,response=await response.json())
   if row['http_status']!=200:
    state['protocol_error']='hint HTTP '+str(row['http_status'])+': '+str(row['response']);return
   if row['response'].get('status')=='accepted':state['binding']=row['response'].get('canonical_event_id') or event
  except asyncio.CancelledError:row['error']='producer_stopped';raise
  except Exception as e:row['error']=repr(e)
  finally:row['completed_ns']=time.monotonic_ns();records.append(row)
  try:await asyncio.wait_for(state['changed'].wait(),.2)
  except asyncio.TimeoutError:pass
async def exercise(handle,p,a,result):
 import aiohttp
 from transformers import AutoTokenizer
 corpus=json.loads((H/'prepared/session-corpus.json').read_text());tok=AutoTokenizer.from_pretrained(corpus['tokenizer'],local_files_only=True)
 sessions=[s for s in corpus['sessions'] if s['cohort']==a.cohort and s['seed_partition']==a.seed][:a.clients]
 assert len(sessions)==a.clients
 result['planned_requests']=sum(1 if a.open_rate else (1 if a.abandon_every and (i+1)%a.abandon_every==0 else min(a.turns,len(s['turns']))) for i,s in enumerate(sessions))
 # The policy coroutine never receives this private wait schedule.
 rng=random.Random(a.seed+893);private_waits={(i,t):(rng.choice([0,2000,4000,8000,12000]) if a.wait=='bursty' else 0) for i,s in enumerate(sessions) for t in range(min(a.turns,len(s['turns']))-1)}
 if a.wait=='longtool':
  assert len(sessions)==24,'longtool pilot fixed24 clients'
  for turn in range(a.turns-1):
   values=[0]*4+[2000]*4+[30000]*6+[45000]*5+[60000]*5;rng.shuffle(values)
   for slot,value in enumerate(values):private_waits[(slot,turn)]=value
 result['workload']={'cohort':a.cohort,'source_ids':[s['source_id'] for s in sessions],'clients':len(sessions),'max_active':a.active,'turn_limit':a.turns,'output_cap':a.max_output,'respect_eos':True,'history':'actual model responses via standard chat template','wait_origin':('synthetic seeded private exact24client fractions 4zero/4two/6thirty/5fortyfive/5sixty seconds perwait' if a.wait=='longtool' else ('synthetic seeded private 0/2/4/8/12 seconds' if a.wait=='bursty' else 'zero wait continuous control')),'arrival_origin':'finite open-loop first turn' if a.open_rate else 'simultaneous clients then actual-output closed loop','policy_estimate_ms':750,'policy_refresh_ms':200,'ack_gating':False,'recorded_tools_executed':False,'abandon_every':a.abandon_every,'pending_flag_assumption':'application knows a tool/agent wait is pending; no future return guarantee','private_wait_ms':{f'{i}:{t}':v for (i,t),v in private_waits.items()}}
 result['prefix_contiguity']=[];result['tool_intervals']=[];result['abandonments']=[];tasks=[]
 result['load_start_ns']=time.monotonic_ns()
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as client:
  async def request(index,tokens,sid,turn,observed,binding,intended):
   params={'campaign_schema':'ep-campaign-serving-v1','campaign_provenance':{'content_origin':'pinned_public_'+a.cohort,'arrival_origin':result['workload']['arrival_origin']},'external_request_id':rid(index),'semantic_request_id':rid(index),'semantic_sample_id':rid(index),'sample_index':index,'benchmark_client_sequence':index,'seed':a.seed,'run_id':RUN,'cell_id':CELL,'session_id':sid,'generation':0,'key_chain_digest':digest(tokens)}
   if observed:params.update(observed_prefix_id=digest({'namespace':p['namespace'],'token_ids':observed}),observed_prefix_boundary_tokens=len(observed))
   if binding:params['wait_hint_event_id']=binding
   body={'model':'matrix-model','request_id':rid(index),'prompt':tokens,'max_tokens':a.max_output,'temperature':0.,'top_p':1.,'top_k':-1,'repetition_penalty':1.,'seed':a.seed+index,'stream':True,'return_tokens_as_token_ids':True,'return_token_ids':True,'kv_transfer_params':params}
   row={'index':index,'role':a.cohort+('-return' if turn else '-initial'),'cohort':a.cohort,'turn':turn,'session_id':sid,'intended_release_ns':intended,'request_start_ns':time.monotonic_ns(),'payload':body}
   try:
    async with client.post(handle.base_url()+'/v1/completions',json=body) as response:
     row['http_status']=response.status
     if response.status!=200:raise RuntimeError((await response.text())[-1500:])
     row.update(await consume_campaign_sse(response.content))
     assert row['terminal_complete']
    row['success']=True
   except Exception as e:row.update(success=False,error=repr(e),output_token_ids=[])
   row['request_end_ns']=time.monotonic_ns();result['requests'].append(row)
   if not row['success']:raise RuntimeError(row['error'])
   row['generated_text']=tok.decode(row['output_token_ids'],skip_special_tokens=True)
   return row
  async def session_worker(slot,session):
   generated=[];previous=None;state=None;policy_task=None
   maxturn=1 if a.open_rate else min(a.turns,len(session['turns']))
   for turn in range(maxturn):
    index=slot*4+turn;sid=session['id']
    if turn:
     wait_ms=private_waits[(slot,turn-1)];intended=previous['request_end_ns']+wait_ms*1_000_000
     # Only completed transcript fields are passed to producer. It cannot see
     # the wait duration, deadline, next message, or this session object.
     source=observed_prefix(previous['payload']['prompt'],previous['output_token_ids'])
     state={'pending':True,'completed_ns':previous['request_end_ns'],'observed':tuple(source),'changed':asyncio.Event()}
     if a.arm!='arc' and wait_ms>0 and source:
      policy_task=asyncio.create_task(producer(client,handle.base_url(),p['namespace'],state,index,sid,result['hints']));tasks.append(policy_task)
     await asyncio.sleep(max(0,(intended-time.monotonic_ns())/1e9));state['pending']=False;state['changed'].set()
    else:
     intended=result['load_start_ns']+int(slot/a.open_rate*1e9) if a.open_rate else result['load_start_ns']
     await asyncio.sleep(max(0,(intended-time.monotonic_ns())/1e9))
    if state and state.get('protocol_error'):raise RuntimeError(state['protocol_error'])
    if turn==1 and a.abandon_every and (slot+1)%a.abandon_every==0:
     result['abandonments'].append({'session_id':sid,'at_ns':time.monotonic_ns(),'pending_started_ns':state['completed_ns'],'binding':state.get('binding'),'producer_learned_at_cancellation_only':True})
     if policy_task and not policy_task.done():policy_task.cancel();await asyncio.gather(policy_task,return_exceptions=True)
     return
    # No future turn is passed to the tokenizer until this release.
    messages=messages_for_turn(session,turn,generated);validate_history(session,turn,generated,messages)
    tokens=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
    assert isinstance(tokens,list) and tokens and all(isinstance(t,int) for t in tokens)
    assert len(tokens)+a.max_output<=p['geometry']['max_model_len']
    observed=tokens[:len(tokens)//16*16];binding=None
    if state:
     source=list(state['observed']);same=tokens[:len(source)]==source
     result['prefix_contiguity'].append({'index':index,'previous_computed_fullblock_tokens':len(source),'contiguous':same})
     if same:observed=source;binding=state.get('binding')
    row=await request(index,tokens,sid,turn,observed,binding,intended)
    if state:result['tool_intervals'].append({'session_id':sid,'start_ns':state['completed_ns'],'intended_release_ns':intended,'return_end_ns':row['request_end_ns'],'simulated':True})
    if policy_task and not policy_task.done():policy_task.cancel();await asyncio.gather(policy_task,return_exceptions=True)
    generated.append(row['generated_text']);previous=row;state=None;policy_task=None
  try:await asyncio.gather(*(session_worker(i,s) for i,s in enumerate(sessions)))
  finally:
   for task in tasks:
    if not task.done():task.cancel()
   await asyncio.gather(*tasks,return_exceptions=True)
   result['load_end_ns']=time.monotonic_ns()
