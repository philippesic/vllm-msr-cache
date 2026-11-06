import asyncio, hashlib, json, time
from benchmarks.eviction_prefetch_campaign import consume_campaign_sse
from pathlib import Path
def namespace(p): return p["namespace"]
def parse(path,marker):
 return [json.loads(x.split("SEMANTIC_PRESERVATION_EVENT ",1)[1]) for x in Path(path).read_text(errors="replace").splitlines() if "SEMANTIC_PRESERVATION_EVENT " in x]
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def rid(index):return f'sc|agenttrial|frozen|fixed|sample-{index}'

def payload(p,index,tokens,n,sid,initial=None,hint=None):
 params={'campaign_schema':'ep-campaign-serving-v1','campaign_provenance':{'content_origin':'simulated_agent_trace','arrival_origin':'closed_loop_agent_or_open_loop_background'},'external_request_id':rid(index),'semantic_request_id':rid(index),'semantic_sample_id':rid(index),'sample_index':index,'benchmark_client_sequence':index,
  'seed':p['trace']['seed'],'run_id':'agenttrial','cell_id':'frozen','session_id':sid,'generation':0,'key_chain_digest':digest(tokens)}
 if initial:
  params.update(observed_prefix_id=digest({'namespace':namespace(p),'token_ids':initial}),observed_prefix_boundary_tokens=len(initial))
 if hint:params['wait_hint_event_id']=hint
 return {'model':'matrix-model','request_id':rid(index),'prompt':tokens,'max_tokens':n,'ignore_eos':True,'temperature':0.0,'top_p':1.0,'top_k':-1,'repetition_penalty':1.0,'seed':p['trace']['seed']+index,'stream':True,'return_tokens_as_token_ids':True,'return_token_ids':True,'kv_transfer_params':params}

async def exercise(handle,p,arm,result):
 import aiohttp
 start=time.monotonic_ns();result['load_start_ns']=start
 trace=p['trace'];results=result['requests'];pending_rpcs=set()
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as client:
  async def request(index,tokens,n,sid,role,intended,initial=None,hint=None):
   body=payload(p,index,tokens,n,sid,initial,hint);begin=time.monotonic_ns();row={'index':index,'role':role,'session_id':sid,'intended_release_ns':intended,'request_start_ns':begin,'payload':body}
   try:
    async with client.post(handle.base_url()+'/v1/completions',json=body,headers={'x-agent-trial-request-id':rid(index)}) as response:
     row['http_status']=response.status
     if response.status!=200:raise RuntimeError((await response.text())[-3000:])
     stream=await consume_campaign_sse(response.content);row.update(stream)
     assert stream['terminal_complete'] and len(stream['output_token_ids'])==n
    row['success']=True
   except Exception as e:row.update(success=False,error=repr(e),output_token_ids=[])
   row['request_end_ns']=time.monotonic_ns();results.append(row);return row
  async def publish(state,index,sid,initial,seq):
   now=time.monotonic_ns();event=digest({'run':'agenttrial','session':sid,'index':index,'refresh':seq})
   hint={'schema':'semantic-known-prefix-wait-hint-v1','event_id':event,'owner_id':rid(index),'run_id':'agenttrial','cell_id':'frozen','session_id':sid,'generation':0,
    'observed_prefix_id':digest({'namespace':namespace(p),'token_ids':initial}),'prefix_boundary_tokens':len(initial),
    'observation_available_ns':now,'input_publication_ns':now,'declared_wait_ms':250,'wait_provenance':'driver_declared_tool_wait_budget','namespace':namespace(p)}
   state['last_published']=event;record={'hint':hint,'started_ns':now,'source_completed_at_ns':state['completed_ns']}
   try:
    async with client.post(handle.base_url()+'/plugins/semantic-offload/wait-hints',json=hint,headers={'x-agent-trial-hint-id':event}) as response:
     record['http_status']=response.status;record['response']=await response.json()
    if record['response'].get('status')=='accepted' and not state.get('accepted'):state['accepted']=event
   except Exception as e:record['error']=repr(e)
   record['completed_ns']=time.monotonic_ns();result['hints'].append(record)
  async def policy(state,index,sid,initial):
   # This producer receives only current pending state and already completed
   # immutable prefix. Private waits/abandonment/next inputs are never arguments.
   deadline=state['completed_ns']+2_000_000_000;seq=0
   while state['pending'] and time.monotonic_ns()<deadline:
    if not arm.startswith('arc') and not state.get('accepted'):
     task=asyncio.create_task(publish(state,index,sid,initial,seq));pending_rpcs.add(task);task.add_done_callback(pending_rpcs.discard)
    seq+=1
    try:await asyncio.wait_for(state['changed'].wait(),timeout=0.25)
    except asyncio.TimeoutError:pass
  async def agent(s,prepared,si):
   intended=start+s['initial_arrival_ms']*1_000_000
   await asyncio.sleep(max(0,(intended-time.monotonic_ns())/1e9))
   tokens=list(prepared['initial_tokens']);binding=None
   for turn in range(len(s['turns'])):
    # This turn's input becomes visible only after the tool's actual release.
    tokens+=prepared['turn_append_tokens'][turn]
    row=await request(si*6+turn,list(tokens),64,s['session_id'],f"{s['session_id']}-turn-{turn}",intended,prepared['initial_tokens'],binding)
    if not row['success']:break
    tokens+=row['output_token_ids']
    if turn==len(s['turns'])-1:break
    state={'pending':True,'completed_ns':row['request_end_ns'],'changed':asyncio.Event()}
    task=asyncio.create_task(policy(state,si*6+turn+1,s['session_id'],prepared['initial_tokens']))
    private=s['load_generator_private']
    if private['abandon_after_turn']==turn:
     # Cancellation is observed only after a pending tool interval, never at
     # the LLM completion where the private abandonment became known to us.
     await asyncio.sleep(2.0);state['pending']=False;state['changed'].set();await task
     result['cancellations'].append({'session_id':s['session_id'],'observed_at_ns':time.monotonic_ns(),'after_completed_turn':turn,'policy_notified_at_completion':False,'endpoint':'no cancellation API; stop refresh, existing leases expire'})
     break
    wait=private['tool_wait_ms_after_turn'][turn]
    intended=row['request_end_ns']+wait*1_000_000
    await asyncio.sleep(max(0,(intended-time.monotonic_ns())/1e9))
    state['pending']=False;state['changed'].set()
    # Read only already received acknowledgments. No release waits for RPC.
    binding=state.get('accepted') or state.get('last_published')
    await task
   return None
  async def background(b,prepared,i):
   intended=start+b['arrival_ms']*1_000_000
   await asyncio.sleep(max(0,(intended-time.monotonic_ns())/1e9))
   await request(36+i,prepared['tokens'],b['max_output_tokens'],b['request_id'],b['request_id'],intended)
  tasks=[asyncio.create_task(agent(s,t,i)) for i,(s,t) in enumerate(zip(trace['sessions'],p['sessions']))]
  tasks += [asyncio.create_task(background(b,t,i)) for i,(b,t) in enumerate(zip(trace['background'],p['background']))]
  await asyncio.gather(*tasks)
  result['load_end_ns']=max(x['request_end_ns'] for x in results)
  if pending_rpcs:await asyncio.gather(*list(pending_rpcs),return_exceptions=True)
  # Terminal transfer/status drain is post-load, excluded from client throughput.
  until=time.monotonic()+8
  while len(parse(handle.log_path,'AGENT_TRIAL_TERMINAL'))<len(results):
   if time.monotonic()>until:break
   await asyncio.sleep(0.05)
  async with client.get(handle.metrics_url()) as response:result['metrics_final']=await response.text()
