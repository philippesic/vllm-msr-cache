import asyncio,hashlib,json,time
from pathlib import Path
from benchmarks.eviction_prefetch_campaign import consume_campaign_sse

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rid(index):return f'sc|prefetchtrial|frozen|fixed|sample-{index}'
def parse(path,marker):
 out=[]
 for s in Path(path).read_text(errors='replace').split(marker+' ')[1:]:
  try:out.append(json.JSONDecoder().raw_decode(s)[0])
  except json.JSONDecodeError:pass
 return out
async def exercise(handle,p,arm,result):
 import aiohttp
 start=time.monotonic_ns();result['load_start_ns']=start;result['tool_intervals']=[]
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as client:
  async def request(index,tokens,n,sid,role,initial=None,binding=None,intended=None):
   intended=intended or time.monotonic_ns();params={'campaign_schema':'ep-campaign-serving-v1','campaign_provenance':{'content_origin':'simulated_agent_trace','arrival_origin':'closed_loop_peer_tool_dag'},'external_request_id':rid(index),'semantic_request_id':rid(index),'semantic_sample_id':rid(index),'sample_index':index,'benchmark_client_sequence':index,'seed':p['trace']['seed'],'run_id':'prefetchtrial','cell_id':'frozen','session_id':sid,'generation':0,'key_chain_digest':digest(tokens)}
   if initial:params.update(observed_prefix_id=digest({'namespace':p['namespace'],'token_ids':initial}),observed_prefix_boundary_tokens=len(initial))
   if binding:params['wait_hint_event_id']=binding
   body={'model':'matrix-model','request_id':rid(index),'prompt':tokens,'max_tokens':n,'ignore_eos':True,'temperature':0.,'top_p':1.,'top_k':-1,'repetition_penalty':1.,'seed':p['trace']['seed']+index,'stream':True,'return_tokens_as_token_ids':True,'return_token_ids':True,'kv_transfer_params':params}
   row={'index':index,'role':role,'session_id':sid,'intended_release_ns':intended,'request_start_ns':time.monotonic_ns(),'payload':body}
   try:
    async with client.post(handle.base_url()+'/v1/completions',json=body) as response:
     row['http_status']=response.status
     if response.status!=200:raise RuntimeError((await response.text())[-1500:])
     stream=await consume_campaign_sse(response.content);row.update(stream);assert stream['terminal_complete'] and len(stream['output_token_ids'])==n
    row['success']=True
   except Exception as e:row.update(success=False,error=repr(e),output_token_ids=[])
   row['request_end_ns']=time.monotonic_ns();result['requests'].append(row)
   if not row['success']:raise RuntimeError(row.get('error'))
   return row
  async def publish(state,index,sid,initial,seq):
   now=time.monotonic_ns();event=digest({'run':'prefetchtrial','session':sid,'index':index,'refresh':seq});hint={'schema':'semantic-known-prefix-wait-hint-v1','event_id':event,'owner_id':rid(index),'run_id':'prefetchtrial','cell_id':'frozen','session_id':sid,'generation':0,'observed_prefix_id':digest({'namespace':p['namespace'],'token_ids':initial}),'prefix_boundary_tokens':len(initial),'observation_available_ns':now,'input_publication_ns':now,'declared_wait_ms':250,'wait_provenance':'driver_declared_tool_wait_budget','namespace':p['namespace']}
   record={'hint':hint,'started_ns':now,'source_completed_at_ns':state['completed_ns']}
   try:
    async with client.post(handle.base_url()+'/plugins/semantic-offload/wait-hints',json=hint) as response:
     record['http_status']=response.status;record['response']=await response.json()
    if record['response'].get('status')=='accepted':state['binding']=record['response'].get('canonical_event_id',event)
   except Exception as e:record['error']=repr(e)
   record['completed_ns']=time.monotonic_ns();result['hints'].append(record)
  async def policy(state,index,sid,initial):
   seq=0
   while state['pending']:
    if result['workload']=='qualify' and state.get('binding'):break
    await publish(state,index,sid,initial,seq);seq+=1
    try:await asyncio.wait_for(state['changed'].wait(),timeout=.2)
    except asyncio.TimeoutError:pass
  for i,r in enumerate(p['rounds']):
   si=i*4;sid=f'agent-{i}';initial=r['source'];warm=await request(si,initial,32,sid,'agent-warm',initial)
   state={'pending':True,'completed_ns':warm['request_end_ns'],'changed':asyncio.Event()};task=None
   if arm!='arc' and result['workload']=='causal':task=asyncio.create_task(policy(state,si+3,sid,initial))
   await request(si+1,r['pressure_a'],32,f'background-{i}-a','background-pressure')
   await request(si+2,r['pressure_b'],32,f'background-{i}-b','background-pressure')
   tool_ready=time.monotonic_ns();lead_ms=int(__import__('os').getenv('EXPERIMENT_LEAD_MS','250'));intended=tool_ready+lead_ms*1_000_000
   if arm!='arc' and result['workload']=='qualify':task=asyncio.create_task(policy(state,si+3,sid,initial))
   await asyncio.sleep(max(0,(intended-time.monotonic_ns())/1e9));state['pending']=False;state['changed'].set()
   # Returning demand never waits for hint acknowledgement or transfer readiness.
   binding=state.get('binding');tokens=initial+warm['output_token_ids']+r['append']
   row=await request(si+3,tokens,64,sid,'agent-return',initial,binding,intended)
   if task:await task
   result['tool_intervals'].append({'session_id':sid,'start_ns':warm['request_end_ns'],'peer_tools_complete_ns':tool_ready,'intended_release_ns':intended,'return_end_ns':row['request_end_ns'],'fixed_post_tool_gap_ms':lead_ms})
  result['load_end_ns']=max(r['request_end_ns'] for r in result['requests'])
  until=time.monotonic()+8
  while len(parse(handle.log_path,'SEMANTIC_PRESERVATION_EVENT'))<len(result['requests']):
   if time.monotonic()>until:break
   await asyncio.sleep(.05)
  async with client.get(handle.metrics_url()) as response:result['metrics_final']=await response.text()
