"""Real CPU-only HTTP control: deliberately drop a reused POST, never a fresh one.
This models the failure class and qualifies non-reuse; it cannot diagnose old runs.
"""
from pathlib import Path
import asyncio,json,collections,hashlib,sys,time
import aiohttp
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parents[1]))
from live_service_workload_nov3 import create_http_session,producer
async def main():
 handlers=set();server_counts=collections.Counter()
 async def serve(reader,writer):
  task=asyncio.current_task();handlers.add(task);ordinal=0
  try:
   while True:
    try:headers=await reader.readuntil(b'\r\n\r\n')
    except asyncio.IncompleteReadError:return
    length=next((int(line.split(b':',1)[1]) for line in headers.split(b'\r\n') if line.lower().startswith(b'content-length:')),0)
    if length:await reader.readexactly(length)
    ordinal+=1;server_counts['received']+=1
    if ordinal>1:
     server_counts['deliberately_dropped_reused_POST']+=1;return
    body=b'{"ok":true}'
    writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '+str(len(body)).encode()+b'\r\nConnection: keep-alive\r\n\r\n'+body);await writer.drain()
  finally:
   writer.close();await writer.wait_closed();handlers.discard(task)
 server=await asyncio.start_server(serve,'127.0.0.1',0);url='http://127.0.0.1:'+str(server.sockets[0].getsockname()[1])+'/post'
 control={'reuses':0,'errors':[]};trace=aiohttp.TraceConfig()
 async def reuse(session,ctx,params):control['reuses']+=1
 trace.on_connection_reuseconn.append(reuse)
 async with aiohttp.ClientSession(trace_configs=[trace]) as client:
  for i in range(2):
   try:
    async with client.post(url,json={'request':i}) as response:assert (await response.json())['ok']
   except aiohttp.ServerDisconnectedError as e:control['errors'].append(type(e).__name__)
 assert control['reuses']==1 and control['errors']==['ServerDisconnectedError'],control
 result={};success=0
 async with create_http_session(result) as client:
  assert client.connector.force_close
  for i in range(100):
   async with client.post(url,json={'request':i},trace_request_ctx={'kind':'qualification','index':i}) as response:
    assert response.status==200 and (await response.json())['ok'];success+=1
 server.close();await server.wait_closed()
 if handlers:await asyncio.gather(*tuple(handlers))
 counts=collections.Counter(e['event'] for e in result['http_connection_events'])
 assert success==100 and counts['connection_create_end']==100 and counts['connection_reuse']==0 and counts['request_exception']==0,(success,counts)
 # Qualify the actual producer's intentional stop independently of network faults.
 accepted=asyncio.Event();cancel_handlers=set()
 async def hold_response(reader,writer):
  task=asyncio.current_task();cancel_handlers.add(task)
  try:
   headers=await reader.readuntil(b'\r\n\r\n');length=next((int(line.split(b':',1)[1]) for line in headers.split(b'\r\n') if line.lower().startswith(b'content-length:')),0)
   if length:await reader.readexactly(length)
   accepted.set();await reader.read()
  finally:writer.close();await writer.wait_closed();cancel_handlers.discard(task)
 cancel_server=await asyncio.start_server(hold_response,'127.0.0.1',0);cancel_url='http://127.0.0.1:'+str(cancel_server.sockets[0].getsockname()[1]);cancel_result={};hint_rows=[]
 state={'pending':True,'observed':tuple([1]*16),'completed_ns':time.monotonic_ns(),'changed':asyncio.Event()}
 async with create_http_session(cancel_result) as client:
  task=asyncio.create_task(producer(client,cancel_url,{},state,1,'cpu-cancellation-control',hint_rows,1000))
  await asyncio.wait_for(accepted.wait(),2);state['pending']=False;state['changed'].set();task.cancel();done=await asyncio.gather(task,return_exceptions=True)
  assert len(done)==1 and isinstance(done[0],asyncio.CancelledError)
 cancel_server.close();await cancel_server.wait_closed()
 if cancel_handlers:await asyncio.wait_for(asyncio.gather(*tuple(cancel_handlers)),2)
 exceptions=[x for x in cancel_result['http_connection_events'] if x['event']=='request_exception']
 assert len(hint_rows)==1 and hint_rows[0]['error']=='producer_stopped' and len(exceptions)==1 and exceptions[0]['error']=='CancelledError()'
 assert exceptions[0]['request']=={'kind':'hint','event_id':hint_rows[0]['hint']['event_id']}
 cancellation={'actual_producer_stopped_record':True,'matching_trace_cancelled_error':True,'generation_failure_hidden':False,'hint_retried':False,'owned_server_closed':not cancel_server.is_serving(),'owned_tasks_remaining':len(cancel_handlers)}
 receipt={'schema':'http-transport-cpu-qualification-v1','positive_failure_control':control,'intentional_hint_cancellation_control':cancellation,'fresh_connection_requests':success,'trace_event_counts':dict(counts),'server_counts':dict(server_counts),'configuration':result['http_client_configuration'],'source_sha256':hashlib.sha256((H/'live_service_workload_nov3.py').read_bytes()).hexdigest(),'no_gpu_used':True,'owned_server_closed':not server.is_serving(),'owned_connection_tasks_remaining':len(handlers),'limitations':['Server deliberately closes when a POST reuses its socket; this models failure class, not historical idle-timeout causation.','No retry hides a request failure; fresh-connection policy is common to both future benchmark arms.']}
 (H/'live-nov3-http-transport-cpu-qualification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
asyncio.run(main())
