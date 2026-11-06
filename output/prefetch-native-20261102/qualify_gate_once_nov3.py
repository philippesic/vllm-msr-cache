import asyncio,hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from live_once_service_workload_nov3 import producer
H=Path(__file__).resolve().parent
class Response:
 status=200
 def __init__(self,ack):self.ack=ack
 async def __aenter__(self):return self
 async def __aexit__(self,*args):pass
 async def json(self):return self.ack
class Client:
 def __init__(self,acks,state,keep_pending=False):self.acks=acks;self.state=state;self.calls=0;self.keep_pending=keep_pending
 def post(self,*args,**kw):
  self.calls+=1
  assert self.calls<=len(self.acks),'Unexpected repeated POST after ready acknowledgement'
  if self.calls==len(self.acks) and not self.keep_pending:self.state['pending']=False;self.state['changed'].set()
  return Response(self.acks[self.calls-1])
def ack(n,reason='cpu_prefix_recency_refreshed',status='no_action'):return {'ready_cpu_prefix_keys':n,'reason':reason,'status':status}
async def case(name,acks,stopped):
 state={'pending':True,'observed':[1]*16,'completed_ns':1,'changed':asyncio.Event()};rows=[];c=Client(acks,state,keep_pending=stopped)
 await asyncio.wait_for(producer(c,'http://unused',{},state,0,'session',rows,0),.5)
 assert state['pending'] is stopped
 assert c.calls==len(acks) and len(rows)==len(acks)
 assert all(r['completed_ns']>=r['started_ns'] for r in rows)
 assert sum(r.get('producer_policy_stop')=='first_positive_cpu_recency_ack' for r in rows)==int(stopped)
 return {'case':name,'POSTs':c.calls,'stop_recorded':stopped,'pending_still_true_on_return':state['pending'],'passed':True}
async def main():
 rows=[]
 for args in [('GPU_full_skip_then_partial_positive',[dict(ack(0),cpu_retention_skipped_full_gpu=True,cpu_retention_ready_prefix_keys=5),dict(ack(2),cpu_retention_skipped_full_gpu=False)],True),('repeated_full_skips_then_zero_until_return',[dict(ack(0),cpu_retention_skipped_full_gpu=True),dict(ack(0),cpu_retention_skipped_full_gpu=True),dict(ack(0),cpu_retention_skipped_full_gpu=False)],False),('first_positive',[ack(2)],True),('zero_then_ready',[ack(0),ack(3)],True),('bool_not_count',[ack(True)],False),('wrong_reason',[ack(3,'not_ready')],False),('wrong_status',[ack(3,status='rejected')],False),('zero_until_return',[ack(0)],False)]:rows.append(await case(*args))
 result={'schema':'GPU-gate-once-positive-producer-CPU-qualification-v1','cases':rows,'source_sha256':hashlib.sha256((H/'live_once_service_workload_nov3.py').read_bytes()).hexdigest(),'scope':'No GPU/model generation; controls verify causal response gating and no additional POST after ready. Does not establish benefit.'}
 (H/'live-nov3-gate-once-cpu-qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
asyncio.run(main())
