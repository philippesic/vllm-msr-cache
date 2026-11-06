"""Retrospective causal-state proxy for proposed client capacity suppression.

No serving code or decisions changed. Replay known dispatch/completion events in
chronological order. Current generation tokens and future prompts/waits are never
read for a decision; shared roots are double counted.
"""
import collections,gzip,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent

def summarize(xs):
 if not xs:return {'observations':0}
 return {'observations':len(xs),'over_capacity':sum(x['full_blocks']>x['CPU_capacity_blocks'] for x in xs),'within_capacity':sum(x['full_blocks']<=x['CPU_capacity_blocks'] for x in xs),'full_blocks_min':min(x['full_blocks'] for x in xs),'full_blocks_max':max(x['full_blocks'] for x in xs),'full_blocks_mean':sum(x['full_blocks'] for x in xs)/len(xs)}
def main():
 rows=[]
 for block in ['gpu-recency','gpu-gate']:
  m=json.loads((H/(block+'-frozen-plan-nov3.json')).read_text())
  for item in m['execution_order']:
   if item['arm']!='cpu':continue
   p=H/'runs'/item['name']/'raw.json.gz';r=json.load(gzip.open(p,'rt'));events=[];turn_limit=r['workload']['turn_limit']
   for q in r['requests']:
    events.append((q['request_start_ns'],'dispatch',q))
    events.append((q['request_end_ns'],'complete',q))
   for hint in r['hints']:events.append((hint['started_ns'],'hint',hint))
   events.sort(key=lambda e:e[0]);state={};snapshots=[];peak=0
   for t,kind,x in events:
    if kind=='dispatch':state[x['session_id']]={'kind':'inflight_prompt','full_blocks':len(x['payload']['prompt'])//16}
    elif kind=='complete':
     if x['turn']+1<turn_limit:state[x['session_id']]={'kind':'pending_completed_prefix','full_blocks':(len(x['payload']['prompt'])+len(x['output_token_ids'])-1)//16}
     else:state.pop(x['session_id'],None)
    else:
     counts=collections.Counter(v['kind'] for v in state.values());parts={k:sum(v['full_blocks'] for v in state.values() if v['kind']==k) for k in ['inflight_prompt','pending_completed_prefix']}
     snapshots.append({'at_ns':t,'session_id':x['hint']['session_id'],'owner_id':x['hint']['owner_id'],'source_completed_at_ns':x['source_completed_at_ns'],'full_blocks':sum(parts.values()),'parts':parts,'sessions':dict(counts),'CPU_capacity_blocks':r['geometry']['cpu_blocks'],'event_id':x['hint']['event_id'],'actual_CPU_ready_refresh':x.get('response',{}).get('ready_cpu_prefix_keys'),'actual_reason':x.get('response',{}).get('reason')})
    peak=max(peak,sum(v['full_blocks'] for v in state.values()))
   assert not state,'Completed workflow must leave no registry state'
   first={}
   for x in snapshots:first.setdefault((x['session_id'],x['owner_id'],x['source_completed_at_ns']),x)
   rows.append({'name':item['name'],'partition':item['partition'],'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'actual_arm':'always_CPU_advice','CPU_capacity_blocks':r['geometry']['cpu_blocks'],'proxy_peak_full_blocks_at_any_event':peak,'all_actual_hint_times':summarize(snapshots),'first_actual_hint_per_wait':summarize(list(first.values())),'snapshots':snapshots})
 out={'schema':'astra-client-capacity-hypothesis-retrospective-v1','rows':rows,'scope':'Unexecuted hypothetical predicate, evaluated only at actual CPU-control hint timestamps. No performance counterfactual or new benchmark. Event state includes dispatched prompts (including server-queued requests) and completed prefixes of known nonterminal workflows; excludes current generation and future messages/wait lengths.','limitations':['Shared prefix roots double counted, CPU/GPU residency ignored; over-capacity is not proof advice is harmful or cannot help a subset.','Unknown in-flight generated KV can undercount; pending prefix registration is proxied by recorded completion time.','Suppressing HTTP would change future timing and policy cadence, so observed trajectories cannot predict execution outcome.','Threshold is proposed engineering heuristic, not learned proof or production eligibility condition.','Inputs are spent; any promising executed candidate needs fresh inputs plus timing/concurrency variation.']}
 (H/'live-nov3-capacity-hypothesis-independent-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{k:v for k,v in x.items() if k!='snapshots'} for x in rows],indent=2))
if __name__=='__main__':main()
