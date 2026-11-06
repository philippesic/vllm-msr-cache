"""Old traces establish potential intervals, never actual demand guard calls."""
import collections,gzip,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
OLD=H.parent/'vram-frontier-20261102'
sys.path.insert(0,str(OLD));import campaign_metrics as metrics

def analyze(raw,meta):
 events=raw.get('wait_hint_events',[]);groups=collections.defaultdict(list)
 for e in events:
  if isinstance(e.get('event_id'),str):groups[e['event_id']].append(e)
 terminals={t['external_request_id']:t for t in raw.get('terminals',[])};rows=[]
 for bind in events:
  if bind.get('event')!='resume_bound':continue
  eid=bind['event_id'];at=bind['bound_at_ns'];related=groups[eid];rid=bind['request_id'];term=terminals.get(rid,{})
  admission=term.get('first_admission',{}).get('admitted_at_ns');ends=[]
  for e in related:
   if e.get('event') in ['staging_ready','staging_retired','staging_retired_pending_fence']:
    stamp=metrics.event_stamp(e)
    if isinstance(stamp,int) and stamp>=at:ends.append((stamp,e['event'],e.get('reason')))
  if isinstance(admission,int) and admission>=at:ends.append((admission,'first_admission',None))
  end=min(ends) if ends else (None,None,None);status=bind.get('staging_status');potential=(end[0]-at)/1e6 if end[0] else None
  rows.append({'event_id':eid,'request_id':rid,'bound_at_ns':at,'bind_status':status,'first_admission_ns':admission,'potential_end_ns':end[0],'potential_end_event':end[1],'potential_end_reason':end[2],'potential_bound_to_endpoint_ms':potential if status in ['pending','in_flight'] else 0,'bound_to_first_admission_ms':(admission-at)/1e6 if isinstance(admission,int) else None,'actual_guard_calls_observed':None})
 spans=[r['potential_bound_to_endpoint_ms'] for r in rows if r['bind_status'] in ['pending','in_flight'] and r['potential_bound_to_endpoint_ms'] is not None]
 return {**meta,'bind_count':len(rows),'bind_states':dict(collections.Counter(r['bind_status'] for r in rows)),'potential_span_ms':metrics.dist(spans),'sum_potential_span_ms':sum(spans),'actual_guard_hits':None,'rows':rows,'limits':['binding is not a scheduler get_num call; guard may never run before state retires','endpoint interval upperbounds possible wall delay only, not causal added latency or throughput loss','invalid-proof runs remain invalid; timing readout does not waive them']}

def main():
 plan=json.loads((OLD/'ANALYSIS_PLAN.json').read_text());rows=[]
 for run in plan['runs']:
  if run['arm']!='candidate':continue
  p=OLD/'runs'/run['name']/'raw.json.gz';raw=json.load(gzip.open(p));s=json.loads((p.parent/'summary.json').read_text());m={'name':run['name'],'GPU_blocks':run['gpu_blocks'],'raw_path':str(p),'strict_proof_valid':s['proofs']['all_passed'],'observed_raw_requests':len(raw['requests'])};row=analyze(raw,m);rows.append(row);print(run['gpu_blocks'],row['bind_states'],row['sum_potential_span_ms'],flush=True)
 out={'scope':'all8semanticfrontierpoints; retrospective potential only','runs':rows,'direct_measurement_required':True,'reason':'old logs never record exact pending/inflight get_num early-return; bind state alone cannot prove a stall','recommended_direct_budgets':[768,2048]};(H/'OFFLINE.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
