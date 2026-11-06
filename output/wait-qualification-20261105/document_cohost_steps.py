import sys
import json,collections,statistics
from pathlib import Path
b=Path(__file__).parent;campaign=sys.argv[1] if len(sys.argv)>1 else 'stage3-documents-cohost';root=b/campaign;result=json.loads((root/'result.json').read_text());out=[]
for case in result['cases']:
 p=root/case['name']/'kv-qualification.jsonl'
 if not p.exists():continue
 rows=[json.loads(l) for l in p.read_text().splitlines()];requests=[json.loads(l) for l in (p.parent/'requests.jsonl').read_text().splitlines()];lo=min(r['queued_ns'] for r in requests);hi=max(r['ended_ns'] for r in requests)
 steps=[r for r in rows if r['event']=='step' and lo<=r['ns']<=hi];groups={}
 for prior,nextstep in zip(steps,steps[1:]):
  kind='zero_tokens' if prior['scheduled_tokens']==0 else 'up_to_12_tokens' if prior['scheduled_tokens']<=12 else 'over_12_tokens'
  d=groups.setdefault(kind,dict(steps=0,inter_step_seconds=0,scheduled_tokens=0,max_gap_s=0,owned_block_seconds=0,running_sequence_seconds=0,seconds_below_32_reclaimable_blocks=0));gap=(nextstep['ns']-prior['ns'])/1e9;d['steps']+=1;d['inter_step_seconds']+=gap;d['scheduled_tokens']+=prior['scheduled_tokens'];d['max_gap_s']=max(d['max_gap_s'],gap);d['owned_block_seconds']+=(prior['total_blocks']-prior['reclaimable'])*gap;d['running_sequence_seconds']+=prior['running']*gap;d['seconds_below_32_reclaimable_blocks']+=gap if prior['reclaimable']<32 else 0
 for d in groups.values():
  d['time_weighted_owned_blocks']=d['owned_block_seconds']/d['inter_step_seconds'] if d['inter_step_seconds'] else None;d['time_weighted_running_sequences']=d['running_sequence_seconds']/d['inter_step_seconds'] if d['inter_step_seconds'] else None
 issued=collections.defaultdict(collections.deque);loads=[]
 for r in rows:
  if r['event']=='load_issued':issued[r['request_id']].append(r['ns'])
  elif r['event']=='load_ready' and issued[r['request_id']]:
   delay=(r['ns']-issued[r['request_id']].popleft())/1e9;assert delay>=0;loads.append(delay)
 out.append(dict(case=case['name'],by_scheduled_token_band=groups,native_load_ready_delays_count=len(loads),native_load_ready_delays_summed_s=sum(loads),load_delay_max_s=max(loads) if loads else None))
(b/(campaign+'-step-diagnosis.json')).write_text(json.dumps(dict(scope='Descriptive scheduler-step intervals by previously reported scheduled-token band, including async overlap. Neither a measured exclusive execution-time decomposition nor direct hint overhead. Loads can overlap.',cases=out),indent=2)+'\n')
for c in out:print(c['case'],c['by_scheduled_token_band'],c['native_load_ready_delays_summed_s'])
