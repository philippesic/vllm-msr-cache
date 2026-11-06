"""Retrospective cadence diagnostic; no policy can access future return times."""
import collections,gzip,json,statistics
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
rows=[];details=[]
for p in sorted((H/'runs').glob('live-nov3-triage-confirm-*-candidate/raw.json.gz')):
 r=json.load(gzip.open(p,'rt'));groups=collections.defaultdict(list)
 for h in r['hints']:groups[(h['hint']['session_id'],h['source_completed_at_ns'])].append(h)
 for (sid,t),hs in groups.items():
  hs.sort(key=lambda h:h['started_ns']);acks=[h.get('response',{}) for h in hs];counts=[a.get('ready_cpu_prefix_keys') for a in acks];valid=[c for c in counts if isinstance(c,int)]
  details.append({'run':r['name'],'session_id':sid,'source_completed_at_ns':t,'hint_count':len(hs),'counts':counts,'ack_reasons':[a.get('reason') for a in acks],'first_ready':counts[0] if counts else None,'first_zero_later_positive':bool(counts and counts[0]==0 and any(isinstance(x,int) and x>0 for x in counts[1:])),'positive_then_zero':bool(counts and isinstance(counts[0],int) and counts[0]>0 and any(x==0 for x in counts[1:])),'refreshed_key_touches':sum(valid),'hint_http_wall_ms':sum((h['completed_ns']-h['started_ns'])/1e6 for h in hs)})
 rows.append({'run':r['name'],'hints':len(r['hints']),'waits_with_hints':len(groups)})
summary={'runs':len(rows),'waits_with_hints':len(details),'hints':sum(x['hint_count'] for x in details),'waits_first_positive':sum(isinstance(x['first_ready'],int) and x['first_ready']>0 for x in details),'waits_first_zero':sum(x['first_ready']==0 for x in details),'waits_first_zero_later_positive':sum(x['first_zero_later_positive'] for x in details),'waits_positive_then_zero':sum(x['positive_then_zero'] for x in details),'repeated_hint_count':sum(x['hint_count']-1 for x in details),'refreshed_key_touches':sum(x['refreshed_key_touches'] for x in details),'first_hint_key_touches':sum(x['first_ready'] for x in details if isinstance(x['first_ready'],int)),'hint_http_summed_wall_s':sum(x['hint_http_wall_ms'] for x in details)/1000}
out={'schema':'retrospective-periodic-CPU-recency-hint-diagnostic-v1','scope':'All twelve candidate runs from closed fresh-input confirmation; no selected subset. Touches may repeat keys and are not unique keys or bytes saved. HTTP durations overlap concurrent requests and are not attributable engine overhead. Cannot infer causal benefit/harm of refreshes from these counters alone.','summary':summary,'runs':rows,'waits':details}
(H/'live-nov3-hint-refresh-root-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
