"""Reproduce the existing-timestamp join; no model execution."""
import gzip,json
from pathlib import Path
from offline import analyze
H=Path(__file__).resolve().parent

def main():
 raw=json.load(gzip.open(H/'runs/probe-agent-b1024-s1103/raw.json.gz'));terms={}
 for t in raw['terminals']:
  for field in ['request_id','external_request_id']:
   if t.get(field):terms[t[field]]=t
 rows=[]
 for b in analyze(raw,{})['rows']:
  if b['bind_status'] not in ['pending','in_flight']:continue
  terminal=terms.get(b['request_id']);stamps=[e['started_at_ns'] for e in (terminal or {}).get('lookup_episodes',[]) if isinstance(e.get('started_at_ns'),int) and e['started_at_ns']>=b['bound_at_ns']];first=min(stamps) if stamps else None
  rows.append({**b,'terminal_joined_by_native_or_external_id':terminal is not None,'first_recorded_actual_lookup_ns':first,'endpoint_to_first_recorded_lookup_ms':(first-b['potential_end_ns'])/1e6 if first and b['potential_end_ns'] else None,'endpoint_before_first_recorded_lookup':first is not None and b['potential_end_ns'] is not None and b['potential_end_ns']<=first})
 out={'scope':'direct1024run4pendingbindings; existing event timestamps only','rows':rows,'first_attempt_error_preserved':'Initial diagnostic join keyed external request IDs only and failed on native bound IDs; no result accepted from that failed join. Corrected explicit native+external ID map, no source/runtime changes.','caution':'First recorded actual lookup episode occurs immediately after semantic guard, not every method entry. Direct branch counter independently proves0semanticguardreturns; no queue-cause attribution.'};(H/'LOOKUP_TIMING.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
