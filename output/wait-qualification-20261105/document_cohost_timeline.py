import sys
import json,statistics
from pathlib import Path
b=Path(__file__).parent;campaign=sys.argv[1] if len(sys.argv)>1 else 'stage3-documents-cohost';root=b/campaign;result=json.loads((root/'result.json').read_text());out=[]
for case in result['cases']:
 rows=[json.loads(l) for l in (root/case['name']/'requests.jsonl').read_text().splitlines()];start=min(r['queued_ns'] for r in rows);end=max(r['ended_ns'] for r in rows);last=sorted(rows,key=lambda r:r['ended_ns'])[-5:]
 gaps=[(y-x)/1e9 for r in rows for x,y in zip(r['token_receipt_ns'],r['token_receipt_ns'][1:])]
 out.append(dict(case=case['name'],whole_request_span_s=(end-start)/1e9,large_stream_gaps_over_1s=sum(g>1 for g in gaps),largest_stream_gap_s=max(gaps),return_ftl=case.get('agent',{}).get('latency',{}).get('tool_to_first_token_s'),tail=[dict(source_id=r['source_id'],turn=r['turn'],output_tokens=len(r['output_token_ids']),end_s=(r['ended_ns']-start)/1e9,ttft_s=r['ttft_s'],latency_s=r['latency_s']) for r in last]))
(b/(campaign+'-timeline-diagnosis.json')).write_text(json.dumps(dict(scope='Descriptive interim/final request timing. Long stream gaps include competing prefill/preemption and do not by themselves identify cause. Recomputed when cases finish; no eligibility change.',cases=out),indent=2)+'\n')
for c in out:print(c['case'],c['whole_request_span_s'],c['large_stream_gaps_over_1s'],c['largest_stream_gap_s'],c['tail'][-1])
