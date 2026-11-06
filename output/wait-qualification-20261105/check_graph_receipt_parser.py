"""Bounded synthetic native-table interpretation checks, not GPU evidence."""
import json,sys,tempfile
from pathlib import Path
b=Path(__file__).resolve().parent;sys.path.insert(0,str(b.parents[1]))
from benchmarks.research_checks.qualification_graph_receipts import report
checks=[]
with tempfile.TemporaryDirectory(prefix='graph-receipt-check-',dir=b) as temporary:
 root=Path(temporary);case=root/'00-wait';case.mkdir()
 (root/'result.json').write_text(json.dumps({'cases':[{'name':'00-wait','evaluation':{'status':'passed'}}]}))
 warm={'payload':{'prompt':[1,2,3],'max_tokens':2}}
 (case/'warmup.json').write_text(json.dumps(warm));(case/'requests.jsonl').write_text('')
 header='**CUDAGraph Config Settings:**\n- Mode: FULL_DECODE_ONLY\n**CUDAGraph Stats:**\n| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |\n'
 def table(count):return header+f'| 1 | 1 | 0 | FULL | {count} |\n| 1000 | 1024 | 24 | NONE | 9000 |\n'
 def check(name,text,expected,bound,lower):
  (case/'server.log').write_text(text);r=report(root);c=r['cases'][0]
  assert r['all_timed_graph_paths_established']==expected,(name,r)
  assert c['untimed_request_step_upper_bound']==bound,(name,r)
  assert c['timed_full_mode_dispatch_lower_bound']==lower,(name,r)
  checks.append(dict(check=name,passed=True,expected_timed_path=expected,untimed_bound=bound,timed_full_mode_lower_bound=lower))
 check('configuration header alone supplies no dispatch evidence',header,False,5,0)
 check('counts within warmup bound do not prove timed dispatch',table(5),False,5,0)
 check('disjoint native intervals accumulate; NONE is excluded',table(3)+table(4),True,5,2)
 (case/'recovery.json').write_text(json.dumps(warm))
 (case/'requests.jsonl').write_text(json.dumps({'payload':{'request_id':'suite-cancel-probe','prompt':[1]*10,'max_tokens':256}})+'\n')
 check('large cancellation/recovery bound cannot be mistaken for timed work',table(270),False,276,0)
 check('counts exceeding all untimed bounds establish a conservative timed lower bound',table(300),True,276,24)
receipt=dict(passed=True,checks=checks,scope='Synthetic fixtures in the exact native table format, testing false-positive prevention and conservative accounting only. No actual GPU graph execution or performance claim.')
(b/'graph-receipt-parser-checks.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Passed',len(checks),'native graph report interpretation checks; owned temporary fixture removed.')
