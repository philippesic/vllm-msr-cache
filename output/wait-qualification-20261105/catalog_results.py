"""Read-only compact inventory; keep every retained attempt, including failures."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.research_checks.qualification_summary import summarize
b=Path(__file__).resolve().parent
entries=[]
for root in sorted(b.iterdir()):
 p=root/'result.json'
 if not root.is_dir() or not p.is_file():continue
 r=json.loads(p.read_text());entry=dict(name=root.name,status=r.get('status') or ('passed' if r.get('passed') else 'failed' if 'passed' in r else 'unknown'),error=r.get('error'),result=str(p))
 if 'cases' in r:
  s=summarize(root);entry.update(cases=len(s['cases']),responses=sum(c['completed'] for c in s['cases']),failed_responses=sum(c['failed'] for c in s['cases']),comparisons=s['comparisons'])
  entry['case_errors']=[dict(name=c['name'],error=c['error'],evaluation=c['evaluation']) for c in s['cases'] if c['error'] or c['evaluation']['status']!='passed']
 else:entry.update(passed=r.get('passed'),scope=r.get('scope'))
 entries.append(entry)
output=b/'all-retained-results-catalog.json';output.write_text(json.dumps(dict(schema='wait-qualification-retained-catalog-v1',scope='All immediate child result.json files; active results are interim and never final decisions.',attempts=entries),indent=2)+'\n')
for e in entries:
 c=e.get('comparisons',{}).get('wait_vs_arc',{});print(e['name'],e.get('status',e.get('passed')),e.get('cases',''),c.get('candidate_batch_time_reduction_pct',''))
