"""Read-only bounded campaign progress; never drives retries or decisions."""
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);spec=json.loads((root/'spec.json').read_text())
result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {}
cases=result.get('cases',[])
print(root.name,result.get('status','starting'),f"{len(cases)}/{len(spec['order'])} cases complete")
if cases:
 c=cases[-1];print('Latest complete:',c['name'],round(c['load_duration_s'],3),'seconds',c['evaluation']['status'],c.get('error') or '')
if len(cases)<len(spec['order']):
 repeat,name=spec['order'][len(cases)];path=root/f'{repeat:02d}-{name}'/'requests.jsonl'
 rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
 print('Current:',f'{repeat:02d}-{name}',len(rows),'responses observed')
for name,c in result.get('qualification_comparisons',{}).items():
 ratio=c.get('geometric_workflow_throughput_ratio')
 if ratio:print(name,'pairs',c['eligible_pairs'],'time reduction',round(100*(1-1/ratio),3),'percent (interim)')
