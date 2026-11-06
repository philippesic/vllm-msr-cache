"""Produce compact portable qualification evidence from sealed suite outputs."""
import hashlib,json
from pathlib import Path
from benchmarks.serving_suite.verify import verify
root=Path('output/suite-qualification')
selected=['smoke-7b-03','smoke-final','screen-final','confirm-final','operational-final','hint-screen-03','public-operational-final','public-operational-7b','startup-timeout','case-timeout','sigterm-startup']
records=[]
for name in selected:
 p=root/name;r=json.loads((p/'result.json').read_text());check=verify(p)
 cases=[]
 for c in r['cases']:
  cases.append({k:c.get(k) for k in ['name','arm','repeat','expected_requests','source_ids','startup_s','wall_s','load_duration_s','summary','telemetry','activity','evaluation','operational','cleanup','smoke_output_match']})
 record=dict(name=name,tier=r['plan']['tier'],status=r['status'],wall_s=r['wall_s'],model=Path(r['plan']['model']).parent.parent.name,plan=r['plan'],cases=cases,offline_verification=check,raw_directory=str(p),manifest_sha256=hashlib.sha256((p/'artifacts.json').read_bytes()).hexdigest(),source_digest=json.loads((p/'source.json').read_text())['digest'],hardware=r.get('hardware'),environment=r.get('environment'),not_started=r.get('not_started'),error=r.get('error'))
 records.append(record)
result=dict(schema='semantic-suite-qualification-v1',date='2026-11-03',scope='Maintained harness qualification on this host; not cache-policy superiority, fresh installation, general answer quality, or production certification.',runs=records,failure_controls=json.loads((root/'failure-controls.json').read_text()))
Path('evidence/benchmark-suite-20261103/qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print([(r['name'],r['status'],round(r['wall_s'],2)) for r in records])
