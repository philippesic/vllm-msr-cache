"""Freeze cadence discovery before any of its model invocations."""
import datetime,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads((H/'bridge-discovery-frozen-plan-nov3.json').read_text());runners={'native':'run_live_service_nov3.py','periodic':'run_live_service_nov3.py','once_positive':'run_live_once_ready_nov3.py','once_complete':'run_live_once_full_nov3.py'}
m={'schema':'once-ready-cadence-discovery-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Exactly 12 invocations on spent7703/8803/9901. Three rotations of four-arm Latin order (incomplete balance across seeds). Discovery only; no inferential CI, replacements, output cap, or production claim. Separately tests cadence using unchanged hardened source, not GPU bridge.','python':base['python'],'runner':str((H/runners['native']).relative_to(R)),'pairs':[],'execution_order':[],'file_pins':{}}
order=list(runners)
for i,seed in enumerate([7703,8803,9901]):
 arms=order[i:]+order[:i];group={'pair':i,'partition':seed,'order':arms,'runs':{}}
 for arm in arms:
  name=f'live-nov3-cadence-s{seed}-{arm}';args=['--name',name,'--source-dir',str((H/'sources/nov3-hardened').relative_to(R)),'--gpu-blocks','1536','--cpu-blocks','2048','--active','8','--batch-tokens','1024','--clients','24','--cohort','agent','--turns','3','--wait','bursty','--support-profile','--service-profile','repository_triage_v2','--corpus-path',str((H/'fresh-agent-nov3/session-corpus.json').relative_to(R)),'--hint-interval-ms','1000','--graphs','--timeout','1800','--startup-timeout','600','--seed',str(seed),'--arm','arc' if arm=='native' else 'candidate']
  args+=['--native-arc'] if arm=='native' else ['--no-audit','--source-label','shared-lookup']
  group['runs'][arm]=name;m['execution_order'].append({'name':name,'pair':i,'partition':seed,'arm':arm,'kind':'spent_input_cadence_discovery','runner':str((H/runners[arm]).relative_to(R)),'args':args})
 m['pairs'].append(group)
paths={R/p for p in base['file_pins']}
for n in [*runners.values(),'live_once_service_workload_nov3.py','live_once_full_service_workload_nov3.py','qualify_once_ready_nov3.py','qualify_once_full_nov3.py','live-nov3-once-ready-cpu-qualification.json','live-nov3-once-full-cpu-qualification.json',Path(__file__).name]:paths.add(H/n)
for p in sorted(paths):
 assert p.is_file(),p
 m['file_pins'][str(p.relative_to(R)) if p.is_relative_to(R) else str(p)]=sha(p)
m['stop_conditions']=base['stop_conditions'];m['producer_rules']={'periodic':'New advice every1000ms while actual wait pending.','once_positive':'Stop after successful no_action/cpu_prefix_recency_refreshed ack with integer actualready>0; otherwise scheduled advice while pending.','once_complete':'Stop after same successful ack with actualready==observed_fullblock_count>0; otherwise scheduled advice while pending.','all':'No future text or private wait duration visible to producer. No added application or transport retry. Every failed HTTP remains ineligible, not replaced.'}
p=H/'cadence-discovery-frozen-plan-nov3.json';assert not p.exists();p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'runs':len(m['execution_order']),'pins':len(m['file_pins'])}))
