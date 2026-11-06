import datetime,hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
D=H/'sources/nov3-gpu-bridge/semantic_offload';D.mkdir(parents=True,exist_ok=True)
for p in (R/'semantic_offload').glob('*.py'):
 if (D/p.name).exists():assert sha(p)==sha(D/p.name),'Existing snapshot differs'
 else:shutil.copyfile(p,D/p.name)
 assert sha(p)==sha(D/p.name)
old=json.loads((H/'triage-http-grid-frozen-plan-nov3.json').read_text())
m={'schema':'gpu-prefix-cpu-suffix-bridge-discovery-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Exactly 9 discovery invocations on spent inputs. No confirmation, replacements, output caps or production claim. Latin 3-arm order. All outcomes retained.','runner':str((H/'run_live_bridge_nov3.py').relative_to(R)),'python':old['python'],'pairs':[],'execution_order':[],'file_pins':{}}
orders=[['native','old','bridge'],['old','bridge','native'],['bridge','native','old']]
for i,(seed,order) in enumerate(zip([7703,8803,9901],orders)):
 group={'pair':i,'partition':seed,'order':order,'runs':{}}
 for arm in order:
  name=f'live-nov3-bridge-s{seed}-{arm}';source=H/'sources'/('nov3-gpu-bridge' if arm=='bridge' else 'nov3-hardened')
  args=['--name',name,'--source-dir',str(source.relative_to(R)),'--gpu-blocks','1536','--cpu-blocks','2048','--active','8','--batch-tokens','1024','--clients','24','--cohort','agent','--turns','3','--wait','bursty','--support-profile','--service-profile','repository_triage_v2','--corpus-path',str((H/'fresh-agent-nov3/session-corpus.json').relative_to(R)),'--hint-interval-ms','1000','--graphs','--timeout','1800','--startup-timeout','600','--seed',str(seed),'--arm','arc' if arm=='native' else 'candidate']
  if arm=='native':args+=['--native-arc']
  else:args+=['--no-audit','--source-label','shared-lookup']
  if arm=='bridge':args+=['--gpu-aware-retention']
  group['runs'][arm]=name;m['execution_order'].append({'name':name,'pair':i,'partition':seed,'arm':arm,'kind':'spent_input_bridge_discovery','args':args})
 m['pairs'].append(group)
# Re-pin the prior complete dependency set at current content, while retaining
# both immutable old source and new source. The preceding confirmation is closed.
paths={R/p for p in old['file_pins']}
paths.update((R/'semantic_offload').glob('*.py'));paths.update(D.glob('*.py'))
paths.update([H/'run_live_bridge_nov3.py',H/'fresh-agent-nov3/session-corpus.json',H/'live-nov3-bridge-tests.xml',Path(__file__)])
for p in sorted(paths):
 assert p.is_file(),p
 m['file_pins'][(str(p.relative_to(R)) if p.is_relative_to(R) else str(p))]=sha(p)
m['source_snapshot']={'path':str(D.parent.relative_to(R)),'exact_checkout_copy':True,'files':len(list(D.glob('*.py'))),'old_source':str((H/'sources/nov3-hardened').relative_to(R))}
m['stop_conditions']=['Source or runtime pin drift','Transfer corruption, measurement instrumentation mismatch, orphan cleanup failure','Three consecutive failed invocations; do not replace failed runs']
(H/'bridge-discovery-frozen-plan-nov3.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'runs':len(m['execution_order']),'pins':len(m['file_pins']),'snapshot':m['source_snapshot']}))
