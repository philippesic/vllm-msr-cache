"""Freeze ready-advice T2 promotion discovery; no prior outcome filtering."""
from pathlib import Path
import json,hashlib,datetime,xml.etree.ElementTree as ET
H=Path(__file__).resolve().parent;R=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
t=ET.parse(H/'live-nov3-advice-promotion-tests.xml').getroot()[0];assert int(t.attrib['tests'])==296 and int(t.attrib['failures'])==int(t.attrib['errors'])==0
base=json.loads((H/'cadence-discovery-frozen-plan-nov3.json').read_text());m={'schema':'predicted-reuse-T2-promotion-discovery-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Exactly12 discovery invocations, all outcomes retained, on spent7703/8803/9901. Predicted reuse promotion is separate from measured demand/cache hits. Three rotations of four-arm order are not fully balanced; no inferential CI or production claim. No output cap, policy future oracle, retries or replacements.','python':base['python'],'runner':base['runner'],'pairs':[],'execution_order':[],'file_pins':{},'stop_conditions':base['stop_conditions']}
arms=['native','periodic','promotion_periodic','promotion_once_positive']
for i,seed in enumerate([7703,8803,9901]):
 order=arms[i:]+arms[:i];group={'pair':i,'partition':seed,'order':order,'runs':{}}
 for arm in order:
  prior=next(x for x in base['execution_order'] if x['partition']==seed and x['arm']==('native' if arm=='native' else 'once_positive' if arm=='promotion_once_positive' else 'periodic'))
  args=prior['args'].copy();name=f'live-nov3-promotion-s{seed}-{arm}';args[args.index('--name')+1]=name
  if arm.startswith('promotion_'):args[args.index('--source-dir')+1]=str((H/'sources/nov3-advice-promotion').relative_to(R))
  group['runs'][arm]=name;m['execution_order'].append({'name':name,'pair':i,'partition':seed,'arm':arm,'kind':'spent_input_ready_advice_promotion_discovery','runner':prior['runner'],'args':args})
 m['pairs'].append(group)
paths={R/p for p in base['file_pins']};paths.update((H/'sources/nov3-advice-promotion/semantic_offload').glob('*.py'))
for n in ['prepare_promotion_variant_nov3.py','add_promotion_telemetry_nov3.py','live-nov3-advice-promotion-source.json','test_advice_promotion_nov3.py','validate_advice_promotion_nov3.py','live-nov3-advice-promotion-tests.xml',Path(__file__).name]:paths.add(H/n)
for p in sorted(paths):
 assert p.is_file(),p
 m['file_pins'][str(p.relative_to(R)) if p.is_relative_to(R) else str(p)]=sha(p)
m['opportunity_telemetry']={'mechanism':'Per valid non-replayed ACK: newly promoted ready T1 key touches and already-T2 refreshed touches. Not unique keys, demand hits, bytes saved or proof of causal latency benefit.','zero_handling':'Missing telemetry in control/invalid ACKs is unavailable, not an inferred zero.','policy_label':'experimental_ready_resident_T2_promotion','persistent_state':'No new manager history; existing bounded ACK registry contains scalar fields.'}
m['test_receipt']={'tests':296,'failures':0,'errors':0,'scope':'CPU ownership/causality/bounds/queue competition controls plus existing completed transcript, prefetch, CPU refs and score-residency tests under pinned block-pool overlay. Not a GPU byte qualification.'}
p=H/'promotion-discovery-frozen-plan-nov3.json';assert not p.exists();p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'runs':len(m['execution_order']),'pins':len(m['file_pins'])}))
