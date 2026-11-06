"""Join fresh performance arms to the separate exact-source byte certificate."""
import fcntl,gzip,hashlib,json
from pathlib import Path
from teacher_busy_frontier import check_pins
H=Path(__file__).resolve().parent
with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=json.loads((H/'CPU-RECENCY-V3-CONFIRM.json').read_text());check_pins(plan)
 progress=json.loads((H/plan['progress_file']).read_text());assert progress['status']=='complete' and len(progress['completed'])==24
 binding=json.loads((H/'runs/cpu-recency-v3-batched-byte-qualified-c24-b2048-candidate-s1103/qualification-binding.json').read_text());assert binding['passed']
 rows=[]
 for run in plan['runs']:
  raw=json.load(gzip.open(H/'runs'/run['name']/'raw.json.gz'))
  assert raw['status']=='complete' and raw['performance_eligible'] is True and not raw['owned_descendants_remaining']
  assert raw['source_pins']==binding['source_pins']
  for key in ['namespace','model_identity','runtime_pins','harness_pins','corpus_identity']:
   if key in binding:assert raw[key]==binding[key],key
  assert raw['geometry']['gpu_blocks']==run['gpu_blocks']
  assert all(raw['geometry'][k]==v for k,v in binding['geometry'].items() if k!='gpu_blocks' and k!='gpu_kv_bytes')
  if run['arm']=='candidate':
   assert raw['config']==binding['config'] and raw['geometry']==binding['geometry']
  else:
   assert raw['config']['kv_connector_extra_config'].get('wait_hint_cpu_retention_only',False) is False
  rows.append({'name':run['name'],'arm':run['arm'],'source_config_identity_passed':True,'geometry':raw['geometry'],'runtime_resolution':raw['runtime_resolution']})
  del raw
 result={'passed':True,'fresh_runs':24,'byte_binding':'runs/cpu-recency-v3-batched-byte-qualified-c24-b2048-candidate-s1103/qualification-binding.json','candidate_exact_source_config_geometry_match_byte_qualification':True,'both_arms_same_source':True,'rows':rows}
 (H/'cpu-recency-v3-configuration-qualification.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='rows'})
