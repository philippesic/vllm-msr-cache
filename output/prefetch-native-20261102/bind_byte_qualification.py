"""Bind an accepted byte qualification to actual source/configuration identities."""
import argparse,fcntl,gzip,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('name');ap.add_argument('--source',required=True);a=ap.parse_args()
with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 d=H/'runs'/a.name;q=json.loads((d/'qualification-summary.json').read_text());assert q['passed'] and q['all_serving_jobs_covered'] and q['invalid_byte_rejected_for_every_component']
 raw=json.load(gzip.open(d/'raw.json.gz'));assert raw['status']=='complete' and raw['performance_eligible'] is False
 expected=json.loads((H/'sources'/a.source/'pins.json').read_text());assert raw['source_pins']==expected
 for rel,digest in expected.items():assert hashlib.sha256((H/'sources'/a.source/rel).read_bytes()).hexdigest()==digest
 result={'passed':True,'byte_qualification':'qualification-summary.json','source_label':a.source,'source_pins':raw['source_pins'],'config':raw['config'],'geometry':raw['geometry'],'namespace':raw['namespace'],'model_identity':raw['model_identity'],'runtime_pins':raw['runtime_pins'],'harness_pins':raw['harness_pins'],'intrusive_not_performance':True,'qualifier_sha256':hashlib.sha256((H/'qualify_transfer_bytes.py').read_bytes()).hexdigest(),'byte_checker_sha256':hashlib.sha256((H/'byte_audit_dynamic.py').read_bytes()).hexdigest()}
 (d/'qualification-binding.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'source_label':a.source,'geometry':result['geometry'],'serving_transfer_jobs':q['serving_transfer_jobs'],'serving_bytes_checked':q['serving_bytes_checked']},indent=2))
