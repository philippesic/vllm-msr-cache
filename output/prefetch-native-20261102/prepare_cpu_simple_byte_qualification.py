"""Intrusive actual transfer equality run; never a performance point."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
p=json.loads((H/'CPU-SIMPLE-CONFIRM.json').read_text());r=json.loads(json.dumps(p['runs'][0]));p['runs']=[r];p.pop('pairs');p['progress_file']='cpu-simple-byte-progress.json'
r['name']='cpu-simple-byte-qualified-c24-b2048-candidate-s1103';c=r['command'];c[c.index('--name')+1]=r['name'];c[c.index('--timeout')+1]='900';c[2]=str(H/'run_cpu_wait_simple_audit.py');c.append('--audit')
p['primary']='Not performance eligible. All72 strict requests and exact recorded input/output IDs, every actual serving D2H/H2D job checked on all28 KV components against owned producer-stream snapshots. One-byte-corruption wrong control for every component. Use qualify_transfer_bytes.py after controller. No speculative/adoption byte claim if zero copies.'
p['hypothesis']='Causal CPU ARC recency changes may affect which existing CPU KV is loaded; require actual GPU transfer equality with strict source/destination physical IDs and complete serving job coverage.'
for name in ['prepare_cpu_simple_byte_qualification.py','run_cpu_wait_simple_audit.py','byte_audit_dynamic.py','hooks-dynamic-byte/sitecustomize.py','qualify_transfer_bytes.py']:
 p['pins'][name]=hashlib.sha256((H/name).read_bytes()).hexdigest()
(H/'CPU-SIMPLE-BYTE-QUALIFICATION.json').write_text(json.dumps(p,indent=2)+'\n');print('Prepared intrusive actual-byte qualification; not run.')
