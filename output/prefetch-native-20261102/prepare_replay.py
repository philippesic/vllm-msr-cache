"""Prepare an evaluation-only trace; future records are not policy inputs."""
from pathlib import Path
import json,gzip,hashlib
H=Path(__file__).resolve().parent;P=H.parent/'prefetch-efficiency-20261101'
for seed,run in [(1103,'cap-b768-arc-f0-s1103-v3'),(2207,'confirm-b2048-arc-s2207'),(3301,'confirm-b2048-arc-s3301')]:
 rawpath=P/'runs'/run/'raw.json.gz'
 r=json.load(gzip.open(rawpath));rows=[]
 for x in r['requests']:
  rows.append(dict(index=x['index'],session_id=x['session_id'],turn=x['turn'],prompt=x['payload']['prompt'],reference_output_token_ids=x['output_token_ids'],reference_max_tokens=x['payload']['max_tokens']))
 assert len(rows)==72 and len({(x['session_id'],x['turn']) for x in rows})==72
 out=dict(schema='fixed-public-agent-evaluation-v1',seed=seed,source=str(rawpath),source_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest(),requests=rows,policy_access='producer receives only completed current prompt and pending flag; future fixed inputs remain in load generator',work='identical prompt token IDs; fixed96 decode tokens via ignore_eos for controlled compute, not actual-output conversation quality')
 (H/'prepared'/f'fixed-agent-{seed}.json').write_text(json.dumps(out,indent=2)+'\n')
