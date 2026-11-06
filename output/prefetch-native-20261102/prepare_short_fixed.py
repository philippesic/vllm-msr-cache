"""Build an exact-work replay path; freezing prompts requires idle GPU later."""
import ast,gzip,hashlib,json,shutil,sys
from pathlib import Path
H=Path(__file__).resolve().parent
def method(text,name):
 tree=ast.parse(text);c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SemanticOffloadingConnectorScheduler')
 n=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return '\n'.join(text.splitlines()[n.lineno-1:n.end_lineno])+'\n\n'
def build():
 source=H/'sources/short-fixed';shutil.copytree(H/'sources/opportunity-input',source)
 f=source/'semantic_offload/connector.py';s=f.read_text();helper=method((H/'sources/idle-pending/semantic_offload/connector.py').read_text(),'ready_staging_idle_timeout');needle='    def has_pending_push_work(self) -> bool:';assert s.count(needle)==1;s=s.replace(needle,helper+needle);ast.parse(s);f.write_text(s)
 pins={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')};(source/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
 s=(H/'fixed_input_workload.py').read_text().replace('prepared/session-corpus.json','prepared/short-tool-corpus.json').replace('fixed-agent-{a.seed}','fixed-short-agent-{a.seed}').replace('index=slot*4+turn','index=slot*16+turn')
 (H/'short_fixed_workload.py').write_text(s)
 s=(H/'run_short_cadence.py').read_text().replace('from short_cadence_workload import','from short_fixed_workload import').replace("'idle-pending' if a.arm=='candidate'","'short-fixed' if a.arm=='candidate'")
 s=s.replace(" env['SEMANTIC_FUSED_MEAN_SUMMARY']", " env['SEMANTIC_REGISTER_FULL_INPUT_PREFIX']='1' if a.arm=='candidate' else '0'\n env['SEMANTIC_FUSED_MEAN_SUMMARY']")
 (H/'run_short_fixed.py').write_text(s)
 s=(H/'fixed_frontier_dynamic.py').read_text().replace('fixed-agent-{run[\'seed\']}','fixed-short-agent-{run[\'seed\']}').replace('clients=run.get("clients",4)','clients=None,turns=6')
 (H/'short_fixed_frontier.py').write_text(s)
 print('Prepared exact-work sources; no traces frozen or runs started.')
def freeze(seed,run):
 # Call only with serving lock held and GPU idle, never during measurements.
 path=H/'runs'/run/'raw.json.gz';raw=json.load(gzip.open(path));assert raw['status']=='complete' and len(raw['requests'])==24
 rows=[dict(index=r['index'],session_id=r['session_id'],turn=r['turn'],prompt=r['payload']['prompt'],reference_output_token_ids=r['output_token_ids']) for r in raw['requests']]
 assert len({(r['session_id'],r['turn']) for r in rows})==24
 out=dict(schema='fixed-public-short-agent-evaluation-v1',seed=seed,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),requests=rows,policy_access='Only previously completed input prefix and pending flag; future canonical prompts remain evaluation-only.',work='Exact input IDs, fixed96decode ignore_eos. Separate serving-compute control, not live agent quality.')
 (H/'prepared'/f'fixed-short-agent-{seed}.json').write_text(json.dumps(out,indent=2)+'\n');print('Frozen',seed)
if __name__=='__main__':
 if len(sys.argv)==1:build()
 else:
  import fcntl
  with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze(int(sys.argv[1]),sys.argv[2])
