from pathlib import Path
import shutil,json,hashlib,copy,difflib,ast
H=Path(__file__).resolve().parent;dst=H/'runtime-prefix-order';shutil.copytree(H/'runtime-shared-lookup',dst)
f=dst/'vllm/v1/core/block_pool.py';before=f.read_text();start=before.index('    def publish_completed_prefix(');end=before.index('    def touch(',start);chunk=before[start:end];assert chunk.count('self.free_blocks(blocks)')==1 and chunk.count('self.free_blocks(suffix_blocks)')==1
chunk=chunk.replace('self.free_blocks(blocks)','self.free_blocks(reversed(blocks))').replace('self.free_blocks(suffix_blocks)','self.free_blocks(reversed(suffix_blocks))');after=before[:start]+chunk+before[end:];f.write_text(after)
(H/'tail-first.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='runtime-shared-lookup/block_pool.py',tofile='runtime-prefix-order/block_pool.py')))
f=H/'run_tailfirst.py';s=(H/'run_frontier_defer.py').read_text();needle="runtime_root=H/(('runtime-'+a.source_label) if a.source_label.startswith('shared-') else 'runtime')";assert needle in s;s=s.replace(needle,"runtime_root=H/'runtime-prefix-order'");f.write_text(s)
p=json.loads((H/'BIND.json').read_text());p['runs']=[copy.deepcopy(r) for r in p['runs']]
for r in p['runs']:
 r['name']='tail-'+r['name'];c=r['command'];c[2]=str(f);c[c.index('--name')+1]=r['name']
p['progress_file']='tail-first-progress.json';p['objective']='Two-line native publication eviction order fix: free tails first as native manager does. Existing completed/fence/identity/publication gates unchanged. Candidate earlybind .15, matched runtime inARC. Preserve previousnegativeearlybindscreen; compare strictserving and actualpartialreuse evidence before success.'
p['pins']={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for f in H.rglob('*') if f.is_file() and f.suffix in ['.py','.json'] and any(n in f.parts for n in ['sources','runtime','runtime-shared-lookup','runtime-prefix-order','hooks-defer','prepared'])}
for n in ['frontier.py','run_tailfirst.py','session_workload.py','runtime_support.py','campaign_metrics.py','readmission_observer.py','lookup_deferral_observer.py','byte_audit.py','model-selection.json']:p['pins'][n]=hashlib.sha256((H/n).read_bytes()).hexdigest()
(H/'TAIL-FIRST.json').write_text(json.dumps(p,indent=2)+'\n')
