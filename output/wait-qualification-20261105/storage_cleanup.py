"""Bounded owned-cache cleanup and measured WSL retention inventory."""
import hashlib,json,os,shutil,stat,subprocess,sys,time
from pathlib import Path
b=Path(__file__).resolve().parent;sys.path.insert(0,str(b.parents[1]))
from harness.gpu_lock import GPULock
assert not json.loads((b/'completion-inventory.json').read_text())['missing_or_running']
assert json.loads((b/'all-retained-verification.json').read_text())['passed']
owned_roots=[];checked_cases=0
for root in b.iterdir():
 p=root/'result.json'
 if not root.is_dir() or not p.is_file():continue
 r=json.loads(p.read_text());assert r.get('status')!='running'
 for c in r.get('cases',[]):
  assert c['cleanup']['verified'] and not c['cleanup']['survivors'];owned_roots.append(root/c['name']);checked_cases+=1
 if 'cleanup' in r:
  assert r['cleanup']['verified'] and not r['cleanup']['survivors'];owned_roots.append(root)
with GPULock():
 q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
 assert not q.stdout.strip(),'GPU has an active compute process; inspect ownership before cleanup'
 gpu=dict(both_locks_acquired=True,compute_processes=q.stdout.strip(),query_stderr=q.stderr)
iso=b/'isolated-reproduction';copy=iso/'source';manifest=json.loads((iso/'source-copy.json').read_text())
assert all(hashlib.sha256((copy/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
def size(path):
 if path.is_file() and not path.is_symlink():return dict(regular_file_bytes=path.stat().st_size,regular_files=1,shared_hardlink_files=int(path.stat().st_nlink>1))
 total=files=hardlinks=0
 for p in path.rglob('*'):
  s=p.lstat()
  if stat.S_ISREG(s.st_mode):total+=s.st_size;files+=1;hardlinks+=s.st_nlink>1
 return dict(regular_file_bytes=total,regular_files=files,shared_hardlink_files=hardlinks)
free_before=shutil.disk_usage(b).free;removed=[];preserved_nonempty=[]
for root in owned_roots:
 p=root/'runtime-staging'
 if not p.exists():continue
 assert not p.is_symlink() and p.resolve().is_relative_to(b)
 if any(p.iterdir()):preserved_nonempty.append(str(p));continue
 p.rmdir();removed.append(dict(path=str(p),kind='empty owned runtime staging',regular_file_bytes=0))
for p in sorted(copy.rglob('__pycache__'),key=lambda x:len(x.parts),reverse=True):
 if p.is_symlink() or not p.is_dir():continue
 assert p.resolve().is_relative_to(copy.resolve())
 contents=list(p.rglob('*'))
 if any(x.is_symlink() or x.is_file() and x.suffix!='.pyc' for x in contents):preserved_nonempty.append(str(p));continue
 measure=size(p);shutil.rmtree(p);removed.append(dict(path=str(p),kind='owned isolated-source bytecode cache',**measure))
p=copy/'.pytest_cache'
if p.is_dir() and not p.is_symlink():
 measure=size(p);shutil.rmtree(p);removed.append(dict(path=str(p),kind='owned isolated pytest metadata',**measure))
free_after=shutil.disk_usage(b).free
assert all(hashlib.sha256((copy/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
model=Path(json.loads((b/'opt125-download-receipt.json').read_text())['path']).parents[1]
targets=[(b,'evidence and protected work','All retained attempts, raw proofs and recovery; contains overlapping categories below'),(b/'preexisting-work.tar.gz','protected recovery','Initial96dirty-file workspace recovery; never bundled'),(iso/'.venv','reusable environment','Exact observed isolated dependency closure and repeatable GPU client'),(copy,'tested source','2874-file manifest unchanged; code/license/docs/test reproduction'),(b/'report-env','reusable environment','Repeat scientific PNG/PDF export without changing primary environment'),(b/'more-itertools-source','public source cache','Pinned public repository/tool inputs; full checkout remains reusable'),(b/'peps-source','public source cache','Pinned public documents/tool inputs; full checkout remains reusable'),(model,'protected shared model cache','Same pinned OPT revision completed for this task; preserve reusable weights and existing configuration')]
for name in json.loads((b/'opt125-download-receipt.json').read_text())['files']:
 target=(Path(json.loads((b/'opt125-download-receipt.json').read_text())['path'])/name).resolve()
 if not target.is_relative_to(model.resolve()):targets.append((target,'protected shared model weight blob','Pinned snapshot resolves to this global blob outside its model directory'))
inventory=[dict(path=str(p.resolve()),category=c,owner='this task; shared model cache retained in place' if p==model else 'this task',retention_reason=reason,next_review='After evidence is durably retained and owner no longer needs reproduction/recovery',**size(p)) for p,c,reason in targets if p.exists()]
r=dict(passed=True,recorded_unix=time.time(),filesystem='Local WSL filesystem',free_bytes_before=free_before,free_bytes_after=free_after,observed_free_bytes_delta=free_after-free_before,removed=removed,preserved_nonempty=preserved_nonempty,retained=inventory,gpu=gpu,owned_case_cleanup_receipts_checked=checked_cases,isolated_source_files_verified=len(manifest['files']),scope='Only empty owned staging and disposable caches inside the newly created isolated source copy removed. Inventoried byte counts are regular-file logical sizes, overlap and can share hardlinks with caches outside these paths; they are not reclaim estimates. Actual filesystem free-byte delta can include concurrent activity. Model weights, raw evidence, initial recovery, useful environments and shared caches retained. Windows backing-volume space and VHDX reclamation are not claimed. Archive size/final retained inventory is recorded adjacent after build.')
(b/'storage-cleanup.json').write_text(json.dumps(r,indent=2)+'\n');print('Cleanup complete:',len(removed),'owned cache/staging directories; filesystem free delta',free_after-free_before,'bytes; both GPU locks available and no compute clients')
