"""Bounded cleanup of this qualification's leaked early launcher copies."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import psutil
root=Path('output/suite-qualification')
first=min(json.loads(p.read_text())['started_unix'] for p in root.glob('*/??-*/case.json'))
current=Path('vendor/vllm/vllm/v1/core/sched/scheduler.py').read_bytes()
original=subprocess.check_output(['git','show','HEAD:vendor/vllm/vllm/v1/core/sched/scheduler.py'])
allowed={hashlib.sha256(b).hexdigest() for b in [current,original]}
references=[]
for proc in psutil.process_iter(['pid','uids','cmdline']):
 try:
  if proc.uids().real==os.getuid():references.append(' '.join(proc.cmdline())+' '+proc.cwd()+' '+' '.join(proc.environ().values()))
 except (psutil.Error,OSError):pass
before=shutil.disk_usage(root).free;removed=[];preserved=[]
for p in sorted(Path('/tmp').glob('semantic-vllm-*')):
 reason=None
 try:
  st=p.lstat()
  scheduler=p/'vllm/v1/core/sched/scheduler.py'
  if p.is_symlink() or not p.is_dir() or st.st_uid!=os.getuid():reason='not an owned regular staging directory'
  elif st.st_mtime < first-60:reason='predates qualification window'
  elif any(str(p) in r for r in references):reason='live process reference'
  elif not scheduler.is_file() or hashlib.sha256(scheduler.read_bytes()).hexdigest() not in allowed:reason='unrecognized source version'
  elif (p/'vllm/v1/core/block_pool.py').read_bytes()!=Path('vendor/vllm/vllm/v1/core/block_pool.py').read_bytes():reason='unrecognized source baseline'
  if reason:preserved.append(dict(path=str(p),reason=reason));continue
  size=sum(f.lstat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
  # Only source staging and generated import cache; model/results are outside it.
  shutil.rmtree(p)
  removed.append(dict(path=str(p),logical_bytes=size,reason='Owned leaked launcher staging in qualification window; source matches recorded repository baseline/current scheduler, no live reference, cheaply reproducible.'))
 except OSError as e:preserved.append(dict(path=str(p),reason=str(e)))
after=shutil.disk_usage(root).free
retained=[]
for p in sorted(root.iterdir()):
 if p.is_dir():size=sum(f.lstat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
 else:size=p.stat().st_size
 retained.append(dict(path=str(p.resolve()),logical_bytes=size,owner='benchmark suite qualification',category='evidence',reason='Useful successful/failed run, protocol, validation, or cleanup record',review='After compact evidence is committed and raw reproduction is no longer needed'))
record=dict(schema='suite-retention-v1',filesystem=str(root.resolve()),free_before_bytes=before,free_after_bytes=after,observed_free_delta_bytes=after-before,scope='WSL filesystem, not Windows VHDX reclamation; concurrent writes may affect free-space delta',removed=removed,preserved_unknown=preserved,retained=retained,shared='Model caches, other campaigns, and reusable uv/ruff cache preserved')
(root/'retention.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['free_before_bytes','free_after_bytes','observed_free_delta_bytes']},indent=2));print('Removed source copies:',len(removed),'Preserved unknown:',len(preserved))
