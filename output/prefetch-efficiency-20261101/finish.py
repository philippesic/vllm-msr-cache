from pathlib import Path
import json,shutil,hashlib,os,subprocess
H=Path(__file__).resolve().parent
rows=json.loads((H/'RESULTS.json').read_text());assert len(rows)==14
assert all(r['completed_strict_proofs']==72 and r['complete_load_eligible'] for r in rows)
cs=json.loads((H/'comparisons.json').read_text());assert len(cs)==10 and all(not c['identity_exact'] for c in cs)
assert (H/'efficiency-frontier.png').stat().st_size>10000
attempts=[]
for d in sorted((H/'runs').iterdir()):
 s=json.loads((d/'summary.json').read_text());passed=(d/'frontier-disposition.json').exists()
 attempts.append(dict(name=d.name,accepted=passed,proofs=s['proofs'],reason='complete strict run' if passed else ('setup before traffic' if not s['proofs']['requests'] else 'client disconnect/incomplete load' if s['proofs']['requests']==33 else 'completed72 but physical proof unavailable')))
(H/'ATTEMPTS.json').write_text(json.dumps(attempts,indent=2)+'\n');assert len(attempts)==17
before=shutil.disk_usage(H).free;removed=[]
for name in ['.plot-packages','.plot-cache','.plot-config']:
 p=H/name
 if p.exists():
  assert p.is_dir() and not p.is_symlink() and p.parent==H
  size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
  shutil.rmtree(p);removed.append(dict(path=str(p),logical_bytes=size,reason='task-local disposable plotting dependencies/cache; exports complete, pinned requirements retained; no live consumer'))
after=shutil.disk_usage(H).free
files=[p for p in H.rglob('*') if p.is_file()];total=sum(p.stat().st_size for p in files)
receipt=dict(filesystem='WSL /dev/sdd; no Windows VHDX reclamation claim',free_before_bytes=before,free_after_bytes=after,observed_free_delta_bytes=after-before,removed=removed,retained=dict(path=str(H),logical_bytes=total,owner='current prefetch efficiency investigation',reason='compressed accepted/failed raw results, frozen sources/runtime, commands/tests and reusable evidence',largest=[dict(path=str(p.relative_to(H)),bytes=p.stat().st_size) for p in sorted(files,key=lambda p:p.stat().st_size,reverse=True)[:8]]),protected=['prior output campaigns','shared model cache','serving venv','pre-existing dirty canonical work'],process_state='all owned server/controller/analysis/plot commands exited; no compute process on NVIDIA GPU',next_review='retain evidence until scientific conclusions/reproduction no longer needed')
(H/'STORAGE.json').write_text(json.dumps(receipt,indent=2)+'\n')
pins={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for f in H.rglob('*.py')}
(H/'SOURCE-PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
(H/'CURRENT.json').write_text(json.dumps(dict(state='screen and selected confirmation complete; no active owned work',gpu_active=False,valid_runs=14,accepted_strict_request_proofs=1008,excluded_attempts=3,goal='waste reduced; repeatable ARC frontier superiority not established',canonical_changes='none by this investigation',storage='STORAGE.json',report='REPORT.md'),indent=2)+'\n')
print(json.dumps(dict(valid_runs=14,proofs=1008,excluded=3,retained_bytes=total,free_delta=after-before)))
