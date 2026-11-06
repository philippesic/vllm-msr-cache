"""Clone the bounded experiment inputs before running; never overwrite evidence."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('--suite',choices=['confirmation','normal','ordinary-confirmation','final-grid'],default='confirmation');p.add_argument('--run',action='store_true');a=p.parse_args()
dest=a.destination.resolve()
assert dest.parent==H.parent,'Destination must be a fresh sibling under the repository output directory'
assert not dest.exists(),'Existing evidence is never overwritten'
protocol=json.loads((H/'confirmation-protocol.json').read_text())
for arm,pins in protocol['source_pins'].items():
 for name,pin in pins.items():
  assert hashlib.sha256((H/'sources'/arm/'semantic_offload'/name).read_bytes()).hexdigest()==pin,(arm,name)
for name,pin in protocol['harness_pins'].items():
 assert hashlib.sha256((H/name).read_bytes()).hexdigest()==pin,name
assert hashlib.sha256((H/'confirm.py').read_bytes()).hexdigest()==protocol['driver_sha256']
for directory in ['sources','prepared','hooks','hooks_future','runtime']:
 shutil.copytree(H/directory,dest/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['run.py','run_future.py','workload.py','confirm.py','normal.py','normal_multi.py','normal_confirm.py','normal_paired_report.py','final_grid.py','analyze.py','paired_report.py','confirmation-protocol.json','normal-protocol.json','ordinary-confirmation-protocol.json','frozen.json','PROTOCOL.md']:
 shutil.copy2(H/name,dest/name)
commands={'confirmation':[[sys.executable,'-B',str(dest/'confirm.py')],[sys.executable,'-B',str(dest/'paired_report.py')]],'normal':[[sys.executable,'-B',str(dest/'normal.py')]],'ordinary-confirmation':[[sys.executable,'-B',str(dest/'normal_confirm.py')],[sys.executable,'-B',str(dest/'normal_paired_report.py')]],'final-grid':[[sys.executable,'-B',str(dest/'final_grid.py')]]}[a.suite]
(dest/'reproduction.json').write_text(json.dumps({'origin':str(H),'suite':a.suite,'commands':commands,'run_requested':a.run,'note':'Use the project venv and same offline model/runtime; one suite at a time, no unrelated host work. Model ref and runtime identity are in frozen.json and per-run raw evidence.'},indent=2)+'\n')
print(json.dumps({'destination':str(dest),'commands':commands,'running':a.run},indent=2))
if a.run:
 for command in commands:subprocess.run(command,check=True,cwd=H.parents[1])
