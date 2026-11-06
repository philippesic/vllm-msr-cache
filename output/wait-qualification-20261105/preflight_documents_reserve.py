"""Real read-only tool preflight, before fresh reserve serving evaluation."""
import json,subprocess,time,hashlib
from pathlib import Path
b=Path(__file__).resolve().parent;destination=b/'documents-reserve-tool-preflight.json'
if destination.exists():raise FileExistsError('Preserve prior attempt')
corpus=json.loads((b/'independent-inputs/documents-reserve.json').read_text());rows=[]
for s in corpus['sessions']:
 for path,sha in s['input_files'].items():
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
 for tool in s['tools']:
  started=time.monotonic();r=subprocess.run(tool['argv'],cwd=s['repository'],capture_output=True,text=True,timeout=90)
  rows.append(dict(source_id=s['id'],kind=tool['kind'],argv=tool['argv'],cwd=s['repository'],returncode=r.returncode,seconds=time.monotonic()-started,stdout=r.stdout,stderr=r.stderr,passed=r.returncode in tool['valid_exit_codes']))
  destination.write_text(json.dumps(dict(started_before_reserve_evaluation=True,rows=rows,passed=all(x['passed'] for x in rows)),indent=2)+'\n')
  if not rows[-1]['passed']:raise RuntimeError('Failed actual reserve tool; inspect retained preflight')
print('Passed',len(rows),'real document reserve tool commands')
