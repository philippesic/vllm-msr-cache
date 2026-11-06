from pathlib import Path
import json,copy,hashlib
H=Path(__file__).resolve().parent;p=json.loads((H/'BIND.json').read_text());runs=[]
for src in p['runs']:
 r=copy.deepcopy(src);r['name']='replay-'+r['name'];r['command'][2]=str(H/'run_replay.py');r['command'][r['command'].index('--name')+1]=r['name'];runs.append(r)
p['runs']=runs;p['progress_file']='replay-progress.json';p['objective']='Fixed prompt IDs and96 decoded tokens per request; earlybind candidate vsARC at768/2048. ProductionMarlin path unchanged. Separate exact output inventories; fixedcompute replay is not application success or full numerical determinism.'
p['pins']={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for f in H.rglob('*') if f.is_file() and f.suffix in ['.py','.json'] and not any(n in f.parts for n in ['runs','qualification']) and not f.name.endswith('progress.json') and f.name not in ['BIND.json','REPLAY.json','CURRENT.json']}
(H/'REPLAY.json').write_text(json.dumps(p,indent=2)+'\n')
