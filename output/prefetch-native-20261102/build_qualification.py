from pathlib import Path
import json,copy
H=Path(__file__).resolve().parent;p=json.loads((H/'FIXED-LOWQUEUE.json').read_text());r=copy.deepcopy(p['runs'][1]);r['name']='qualify-fixed-b520-candidate-bytes';r['command'][r['command'].index('--name')+1]=r['name'];r['command'].append('--audit');r['command'][r['command'].index('--timeout')+1]='600';p['runs']=[r];p['progress_file']='byte-qualification-progress.json';p['objective']='Intrusive actual D2H/H2D all-component byte comparisons plus rejected one-byte corruptions and fixed-work/strictKVproofs; never performance eligible. Requires actual prefetch/adoption evidence.'
(H/'BYTE-QUALIFICATION.json').write_text(json.dumps(p,indent=2)+'\n')
