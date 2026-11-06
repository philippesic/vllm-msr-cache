from pathlib import Path
import json,copy,hashlib
H=Path(__file__).resolve().parent
s=(H/'fixed_workload.py').read_text().replace('past=list(prompt);return past[:len(past)//16*16]', 'past=list(prompt)+list(output[:-1]);return past[:len(past)//16*16]')
(H/'fixed_transcript_workload.py').write_text(s)
s=(H/'run_replay.py').read_text().replace('from fixed_workload import','from fixed_transcript_workload import').replace("H/'fixed_workload.py'","H/'fixed_transcript_workload.py'")
(H/'run_fixed_transcript.py').write_text(s)
p=json.loads((H/'FIXED-LOWQUEUE.json').read_text());p['runs']=p['runs'][:2]
for r in p['runs']:
 r['name']=r['name'].replace('fixed-lowqueue','fixed-transcript');r['command'][2]=str(H/'run_fixed_transcript.py');r['command'][r['command'].index('--name')+1]=r['name']
p['progress_file']='fixed-transcript-progress.json';p['objective']='Preserve actual computed-transcript nomination in fixed prompt/96decode replay. Policy sees own actual observed output only, never reference future prompt. Return binding requires actual exact contiguous match; output differences are inventoried, not waived. Input-only screen had zero transfers and does not qualify prefetch.'
for n in ['run_fixed_transcript.py','fixed_transcript_workload.py']:p['pins'][n]=hashlib.sha256((H/n).read_bytes()).hexdigest()
(H/'FIXED-TRANSCRIPT.json').write_text(json.dumps(p,indent=2)+'\n')
