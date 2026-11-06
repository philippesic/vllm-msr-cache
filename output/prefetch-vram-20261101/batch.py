"""Execute a finite, pinned JSON plan sequentially on the GPU owner host."""
import argparse,gzip,hashlib,json,os,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);a=parser.parse_args()
plan=json.loads(a.plan.read_text());assert plan['cases']
receipt={'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'started_ns':time.monotonic_ns(),'cases':[]}
for c in plan['cases']:
    env={**os.environ,**plan['environment'],**c.get('environment',{})}
    source=H/'sources'/c.get('source',plan['source']);env['EXPERIMENT_SOURCE']=str(source)
    for name,value in plan.get('harness_pins',{}).items():
        path=H.parents[1]/name[5:] if name.startswith('root:') else H/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==value, name
    expected=json.loads((source/'pins.json').read_text())
    expected=expected.get('files',expected)
    assert all(hashlib.sha256((source/name if '/' in name else source/'semantic_offload'/name).read_bytes()).hexdigest()==value for name,value in expected.items())
    command=[sys.executable,'-B',str(H/'run.py'),'run','--workload',c['workload'],'--seed',str(c['seed']),'--policy',c['policy'],'--phase',c['phase'],'--threshold','0']
    print('START',c,flush=True)
    r=subprocess.run(command,env=env)
    receipt['cases'].append({'case':c,'returncode':r.returncode,'finished_ns':time.monotonic_ns()})
    (H/(a.plan.stem+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    if r.returncode:raise SystemExit(r.returncode)
receipt['finished_ns']=time.monotonic_ns();(H/(a.plan.stem+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
