"""Clone finite pinned experiments offline into a new sibling output directory.
Uses this checkout, its installed venv and cached model; never overwrites them.
"""
import argparse,hashlib,importlib.metadata,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path)
    parser.add_argument('--plan',action='append',help='Plan filename in this evidence directory; repeat for a custom subset, including negative screens.')
    args=parser.parse_args();dest=args.destination.resolve()
    if dest.parent!=ROOT/'output' or dest.exists():parser.error('use a new directory directly under this checkout/output')
    plan_names=args.plan or ['confirmation-long.json','confirmation-short-combined.json','quality-prefix-enabled-plan.json','quality-prefix-more-plan.json']
    if any(Path(n).name!=n for n in plan_names):parser.error('plan names must be local filenames')
    plans=[json.loads((H/n).read_text()) for n in plan_names]
    for plan in plans:
        for name,value in plan['harness_pins'].items():
            path=ROOT/name[5:] if name.startswith('root:') else H/name
            if sha(path)!=value:parser.error('pinned dependency changed: '+name)
    sources={c.get('source',p['source']) for p in plans for c in p['cases']}
    for name in sources:
        path=H/'sources'/name;pins=json.loads((path/'pins.json').read_text());pins=pins.get('files',pins)
        for n,value in pins.items():
            f=path/n if '/' in n else path/'semantic_offload'/n
            if sha(f)!=value:parser.error('source changed: '+str(f))
    frozen=json.loads((H/'frozen.json').read_text())
    model_ref=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/refs/main'
    if model_ref.read_text().strip()!=frozen['revision']:parser.error('cached model revision changed')
    expected=json.loads((H/'host.json').read_text())['packages']
    if any(importlib.metadata.version(name)!=version for name,version in expected.items()):parser.error('installed package versions differ from the measured host')
    dest.mkdir()
    names={name for plan in plans for name in plan['harness_pins'] if not name.startswith('root:')}
    names.update(plan_names);names.update(['confirm_results.py','frozen.json','host.json','qualify_prefix_mean.py','qualify_gather_cuda.py','qualify_native_bytes.py','check_quality_transport.py','reproduce.py'])
    for name in sorted(names):
        path=H/name;target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for name in sorted(sources):shutil.copytree(H/'sources'/name,dest/'sources'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Preserve external recovery evidence without altering the live checkout.
    for name in {n[5:] for plan in plans for n in plan['harness_pins'] if n.startswith('root:')}:
        target=dest/'external-checkout-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    for plan in plans:
        for name,value in plan['harness_pins'].items():
            path=ROOT/name[5:] if name.startswith('root:') else dest/name
            assert sha(path)==value,name
    pins={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
    (dest/'reproduction-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    (dest/'reproduction-design.json').write_text(json.dumps({'plans_in_order':plan_names,'sources':sorted(sources),'performance_plans':[n for n in plan_names if n.startswith('confirmation-')],'quality_or_negative_plans':[n for n in plan_names if not n.startswith('confirmation-')],'same_checkout_venv_cached_model_required':True},indent=2)+'\n')
    print(dest)
    for name in plan_names:print('bin/python -B '+str(dest/'batch.py')+' '+str(dest/name))
    print('Run plans serially. Analyze confirmation plans only after GPU cleanup; quality and negative screens never enter performance confirmation.')
if __name__=='__main__':main()
