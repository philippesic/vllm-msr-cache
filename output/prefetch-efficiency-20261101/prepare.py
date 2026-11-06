import pathlib,shutil,json,hashlib
H=pathlib.Path(__file__).resolve().parent
P=H.parent/'vram-frontier-20261102'
for n in ['frontier.py','run_frontier_defer.py','runtime_support.py','campaign_metrics.py','session_workload.py','readmission_observer.py','lookup_deferral_observer.py','byte_audit.py','corpus.py','analyze.py','model-selection.json','REUSE_MANIFEST.json']:
 shutil.copy2(P/n,H/n)
for n in ['sources','runtime','runtime-shared-lookup','hooks-defer','prepared']:
 shutil.copytree(P/n,H/n,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for n in ['runs','qualification']: (H/n).mkdir()
f=H/'run_frontier_defer.py';s=f.read_text().replace("EXPERIMENT_PREFETCH_FRACTION='.6'","EXPERIMENT_PREFETCH_FRACTION=str(a.prefetch_fraction)")
s=s.replace("ap=argparse.ArgumentParser();", "ap=argparse.ArgumentParser();ap.add_argument('--prefetch-fraction',type=float,default=.6);")
f.write_text(s)
runs=[]
for arm,fraction in [('arc',0),('candidate',.6),('candidate',.15),('candidate',.3)]:
 name=f'cap-b768-{arm}-f{fraction}-s1103'
 command=['/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python','-B',str(f),'--name',name,'--arm',arm,'--source-label','shared-lookup','--gpu-blocks','768','--cpu-blocks','4096','--active','8','--clients','24','--cohort','agent','--turns','3','--max-output','96','--wait','bursty','--seed','1103','--startup-timeout','180','--timeout','240','--prefetch-fraction',str(fraction)]
 runs.append(dict(name=name,arm=arm,gpu_blocks=768,seed=1103,prefetch_fraction=fraction,command=command))
pins={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for f in H.rglob('*') if f.is_file() and f.suffix in ['.py','.json']}
(H/'PLAN.json').write_text(json.dumps(dict(schema='efficiency-cap-screen-v1',progress_file='progress.json',runs=runs,pins=pins,controls='Frozen previous frontier model, corpus, policy, scorer, workload and strict validator. Only speculative budget fraction changes. Fresh ARC and current prefetch controls. Single-partition screen; repeat promising candidate at another budget and partition before a benefit claim.',selection='Prefer strictly valid active prefetch with reduced waste AND improved throughput/TTFT over .6; compare matched ARC. No zero-prefetch utilization success claim.'),indent=2)+'\n')
