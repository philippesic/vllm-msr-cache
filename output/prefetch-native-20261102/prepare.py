from pathlib import Path
import shutil,json,hashlib
H=Path(__file__).resolve().parent;P=H.parent/'prefetch-efficiency-20261101'
for n in ['frontier.py','run_frontier_defer.py','runtime_support.py','campaign_metrics.py','session_workload.py','readmission_observer.py','lookup_deferral_observer.py','byte_audit.py','corpus.py','analyze.py','model-selection.json','REUSE_MANIFEST.json']:
 shutil.copy2(P/n,H/n)
for n in ['sources','runtime','runtime-shared-lookup','hooks-defer','prepared']:
 shutil.copytree(P/n,H/n,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for n in ['runs','qualification']: (H/n).mkdir()
f=H/'run_frontier_defer.py';s=f.read_text().replace("if a.source_label=='shared-lookup':cfg['kv_connector_extra_config']['wait_hint_publish_at_lookup']=True", "if a.source_label=='shared-lookup':cfg['kv_connector_extra_config']['wait_hint_publish_at_lookup']=not a.publish_at_bind")
s=s.replace("ap=argparse.ArgumentParser();", "ap=argparse.ArgumentParser();ap.add_argument('--publish-at-bind',action='store_true');")
f.write_text(s)
runs=[]
for budget,arm,early in [(768,'arc',False),(768,'candidate',True),(2048,'candidate',True),(2048,'arc',False)]:
 name=f'bind-b{budget}-{arm}-s1103';cmd=['/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python','-B',str(f),'--name',name,'--arm',arm,'--source-label','shared-lookup' if arm=='candidate' else 'candidate','--gpu-blocks',str(budget),'--cpu-blocks','4096','--active','8','--clients','24','--cohort','agent','--turns','3','--max-output','96','--wait','bursty','--seed','1103','--startup-timeout','180','--timeout','240','--prefetch-fraction','.15']
 if early:cmd.append('--publish-at-bind')
 runs.append(dict(name=name,arm=arm,gpu_blocks=budget,seed=1103,prefetch_fraction=.15,command=cmd))
pins={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for f in H.rglob('*') if f.is_file() and f.suffix in ['.py','.json']}
(H/'BIND.json').write_text(json.dumps(dict(schema='prefetch-native-screen-v1',progress_file='bind-progress.json',runs=runs,pins=pins,objective='User explicitly requests continued iteration until proven throughput/VRAM or TTFT/VRAM superiority. No assistant experiment quota ends the objective. This screen tests existing immediate-bind native evictability at15% speculative cap; canonical defaults untouched. Strict failures stop for diagnosis; no waivers. Fixed-input replay qualification proceeds next.'),indent=2)+'\n')
(H/'CURRENT.json').write_text(json.dumps(dict(state='active',objective='proven serving throughput/VRAM or first-token-latency/VRAM improvement against ARC',next='bind publication screen, fixed-input replay, iterate on measured results',resource_owner='root on current WSL GPU; all GPU runs serial',prior_evidence=str(P),success=False),indent=2)+'\n')
