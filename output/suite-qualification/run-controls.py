"""Retained qualification driver: calls maintained CLI without runtime hooks."""
import json, subprocess, sys, time
from pathlib import Path
root=Path('output/suite-qualification')
model='/home/philip/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
def command(name,tier='smoke',extra=()):
 return [sys.executable,'-m','benchmarks.serving_suite.runner',tier,'--model',model,'--output',str(root/name),*extra]
controls=[]
for name,extra in [('startup-timeout',['--startup-timeout-s','.1']),('case-timeout',['--case-timeout-s','.01'])]:
 print('CONTROL',name,flush=True)
 rc=subprocess.call(command(name,extra=extra));r=json.loads((root/name/'result.json').read_text())
 passed=rc==2 and r['status']=='failed' and all(c['cleanup']['verified'] for c in r['cases'])
 controls.append(dict(control=name,expected_failure_observed=passed,exit_code=rc))
 if not passed:raise RuntimeError(name)
name='sigterm-startup';p=subprocess.Popen(command(name));deadline=time.monotonic()+60
try:
 log=root/name/'00-candidate/server.log'
 while (not log.exists() or not log.stat().st_size) and time.monotonic()<deadline:
  if p.poll() is not None:raise RuntimeError('Child exited before interrupt control')
  time.sleep(.1)
 if not log.exists() or not log.stat().st_size:raise TimeoutError('No startup log')
 p.terminate();rc=p.wait(timeout=20)
 r=json.loads((root/name/'result.json').read_text())
 passed=rc==2 and r['status']=='failed' and all(c['cleanup']['verified'] for c in r['cases']) and not list((root/name/'00-candidate/runtime-staging').iterdir())
 controls.append(dict(control=name,expected_failure_observed=passed,exit_code=rc))
 if not passed:raise RuntimeError(name)
finally:
 if p.poll() is None:p.terminate();p.wait(timeout=20)
(root/'failure-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
for name,tier,extra in [
 ('hint-screen','screen',['--sessions','4','--candidate-config',str(root/'cpu-hint-config.json'),'--publish-hints','--require-activity','store,restore,hint_accepted']),
 ('public-operational-final','operational',['--corpus',str(root/'public-corpus.json'),'--sessions','4','--turns','2','--concurrency','2','--waves','2','--wave-interval-s','1','--arrival-interval-s','.1','--gpu-blocks','512','--max-model-len','8192','--case-timeout-s','360','--request-timeout-s','90','--require-activity','store','--execution','graphs'])]:
 print('QUALIFY',name,flush=True)
 rc=subprocess.call(command(name,tier,extra))
 print('EXIT',name,rc,flush=True)
 if rc:raise RuntimeError(name)
