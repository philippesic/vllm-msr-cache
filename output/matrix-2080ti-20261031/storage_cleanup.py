"""Bounded cleanup of this task's bytecode; retain reusable caches and evidence."""
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tarfile
import time
from pathlib import Path
import psutil

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
def write(name, data): (H/name).write_text(json.dumps(data, indent=2)+'\n')
def inventory(path, resolve_links=False):
    path = Path(path)
    if not path.exists(): return {'path':str(path),'exists':False}
    seen=set();logical=allocated=count=0
    paths=[path] if path.is_file() else path.rglob('*')
    for p in paths:
        if p.is_symlink() and not resolve_links:continue
        if not p.is_file():continue
        s=p.stat();ident=(s.st_dev,s.st_ino)
        if ident in seen:continue
        seen.add(ident);logical+=s.st_size;allocated+=s.st_blocks*512;count+=1
    return {'path':str(path),'logical_bytes':logical,'allocated_file_bytes':allocated,'unique_files':count,'followed_file_symlinks_for_inventory_only':resolve_links}

identities=set()
for file in [*(H/'runs').glob('*/raw.json.gz'),*(H/'failed-attempts').glob('*/raw.json.gz')]:
    r=json.loads(gzip.decompress(file.read_bytes()))
    for sample in r.get('resource_samples',[]):
        identities.update(p['identity'] for p in sample['processes'])
remaining=[]
for ident in identities:
    pid,created=ident.split(':',1)
    try:
        p=psutil.Process(int(pid))
        if abs(p.create_time()-float(created))<.001 and p.status()!=psutil.STATUS_ZOMBIE:
            remaining.append({'identity':ident,'command':p.cmdline()})
    except psutil.Error:pass
assert not remaining,remaining
with open('/tmp/vllm-semantic-cache-gpu.lock','a') as f:
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fcntl.flock(f,fcntl.LOCK_UN)
with socket.socket() as s: port_open=s.connect_ex(('127.0.0.1',19371))==0
assert not port_open

snapshot_changes=[];snapshot_files=0
with tarfile.open(H/'source-snapshot.tar.gz') as t:
    for item in t:
        if not item.isfile():continue
        snapshot_files+=1;p=ROOT/item.name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).digest()!=hashlib.sha256(t.extractfile(item).read()).digest():snapshot_changes.append(item.name)
assert not snapshot_changes,snapshot_changes
initial=json.loads((H/'environment.json').read_text())
current_status=subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True)
assert current_status==initial['git_status']

before_inventory=inventory(H)
free_before=shutil.disk_usage(H).free
removed=[]
for rel in ['__pycache__/workload.cpython-312.pyc','__pycache__/run.cpython-312.pyc','hooks/__pycache__/sitecustomize.cpython-312.pyc']:
    p=H/rel
    if not p.exists():continue
    assert not p.is_symlink() and p.is_file() and p.stat().st_uid==os.getuid()
    p.resolve().relative_to(H)
    st=p.stat();removed.append({'path':str(p),'logical_bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'reason':'Task-owned bytecode, fully recoverable from retained pinned Python source; no live consumers.'});p.unlink()
for rel in ['__pycache__','hooks/__pycache__']:
    p=H/rel
    if p.exists() and not any(p.iterdir()):p.rmdir()
tmp=H/'failed-attempts/diagnostic-multi-1103-semantic/components/components-33774.tmp'
preserved_write=None
if tmp.exists():
    assert not tmp.is_symlink();json.loads(tmp.read_text())
    dst=tmp.parent/'interrupted-profile-write.json';assert not dst.exists();tmp.rename(dst)
    preserved_write={'from':str(tmp),'to':str(dst),'reason':'Valid newer profile counters unique to the failed post-load export. Preserve as failure evidence rather than delete as scratch.'}
free_after=shutil.disk_usage(H).free
receipts=[]
for p in H.rglob('compression.json'):receipts.extend(json.loads(p.read_text()))
model=Path('/home/philip/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct')
retained=[{**inventory(H),'category':'evidence','owner':'this benchmark task','reason':'Pinned input/source recovery, successful and failed raw records, compressed logs, derived metrics and report; needed to audit negative results.','review_when':'After results are archived or superseded with an agreed evidence retention decision.'},{**inventory(model,True),'category':'reproducible cache','owner':'shared project model cache; downloaded for this task','reason':'Reusable 1.5B model and tokenizer; download took about ten minutes. Inventory follows known file symlinks and counts the shared hub weight blob once.','resolved_weights':str(next(model.glob('snapshots/*/model.safetensors')).resolve()),'review_when':'Model no longer needed by this or other work.'}]
for path in ['/home/philip/.triton','/home/philip/.cache/vllm','/home/philip/.cache/huggingface/xet','/tmp/torchinductor_philip']:
    retained.append({**inventory(path),'category':'reproducible cache','owner':'shared runtime cache; task-specific growth unknown','reason':'Reusable compilation/download metadata; preserve shared cache and avoid cold recompilation.','review_when':'A coordinated cache cleanup is requested.'})
record={'at_unix_s':time.time(),'host':initial['platform'],'filesystem_scope':'WSL ext4 only; Windows backing volume not measured or compacted','inventory_before':before_inventory,'removed':removed,'removed_logical_bytes':sum(x['logical_bytes'] for x in removed),'free_bytes_before':free_before,'free_bytes_after':free_after,'observed_free_delta_bytes':free_after-free_before,'free_delta_note':'Same filesystem measurement immediately around cleanup; concurrent activity can affect observed delta. Logical artifact sizes are not a Windows reclaim estimate.','interrupted_profile_preserved':preserved_write,'retained':retained,'existing_compression':{'uncompressed_bytes':sum(x['uncompressed_bytes'] for x in receipts),'compressed_bytes':sum(x['compressed_bytes'] for x in receipts),'lossless_verification':'Each receipt was hash-verified against its original before removal of the uncompressed file.'},'processes':{'sampled_owned_identities_checked':len(identities),'remaining_owned_processes':remaining,'benchmark_port_open':port_open,'project_gpu_lock_acquirable':True,'gpu_after':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],text=True).strip()},'source_verification':{'snapshot_files_compared':snapshot_files,'snapshot_changes':snapshot_changes,'git_status_unchanged':True}}
write('STORAGE.json',record)
print(json.dumps({k:record[k] for k in ['removed_logical_bytes','free_bytes_before','free_bytes_after','observed_free_delta_bytes','processes','source_verification']},indent=2))
