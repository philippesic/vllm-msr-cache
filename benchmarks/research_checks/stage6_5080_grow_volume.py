"""Grow the existing project ext4 image by8GiB after every timed phase is terminal.

Append-only backing-file extension, loop-capacity refresh and online ext4 growth.
Never formats/replaces/unmounts the image. A retained before-receipt allows an
interrupted append/capacity/resize sequence to resume at its actual state.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
from benchmarks.serving_suite.campaign_contracts import Ledger,atomic_json,load
from benchmarks.research_checks.stage6_healthcheck import coordinators,gpu_snapshot
from harness.gpu_lock import GPULock

REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'output/stage6-5080-20261109'
VOLUME=REPO/'output/stage6-5080-20261108/linux-volume-01'
OUT=VOLUME/'primary-20261109'
IMAGE=Path('/mnt/c/Users/philip/AppData/Local/vllm-semantic-cache-ext-stage6-5080/campaigns/stage6-5080-outputs-01.ext4')
OLD=20*1024**3
TARGET=28*1024**3


def root(args):
    return subprocess.check_output(['/mnt/c/Windows/System32/wsl.exe','-d','Ubuntu','-u','root','--',*args],text=True)


def mount():
    v=json.loads(subprocess.check_output(['findmnt','--json','--target',str(VOLUME),'-o','TARGET,SOURCE,FSTYPE,OPTIONS'],text=True))['filesystems']
    assert len(v)==1 and v[0]['target']==str(VOLUME) and v[0]['source']=='/dev/loop0' and v[0]['fstype']=='ext4' and VOLUME.is_mount()
    loops=json.loads(root(['/usr/sbin/losetup','--list','--json','--output','NAME,BACK-FILE,SIZELIMIT']))['loopdevices']
    loop=next(x for x in loops if x['name']=='/dev/loop0')
    assert loop['back-file']==str(IMAGE) and loop['sizelimit']==0
    return dict(mount=v[0],loop=loop)


def main():
    assert Path.cwd()==REPO and sys.dont_write_bytecode
    assert sys.executable=='/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
    evidence=REPO/'evidence/stage6-5080-20261109/volume-growth-01.json'
    before_path=BASE/'volume-growth-before-01.private.json'
    assert not evidence.exists(),'Preserve completed growth proof'
    assert IMAGE.is_file() and not IMAGE.is_symlink() and IMAGE.stat().st_uid==os.getuid()
    for phase in('overhead','operational'):
        b=load(REPO/f'evidence/stage6-5080-20261109/{phase}-boundary-01.json')
        assert b['execution_complete'] and b['frozen_phase_complete']
    assert load(REPO/'evidence/stage6-5080-20261109/confirmation-attempts-complete.json')['execution_complete']
    assert not coordinators(OUT) and not any(e['state']in('planned','running')for e in Ledger(OUT).index().values())
    topology=mount();stat=IMAGE.stat();assert OLD<=stat.st_size<=TARGET
    raw_bytes=int(subprocess.check_output(['du','-s','-B1',str(OUT)],text=True).split()[0])
    original_bundle=REPO/'output/stage6-5080-20261108/primary-execution-bundle-01'
    bundle_bytes=int(subprocess.check_output(['du','-s','-B1',str(original_bundle)],text=True).split()[0])
    windows=shutil.disk_usage('/mnt/c');remaining=TARGET-stat.st_size
    assert windows.free>remaining+2*1024**3,'Insufficient backing volume for declared extension and existing host work'
    projected=shutil.disk_usage(VOLUME).free+(TARGET-OLD)
    assert projected>raw_bytes+3*bundle_bytes+1024**3,'Re-size the concrete relocation budget before growing'
    if before_path.exists():
        before=load(before_path);assert before['image']==str(IMAGE) and before['inode']==stat.st_ino and before['target_bytes']==TARGET
    else:
        assert stat.st_size==OLD
        before=dict(recorded_unix=time.time(),image=str(IMAGE),inode=stat.st_ino,old_bytes=OLD,target_bytes=TARGET,
            allocated_bytes=stat.st_blocks*512,topology=topology,raw_campaign_bytes=raw_bytes,
            source_bundle_bytes=bundle_bytes,relocation_budget_bytes=raw_bytes+3*bundle_bytes+1024**3,
            free_bytes={k:shutil.disk_usage(p).free for k,p in [('ext4',VOLUME),('root','/'),('Windows','/mnt/c')]})
        atomic_json(before_path,before);before_path.chmod(0o600)
    with GPULock():
        assert not gpu_snapshot()['processes'] and not coordinators(OUT)
        with IMAGE.open('r+b',buffering=0)as f:
            f.seek(0,os.SEEK_END);assert OLD<=f.tell()<=TARGET
            zero=bytes(64*1024**2)
            while f.tell()<TARGET:
                part=zero[:min(len(zero),TARGET-f.tell())]
                assert f.write(part)==len(part)
            os.fsync(f.fileno())
        assert IMAGE.stat().st_size==TARGET
        mount()
        capacity=root(['/usr/sbin/losetup','--set-capacity','/dev/loop0'])
        assert int(root(['/usr/sbin/blockdev','--getsize64','/dev/loop0']).strip())==TARGET
        resize=root(['/usr/sbin/resize2fs','/dev/loop0'])
        after_topology=mount();after=IMAGE.stat();free={k:shutil.disk_usage(p).free for k,p in [('ext4',VOLUME),('root','/'),('Windows','/mnt/c')]}
        assert after.st_ino==before['inode'] and after.st_size==TARGET
        assert free['ext4']>before['free_bytes']['ext4']+7*1024**3
        assert free['ext4']>raw_bytes+3*bundle_bytes+1024**3
        result=dict(schema='stage6-5080-existing-volume-growth-v1',recorded_unix=time.time(),image=str(IMAGE),
            old_bytes=OLD,new_bytes=TARGET,append_bytes=TARGET-OLD,allocated_bytes=after.st_blocks*512,
            same_image_inode=True,never_formatted_replaced_or_unmounted=True,all_timed_phases_terminal=True,
            model_weights_not_copied_by_portability=True,raw_campaign_bytes=raw_bytes,source_bundle_bytes=bundle_bytes,
            relocation_budget_bytes=before['relocation_budget_bytes'],topology=after_topology,
            before_free_bytes=before['free_bytes'],after_free_bytes=free,resize_output=resize.strip(),
            scope='Declared storage repair for full materialized relocation. Appended zero-backed space to the existing20GiB image, refreshed its exact loop capacity and grew mounted ext4 to28GiB. All November8/November9 data retained in place; subsequent campaign/history seal audits still required. No model/source/study-design change.')
        atomic_json(evidence,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
