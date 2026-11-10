"""Exec the frozen GPU-free portability proof under an independent user service."""
from pathlib import Path
import os
import subprocess
import sys
import time
import psutil
from benchmarks.serving_suite.campaign_contracts import Ledger,atomic_json,load
from benchmarks.research_checks.stage6_healthcheck import coordinators,gpu_snapshot

REPO=Path(__file__).resolve().parents[2]
BASE=REPO/'output/stage6-5080-20261109'
VOLUME=REPO/'output/stage6-5080-20261108/linux-volume-01'
SOURCE=REPO/'output/stage6-5080-20261108/primary-execution-bundle-01/source'
OUT=VOLUME/'primary-20261109'
PYTHON='/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'


def main():
    assert Path.cwd()==REPO and sys.executable==PYTHON and sys.dont_write_bytecode and VOLUME.is_mount()
    ready=load(REPO/'evidence/stage6-5080-20261109/post-acquisition-audit-01.json')
    assert ready['execution_complete'] and ready['verification']['verified_files']==67120
    cap=load(BASE/'caps-all-final-01/summary.json');assert cap['complete_inventory'] and cap['status']=='audited'
    assert not any(e['state']in('running','planned')for e in Ledger(OUT).index().values())
    assert not coordinators(OUT)and not gpu_snapshot()['processes']
    source=subprocess.check_output([PYTHON,'-c','from benchmarks.serving_suite.campaign import source_snapshot; print(source_snapshot()["digest"])'],cwd=SOURCE,text=True).strip()
    assert source==ready['frozen_acquisition_source']
    destination=VOLUME/'portability-20261109-final-01'
    owner=BASE/'portability-owner-01.private.json';log=BASE/'portability-final-01.log'
    assert not destination.exists()and not owner.exists()and not log.exists()
    assert not OUT.with_name(OUT.name+'-portability-parked').exists()
    command=[PYTHON,'-m','benchmarks.serving_suite.campaign_portability','--out',str(OUT),'--destination',str(destination),
        '--runtimes','/home/philip/.cache/semantic-offload/stage6-5080-runtimes-02']
    process=psutil.Process()
    atomic_json(owner,dict(pid=process.pid,birth=process.create_time(),command=command,cwd=str(SOURCE),unit='stage6-5080-portability-20261110-01.service',
        recorded_unix=time.time(),source=source,destination=str(destination),scope='GPU-free exact frozen relocation/import proof, independent of tool-host lifetime; preserves the original path through a reversible park/restore and never executes imported models.'))
    owner.chmod(0o600)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONPATH',None)
    with log.open('xb')as stream:os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
    os.chdir(SOURCE);os.execvpe(PYTHON,command,env)


if __name__=='__main__':main()
