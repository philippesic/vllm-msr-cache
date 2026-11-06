import subprocess
import sys

import psutil
import pytest

from benchmarks.serving_suite.campaign_execution import run_worker


@pytest.mark.parametrize('mode', ['timeout','crash','orphan'])
def test_worker_cleanup_reaps_detached_descendant(tmp_path, mode):
    receipt = tmp_path/'child.pid'
    child = 'import time; time.sleep(120)'
    script = ("import subprocess,sys,time; from pathlib import Path; "
              f"p=subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True); "
              f"Path({str(receipt)!r}).write_text(str(p.pid)); "
              + ('time.sleep(120)' if mode=='timeout' else "time.sleep(.1); sys.exit(7)" if mode=='crash' else 'time.sleep(.1)'))
    with (tmp_path/'worker.log').open('wb') as log:
        with pytest.raises((subprocess.TimeoutExpired,RuntimeError)):
            run_worker([sys.executable,'-c',script],cwd=tmp_path,log=log,timeout=1)
    assert not psutil.pid_exists(int(receipt.read_text()))


def test_worker_preserves_preexisting_child(tmp_path):
    child = subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
    try:
        with (tmp_path/'worker.log').open('wb') as log:
            run_worker([sys.executable,'-c','pass'],cwd=tmp_path,log=log,timeout=5)
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)
