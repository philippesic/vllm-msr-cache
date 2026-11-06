"""Deliberately interrupt one owned screen after a partial comparison block.

Invoke explicitly during development. This never signals an unrelated process,
changes sealed attempts, or resumes a process from partial measurements.
"""
import argparse
import json
import signal
import time
from pathlib import Path

import psutil

from .campaign_contracts import Ledger,atomic_json,load,local_path,verify_tree


def interrupt(out,only,timeout=600):
    out = Path(out).resolve()
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        try:
            entries = list(Ledger(out).index().values())
        except ValueError:
            time.sleep(.2)
            continue  # A concurrent append can briefly expose a torn last line.
        for row in entries:
            identity = row['details'].get('identity',{})
            if row['state']!='running' or identity.get('phase')!='screen' or identity['model']+':'+identity['cell']!=only:
                continue
            prior = [e for e in entries if e['state']=='verified' and e['details'].get('block_id')==row['details']['block_id']]
            directory = local_path(out,row['details']['path'])
            if not prior or not any(p.stat().st_size for p in directory.glob('measurement/*/resources.jsonl')):
                continue
            workers = []
            for process in psutil.process_iter(['cmdline']):
                args = process.info['cmdline'] or []
                if '_worker' in args and '--attempt' in args and args[args.index('--attempt')+1]==str(directory):
                    workers.append(process)
            if len(workers)!=1:
                raise RuntimeError('Owned worker identity is ambiguous')
            worker = workers[0]
            parent = worker.parent()
            args = parent.cmdline()
            if ('benchmarks.serving_suite.campaign' not in args or '--out' not in args or
                args[args.index('--out')+1]!=str(out) or '--phase' not in args or
                args[args.index('--phase')+1]!='screen'):
                raise RuntimeError('Worker parent is not the selected screen coordinator')
            receipt = dict(scope='Deliberate development interruption after a verified partial comparison block',
                attempt=row['attempt_id'],artifacts=row['details']['path'],coordinator_pid=parent.pid,
                coordinator_birth=parent.create_time(),worker_pid=worker.pid,command=args,
                prior_partial_attempts=[e['attempt_id'] for e in prior],signal='SIGINT',started_unix=time.time())
            atomic_json(out/'analysis/recovery-interruption-intent.json',receipt)
            parent.send_signal(signal.SIGINT)
            try:
                parent.wait(timeout=90)
            except psutil.TimeoutExpired as exc:
                raise RuntimeError('Interrupted coordinator did not finish its owned cleanup') from exc
            event = Ledger(out).index()[row['attempt_id']]
            if event['state']!='interrupted':
                raise RuntimeError('Coordinator did not retain an interrupted attempt')
            verify_tree(directory,event['details']['seal'])
            result = directory/'measurement/result.json'
            receipt.update(coordinator_exited=True,interrupted_attempt_sealed=True,
                owned_cleanup_verified=result.exists() and load(result)['cases'][0]['cleanup']['verified'])
            atomic_json(out/'analysis/recovery-interruption-receipt.json',receipt)
            return receipt
        time.sleep(.2)
    raise TimeoutError('No eligible active partial comparison block was observed')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--only',required=True)
    parser.add_argument('--timeout',type=float,default=600)
    args = parser.parse_args()
    print(json.dumps(interrupt(args.out,args.only,args.timeout),indent=2))
