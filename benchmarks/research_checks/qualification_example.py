"""Own and clean a server while checking the documented minimal client example."""
from __future__ import annotations
import argparse
import copy
import contextlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from harness.gpu_lock import gpu_reserved,GPULock,LEGACY_LOCK
from benchmarks.serving_suite.contracts import write_json
from benchmarks.serving_suite.resources import OwnedServer,available_gpu,free_port
from benchmarks.serving_suite.runner import command,wait_quiet,freeze_source
from benchmarks.research_checks.wait_hint_qualification import identity


@contextlib.contextmanager
def handled_termination():
    def interrupt(signum,frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    old=signal.signal(signal.SIGTERM,interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM,old)


@gpu_reserved
def execute(spec,output):
    output.mkdir(parents=True,exist_ok=False)
    plan=copy.deepcopy(spec['plan']);plan.update(spec['conditions']['wait'])
    plan.update(qualification_conditions=['wait'])
    # The unaudited integration profile keeps ordinary registry bounds.
    extra=plan['candidate_config']['kv_connector_extra_config']
    extra.update(preservation_audit=False,preservation_events=True,benchmark_telemetry=True)
    cmd,cfg=command(plan,'candidate',free_port())
    write_json(output/'configuration.json',cfg);write_json(output/'plan.json',plan)
    initial=identity();write_json(output/'source.json',initial)
    freeze_source(initial,output)
    example=Path('examples/wait_hint_client.py').resolve()
    (output/'wait_hint_client.py').write_bytes(example.read_bytes())
    (output/'supervisor.py').write_bytes(Path(__file__).read_bytes())
    env={k:v for k,v in os.environ.items() if not k.startswith(('SEMANTIC_','EXPERIMENT_'))}
    env.update(VLLM_WSL2_ENABLE_PIN_MEMORY='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VLLM_PLUGINS='semantic_wait_hint',
               SEMANTIC_VLLM_STAGE_PARENT=str((output/'runtime-staging').resolve()))
    env.pop('VLLM_KV_QUALIFICATION_TRACE',None)
    env.pop('PYTHONPATH',None)
    server=OwnedServer(cmd,output,env,int(cmd[cmd.index('--port')+1]))
    result=dict(passed=False,scope='Author-run minimal integration check, unaudited bounded-registry profile; not independent reproduction or performance evidence.')
    with handled_termination(),GPULock(path=LEGACY_LOCK):
        result['hardware']=available_gpu()
        try:
            result['startup_s']=server.start(plan['startup_timeout_s'])
            invocation=[sys.executable,str(example),'--url',server.url,'--model',plan['model'],
                        '--config',str((output/'configuration.json').resolve())]
            run=subprocess.run(invocation,cwd=Path(__file__).resolve().parents[2],
                env=env,capture_output=True,text=True,timeout=240)
            (output/'client.stdout').write_text(run.stdout);(output/'client.stderr').write_text(run.stderr)
            result['client_exit_code']=run.returncode
            if run.returncode:raise RuntimeError('Integration client failed; inspect retained stdout/stderr')
            payload=json.loads(run.stdout)
            result['client']=payload
            if payload['hint_acknowledgements']<1 or payload['ready_cpu_key_refresh_observations']<1:
                raise RuntimeError('Example did not establish active CPU prefix refresh')
            (output/'metrics-drained.prom').write_text(wait_quiet(server))
            if identity()['digest']!=initial['digest'] or example.read_bytes()!=(output/'wait_hint_client.py').read_bytes():
                raise RuntimeError('Executed source changed')
            result['passed']=True
        except BaseException as exc:
            result['error']=f'{type(exc).__name__}: {exc}'
        finally:
            result['cleanup']=server.close()
            result['passed']=result['passed'] and result['cleanup']['verified']
            write_json(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=execute(json.loads(args.spec.read_text()),args.output)
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
