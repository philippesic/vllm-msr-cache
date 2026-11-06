"""Local hint hot-path profile; excludes network/scheduler queue delay."""
import argparse
import cProfile
import io
import json
import pstats
import statistics
import time
from contextlib import redirect_stdout
from pathlib import Path
from semantic_offload.manager import PreservationOffloadingManager
from tests.test_step_1_5_prefetch import _StubBlockPool, _insert_resident, _make_connector_scheduler, _register_wait_prefix, _wait_hint, to_key
from benchmarks.serving_suite.contracts import digest


def run(output):
    output.mkdir(exist_ok=False)
    rows=[]
    profile=cProfile.Profile()
    for count in (64,256,512):
        m=PreservationOffloadingManager(num_blocks=4096,cache_policy='arc')
        keys=[to_key(i) for i in range(count)]
        for i,k in enumerate(keys):_insert_resident(m,k,i)
        s=_make_connector_scheduler(m,_StubBlockPool(8192,free_blocks=4096))
        s._wait_hint_cpu_retention_only=True
        s._preservation_audit_enabled=False
        prefix,boundary=_register_wait_prefix(s,keys)
        values=[]
        for i in range(501):
            raw=_wait_hint(s,prefix,boundary,event_id=f'{i:064x}')
            start=time.perf_counter_ns()
            result=s.publish_wait_hint(raw)
            values.append(time.perf_counter_ns()-start)
            assert result['ready_cpu_prefix_keys']==count
        rows.append(dict(kind='scheduler_hint',keys=count,median_ns=statistics.median(values),p95_ns=sorted(values)[475]))
        profile.enable()
        for i in range(501,701):s.publish_wait_hint(_wait_hint(s,prefix,boundary,event_id=f'{i:064x}'))
        profile.disable()
    for count in (1024,4096,8192):
        data=dict(namespace={'model':'development'},token_ids=list(range(count)));times=[]
        for _ in range(501):
            start=time.perf_counter_ns();digest(data);times.append(time.perf_counter_ns()-start)
        rows.append(dict(kind='client_prefix_digest',tokens=count,median_ns=statistics.median(times)))
    profile.dump_stats(output/'scheduler.prof')
    out=io.StringIO();pstats.Stats(profile,stream=out).sort_stats('cumulative').print_stats(35)
    (output/'scheduler.txt').write_text(out.getvalue())
    (output/'costs.json').write_text(json.dumps(dict(scope='CPU-only hot path, audit disabled; network and scheduler queue excluded',rows=rows),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
