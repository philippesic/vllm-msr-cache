"""Link diagnostic CPU queue movement to committed-key survival and demand reuse.

This is observed lifecycle evidence, not an invented per-key counterfactual.
Use controlled paired replay for comparative work saved; keep tracing out of
primary performance measurements.
"""
from __future__ import annotations
import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from benchmarks.serving_suite.contracts import write_json


def events(path,marker):
    result=[]
    with Path(path).open() as stream:
        for line in stream:
            if marker in line:
                result.append(json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0])
    return result


def analyze(directory):
    directory=Path(directory);log=directory/'server.log'
    traces=events(log,'WAIT_HINT_RETENTION_TRACE ')
    preservation=events(log,'SEMANTIC_PRESERVATION_EVENT ')
    commits=defaultdict(set);evictions=defaultdict(set);loads=defaultdict(dict)
    for record in preservation:
        lifecycle=record.get('key_lifecycle',{})
        for commit in lifecycle.get('cpu_commits',[]):
            commits[commit['key']].add(commit['committed_at_ns'])
        for eviction in lifecycle.get('cpu_evictions',[]):
            evictions[eviction['key']].add(eviction['observed_at_ns'])
        for job in record.get('transfer_job_statuses',[]):
            if job.get('direction')=='load' and job.get('success') is True and job.get('status')=='success' and not job.get('cancelled') and not job.get('is_prefetch'):
                for key in job['keys']:
                    loads[key][job['job_id']]=dict(request_id=record['external_request_id'],
                        job_id=job['job_id'],started_at_ns=job['started_at_ns'],completed_at_ns=job['completed_at_ns'])
    chains=[];touched=set();moved=set()
    for trace in traces:
        for encoded in trace['prefix_keys']:
            value=ast.literal_eval(encoded)
            if not isinstance(value,bytes):
                raise ValueError('Expected canonical bytes OffloadKey')
            key=value.hex();touched.add(key)
            before=trace['before'][encoded];after=trace['after'][encoded]
            changed=any(segment in after and after[segment]>rank for segment,rank in before.items())
            if changed:moved.add(key)
            now=trace['monotonic_ns']
            prior=[t for t in commits[key] if t<=now]
            for load in loads[key].values():
                if not prior or load['started_at_ns']<=now:
                    continue
                intervening=[t for t in evictions[key] if now<t<=load['started_at_ns']]
                recommitted=[t for t in commits[key] if now<t<=load['started_at_ns']]
                if not intervening and not recommitted and changed:
                    chains.append(dict(key=key,hint_event_id=trace['event_id'],
                        latest_commit_ns=max(prior),refresh_ns=now,queue_before=before,queue_after=after,
                        eviction_observed_between_refresh_and_load=False,
                        recommit_observed_between_refresh_and_load=False,**load))
    # One representative chain per key avoids counting repeated hints as many
    # independent saved blocks. Preserve totals separately for trace activation.
    unique={}
    for chain in chains:unique.setdefault(chain['key'],chain)
    return dict(schema='wait-retention-lifecycle-links-v2',trace_events=len(traces),
        distinct_traced_keys=len(touched),distinct_keys_moved_toward_mru=len(moved),
        distinct_moved_survived_and_demand_loaded_keys=len(unique),examples=list(unique.values())[:24],
        limits='Observed queue moves, earlier CPU commitment, no recorded intervening eviction or recommit and later successful demand load. Observation coverage is not an independent continuous-residency proof and does not prove that each key would have been evicted without advice; paired replay supplies comparative evidence. Trace cost is excluded from primary timing.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();write_json(args.output,analyze(args.directory))

if __name__=='__main__':main()
