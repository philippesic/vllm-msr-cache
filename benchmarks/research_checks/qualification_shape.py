"""Descriptive context growth and real tool/return timing for frozen workflows."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from benchmarks.serving_suite.contracts import write_json
from benchmarks.serving_suite.report import percentile


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def distribution(values):
    return dict(count=len(values),mean=statistics.mean(values) if values else None,
        p50=percentile(values,.5),p95=percentile(values,.95),minimum=min(values) if values else None,
        maximum=max(values) if values else None)


def report(directory):
    directory=Path(directory);result=json.loads((directory/'result.json').read_text());cases=[]
    for case in result['cases']:
        root=directory/case['name'];requests=rows(root/'requests.jsonl');tools=[]
        for path in root.glob('workflow-*.jsonl'):
            tools.extend(r for r in rows(path) if r['event']=='tool')
        cases.append(dict(name=case['name'],condition=case['variant'],
            expected_initial_sessions=result['plan']['sessions']*result['plan'].get('waves',1),
            offered_initial_requests=sum(r['turn']==0 for r in requests),
            offered_return_requests=sum(r['turn']>0 for r in requests),
            failed_requests=sum(not r['success'] for r in requests),
            prompt_tokens_by_turn={str(turn):distribution([len(r['payload']['prompt']) for r in requests if r['turn']==turn]) for turn in sorted({r['turn'] for r in requests})},
            actual_tool_duration_s=distribution([r['duration_s'] for r in tools]),
            actual_tool_queue_s=distribution([r['queue_s'] for r in tools if 'queue_s' in r]),
            tool_output_excerpt_truncations=sum(bool(r.get('prompt_output_truncated')) for r in tools),
            tool_to_first_token_s=case.get('agent',{}).get('latency',{}).get('tool_to_first_token_s'),
            observed_server_peak_running=case.get('agent',{}).get('opportunity',{}).get('peak_running'),
            observed_server_peak_waiting=case.get('agent',{}).get('opportunity',{}).get('peak_waiting')))
    return dict(schema='wait-qualification-workload-shape-v1',cases=cases,
        scope='Descriptive within-run measurements, no request-level uncertainty. Authored pipelines are designed to return after every actual tool; no real-world return-probability or independently offered-load claim. Return latency includes server queueing; short tool executions must not be described as long external waits.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();write_json(a.output,report(a.directory))

if __name__=='__main__':main()
