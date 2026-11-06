"""Read native graph-mode interval tables; configuration alone is insufficient."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

TABLE_ROW = re.compile(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|')


def report(directory):
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text())
    cases = []
    for case in result['cases']:
        root = directory / case['name']
        text = (root / 'server.log').read_text(errors='replace')
        rows = [dict(unpadded_tokens=int(m[1]), padded_tokens=int(m[2]),
                     paddings=int(m[3]), runtime_mode=m[4].strip(), count=int(m[5]))
                for m in TABLE_ROW.finditer(text)]
        full = sum(r['count'] for r in rows
                   if r['runtime_mode'].rsplit('.', 1)[-1] == 'FULL')
        # Count the entire prompt plus permitted output of untimed requests as
        # a deliberately generous bound, including their prefill work. Graph
        # capture/profiling does not emit SchedulerStats to this native logger.
        untimed_bound = 0
        untimed = []
        for name in ('warmup', 'recovery'):
            path = root / (name + '.json')
            if path.exists():
                record = json.loads(path.read_text())
                payload = record['payload']
                bound = len(payload['prompt']) + payload['max_tokens']
                untimed_bound += bound
                untimed.append(dict(phase=name, conservative_step_bound=bound))
        requests = root / 'requests.jsonl'
        for line in requests.read_text().splitlines():
            record = json.loads(line)
            payload = record.get('payload', {})
            if payload.get('request_id') != 'suite-cancel-probe':
                continue
            bound = len(payload['prompt']) + payload['max_tokens']
            untimed_bound += bound
            untimed.append(dict(phase='after-timing cancellation', conservative_step_bound=bound))
        lower = max(0, full - untimed_bound)
        cases.append(dict(case=case['name'], measurement_status=case['evaluation']['status'],
                          native_interval_rows=rows, full_mode_dispatch_count=full,
                          untimed_request_step_upper_bound=untimed_bound,
                          untimed_requests=untimed,
                          timed_full_mode_dispatch_lower_bound=lower,
                          timed_graph_path_established=lower > 0))
    return dict(schema='wait-hint-native-graph-receipts-v1', cases=cases,
                all_timed_graph_paths_established=bool(cases) and all(c['timed_graph_path_established'] for c in cases),
                scope='Native CUDAGraphLogging disjoint interval counts of actual runtime dispatch mode, less a conservative bound for every untimed warmup/recovery/cancellation token. Not a CUDA kernel profiler, exact replay count or graph-versus-eager performance estimate. Required native --cudagraph-metrics must be enabled before measurement.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = report(args.directory)
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    print('Actual timed graph paths established:', value['all_timed_graph_paths_established'])
    return 0 if value['all_timed_graph_paths_established'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
