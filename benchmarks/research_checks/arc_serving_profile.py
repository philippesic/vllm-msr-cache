"""Opt-in serving eviction accounting and exact-state offline replay.

Body clocks exclude snapshot/log overhead. Store intervals include eviction and
must never be added to it. Sampling is deliberately bounded and not unbiased.
"""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import statistics
import time


def snapshot(policy, n, protected):
    return dict(capacity=policy.cache_capacity, target=policy.target_t1_size,
                n=n, protected=[k.hex() for k in protected],
                **{name: [[k.hex(), b.block_id, b.ref_cnt] for k, b in getattr(policy, name).items()]
                   for name in ('t1', 't2')},
                **{name: [k.hex() for k in getattr(policy, name)] for name in ('b1', 'b2')})


class EvictionProfile:
    def __init__(self, policy, directory):
        from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
        if not isinstance(policy, ARCCachePolicy):
            raise ValueError('Eviction profiling requires ARC')
        self.policy = policy
        self.original = policy.evict
        self.directory = Path(directory) / str(os.getpid())
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.snapshots = []
        self.buckets = set()
        self.calls = 0
        self.body_wall_ns = self.body_cpu_ns = 0
        self.observer_wall_ns = self.flush_wall_ns = 0
        self.store_calls = self.store_wall_ns = self.store_cpu_ns = 0
        policy.evict = self.evict

    def evict(self, n, protected):
        start = time.perf_counter_ns()
        row = dict(call=self.calls, timestamp_ns=time.monotonic_ns(), n=n,
                   protected=len(protected), t1=len(self.policy.t1), t2=len(self.policy.t2),
                   target=self.policy.target_t1_size)
        bucket = n.bit_length()
        if n and len(self.snapshots) < 12 and (bucket not in self.buckets or self.calls % 1000 == 0):
            self.buckets.add(bucket)
            row['snapshot'] = len(self.snapshots)
            self.snapshots.append(dict(call=self.calls, state=snapshot(self.policy, n, protected)))
        body_start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        try:
            result = self.original(n, protected)
        except BaseException as exc:
            row['error'] = type(exc).__name__
            raise
        else:
            row['victims'] = None if result is None else len(result)
            return result
        finally:
            cpu = time.thread_time_ns() - cpu_start
            wall = time.perf_counter_ns() - body_start
            row.update(body_wall_ns=wall, body_cpu_ns=cpu)
            self.calls += 1
            self.body_wall_ns += wall
            self.body_cpu_ns += cpu
            self.rows.append(row)
            self.observer_wall_ns += time.perf_counter_ns() - start - wall

    def record_store(self, wall_ns, cpu_ns):
        self.store_calls += 1
        self.store_wall_ns += wall_ns
        self.store_cpu_ns += cpu_ns

    def flush(self):
        start = time.perf_counter_ns()
        with (self.directory / 'calls.jsonl').open('a') as f:
            for row in self.rows:
                f.write(json.dumps(row) + '\n')
        self.rows.clear()
        (self.directory / 'snapshots.json').write_text(json.dumps(self.snapshots))
        self.flush_wall_ns += time.perf_counter_ns() - start
        summary = dict(policy=type(self.policy).__name__, calls=self.calls,
                       body_wall_ns=self.body_wall_ns, body_cpu_ns=self.body_cpu_ns,
                       observer_wall_ns=self.observer_wall_ns, flush_wall_ns=self.flush_wall_ns,
                       store_calls=self.store_calls, store_wall_ns=self.store_wall_ns,
                       store_cpu_ns=self.store_cpu_ns, snapshots=len(self.snapshots),
                       note='Store includes eviction. Flush excludes summary serialization. Samples are bounded, not unbiased.')
        temp = self.directory / 'summary.tmp'
        temp.write_text(json.dumps(summary, indent=2) + '\n')
        temp.replace(self.directory / 'summary.json')


class CountedQueue(OrderedDict):
    visits = 0
    def items(self):
        for item in super().items():
            self.visits += 1
            yield item


def restore(state, cls, counted=False):
    from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
    p = cls(state['capacity'])
    p.target_t1_size = state['target']
    for name in ('t1', 't2'):
        queue = CountedQueue() if counted else OrderedDict()
        for key, block_id, refs in state[name]:
            b = BlockStatus(block_id)
            b.ref_cnt = refs
            queue[bytes.fromhex(key)] = b
        setattr(p, name, queue)
    for name in ('b1', 'b2'):
        getattr(p, name).update((bytes.fromhex(k), None) for k in state[name])
    return p


def replay(state, repeats=21):
    from semantic_offload.arc_optimization import LinearARCCachePolicy
    from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
    protected = {bytes.fromhex(k) for k in state['protected']}
    outcomes, report = [], {}
    for cls in (ARCCachePolicy, LinearARCCachePolicy):
        counted = restore(state, cls, True)
        result = counted.evict(state['n'], protected)
        visits = counted.t1.visits + counted.t2.visits
        outcomes.append((None if result is None else [(k.hex(), b.block_id, b.ref_cnt) for k, b in result],
                         snapshot(counted, state['n'], protected)))
        wall, cpu = [], []
        for _ in range(repeats):
            p = restore(state, cls)
            start = time.perf_counter_ns(); cstart = time.thread_time_ns()
            p.evict(state['n'], protected)
            cpu.append(time.thread_time_ns() - cstart)
            wall.append(time.perf_counter_ns() - start)
        report[cls.__name__] = dict(median_wall_ns=statistics.median(wall),
                                   median_cpu_ns=statistics.median(cpu), visits=visits)
    # protected order is arbitrary but identical within a process.
    if outcomes[0] != outcomes[1]:
        raise AssertionError('Native/linear replay decision or state mismatch')
    return dict(n=state['n'], protected=len(protected), t1=len(state['t1']), t2=len(state['t2']),
                referenced=sum(b[2] != 0 for name in ('t1', 't2') for b in state[name]),
                equivalent=True, timing=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = []
    for path in sorted(args.directory.rglob('arc-profile/*/summary.json')):
        summary = json.loads(path.read_text())
        calls = [json.loads(line) for line in path.with_name('calls.jsonl').read_text().splitlines()]
        assert len(calls) == summary['calls']
        assert sum(r['body_wall_ns'] for r in calls) == summary['body_wall_ns']
        summary['n_histogram'] = {str(n): sum(r['n'] == n for r in calls) for n in sorted({r['n'] for r in calls})}
        summary['max_call_wall_ns'] = max((r['body_wall_ns'] for r in calls), default=0)
        summary['failed_calls'] = sum(r.get('victims') is None for r in calls)
        summary['replays'] = [dict(call=s['call'], **replay(s['state'])) for s in json.loads(path.with_name('snapshots.json').read_text())]
        report.append(dict(path=str(path), **summary))
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
