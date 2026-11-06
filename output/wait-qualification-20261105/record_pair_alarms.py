"""Record every completed ARC/wait point loss, without changing the run or gates."""
import json, sys, time
from pathlib import Path

b = Path(__file__).resolve().parent
for name in sys.argv[1:]:
    root = b / name
    result = json.loads((root / 'result.json').read_text())
    path = b / (name + '-pair-alarms.json')
    previous = json.loads(path.read_text()) if path.exists() else {'alarms': []}
    known = {a['repeat']: a for a in previous['alarms']}
    cases = {(c['repeat'], c['variant']): c for c in result['cases']}
    alarms = []
    for repeat in sorted({c['repeat'] for c in result['cases']}):
        arc, wait = cases.get((repeat, 'arc')), cases.get((repeat, 'wait'))
        if not arc or not wait or wait['load_duration_s'] <= arc['load_duration_s']:
            continue
        details = []
        for c in (arc, wait):
            a = c.get('agent', {})
            cache, opportunity = a.get('cache', {}), a.get('opportunity', {})
            p = root / c['name'] / 'resources.jsonl'
            resources = [json.loads(x) for x in p.read_text().splitlines() if x] if p.exists() else []
            def delta(key):
                return resources[-1][key] - resources[0][key] if resources and key in resources[0] and key in resources[-1] else None
            details.append(dict(case=c['name'], batch_s=c['load_duration_s'],
                output_tokens=c['summary']['output_tokens'], failed=c['summary']['failed'],
                preemptions=cache.get('preemptions'),
                native_scheduled_prefill_tokens=cache.get('locally_executed_prefill_tokens'),
                physical_demand_load_bytes=cache.get('physical_demand_load_bytes'),
                returning_requests_with_prior_cpu_eviction=opportunity.get('returning_requests_with_prior_cpu_eviction'),
                server_cpu_s=c.get('telemetry', {}).get('process_cpu_elapsed_s'),
                client_tool_cpu_s=c.get('telemetry', {}).get('client_tool_cpu_elapsed_s'),
                swap_in_bytes=delta('host_swap_in_bytes'), swap_out_bytes=delta('host_swap_out_bytes'),
                evaluation=c['evaluation'], error=c.get('error')))
        entry = dict(repeat=repeat, first_observed_unix=known.get(repeat, {}).get('first_observed_unix', time.time()),
            wait_overhead_pct=100*(wait['load_duration_s']/arc['load_duration_s']-1), cases=details)
        alarms.append(entry)
        if repeat not in known:
            print(name, 'NEW ARC LOSS', repeat, round(entry['wait_overhead_pct'], 4), 'percent')
            for detail in details:
                print(json.dumps({k: detail[k] for k in ('case', 'native_scheduled_prefill_tokens', 'output_tokens', 'preemptions', 'returning_requests_with_prior_cpu_eviction', 'swap_in_bytes')}))
    path.write_text(json.dumps(dict(campaign=name, recorded_unix=time.time(),
        measurement_status=result['status'], alarms=alarms,
        scope='Every completed pair with a slower wait point estimate. These covariates do not establish a cause, reject a mechanism, or replace fixed whole-run inference. Preserve all outcomes and diagnose before advancing regimes.'), indent=2)+'\n')
    print(name, 'retained ARC-loss pairs:', len(alarms))
