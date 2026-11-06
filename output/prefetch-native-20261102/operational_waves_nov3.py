"""Finite serial persistent-engine waves; no between-wave cache/counter reset."""
import asyncio
import copy
import time
import importlib.util
from pathlib import Path
from runtime_support import control, sha


def wave_identity(case_name, wave_index):
    if not isinstance(case_name, str) or not case_name or '|' in case_name:
        raise ValueError('case_name must be a nonempty request-id-safe string')
    if type(wave_index) is not int or wave_index < 0:
        raise ValueError('wave_index must be a nonnegative integer')
    return case_name + '-wave-' + str(wave_index)


def classify_capacity(hints):
    counts = {'session_registry_capacity': 0, 'other_no_action': 0, 'accepted': 0, 'other': 0}
    for row in hints:
        response = row.get('response')
        if not isinstance(response, dict):
            counts['other'] += 1
        elif response.get('reason') == 'session_registry_capacity':
            counts['session_registry_capacity'] += 1
        elif response.get('status') == 'no_action':
            counts['other_no_action'] += 1
        elif response.get('status') == 'accepted':
            counts['accepted'] += 1
        else:
            counts['other'] += 1
    return counts


def load_wave_module(case_name, index):
    identity = wave_identity(case_name, index)
    path = Path(__file__).with_name('live_operational_workload_nov3.py')
    spec = importlib.util.spec_from_file_location('_operational_' + identity.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN = identity
    module.CELL = 'persistent-wave'
    return module


async def exercise(handle, p, a, result):
    seeds = [int(x) for x in a.wave_seeds.split(',')]
    if not seeds or a.waves < 1 or a.wave_settle_s < 0 or a.wave_release_interval_s < 0:
        raise ValueError('Invalid finite wave plan')
    if a.wave_release_interval_s and (a.waves != 3 or a.cancel_probe):
        raise ValueError('Overlap qualification is exactly three waves without cancellation probe')
    result['waves'] = []
    result['load_start_ns'] = time.monotonic_ns()
    origin_ns = result['load_start_ns']
    from semantic_offload.request_priority import gateway_priority
    result['operational_contract'] = {
        'waves': a.waves, 'seed_cycle': seeds, 'gateway_origin_ns': origin_ns,
        'same_server_all_waves': True, 'between_wave_cache_reset': False,
        'between_wave_counter_reset': False, 'unique_application_identity_per_wave': True,
        'model_admission_namespace_unchanged': True, 'actual_histories_regenerated': True,
        'arrival_mode': 'fixed finite overlapping wave releases' if a.wave_release_interval_s else 'serial completed waves; not sustained open arrivals',
        'wave_release_interval_s': a.wave_release_interval_s,
        'wave_settle_s': a.wave_settle_s,
        'registry_limit': a.registry_limit,
        'accelerated_capacity_case': a.registry_limit != 4096,
        'public_helper_path': str(Path(gateway_priority.__code__.co_filename).resolve()),
        'public_helper_sha256': sha(Path(gateway_priority.__code__.co_filename)),
    }
    async def one_wave(index):
        args = copy.copy(a)
        args.seed = seeds[index % len(seeds)]
        args.application_namespace = wave_identity(a.name, index)
        args.gateway_origin_ns = origin_ns
        module = load_wave_module(a.name, index)
        wave = {'wave_index': index, 'seed': args.seed, 'application_namespace': args.application_namespace,
                'requests': [], 'hints': [], 'cancellations': [], 'status': 'started'}
        result['waves'].append(wave)
        intended = origin_ns + int(index * a.wave_release_interval_s * 1e9)
        wave['intended_wave_release_ns'] = intended if a.wave_release_interval_s else None
        if a.wave_release_interval_s:
            await asyncio.sleep(max(0, (intended - time.monotonic_ns()) / 1e9))
        wave['actual_wave_start_ns'] = time.monotonic_ns()
        try:
            await module.exercise(handle, p, args, wave)
            wave['status'] = 'complete' if len(wave['requests']) == a.clients * a.turns and all(r.get('success') and r.get('natural_eos') for r in wave['requests']) else 'request_failures'
        finally:
            wave['capacity_ack_counts'] = classify_capacity(wave['hints'])
            for key in ('requests', 'hints', 'cancellations', 'http_connection_events', 'prefix_contiguity', 'tool_intervals', 'abandonments', 'submission_events'):
                result.setdefault(key, []).extend(dict(row, operational_wave=index) for row in wave.get(key, []))
            if not a.wave_release_interval_s:
                # Synchronizing export only after this sole wave has finished.
                wave['immediate_export'] = control(handle, 'export')
            else:
                wave['immediate_export'] = None
                wave['export_unavailable_reason'] = 'No synchronization/export while sibling waves remain active'
        if not a.wave_release_interval_s:
            await asyncio.sleep(a.wave_settle_s)
            wave['settled_export'] = control(handle, 'export')
        result['http_client_configuration'] = wave.get('http_client_configuration')
        if wave['status'] != 'complete':
            raise RuntimeError('Persistent wave did not complete all natural-EOS requests')
    try:
        if a.wave_release_interval_s:
            tasks = [asyncio.create_task(one_wave(i)) for i in range(a.waves)]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(a.wave_settle_s)
            result['overlap_settled_export'] = control(handle, 'export')
        else:
            for index in range(a.waves):
                await one_wave(index)
        if a.cancel_probe:
            from operational_cancel_probe_nov3 import run
            await run(handle, a, result, load_wave_module(a.name, a.waves), origin_ns)
    finally:
        result['load_end_ns'] = time.monotonic_ns()
        result['planned_requests'] = a.waves * a.clients * a.turns
