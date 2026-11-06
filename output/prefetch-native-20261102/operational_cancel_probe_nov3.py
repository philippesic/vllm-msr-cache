"""One deliberate streaming disconnect, then four natural-EOS health requests."""
import copy
import asyncio
import json
import time
from benchmarks.eviction_prefetch_campaign import consume_campaign_sse
from semantic_offload.request_priority import gateway_priority
from runtime_support import control


def probe_payload(source, name, ordinal, origin_ns):
    body = copy.deepcopy(source)
    rid = 'sc|' + name + '|postwave|fixed|sample-' + str(ordinal)
    body['request_id'] = rid
    body['priority'] = gateway_priority(time.monotonic_ns(), origin_ns, 0, 8000)
    params = body['kv_transfer_params']
    for key in ('external_request_id', 'semantic_request_id', 'semantic_sample_id'):
        params[key] = rid
    params.update(run_id=name, cell_id='postwave', session_id=name+'-probe-'+str(ordinal),
                  sample_index=ordinal, benchmark_client_sequence=ordinal)
    for key in ('wait_hint_event_id', 'observed_prefix_id', 'observed_prefix_boundary_tokens'):
        params.pop(key, None)
    return body


async def run(handle, a, result, workload, origin_ns):
    sources = [r['payload'] for r in result['waves'][0]['requests'] if r['turn'] == 0]
    if len(sources) < 4:
        raise RuntimeError('Cancellation health probes require four known first prompts')
    outcome = {'kind': 'deliberate_client_disconnect', 'health_requests': [], 'status': 'started'}
    result['cancellation_probe'] = outcome
    async with workload.create_http_session(outcome) as client:
        body = probe_payload(sources[0], a.name, 0, origin_ns)
        abort = {'payload': body, 'request_start_ns': time.monotonic_ns(), 'terminal_before_close': False}
        outcome['abort_request'] = abort
        async with client.post(handle.base_url()+'/v1/completions', json=body,
                               trace_request_ctx={'kind': 'intentional_disconnect', 'request_id': body['request_id']}) as response:
            abort['http_status'] = response.status
            if response.status != 200:
                raise RuntimeError('Abort probe HTTP error: '+str(response.status))
            async for raw in response.content:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    abort['terminal_before_close'] = True
                    break
                chunk = json.loads(data)
                abort['response_id'] = chunk.get('id')
                choices = chunk.get('choices', [])
                if any(c.get('finish_reason') is not None for c in choices):
                    abort['terminal_before_close'] = True
                    break
                if any(c.get('token_ids') for c in choices):
                    abort['first_token_chunk'] = chunk
                    abort['disconnect_ns'] = time.monotonic_ns()
                    response.close()
                    break
        abort['request_end_ns'] = time.monotonic_ns()
        # No retry/replacement if the abort was not exercised. Health evidence is
        # still collected and retained before the precise qualification fails.
        for ordinal, source in enumerate(sources[:4], 1):
            payload = probe_payload(source, a.name, ordinal, origin_ns)
            row = {'payload': payload, 'request_start_ns': time.monotonic_ns()}
            outcome['health_requests'].append(row)
            async with client.post(handle.base_url()+'/v1/completions', json=payload,
                                   trace_request_ctx={'kind': 'health_generation', 'request_id': payload['request_id']}) as response:
                row['http_status'] = response.status
                if response.status != 200:
                    raise RuntimeError('Health HTTP error: '+str(response.status))
                row.update(await consume_campaign_sse(response.content))
            row['request_end_ns'] = time.monotonic_ns()
            row['natural_eos'] = row['terminal_complete'] and row['finish_reasons'] == ['stop']
        outcome['after_health_export'] = control(handle, 'export')
        await asyncio.sleep(a.wave_settle_s)
        outcome['settled_after_health_export'] = control(handle, 'export')
    # Pinned installed completion serving names one-prompt engine requests as
    # response_id + '-0'; exact equality avoids accidental substring matches.
    abort['expected_engine_request_id'] = (abort.get('response_id') or '') + '-0'
    observed = outcome['settled_after_health_export']['scheduler'].get('operational_abort_transitions')
    outcome['abort_observer'] = observed
    matches = [r for r in (observed or {}).get('records', []) if r['request_id'] == abort['expected_engine_request_id'] and r['status'] == 'FINISHED_ABORTED']
    outcome['actual_abort_transition_matches'] = matches
    outcome['qualified'] = bool(abort.get('disconnect_ns') and not abort['terminal_before_close'] and len(matches) == 1 and len(outcome['health_requests']) == 4 and all(r['natural_eos'] for r in outcome['health_requests']))
    outcome['status'] = 'qualified' if outcome['qualified'] else 'not_qualified_no_retry'
    if not outcome['qualified']:
        raise RuntimeError('Actual backend abort / natural-EOS health qualification failed or was not exercised')
