"""Causal two-source peer-tool competition; driver-private releases stay private.

Each round warms two disjoint observed prefixes, executes two serial peer tools,
then releases both continuations concurrently. Policy producers receive only a
pending flag, completed-source metadata, that observed prefix and a constant wait
estimate. They receive no peer completion signal, future input, return deadline,
private skew or selected winner. Return requests never await hint acknowledgments.
"""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from benchmarks.eviction_prefetch_campaign import consume_campaign_sse


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rid(index):
    return f'sc|competingtrial|frozen|fixed|sample-{index}'


def parse(path, marker):
    records = []
    for segment in Path(path).read_text(errors='replace').split(marker + ' ')[1:]:
        try:
            records.append(json.JSONDecoder().raw_decode(segment)[0])
        except json.JSONDecodeError:
            pass
    return records


async def _publish_hint(client, base_url, namespace, state, index, sid,
                        observed, sequence, records, declared_wait_ms):
    now = time.monotonic_ns()
    event = digest({'run': 'competingtrial', 'session': sid,
                    'index': index, 'refresh': sequence})
    hint = {
        'schema': 'semantic-known-prefix-wait-hint-v1', 'event_id': event,
        'owner_id': rid(index), 'run_id': 'competingtrial', 'cell_id': 'frozen',
        'session_id': sid, 'generation': 0,
        'observed_prefix_id': digest({'namespace': namespace, 'token_ids': observed}),
        'prefix_boundary_tokens': len(observed),
        'observation_available_ns': now, 'input_publication_ns': now,
        'declared_wait_ms': declared_wait_ms,
        'wait_provenance': 'driver_declared_tool_wait_budget', 'namespace': namespace,
    }
    record = {'hint': hint, 'started_ns': now,
              'source_completed_at_ns': state['completed_ns']}
    try:
        async with client.post(base_url + '/plugins/semantic-offload/wait-hints', json=hint) as response:
            record['http_status'] = response.status
            record['response'] = await response.json()
        reply = record['response']
        if reply.get('status') == 'accepted':
            # A refresh can acknowledge an already-owned canonical event. The
            # demand must bind that identity, not this refresh's new event ID.
            state['binding'] = reply.get('canonical_event_id') or event
    except asyncio.CancelledError:
        record['error'] = 'producer_stopped_after_pending_interval'
        raise
    except Exception as error:
        record['error'] = repr(error)
    finally:
        record['completed_ns'] = time.monotonic_ns()
        records.append(record)


async def _hint_policy(client, base_url, namespace, state, index, sid, observed,
                       records, declared_wait_ms=250, activation=None):
    if activation is not None:
        await activation.wait()  # Public peer-start event, never a return timer.
    sequence = 0
    while state['pending']:
        await _publish_hint(client, base_url, namespace, state, index, sid,
                            observed, sequence, records, declared_wait_ms)
        sequence += 1
        try:
            await asyncio.wait_for(state['changed'].wait(), timeout=.2)
        except asyncio.TimeoutError:
            pass


def _completed_observed_prefix(source, completed_output):
    # The last sampled token need not have computed KV yet. Only full blocks
    # of the already completed transcript are nominated; no next-turn input.
    past = tuple(source) + tuple(completed_output[:-1])
    return past[:len(past)//16*16]


async def exercise(handle, p, arm, result):
    import aiohttp

    start = time.monotonic_ns()
    result['load_start_ns'] = start
    result['tool_intervals'] = []
    private = p.get('driver_private', {})
    hint_start = os.getenv('EXPERIMENT_HINT_START', 'pending')
    if hint_start not in ('pending', 'last_peer'):
        raise ValueError('EXPERIMENT_HINT_START must be pending or last_peer')
    lead_ms = int(os.getenv('EXPERIMENT_LEAD_MS', str(private.get('post_peer_gap_ms', 250))))
    delays = list(private.get('return_delays_ms', [0, 0]))
    if 'EXPERIMENT_RETURN_SKEW_MS' in os.environ:
        skew = int(os.environ['EXPERIMENT_RETURN_SKEW_MS'])
        first = int(os.getenv('EXPERIMENT_RETURN_FIRST', '0'))
        if first not in (0, 1) or skew < 0:
            raise ValueError('private return first must be 0/1 and skew nonnegative')
        delays = [0 if slot == first else skew for slot in range(2)]
    if lead_ms < 0 or len(delays) != 2 or any(not isinstance(x, int) or x < 0 for x in delays):
        raise ValueError('peer gap and two private return delays must be nonnegative milliseconds')
    result['competing_protocol'] = {
        'schema': 'causal-competing-peer-tools-v1', 'rounds': len(p['rounds']),
        'fixed_post_peer_gap_ms': lead_ms, 'driver_private_return_delays_ms': delays,
        'policy_constant_wait_estimate_ms': 250, 'policy_refresh_ms': 200,
        'hint_start_mode': hint_start,
        'hint_prefix_mode': 'completed_transcript' if os.getenv('EXPERIMENT_COMPLETED_TRANSCRIPT','0')=='1' else 'warm_input',
        'return_launch': 'independent timers; no wait for hint acknowledgment',
        'policy_inputs': ['pending flag', 'source completion time', 'source identity',
                          'already observed prefix', 'constant wait estimate',
                          'optional public last-peer-start event'],
        'policy_excludes': ['peer completion', 'future input', 'return deadline',
                            'private return skew', 'selected winner'],
        'arc_sends_hint_rpcs': False,
    }
    policy_tasks = []
    states = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as client:
        async def request(index, tokens, output_count, sid, role, observed=None,
                          binding=None, intended=None, started=None):
            if intended is None:
                intended = time.monotonic_ns()
            params = {
                'campaign_schema': 'ep-campaign-serving-v1',
                'campaign_provenance': {'content_origin': 'simulated_agent_trace',
                                        'arrival_origin': 'closed_loop_competing_peer_tool_dag'},
                'external_request_id': rid(index), 'semantic_request_id': rid(index),
                'semantic_sample_id': rid(index), 'sample_index': index,
                'benchmark_client_sequence': index, 'seed': p['trace']['seed'],
                'run_id': 'competingtrial', 'cell_id': 'frozen', 'session_id': sid,
                'generation': 0, 'key_chain_digest': digest(tokens),
            }
            if observed is not None:
                params.update(observed_prefix_id=digest({'namespace': p['namespace'], 'token_ids': observed}),
                              observed_prefix_boundary_tokens=len(observed))
            if binding:
                params['wait_hint_event_id'] = binding
            body = {
                'model': 'matrix-model', 'request_id': rid(index), 'prompt': tokens,
                'max_tokens': output_count, 'ignore_eos': True, 'temperature': 0.,
                'top_p': 1., 'top_k': -1, 'repetition_penalty': 1.,
                'seed': p['trace']['seed'] + index, 'stream': True,
                'return_tokens_as_token_ids': True, 'return_token_ids': True,
                'kv_transfer_params': params,
            }
            row = {'index': index, 'role': role, 'session_id': sid,
                   'intended_release_ns': intended, 'request_start_ns': time.monotonic_ns(),
                   'payload': body}
            if started is not None:
                started.set()  # This peer request has begun, after prior completion.
            try:
                async with client.post(handle.base_url() + '/v1/completions', json=body) as response:
                    row['http_status'] = response.status
                    if response.status != 200:
                        raise RuntimeError((await response.text())[-1500:])
                    stream = await consume_campaign_sse(response.content)
                    row.update(stream)
                    assert stream['terminal_complete'] and len(stream['output_token_ids']) == output_count
                row['success'] = True
            except Exception as error:
                row.update(success=False, error=repr(error), output_token_ids=[])
            row['request_end_ns'] = time.monotonic_ns()
            result['requests'].append(row)
            if not row['success']:
                raise RuntimeError(row.get('error'))
            return row

        try:
            for round_index, round_data in enumerate(p['rounds']):
                base = round_index * 6
                warm = []
                round_states = []
                round_tasks = []
                activation = asyncio.Event() if hint_start == 'last_peer' else None
                for slot, session in enumerate(round_data['sessions']):
                    sid = f'agent-{round_index}-{slot}'
                    completed = await request(base + slot, session['source'], 32,
                                              sid, 'agent-warm', session['source'])
                    warm.append(completed)
                    observed = list(session['source'])
                    if os.getenv('EXPERIMENT_COMPLETED_TRANSCRIPT','0')=='1':
                        # Only already received output tokens, excluding the final
                        # sampled token whose KV need not have been computed.
                        observed = list(_completed_observed_prefix(observed, completed['output_token_ids']))
                        assert session['source'] == observed[:len(session['source'])]
                    state = {'pending': True, 'completed_ns': completed['request_end_ns'],
                             'changed': asyncio.Event(), 'observed_prefix': tuple(observed)}
                    states.append(state)
                    round_states.append(state)
                    if not arm.startswith('arc'):
                        task = asyncio.create_task(_hint_policy(
                            client, handle.base_url(), p['namespace'], state,
                            base + 4 + slot, sid, state['observed_prefix'], result['hints'],
                            activation=activation))
                        round_tasks.append(task)
                        policy_tasks.append(task)
                await request(base + 2, round_data['pressure_a'], 32,
                              f'background-{round_index}-a', 'background-pressure')
                await request(base + 3, round_data['pressure_b'], 32,
                              f'background-{round_index}-b', 'background-pressure',
                              started=activation)
                peer_complete = time.monotonic_ns()
                common_release = peer_complete + lead_ms * 1_000_000

                async def resume(slot):
                    session = round_data['sessions'][slot]
                    state = round_states[slot]
                    sid = f'agent-{round_index}-{slot}'
                    intended = common_release + delays[slot] * 1_000_000
                    await asyncio.sleep(max(0, (intended - time.monotonic_ns()) / 1e9))
                    state['pending'] = False
                    state['changed'].set()
                    # Read only acknowledgments already received. Neither this
                    # request nor the other continuation waits for an RPC.
                    binding = state.get('binding')
                    tokens = session['source'] + warm[slot]['output_token_ids'] + session['append']
                    completed = await request(base + 4 + slot, tokens, 64, sid,
                                              'agent-return', state['observed_prefix'], binding, intended)
                    result['tool_intervals'].append({
                        'session_id': sid, 'start_ns': state['completed_ns'],
                        'peer_tools_complete_ns': peer_complete,
                        'common_intended_release_ns': common_release,
                        'intended_release_ns': intended,
                        'return_end_ns': completed['request_end_ns'],
                        'fixed_post_tool_gap_ms': lead_ms,
                        'driver_private_return_delay_ms': delays[slot],
                    })
                    return completed

                await asyncio.gather(resume(0), resume(1))
                # Producers no longer have a pending tool. Cancel an outstanding
                # response after demand completion rather than delay the next
                # round awaiting an acknowledgment that cannot bind either demand.
                for task in round_tasks:
                    if not task.done():
                        task.cancel()
                if round_tasks:
                    await asyncio.gather(*round_tasks, return_exceptions=True)
            result['load_end_ns'] = max(row['request_end_ns'] for row in result['requests'])
        finally:
            for state in states:
                state['pending'] = False
                state['changed'].set()
            for task in policy_tasks:
                if not task.done():
                    task.cancel()
            if policy_tasks:
                await asyncio.gather(*policy_tasks, return_exceptions=True)
        until = time.monotonic() + 8
        while len(parse(handle.log_path, 'SEMANTIC_PRESERVATION_EVENT')) < len(result['requests']):
            if time.monotonic() > until:
                break
            await asyncio.sleep(.05)
        async with client.get(handle.metrics_url()) as response:
            result['metrics_final'] = await response.text()
