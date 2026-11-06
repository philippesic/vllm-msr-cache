"""Causal closed-loop replay with frozen input/output IDs and recorded tool waits.

Artificial output forcing pays for normal model forwards plus a logits processor.
Recorded delays are diagnostics, not executed live tools or application evidence.
Future tokens and delays are never passed to the retention policy.
"""
import asyncio
import copy
import json
import time
from pathlib import Path

import aiohttp

from .contracts import digest
from .repo_workload import publish_hint, known_prefix_tokens
from .workload import completion


async def run_wave(url, plan, sessions, tokenizer, wave, writer, arm, namespace, hints, stream_dir):
    directory = Path(stream_dir).parent
    rows = []
    abandoned_tasks = []
    events = (directory / f'workflow-{wave}.jsonl').open('x')
    def emit(event, **fields):
        events.write(json.dumps(dict(event=event, ns=time.monotonic_ns(), wave=wave,
                                    diagnostic_replay=True, **fields))+'\n')
        events.flush()
    origin = time.monotonic_ns()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=plan['request_timeout_s']),
                                    connector=aiohttp.TCPConnector(force_close=True)) as client:
        async def session(slot, source):
            sid = f"w{wave}:{source['id']}"
            session_publish = plan['publish_hints'] and source.get('publish_hints', True)
            previous = None
            tool_end = None
            emit('session_start', session_id=sid, source_id=source['id'])
            for turn, frozen in enumerate(source['replay']):
                rid = f'sc|suite|case|w{wave}-s{slot}|sample-{turn}'
                first = asyncio.Event()
                receipts = []
                async def announce():
                    receipt = await publish_hint(client, url, previous, sid, rid, namespace, hints,
                        declared_wait_ms=plan.get('hint_declared_wait_ms',750),
                        completed_transcript=plan.get('hint_completed_transcript',True))
                    receipts.append(receipt)
                    return receipt
                if turn:
                    if session_publish:
                        await announce()
                    start = time.monotonic_ns()
                    async def repeat_hints():
                        while True:
                            await asyncio.sleep(plan['hint_refresh_s'])
                            await announce()
                    refresh = asyncio.create_task(repeat_hints()) if session_publish and plan.get('hint_during_wait',True) and not plan.get('hint_initial_only') else None
                    try:
                        await asyncio.sleep(frozen['delay_s'])
                    finally:
                        if refresh:
                            refresh.cancel()
                            await asyncio.gather(refresh, return_exceptions=True)
                    tool_end = time.monotonic_ns()
                    emit('tool', session_id=sid, turn=turn, kind=frozen['tool']['kind'],
                         started_ns=start, ended_ns=tool_end, duration_s=(tool_end-start)/1e9,
                         success=True, exit_code=frozen['tool']['exit_code'],
                         replayed_not_executed=True, queue_s=0,
                         output_bytes=0, prompt_output_truncated=False)
                    emit('hint_window', session_id=sid, turn=turn, tool_started_ns=start,
                         tool_ended_ns=tool_end, receipts=receipts)
                payload = copy.deepcopy(frozen['payload'])
                payload.update(request_id=rid, max_tokens=len(frozen['tokens']),
                               min_tokens=len(frozen['tokens']), ignore_eos=True,
                               vllm_xargs={'diagnostic_replay_tokens':frozen['tokens']})
                params = payload['kv_transfer_params']
                for key in ('observed_prefix_id','observed_prefix_boundary_tokens','wait_hint_event_id'):
                    params.pop(key,None)
                params.update(session_id=sid, external_request_id=rid)
                params['campaign_provenance'] = {**params.get('campaign_provenance', {}),
                    'content_origin':'frozen_trace_diagnostic_replay',
                    'wait_origin':plan.get('replay_wait_origin','recorded_tool_delays')}
                if session_publish:
                    observed = known_prefix_tokens(previous, plan.get('hint_completed_transcript',True)) if previous else payload['prompt']
                    observed = observed[:len(observed)//16*16]
                    params.update(observed_prefix_id=digest(dict(namespace=namespace,token_ids=observed)),
                                  observed_prefix_boundary_tokens=len(observed))
                now = time.monotonic_ns()
                # Frozen payloads may contain the historical gateway priority.
                # A zero-credit/FCFS intervention must not inherit that value.
                payload.pop('priority', None)
                if plan.get('continuation_credit_ms'):
                    from semantic_offload.request_priority import gateway_priority
                    payload['priority'] = gateway_priority(now,origin,turn,plan['continuation_credit_ms'])
                async def queued():
                    while not first.is_set():
                        await asyncio.sleep(plan['hint_refresh_s'])
                        if not first.is_set():
                            queue_receipts.append(await announce())
                queue_receipts = []
                refresh = asyncio.create_task(queued()) if turn and session_publish and plan.get('hint_through_queue',True) and not plan.get('hint_initial_only') else None
                try:
                    row = await completion(client,url,payload,dict(session_id=sid,source_id=source['id'],
                        wave=wave,turn=turn,queued_ns=now,tool_completed_ns=tool_end,
                        diagnostic_replay=True),writer,
                        stream_path=Path(stream_dir)/f'replay-{wave}-{slot}-{turn}.sse',first_token_event=first)
                finally:
                    if refresh:
                        refresh.cancel()
                        await asyncio.gather(refresh, return_exceptions=True)
                    if turn:
                        emit('queue_hint_window',session_id=sid,turn=turn,
                             publications=len(queue_receipts),receipts=queue_receipts,
                             first_token_observed=first.is_set())
                rows.append(row)
                if not row['success'] or row.get('output_token_ids') != frozen['tokens']:
                    raise RuntimeError(f'Fixed-token replay mismatch: {rid}')
                previous = row
            emit('session_end',session_id=sid,source_id=source['id'],success=True)
            if session_publish and source.get('abandoned_hint', False):
                # The policy sees an ordinary causal hint for a possible next
                # turn. Only the diagnostic log knows no such turn is offered.
                async def abandoned():
                    owner = f'sc|suite|case|w{wave}-s{slot}|sample-{len(source["replay"])}'
                    while True:
                        before = len(hints)
                        try:
                            await publish_hint(client, url, previous, sid, owner, namespace, hints,
                                declared_wait_ms=plan.get('hint_declared_wait_ms',750),
                                completed_transcript=plan.get('hint_completed_transcript',True))
                        finally:
                            own = [h for h in hints[before:] if h['hint']['owner_id']==owner]
                            for receipt in own:
                                receipt['abandoned_advice_diagnostic'] = True
                                emit('abandoned_hint',session_id=sid,future_owner_id=owner,receipt=receipt)
                        await asyncio.sleep(plan['hint_refresh_s'])
                abandoned_tasks.append(asyncio.create_task(abandoned()))
        tasks = [asyncio.create_task(session(i,s)) for i,s in enumerate(sessions)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks + abandoned_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks,*abandoned_tasks,return_exceptions=True)
            events.close()
    return rows
