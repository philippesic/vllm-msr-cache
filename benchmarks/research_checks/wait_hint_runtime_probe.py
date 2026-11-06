"""Live cache-integrity and malformed-advice probe using actual document excerpts.

This is a functional diagnostic, not application performance evidence. It forces
cache churn, returns a hinted prefix and compares generation against fresh
computation under a distinct cache salt in the same runtime/model/precision.
"""
from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path

import aiohttp
from benchmarks.serving_suite.contracts import digest, write_json
from benchmarks.serving_suite.repo_workload import publish_hint
from benchmarks.serving_suite.workload import completion



ACK_SCHEMA = 'semantic-known-prefix-wait-hint-ack-v1'
FUTURE_FIELD_DETAIL = 'wait hint contains forbidden future or diagnostic fields'


def valid_ack(response, event_id):
    body = response.get('body')
    return (response.get('http_status') == 200 and isinstance(body, dict)
            and body.get('schema') == ACK_SCHEMA and body.get('event_id') == event_id)


def valid_rejection(response, event_id, reason):
    return (valid_ack(response, event_id) and response['body'].get('status') == 'rejected'
            and response['body'].get('reason') == reason
            and response['body'].get('ready_cpu_prefix_keys', 0) == 0)


def valid_duplicate(response, event_id, original):
    return (valid_ack(response, event_id) and isinstance(original, dict)
            and response['body'].get('idempotent_replay') is True
            and response['body'] == {**original, 'idempotent_replay': True})


def valid_future_rejection(response):
    body = response.get('body')
    return response.get('http_status') == 400 and body in (
        {'detail': FUTURE_FIELD_DETAIL},
        {'error': {'message': FUTURE_FIELD_DETAIL, 'type': 'Bad Request',
                   'param': None, 'code': 400}},
    )


def scoped_metric_deltas(before, after):
    """Keep eager directional counters and missing/reset evidence explicit."""
    from harness.metrics import parse_metrics_evidence, diff_with_provenance
    names = ('vllm:request_prefill_kv_computed_tokens_sum',
             'vllm:external_prefix_cache_hits_total',
             'vllm:kv_offload_load_bytes_total')
    left, right = parse_metrics_evidence(before), parse_metrics_evidence(after)
    values, provenance = diff_with_provenance(left['values'], right['values'])
    return ({name: values[name] for name in names},
            {name: dict(before=left['sources'][name], after=right['sources'][name],
                        difference=provenance[name]) for name in names})


async def run_wave(url,plan,sessions,tokenizer,wave,writer,arm,namespace,hints,stream_dir):
    directory=Path(stream_dir).parent
    probe=plan['runtime_probe']
    documents=[Path(p).read_text() for p in probe['documents']]
    rows=[];checks=[];request_metrics=[]
    origin=time.monotonic_ns()
    sid=f'probe-{wave}'
    salt=f'qualification-cached-{wave}'
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=plan['request_timeout_s']),connector=aiohttp.TCPConnector(force_close=True)) as client:
        from benchmarks.serving_suite.runner import metric
        async def metrics():
            async with client.get(url+'/metrics') as response:
                if response.status!=200:
                    raise RuntimeError('Probe metrics unavailable')
                return await response.text()
        baseline_metrics=await metrics()
        completed_baseline=metric(baseline_metrics,'vllm:request_success_total')
        if completed_baseline is None:
            raise RuntimeError('Probe completion counter unavailable')
        last_metrics=baseline_metrics
        async def request(prompt,turn,*,fresh=False,register=False):
            nonlocal last_metrics
            rid=f'sc|suite|case|w{wave}-s0|sample-{turn}'
            params=dict(run_id='suite',cell_id='case',session_id=sid,generation=0,
                        external_request_id=rid,benchmark_client_sequence=turn)
            if register:
                observed=prompt[:len(prompt)//16*16]
                params.update(observed_prefix_id=digest(dict(namespace=namespace,token_ids=observed)),
                              observed_prefix_boundary_tokens=len(observed))
            payload=dict(model='suite-model',request_id=rid,prompt=prompt,
                max_tokens=probe.get('output_tokens',24),temperature=0,seed=plan['seed'],
                stream=True,return_token_ids=True,kv_transfer_params=params,
                cache_salt=salt+'-fresh' if fresh else salt)
            row=await completion(client,url,payload,dict(session_id=sid,source_id=sessions[0]['id'],
                wave=wave,turn=turn,queued_ns=time.monotonic_ns(),functional_probe=True),writer,
                stream_path=Path(stream_dir)/f'probe-{wave}-{turn}.sse')
            rows.append(row)
            if row['success']:
                deadline=time.monotonic()+20
                while True:
                    after=await metrics()
                    count=metric(after,'vllm:request_success_total')
                    if count is not None and count>=completed_baseline+len(rows):
                        break
                    if time.monotonic()>=deadline:
                        raise RuntimeError('Probe metrics did not settle')
                    await asyncio.sleep(.1)
                deltas, provenance = scoped_metric_deltas(last_metrics, after)
                before_path = directory / f'probe-metrics-{wave}-{turn}-before.prom'
                after_path = directory / f'probe-metrics-{wave}-{turn}-after.prom'
                before_path.write_text(last_metrics)
                after_path.write_text(after)
                request_metrics.append(dict(request_id=rid,turn=turn,
                    prompt_tokens=len(prompt),fresh_salt=fresh,metrics=deltas,
                    metric_provenance=provenance,
                    before_exposition=before_path.name,after_exposition=after_path.name))
                last_metrics=after
            if not row['success']:
                raise RuntimeError(f'Probe generation failed: {rid}')
            return row
        abandoning=probe.get('abandon_after_initial_hint',False)
        target_text=('Unique abandoned qualification target '+sid+'\n' if abandoning else '')+documents[0]
        target=tokenizer.encode(target_text)[:probe.get('prompt_tokens',512)]
        previous=await request(target,0,register=plan['publish_hints'])
        last_hint=None
        for index,document in enumerate(documents[1:],1):
            if plan['publish_hints'] and (not abandoning or index==1):
                last_hint=await publish_hint(client,url,previous,sid,
                    f'sc|suite|case|w{wave}-s0|sample-1',namespace,hints)
                if last_hint.get('error'):
                    raise RuntimeError('Probe hint transport failed')
            await request(tokenizer.encode(document)[:probe.get('prompt_tokens',512)],index)
        if plan['publish_hints']:
            if not abandoning:
                last_hint=await publish_hint(client,url,previous,sid,
                    f'sc|suite|case|w{wave}-s0|sample-1',namespace,hints)
            if last_hint.get('error'):
                raise RuntimeError('Probe final hint transport failed')
            raw=last_hint['hint']
            async def advice(body):
                async with client.post(url+'/plugins/semantic-offload/wait-hints',json=body) as response:
                    return dict(http_status=response.status,body=await response.json())
            duplicate=await advice(raw)
            checks.append(dict(check='duplicate_is_idempotent',passed=valid_duplicate(duplicate,raw['event_id'],last_hint.get('response')),response=duplicate))
            bad=dict(raw,event_id=digest([raw['event_id'],'wrong-namespace']),
                     namespace={**namespace,'model_revision':'wrong-model'})
            wrong_namespace=await advice(bad)
            checks.append(dict(check='wrong_namespace_rejected',passed=valid_rejection(wrong_namespace,bad['event_id'],'namespace_mismatch'),response=wrong_namespace))
            bad=dict(raw,event_id=digest([raw['event_id'],'future-field']),actual_resume_ns=time.monotonic_ns()+10**12)
            future=await advice(bad)
            checks.append(dict(check='future_information_rejected',passed=valid_future_rejection(future),response=future))
            bad=dict(raw,event_id=digest([raw['event_id'],'stale-clock']),observation_available_ns=1,input_publication_ns=1)
            stale=await advice(bad)
            checks.append(dict(check='stale_advice_rejected',passed=valid_rejection(stale,bad['event_id'],'causal_clock_mismatch'),response=stale))
        returning=previous['payload']['prompt']+previous['output_token_ids'][:-1]+tokenizer.encode('\nContinue this explanation briefly:')
        cached=await request(returning,len(documents),register=plan['publish_hints'])
        fresh=await request(returning,len(documents)+1,fresh=True)
        cached_metrics=request_metrics[-2]['metrics']; fresh_metrics=request_metrics[-1]['metrics']
        computed=fresh_metrics['vllm:request_prefill_kv_computed_tokens_sum']
        checks.append(dict(check='fresh_reference_recomputed_entire_prompt',
            passed=computed==len(returning) and fresh_metrics['vllm:external_prefix_cache_hits_total']==0,
            prompt_tokens=len(returning),computed_tokens=computed,metrics=fresh_metrics))
        if plan['publish_hints'] and probe.get('require_restore',True):
            checks.append(dict(check='return_really_restored_cpu_prefix',
                passed=(cached_metrics['vllm:external_prefix_cache_hits_total'] or 0)>0
                       and (cached_metrics['vllm:kv_offload_load_bytes_total'] or 0)>0,
                metrics=cached_metrics))
        if probe.get('require_no_cache_after_abandon'):
            checks.append(dict(check='abandoned_advice_did_not_prevent_later_eviction',
                passed=cached_metrics['vllm:request_prefill_kv_computed_tokens_sum']==len(returning)
                       and cached_metrics['vllm:external_prefix_cache_hits_total']==0
                       and cached_metrics['vllm:kv_offload_load_bytes_total']==0,
                prompt_tokens=len(returning),metrics=cached_metrics,
                scope='Initial advice only, then distinct-prefix churn and a late diagnostic return; not a live application performance result.'))
        checks.append(dict(check='cached_equals_fresh_generation' ,passed=cached['output_token_ids']==fresh['output_token_ids'],
            cached_request_id=cached['request_id'],fresh_request_id=fresh['request_id'],
            cached_output_ids=cached['output_token_ids'],fresh_output_ids=fresh['output_token_ids']))
        result=dict(schema='wait-hint-runtime-probe-v4',wave=wave,checks=checks,request_metrics=request_metrics,
            started_ns=origin,ended_ns=time.monotonic_ns(),passed=all(c['passed'] for c in checks),
            note='Actual CPU restore and fresh-reference computed tokens must additionally be checked from native request receipts. This probe alone does not infer KV byte equality.')
        write_json(directory/f'runtime-probe-{wave}.json',result)
        if not result['passed']:
            raise RuntimeError('Live functional probe failed; inspect retained checks')
    return rows
