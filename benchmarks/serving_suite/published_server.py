"""Small HTTP transport for unmodified published AsyncLLM implementations.

The upstream draft exposes kv_hints on AsyncLLM, not its completions schema.
This bridge forwards that field without replacing scheduler/cache methods.
Run with the published checkout's interpreter, not the project's runtime.
"""
import asyncio
import inspect
import json


def main():
    from aiohttp import web
    from prometheus_client import generate_latest
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    parser.add_argument('--policy', choices=['continuum', 'upstream'], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    async def app_factory():
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs.from_cli_args(args))
        decoder = None
        if args.policy == 'upstream':
            import msgspec
            from vllm.v1.kv_hints import KvHintsEnvelope
            decoder = lambda raw: msgspec.convert(raw, type=KvHintsEnvelope)
        return await create_app(engine, args.policy, SamplingParams, decoder)

    web.run_app(app_factory(), host='127.0.0.1', port=args.port)


async def create_app(engine, policy, sampling_class, envelope_decoder=None, metrics_body=None):
    from aiohttp import web
    from prometheus_client import generate_latest
    completed = 0
    app = web.Application()

    async def health(request):
        await engine.check_health()
        return web.json_response({'status': 'ready', 'policy': policy})

    async def metrics(request):
        return web.Response(body=(metrics_body or generate_latest)()+
            f'\nsuite_completed_requests_total {completed}\n'.encode(),
            content_type='text/plain')

    async def complete(request):
        nonlocal completed
        raw = await request.json()
        rid = raw['request_id']
        extra = {}
        kwargs = {}
        if policy == 'upstream' and 'priority' in raw:
            kwargs['priority'] = raw['priority']
        if policy == 'continuum':
            extra = {k: raw[k] for k in ('job_id', 'is_last_step') if k in raw}
            # Warmup is an independent finished job, never a tool wait.
            extra.setdefault('job_id', rid)
            extra.setdefault('is_last_step', True)
        elif raw.get('kv_hints'):
            kwargs['kv_hints'] = envelope_decoder(raw['kv_hints'])
        params = sampling_class(max_tokens=raw['max_tokens'],
            temperature=raw.get('temperature', 0), seed=raw.get('seed'),
            extra_args=extra or None)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        count = 0
        finished = False
        try:
            async for output in engine.generate(
                {'prompt_token_ids': raw['prompt']}, params, rid, **kwargs
            ):
                item = output.outputs[0]
                tokens = list(item.token_ids[count:])
                count = len(item.token_ids)
                event = dict(id=rid, choices=[dict(index=0, text='', token_ids=tokens,
                    finish_reason=item.finish_reason)])
                await response.write(('data: '+json.dumps(event)+'\n\n').encode())
                finished = output.finished
            if not finished:
                raise RuntimeError('Engine ended without a terminal output')
            completed += 1
            await response.write(b'data: [DONE]\n\n')
            return response
        finally:
            if not finished:
                await engine.abort(rid)

    async def stats():
        while True:
            await asyncio.sleep(0.5)
            await engine.do_log_stats()

    stats_task = asyncio.create_task(stats())

    async def cleanup(app):
        stats_task.cancel()
        await asyncio.gather(stats_task, return_exceptions=True)
        result = engine.shutdown()
        if inspect.isawaitable(result):
            await result

    app.router.add_get('/health', health)
    app.router.add_get('/metrics', metrics)
    app.router.add_post('/v1/completions', complete)
    app.on_cleanup.append(cleanup)
    return app


if __name__ == '__main__':
    main()
