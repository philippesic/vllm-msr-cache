"""Two-turn, co-located wait-hint integration example against an existing server.

Use the pinned runtime and CPU-only wait profile documented in WAIT_HINT_SUPPORT.
The application executes a fixed real SHA-256 tool; generated shell is never run.
Hint activity demonstrates integration, not saved work or a performance win.
"""
import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from transformers import AutoTokenizer


def prefix_id(tokens,namespace):
    return hashlib.sha256(json.dumps(dict(namespace=namespace,token_ids=tokens),
        sort_keys=True,separators=(',',':')).encode()).hexdigest()


async def main(args):
    if urlparse(args.url).hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('This example uses a shared host monotonic clock and requires a loopback server')
    namespace=json.loads(args.config.read_text())['kv_connector_extra_config']['admission_namespace']
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    session_id='integration-'+str(time.monotonic_ns())
    history=[dict(role='user',content='In one short sentence, explain exact-prefix KV caching.')]
    first_token=asyncio.Event();acknowledgements=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),
                                    connector=aiohttp.TCPConnector(force_close=True)) as client:
        async def generate(turn,observed=None):
            prompt=tokenizer.apply_chat_template(history,tokenize=True,add_generation_prompt=True,return_dict=False)
            known=observed if observed is not None else prompt[:len(prompt)//16*16]
            rid=f'sc|example|client|{session_id}|sample-{turn}'
            params=dict(run_id='example',cell_id='client',session_id=session_id,generation=0,
                external_request_id=rid,observed_prefix_id=prefix_id(known,namespace),
                observed_prefix_boundary_tokens=len(known))
            payload=dict(model='suite-model',request_id=rid,prompt=prompt,max_tokens=128,
                         temperature=0,stream=True,return_token_ids=True,kv_transfer_params=params)
            outputs=[];done=False
            async with client.post(args.url+'/v1/completions',json=payload) as response:
                if response.status!=200:
                    raise RuntimeError(await response.text())
                async for line in response.content:
                    if not line.startswith(b'data: '):continue
                    if line[6:].strip()==b'[DONE]':done=True;break
                    chunk=json.loads(line[6:])
                    for choice in chunk.get('choices',[]):
                        tokens=choice.get('token_ids') or []
                        if tokens and turn:first_token.set()
                        outputs.extend(tokens)
            if not done or not outputs:
                raise RuntimeError('Incomplete generation')
            text=tokenizer.decode(outputs,skip_special_tokens=True)
            history.append(dict(role='assistant',content=text))
            return dict(payload=payload,output_token_ids=outputs,text=text)
        previous=await generate(0)
        known=(previous['payload']['prompt']+previous['output_token_ids'][:-1])
        known=known[:len(known)//16*16]
        async def hint():
            now=time.monotonic_ns()
            raw=dict(schema='semantic-known-prefix-wait-hint-v1',event_id=hashlib.sha256(str(now).encode()).hexdigest(),
                owner_id=f'sc|example|client|{session_id}|sample-1',run_id='example',cell_id='client',
                session_id=session_id,generation=0,namespace=namespace,
                observed_prefix_id=prefix_id(known,namespace),prefix_boundary_tokens=len(known),
                observation_available_ns=now,input_publication_ns=now,declared_wait_ms=750,
                wait_provenance='driver_declared_tool_wait_budget')
            async with client.post(args.url+'/plugins/semantic-offload/wait-hints',json=raw) as response:
                ack=await response.json()
                if response.status!=200 or ack.get('event_id')!=raw['event_id']:
                    raise RuntimeError('Hint acknowledgement failed: '+str(ack))
                acknowledgements.append(ack)
        await hint()
        async def refresh():
            while not first_token.is_set():
                await asyncio.sleep(1)
                if not first_token.is_set():await hint()
        refresher=asyncio.create_task(refresh())
        proc=None
        try:
            proc=await asyncio.create_subprocess_exec(sys.executable,'-c',
                'import hashlib; print(hashlib.sha256(b"wait-hint integration probe").hexdigest())',
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,start_new_session=True)
            stdout,stderr=await asyncio.wait_for(proc.communicate(),10)
            if proc.returncode:
                raise RuntimeError(stderr.decode())
            history.append(dict(role='user',content='The application actually ran SHA-256 and got: '+stdout.decode()+
                                '\nIn one short sentence, explain what this output establishes and one limitation.'))
            returning=await generate(1,known)
        finally:
            refresher.cancel()
            await asyncio.gather(refresher,return_exceptions=True)
            if proc is not None and proc.returncode is None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                await proc.communicate()
        if refresher.done() and not refresher.cancelled() and refresher.exception():
            raise refresher.exception()
        print(json.dumps(dict(first_response=previous['text'],returning_response=returning['text'],
            hint_acknowledgements=len(acknowledgements),
            ready_cpu_key_refresh_observations=sum(a.get('ready_cpu_prefix_keys',0) for a in acknowledgements),
            scope='Integration activity only; not a saved-work or performance measurement.'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
