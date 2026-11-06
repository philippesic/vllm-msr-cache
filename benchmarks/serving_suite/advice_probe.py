"""Bounded live advice perturbations for persistent-server operational checks."""
import asyncio
import time

from .campaign_contracts import digest


async def exercise(client, url, original, wave):
    raw = original['hint']
    async def send(name, payload, checker):
        async with client.post(url+'/plugins/semantic-offload/wait-hints',json=payload) as response:
            body = await response.json()
            return dict(check=name,passed=checker(response.status,body),http_status=response.status,
                        request=payload,response=body,observed_ns=time.monotonic_ns())
    checks = []
    if wave == 0:
        checks.append(await send('duplicate_is_idempotent',raw,
            lambda status,body: status==200 and body.get('idempotent_replay') is True))
        bad = dict(raw,event_id=digest([raw['event_id'],'unknown-session']),session_id=raw['session_id']+'-unobserved')
        checks.append(await send('unknown_session_cannot_protect_prefix',bad,
            lambda status,body: status==200 and body.get('reason')=='observed_prefix_unavailable'))
        bad = dict(raw,event_id=digest([raw['event_id'],'future']),actual_resume_ns=time.monotonic_ns()+10**12)
        checks.append(await send('future_information_rejected',bad,lambda status,body: status==400))
    elif wave == 1:
        # Reuse a real observation/publication clock, delaying transport beyond
        # its declared nomination deadline. No invented future arrival time.
        await asyncio.sleep(1.1)
        delayed = dict(raw,event_id=digest([raw['event_id'],'expired']))
        checks.append(await send('delayed_advice_expires',delayed,
            lambda status,body: status==200 and body.get('status')=='expired'
                               and body.get('reason')=='nomination_deadline_elapsed'))
    return checks
