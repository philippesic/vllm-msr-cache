"""Declared false-positive advice experiment; disabled outside the addendum.

All nominations still bind to real observed token IDs. This changes application
prediction timing, never the selected retention policy or validation contract.
"""
from __future__ import annotations

import asyncio
import time


async def schedule(*, mode, scenario, index, later, event, announce, emit,
                   late_delay_s=44.25, abandoned_refresh_s=30.0):
    if mode not in ('accurate', 'misleading'):
        raise ValueError('Unknown prespecified advice mode')
    false_positive = scenario == 'abandoned' and index == 0
    if not later and not false_positive:
        return None
    if false_positive and mode == 'accurate':
        emit('advice_prediction', scenario=scenario, mode=mode, action='abstain')
        return None

    delay = late_delay_s if mode == 'accurate' and scenario == 'late' and index == 0 else 0
    event.clear()

    async def publish():
        row = await announce()
        row['prediction_experiment'] = dict(mode=mode, scenario=scenario,
            false_positive=false_positive, request_index=index)
        emit('advice_prediction', scenario=scenario, mode=mode, action='publish',
             false_positive=false_positive, request_index=index,
             event_id=row['hint']['event_id'], response=row.get('response'), error=row.get('error'))

    # The normal immediate hint remains awaited before releasing the dependency
    # to tools. A truthful long-wait prediction does not delay tool execution.
    if not delay:
        await publish()

    async def refresh():
        if delay:
            await asyncio.sleep(delay)
            if event.is_set():
                return
            await publish()
        deadline = time.monotonic() + abandoned_refresh_s if false_positive else None
        while True:
            await asyncio.sleep(1.0)
            if event.is_set() or deadline is not None and time.monotonic() >= deadline:
                return
            await publish()

    return asyncio.create_task(refresh())
