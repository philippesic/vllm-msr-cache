"""Default-off native scheduler observations for prefetch qualification.

No cache lookups, reservations, policy changes, or token/content capture. Free
blocks include idle cached blocks; they are reclaimable, not necessarily empty.
All timestamps use the scheduler's monotonic clock. File I/O is deliberately
charged to instrumented runs and is not a production-performance configuration.
"""

import json
import os
import time
from itertools import chain, islice


class SchedulerTrace:
    def __init__(self, path):
        # Exclusive creation prevents accidentally pooling separate engines.
        self.stream = open(path, "x", buffering=1)
        self.emit("start", pid=os.getpid(), schema="kv-qualification-v1")

    def emit(self, event, **fields):
        self.stream.write(json.dumps(dict(event=event, ns=time.monotonic_ns(), **fields)) + "\n")

    def step(self, scheduler, scheduled):
        waiting_count = len(scheduler.waiting) + len(scheduler.skipped_waiting)
        waiting = islice(chain(scheduler.waiting, scheduler.skipped_waiting), 64)
        self.emit(
            "step", step=scheduler.current_step,
            running=len(scheduler.running), max_running=scheduler.max_num_running_reqs,
            reclaimable=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
            total_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks,
            waiting_count=waiting_count,
            waiting=[[r.request_id, r.status.name] for r in waiting],
            waiting_truncated=waiting_count > 64,
            scheduled_tokens=sum(scheduled.values()),
        )

    def close(self):
        self.stream.close()


def create_scheduler_trace():
    path = os.environ.get("VLLM_KV_QUALIFICATION_TRACE")
    return SchedulerTrace(path) if path else None
