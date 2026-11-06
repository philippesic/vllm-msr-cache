"""Bounded observation of actual native abort transitions, without policy changes."""
import functools
import time
from collections import deque

ABORTS = deque(maxlen=32)
ABORT_COUNT = 0


def install(cls):
    original = cls.finish_requests
    @functools.wraps(original)
    def finish(self, *args, **kwargs):
        global ABORT_COUNT
        result = original(self, *args, **kwargs)
        # The native API returns only requests actually transitioned by this call.
        # Never consume input iterables, inspect all active requests, or alter result.
        for req in result:
            status = getattr(req, 'status', None)
            if getattr(status, 'name', None) == 'FINISHED_ABORTED':
                ABORT_COUNT += 1
                ABORTS.append({'request_id': req.request_id, 'status': status.name,
                               'observed_ns': time.monotonic_ns()})
        return result
    cls.finish_requests = finish
    return original


def snapshot():
    return {'count': ABORT_COUNT, 'records': list(ABORTS), 'record_capacity': 32,
            'scope': 'Only requests returned by native finish_requests in FINISHED_ABORTED state'}
