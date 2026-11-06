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
                params = getattr(req, 'kv_transfer_params', None)
                external_id = params.get('external_request_id') if isinstance(params, dict) else None
                ABORTS.append({'request_id': req.request_id, 'external_request_id': external_id if type(external_id) is str else None, 'external_mapping_available': type(external_id) is str, 'status': status.name,
                               'observed_ns': time.monotonic_ns()})
        return result
    cls.finish_requests = finish
    return original


def snapshot():
    return {'count': ABORT_COUNT, 'records': list(ABORTS), 'record_capacity': 32,
            'scope': 'Only actual returned FINISHED_ABORTED Requests; external_request_id read directly from Request.kv_transfer_params, never parsed from internal ID'}
