import threading
import torch
from unittest.mock import patch
from tests.test_async_scoring import _worker, _key, _summary
from semantic_offload.async_scoring import AsyncScoringController

def gated(w, budget=1_000_000):
    w._scoring_schedule = 'continuous'
    w._async_scoring.shutdown()
    release = threading.Event()
    original = w._score_async_payload
    def scorer(payload):
        assert release.wait(5)
        return original(payload)
    w._async_scoring = AsyncScoringController(scorer, max_queue_items=2,
        max_snapshot_bytes=budget, workers=1)
    return release

def test_packed_bank_reuses_immutable_tensor_and_copies_queries():
    w = _worker('asynchronous'); release = gated(w)
    try:
        query = torch.ones(1, 1, 4)
        with patch('torch.stack', wraps=torch.stack) as stack:
            assert w._submit_async_scoring(['a'], query) == 0
            assert w._submit_async_scoring(['b'], query) == 1
        assert stack.call_count == 1
        pending = list(w._async_scoring._pending.values())
        a, b = [entry.snapshot.payload for entry in pending]
        assert a.summaries == b.summaries == ()
        assert a.packed_fields[0][1] is b.packed_fields[0][1]
        assert a.queries.data_ptr() != query.data_ptr()
        query.zero_(); release.set()
        for entry in pending:
            scores, heads = entry.future.result(timeout=5).result
            values = next(iter(scores['mean'].values()))
            assert values == [(_key(1), 8.), (_key(2), -4.)]
        w._drain_async_scores(wait=True)
    finally:
        release.set(); w._async_scoring.shutdown()

def test_packed_overwrite_leaves_pending_old_bank_intact_and_filters_stale_key():
    w = _worker('asynchronous'); release = gated(w)
    try:
        w._submit_async_scoring(['a'], torch.ones(1, 1, 4))
        first = next(iter(w._async_scoring._pending.values()))
        old_tensor = first.snapshot.payload.packed_fields[0][1]
        w.durable_summaries[_key(1)] = [_summary(9.)]
        w._durable_summary_publications[_key(1)] = (0, 9, 3)
        w._submit_async_scoring(['b'], torch.ones(1, 1, 4))
        second = list(w._async_scoring._pending.values())[1]
        assert second.snapshot.payload.packed_fields[0][1] is not old_tensor
        release.set()
        assert first.future.result(timeout=5).result[0]['mean']['a'][0] == (_key(1), 8.)
        assert second.future.result(timeout=5).result[0]['mean']['b'][0] == (_key(1), 36.)
        w._drain_async_scores(wait=True)
        observations = w.pop_async_score_observations()
        first_scores = observations[0]['scores']['mean']['a']
        assert all(key != _key(1) for key, _ in first_scores)
        assert observations[1]['scores']['mean']['b'][0] == (_key(1), 36.)
    finally:
        release.set(); w._async_scoring.shutdown()

def test_full_queue_rejects_before_bank_scan():
    w = _worker('asynchronous'); release = gated(w)
    try:
        w._submit_async_scoring(['a'], torch.ones(1,1,4))
        w._submit_async_scoring(['b'], torch.ones(1,1,4))
        class NoScan(dict):
            def __iter__(self): raise AssertionError('full queue bank scan')
        w.durable_summaries = NoScan(w.durable_summaries)
        assert w._submit_async_scoring(['c'], torch.ones(1,1,4)) is None
    finally:
        release.set(); w._async_scoring.shutdown()

def test_byte_rejection_precedes_packed_allocation():
    w = _worker('asynchronous'); release = gated(w, budget=16)
    try:
        with patch('torch.stack', side_effect=AssertionError('byte-rejected bank allocation')):
            assert w._submit_async_scoring(['a'], torch.ones(1,1,4)) is None
        assert w._async_scoring.stats()['pending_items'] == 0
        assert getattr(w, '_async_packed_bank', None) is None
    finally:
        release.set(); w._async_scoring.shutdown()

def test_packed_view_charges_entire_backing_storage():
    w = _worker('asynchronous'); release = gated(w)
    try:
        keys = tuple(w.durable_summaries)
        signature = (w._summary_generation, id(w.durable_summaries),
            tuple((key,w._durable_summary_publications.get(key)) for key in keys))
        backing = torch.ones(4096)
        view = backing[:8].reshape(2,1,4)
        w._async_packed_bank = (signature, view, ())
        w._submit_async_scoring(['a'], torch.ones(1,1,4))
        snapshot = next(iter(w._async_scoring._pending.values())).snapshot
        assert snapshot.snapshot_bytes >= backing.untyped_storage().nbytes()
        events = w.pop_async_scoring_telemetry()
        captured = next(event for event in events if event['event'] == 'snapshot_captured')
        assert captured['retained_unique_storage_bytes'] == backing.untyped_storage().nbytes()
    finally:
        release.set(); w._async_scoring.shutdown()

def test_cache_release_fences_pending_packed_producer():
    w = _worker('asynchronous'); seen=[]
    class Fence:
        def synchronize(self): seen.append('fenced')
    try:
        w._async_packed_bank = ((), torch.ones(2), (Fence(),))
        w._release_async_packed_bank()
        assert seen == ['fenced'] and w._async_packed_bank is None
    finally: w._async_scoring.shutdown()

def test_selected_producers_precede_pack_and_source_storage_tracks_reader_stream():
    from types import SimpleNamespace
    w = _worker('asynchronous'); w._async_scoring.shutdown(); seen=[]; submitted=[]
    class Stream:
        def wait_event(self, event): seen.append(('wait', event))
    current = Stream()
    class Event:
        def record(self, stream): seen.append(('record', self, stream))
    selected = object(); unrelated = object()
    w._summary_job_events = {0: selected, 9: unrelated}
    w._async_scoring = SimpleNamespace(can_accept=lambda _: True,
        submit=lambda payload, **kw: submitted.append((payload, kw)) or len(submitted)-1,
        record_event=lambda **kw: None)
    stack = torch.stack
    def tracked_stack(*args, **kwargs):
        seen.append(('stack',))
        return stack(*args, **kwargs)
    try:
        with patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)), \
             patch.object(torch.Tensor, 'record_stream', lambda tensor, stream: seen.append(('reader', tensor, stream))), \
             patch('torch.cuda.current_stream', return_value=current), \
             patch('torch.cuda.Event', side_effect=Event), \
             patch('torch.stack', side_effect=tracked_stack):
            w._submit_async_scoring(['a'], torch.ones(1,1,4))
            w._submit_async_scoring(['b'], torch.ones(1,1,4))
        assert [item for item in seen if item[0]=='wait'] == [('wait',selected)]
        stack_index = next(index for index, item in enumerate(seen) if item[0]=='stack')
        assert len([item for item in seen[:stack_index] if item[0]=='reader']) == 2
        assert all(item[2] is current for item in seen if item[0]=='reader')
        first, second = [payload for payload, _ in submitted]
        assert first.packed_fields[0][1] is second.packed_fields[0][1]
        first_fences, second_fences = [kw['readiness_fences'] for _, kw in submitted]
        assert first_fences[0] is second_fences[0]  # Immutable bank's producer.
        assert first_fences[1] is not second_fences[1]  # Distinct owned queries.
        assert selected not in first_fences and unrelated not in first_fences
    finally:
        w._async_packed_bank = None

def test_packed_scorer_error_fences_reader_stream_before_future_failure():
    w = _worker('asynchronous'); release = gated(w); seen=[]
    try:
        w._submit_async_scoring(['a'], torch.ones(1,1,4))
        payload = next(iter(w._async_scoring._pending.values())).snapshot.payload
        class Stream:
            def synchronize(self): seen.append('reader_fenced')
        class Context:
            def __enter__(self): pass
            def __exit__(self,*a): pass
        w._async_scoring_stream = Stream()
        import semantic_offload.worker as module
        with patch('torch.cuda.stream', return_value=Context()), \
             patch.object(module, 'score_mean_batch', side_effect=RuntimeError('injected score failure')):
            try:
                w._score_async_payload(payload)
                raise AssertionError('failure was swallowed')
            except RuntimeError as error:
                assert str(error) == 'injected score failure'
        assert seen == ['reader_fenced']
    finally:
        w._async_scoring_stream = None
        release.set(); w._async_scoring.shutdown()

def test_subset_and_opportunity_keep_existing_payload_projection_fallback():
    for schedule, subset in [('continuous', True), ('opportunity', False)]:
        w = _worker('asynchronous'); release = gated(w)
        w._scoring_schedule = schedule
        w._query_snapshot_max_age_ns = 1_000_000_000
        try:
            keys = [_key(1)] if subset else None
            w._submit_async_scoring(['a'], torch.ones(1,1,4), candidate_keys=keys)
            payload = next(iter(w._async_scoring._pending.values())).snapshot.payload
            assert not payload.packed_fields and payload.summaries
            assert len(payload.keys) == (1 if subset else 2)
        finally:
            release.set(); w._async_scoring.shutdown()

def test_reset_drains_payload_and_releases_packed_bank():
    w = _worker('asynchronous')
    try:
        w._submit_async_scoring(['a'], torch.ones(1,1,4))
        assert w._async_packed_bank is not None
        w.reset_semantic_state()
        assert w._async_packed_bank is None
        assert w._async_scoring.stats()['pending_snapshot_bytes'] == 0
        assert not w._pending_async_score_observations
    finally: w._async_scoring.shutdown()

def test_pack_does_not_retain_autograd_source_graphs():
    w = _worker('asynchronous'); release = gated(w)
    try:
        for heads in w.durable_summaries.values():
            for head in heads:
                head.mean = torch.ones(4, requires_grad=True) * head.mean
        w._submit_async_scoring(['a'], torch.ones(1,1,4))
        payload = next(iter(w._async_scoring._pending.values())).snapshot.payload
        packed = payload.packed_fields[0][1]
        assert packed.requires_grad is False and packed.grad_fn is None
    finally:
        release.set(); w._async_scoring.shutdown()
