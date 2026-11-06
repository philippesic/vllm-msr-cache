"""Constant-number scalar metadata reads; never enumerate cache residency."""
from collections import OrderedDict, Counter, deque
import os
import time

SCHEDULER_FIELDS = (
    '_observed_prefix_generations', '_observed_prefixes', '_wait_hint_results',
    '_wait_hint_request_bindings', '_published_wait_prefixes', '_prefetched',
    '_wait_prefetch', '_wait_prefetch_jobs', '_pending_job_keys', '_pending_job_blocks',
    '_pending_splice_jobs', '_pending_splice_releases', '_pending_splice_protection_ids',
    '_pending_splice_request_ids', '_preempted_pending', '_preservation_requests',
    '_preservation_finished', '_preservation_jobs', '_preservation_job_events',
    '_preservation_known_stored_keys', '_preservation_evicted_keys',
    '_score_source_tokens', '_score_source_deadlines', '_lifecycle_request_params',
    '_pending_lifecycle_releases', '_decision_scoring_requests', '_decision_scoring_attempts',
)
MANAGER_FIELDS = (
    '_score_residencies', '_pending_evicted_keys', '_lifecycle_release_ledger',
    '_lifecycle_owner_generations', '_lifecycle_owner_states', '_lifecycle_owner_keys',
    '_lifecycle_owners_by_key', '_pending_admission_records', '_pending_chain_admission_records',
    '_chain_admission_pending_stores', '_staging_protection_keys', '_pending_staging_protection_records',
)
WORKER_FIELDS = (
    'durable_summaries', '_durable_summary_publications', '_summary_job_publications',
    '_pending_job_keys', '_pending_job_blocks', '_transfer_directions',
    '_transfer_started_at_ns', '_transfer_start_times', '_transfer_completion_times',
    '_transfer_destination_blocks', '_pending_transfer_events', '_failed_load_block_ids',
    '_pending_scores', '_pending_score_head_counts', '_pending_async_score_observations',
    '_first_forward_times_ns', '_preempted_request_ids', '_preempted_forward_times_ns',
    '_query_snapshots', '_query_snapshot_events', '_retired_async_request_ids',
    '_decision_async_signatures', '_summary_job_events', '_summary_job_generations',
    '_deferred_transfer_results', '_async_request_versions', '_async_submission_requests',
    '_splice_events', '_splice_event_evidence', '_completed_splice_evidence',
)
_COLLECTIONS = (dict, list, tuple, set, frozenset, OrderedDict, Counter, deque)

def lengths(obj, fields):
    """Missing and unexpected types are unavailable, not manufactured zeros."""
    values = vars(obj) if obj is not None and hasattr(obj, '__dict__') else {}
    return {key: len(values[key]) if type(values.get(key)) in _COLLECTIONS else None for key in fields}

def snapshot(scheduler=None, manager=None, native_scheduler=None, worker=None):
    native = vars(native_scheduler) if native_scheduler is not None else {}
    sched = vars(scheduler) if scheduler is not None else {}
    return {
        'schema': 'operational-scalar-state-v1', 'at_ns': time.monotonic_ns(), 'pid': os.getpid(),
        'scope': 'O(1) lengths of explicitly named metadata containers; no cache or block enumeration',
        'scheduler_type': type(scheduler).__name__ if scheduler is not None else None,
        'scheduler_lengths': lengths(scheduler, SCHEDULER_FIELDS),
        'manager_type': type(manager).__name__ if manager is not None else None,
        'manager_lengths': lengths(manager, MANAGER_FIELDS),
        'worker_type': type(worker).__name__ if worker is not None else None,
        'worker_lengths': lengths(worker, WORKER_FIELDS),
        'session_registry_limit': sched.get('_observed_session_limit'),
        'observed_prefix_limit': sched.get('_observed_prefix_limit'),
        'hint_result_limit': sched.get('_wait_hint_result_limit'),
        'preservation_audit_enabled': sched.get('_preservation_audit_enabled'),
        'native_request_count': len(native['requests']) if type(native.get('requests')) is dict else None,
        'native_running_count': len(native['running']) if type(native.get('running')) is list else None,
        # The installed request queue supports O(1) __len__; missing remains unavailable.
        'native_waiting_count': len(native['waiting']) if native.get('waiting') is not None else None,
    }
