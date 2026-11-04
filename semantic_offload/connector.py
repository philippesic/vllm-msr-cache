# SPDX-License-Identifier: Apache-2.0
"""SemanticOffloadingConnector: carries relevance-score plumbing alongside
vLLM's stock OffloadingConnector.

Two extra channels, piggybacking on the existing worker<->scheduler metadata
protocol (see .claude/docs/semantic-eviction-plan.md, Step 1.3):

- scheduler -> worker: which OffloadKey(s) each store job_id represents
  (`SemanticOffloadingConnectorMetadata.store_job_blocks`). The worker needs
  this to re-key its summaries by the stable OffloadKey identity instead of
  the transient GPU block_id they're computed from -- GPU blocks get reused
  for unrelated content once freed, but the CPU-tier block (and the need to
  score it) long outlives that. The layout is exact for single-group jobs,
  including multi-block chunks; ambiguous hybrid/sliding layouts are skipped
  rather than assigned to the wrong key.
- worker -> scheduler: computed relevance scores
  (`SemanticWorkerMetadata.pending_scores`), consumed by
  `SemanticOffloadingManager.update_relevance()` for the EMA-smoothed
  relevance state Step 1.4 will read from.
- scheduler -> server log (opt-in): request-owned transfer completions are
  joined back to deterministic request/sample identity and emitted as
  `SEMANTIC_PRESERVATION_EVENT` JSONL. The default remains inert.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from semantic_offload.profiling import PROFILE, profiled
from semantic_offload.benchmark_metrics import record_relevance_batch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.outputs import KVConnectorOutput

from semantic_offload._debug import DISABLE_PREFETCH as _DISABLE_PREFETCH
from semantic_offload._debug import ENABLED as _DEBUG_ENABLED
from semantic_offload._debug import TIMING as _TIMING
from semantic_offload._debug import (
    debug_print,
    record_async_scoring_event,
    record_count,
    record_decision_feature,
    record_preservation_event,
    record_timing,
    record_wait_hint_event,
)
from semantic_offload._vllm_compat import (
    config_blocks_per_chunk,
    construct_scheduler_base,
    construct_worker,
    create_offloading_spec,
    group_config_tokens_per_block,
    group_config_tokens_per_chunk,
    scheduler_being_loaded_set,
    scheduler_key_byte_layout,
    set_next_stored_chunk_idx,
)
from semantic_offload.lifecycle import (
    LifecycleHintResult,
    LifecycleReleaseController,
    LifecycleReleaseHint,
)
from semantic_offload.manager import (
    ComposedRelevanceUpdates,
    LatestRelevance,
    compose_latest_relevance,
    compose_relevance_updates,
)
from semantic_offload.store_layout import build_store_job_layout
from semantic_offload.wait_hint import (
    KnownPrefixWaitHint,
    WaitHintLease,
    select_useful_on_time,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.request import Request

# method -> req_id -> list[(offload_key, score)]
RelevanceScores = dict[str, dict[str, list[tuple[OffloadKey, float]]]]
ScoreHeadCounts = dict[str, dict[str, int]]
ScoreWorkerCounts = dict[str, dict[str, int]]
_MISSING_LIFECYCLE = object()
_CAMPAIGN_SCHEMA = "ep-campaign-serving-v1"
_CAMPAIGN_POLICY_PARAM_FIELDS = frozenset(
    {
        "campaign_schema",
        "external_request_id",
        "semantic_request_id",
        "semantic_sample_id",
        "sample_index",
        "benchmark_client_sequence",
        "seed",
        "run_id",
        "cell_id",
        "session_id",
        "generation",
        "key_chain_digest",
        "candidate_config_hash",
        "max_offload_tokens",
        "admission_cost_per_byte",
        "admission_cost_provenance",
        "admission_cost_observation_available_ns",
        "admission_cost_input_publication_ns",
        "observed_prefix_id",
        "observed_prefix_boundary_tokens",
        "wait_hint_event_id",
    }
)


def _campaign_policy_params(params: object) -> dict | None:
    """Strip the sealed post-hoc sidecar before constructing ReqContext."""
    if (
        not isinstance(params, dict)
        or params.get("campaign_schema") != _CAMPAIGN_SCHEMA
    ):
        return params
    allowed_input = _CAMPAIGN_POLICY_PARAM_FIELDS | {"campaign_provenance"}
    unexpected = set(params) - allowed_input
    if unexpected:
        raise ValueError(
            "campaign kv_transfer_params contain unexpected policy-visible fields: "
            + ", ".join(sorted(unexpected))
        )
    provenance = params.get("campaign_provenance")
    if not isinstance(provenance, dict):
        raise TypeError("campaign_provenance must be a sealed object")
    return {key: value for key, value in params.items() if key != "campaign_provenance"}


@dataclass(frozen=True, slots=True)
class SemanticTransferCompletion:
    """One worker's completed transfer contribution for a scheduler job."""

    is_load: bool
    bytes: int | None
    time_s: float | None
    max_time_s: float | None = None
    contributors: int = 1
    success: bool = True
    status: str = "success"
    cancelled: bool = False
    start_time_s: float | None = None
    completion_time_s: float | None = None
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    contributor_ids: tuple[str, ...] = ()


PreservationTransferEvents = dict[int, SemanticTransferCompletion]


def _earliest_timestamp(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _latest_timestamp(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _event_key(key: OffloadKey):
    """Return a deterministic JSON-safe representation of an OffloadKey."""
    if isinstance(key, bytes):
        return key.hex()
    if isinstance(key, tuple):
        return [_event_key(item) for item in key]
    if isinstance(key, list):
        return [_event_key(item) for item in key]
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key
    return repr(key)


def _event_key_sort(key: OffloadKey) -> str:
    return repr(_event_key(key))


def _observed_prefix_digest(token_ids: list[int], namespace: Mapping[str, str]) -> str:
    payload = json.dumps(
        {"namespace": dict(namespace), "token_ids": token_ids},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _new_prefetch_attribution_state() -> dict[str, object]:
    """Create a nested transfer envelope for legacy/manual states."""
    return {
        "status": "unavailable",
        "unavailable_reasons": [
            "t0_preemption_timestamp_unavailable",
            "t1_reservation_ownership_unavailable",
            "t4_readmission_timestamp_unavailable",
            "t5_blocking_remainder_boundaries_unavailable",
            "t6_first_forward_timestamp_unavailable",
            "cpu_residency_proof_unavailable",
            "splice_acknowledgement_unavailable",
            "request_queue_join_unavailable",
        ],
        "transfer_jobs": [],
        "attempts": [],
    }


def _prefetch_attribution_state(state: dict[str, object]) -> dict[str, object]:
    attribution = state.get("prefetch_attribution")
    if not isinstance(attribution, dict):
        attribution = _new_prefetch_attribution_state()
        state["prefetch_attribution"] = attribution
    attempts = attribution.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
        attribution["attempts"] = attempts
    return attribution


def _append_prefetch_attempt(
    state: dict[str, object],
    *,
    attempt_id: str,
    episode_id: int,
    selector: str,
    candidate_keys: list[OffloadKey],
    selected_keys: list[OffloadKey],
    cpu_resident_ready_keys: list[OffloadKey],
    reservation_id: str,
    reservation_block_ids: list[int],
    reservation_submit_time_s: float,
) -> None:
    attribution = _prefetch_attribution_state(state)
    attribution["attempts"].append(
        {
            "attempt_id": attempt_id,
            "episode_id": episode_id,
            "selector": selector,
            "candidate_keys": [_event_key(key) for key in candidate_keys],
            "selected_keys": [_event_key(key) for key in selected_keys],
            "cpu_resident_ready_keys": [
                _event_key(key) for key in cpu_resident_ready_keys
            ],
            "reservation_id": reservation_id,
            "reservation_block_ids": [
                str(block_id) for block_id in reservation_block_ids
            ],
            "reservation_submit_time_s": reservation_submit_time_s,
            "status": "reserved",
            "actual_load_plan_keys": None,
            "actual_destination_block_ids": None,
            "intersection_keys": None,
            "remainder_keys": None,
            "readmission_observed_at_s": None,
        }
    )


def _update_prefetch_attempt_plan(
    state: dict[str, object],
    attempt_id: str | None,
    actual_load_plan_keys: list[OffloadKey],
    *,
    status: str,
    actual_destination_block_ids: list[int] | None = None,
) -> None:
    if attempt_id is None:
        return
    attribution = _prefetch_attribution_state(state)
    for attempt in attribution["attempts"]:
        if not isinstance(attempt, dict) or attempt.get("attempt_id") != attempt_id:
            continue
        selected = set(attempt.get("selected_keys") or ())
        actual = [_event_key(key) for key in actual_load_plan_keys]
        attempt.update(
            {
                "actual_load_plan_keys": actual,
                "actual_destination_block_ids": (
                    [str(block_id) for block_id in actual_destination_block_ids]
                    if actual_destination_block_ids is not None
                    else None
                ),
                "intersection_keys": [key for key in actual if key in selected],
                "remainder_keys": [key for key in actual if key not in selected],
                "readmission_observed_at_s": time.monotonic(),
                "status": status,
            }
        )
        return


def _set_prefetch_attempt_status(
    state: dict[str, object], attempt_id: str | None, status: str
) -> None:
    if attempt_id is None:
        return
    attribution = _prefetch_attribution_state(state)
    for attempt in attribution["attempts"]:
        if isinstance(attempt, dict) and attempt.get("attempt_id") == attempt_id:
            attempt["status"] = status
            return


def _transfer_event_record(
    job_id: int,
    info: dict[str, object],
    transfer: SemanticTransferCompletion | None,
    *,
    status: str | None = None,
    success: bool | None = None,
    cancelled: bool | None = None,
) -> dict[str, object]:
    """Serialize one request-owned transfer without inferred measurements."""
    is_load = bool(info.get("is_load"))
    transfer_status = transfer.status if transfer is not None else status or "failure"
    transfer_success = transfer.success if transfer is not None else bool(success)
    transfer_cancelled = transfer.cancelled if transfer is not None else bool(cancelled)
    return {
        "job_id": job_id,
        "request_id": info.get("req_id"),
        "is_prefetch": bool(info.get("is_prefetch")),
        "direction": "load" if is_load else "store",
        "status": transfer_status,
        "success": transfer_success,
        "cancelled": transfer_cancelled,
        "contributors": transfer.contributors if transfer is not None else 0,
        "expected_contributors": info.get("expected_contributors"),
        "keys": [
            _event_key(key) for key in sorted(info.get("keys", ()), key=_event_key_sort)
        ],
        "source_block_ids": info.get("src_block_ids"),
        "destination_block_ids": info.get("dst_block_ids"),
        "bytes": transfer.bytes if transfer is not None else None,
        "time_s": transfer.time_s if transfer is not None else None,
        "time_source": (
            "cuda_event_elapsed_time"
            if transfer is not None and transfer.time_s is not None
            else None
        ),
        "max_time_s": transfer.max_time_s if transfer is not None else None,
        "start_time_s": transfer.start_time_s if transfer is not None else None,
        "completion_time_s": (
            transfer.completion_time_s if transfer is not None else None
        ),
        "started_at_ns": transfer.started_at_ns if transfer is not None else None,
        "completed_at_ns": (transfer.completed_at_ns if transfer is not None else None),
        "contributor_ids": (
            list(transfer.contributor_ids) if transfer is not None else []
        ),
    }


def _speculative_transfer_accounting(
    state: dict[str, object], bytes_per_key: int
) -> dict[str, object]:
    """Summarize physical speculative work without treating it as reuse."""
    records = list(state.get("speculative_transfer_job_statuses", []))
    successful = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("status") == "success"
        and record.get("success") is True
        and record.get("cancelled") is False
    ]
    completed_bytes = sum(
        int(record["bytes"])
        for record in successful
        if isinstance(record.get("bytes"), int)
        and not isinstance(record.get("bytes"), bool)
        and int(record["bytes"]) >= 0
    )
    device_times = [
        float(record["time_s"])
        for record in successful
        if isinstance(record.get("time_s"), (int, float))
        and not isinstance(record.get("time_s"), bool)
        and float(record["time_s"]) >= 0.0
    ]
    attempted_blocks = int(state.get("prefetch_attempted_blocks", 0))
    # Legacy/manual request states predate the explicit submission counter;
    # retain their old attempted==submitted interpretation. New producer
    # states always initialize the field, including an authoritative zero.
    submitted_blocks = int(state.get("prefetch_submitted_blocks", attempted_blocks))
    # Attempted reservations, jobs actually handed to worker metadata, and
    # completed worker transfers are distinct lifecycle boundaries.  None of
    # the first two is evidence of physical H2D work.
    submitted_bytes = submitted_blocks * bytes_per_key
    demand_boundary_ns = state.get("prefetch_t4_ns")
    demand_boundary_source = state.get("prefetch_t4_source") or (
        "readmission_allocation"
    )
    if not isinstance(demand_boundary_ns, int) and isinstance(
        state.get("wait_hint"), dict
    ):
        demand_boundary_ns = state.get("connector_observed_ns")
        demand_boundary_source = "hinted_request_connector_observation"
    late_records: list[dict[str, object]] | None = None
    if isinstance(demand_boundary_ns, int) and not isinstance(demand_boundary_ns, bool):
        late_records = [
            record
            for record in successful
            if isinstance(record.get("completed_at_ns"), int)
            and int(record["completed_at_ns"]) > demand_boundary_ns
        ]
        late_bytes = sum(
            int(record.get("bytes", 0))
            for record in late_records
            if isinstance(record.get("bytes"), int)
            and not isinstance(record.get("bytes"), bool)
        )
    else:
        demand_boundary_ns = None
        demand_boundary_source = None
        late_bytes = None
    consumed_bytes = int(state.get("prefetch_consumed_blocks", 0)) * bytes_per_key
    released_reservation_bytes = (
        int(state.get("prefetch_discarded_blocks", 0)) * bytes_per_key
    )
    discarded_bytes = min(
        max(0, completed_bytes - consumed_bytes), released_reservation_bytes
    )
    return {
        "job_count": len(records),
        "successful_job_count": len(successful),
        "failed_job_count": len(records) - len(successful),
        "cancelled_job_count": sum(
            record.get("cancelled") is True
            for record in records
            if isinstance(record, dict)
        ),
        "attempted_blocks": attempted_blocks,
        "attempted_bytes": attempted_blocks * bytes_per_key,
        "submitted_bytes": submitted_bytes,
        "submitted_jobs": int(state.get("prefetch_submitted_jobs", len(records))),
        "submitted_blocks": submitted_blocks,
        "completed_bytes": completed_bytes,
        "consumed_bytes": consumed_bytes,
        "consumption_events": int(consumed_bytes > 0),
        "discarded_bytes": discarded_bytes,
        "late_job_count": len(late_records) if late_records is not None else None,
        "late_bytes": late_bytes,
        "on_time_job_count": (
            len(successful) - len(late_records) if late_records is not None else None
        ),
        "on_time_bytes": (
            completed_bytes - late_bytes if late_bytes is not None else None
        ),
        "demand_boundary_ns": demand_boundary_ns,
        "demand_boundary_source": demand_boundary_source,
        "released_reservation_bytes": released_reservation_bytes,
        "uncompleted_submitted_bytes": max(0, submitted_bytes - completed_bytes),
        "unconsumed_completed_bytes": completed_bytes
        - consumed_bytes
        - discarded_bytes,
        "actual_contributors": sum(
            int(record.get("contributors", 0))
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("contributors"), int)
            and not isinstance(record.get("contributors"), bool)
        ),
        "expected_contributors": sum(
            int(record.get("expected_contributors", 0))
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("expected_contributors"), int)
            and not isinstance(record.get("expected_contributors"), bool)
        ),
        "device_time_s": (
            sum(device_times) if len(device_times) == len(successful) else None
        ),
    }


def _speculative_transfer_cost_accounting(
    state: dict[str, object], bytes_per_key: int, accounting: Mapping[str, object]
) -> dict[str, int]:
    """Count completed speculative writes duplicated by later physical work."""
    successful = [
        record
        for record in state.get("speculative_transfer_job_statuses", [])
        if isinstance(record, dict)
        and record.get("status") == "success"
        and record.get("success") is True
        and record.get("cancelled") is False
    ]
    demand_loaded_keys = {
        str(key)
        for record in state.get("transfer_job_statuses", [])
        if isinstance(record, dict)
        and record.get("direction") == "load"
        and record.get("is_prefetch") is not True
        and record.get("success") is True
        for key in record.get("keys", [])
    }
    seen_speculative_keys: set[str] = set()
    duplicated_jobs = 0
    duplicated_occurrences = 0
    for record in successful:
        record_keys = [str(key) for key in record.get("keys", [])]
        duplicates = [
            key
            for key in record_keys
            if key in demand_loaded_keys or key in seen_speculative_keys
        ]
        duplicated_jobs += int(bool(duplicates))
        duplicated_occurrences += len(duplicates)
        seen_speculative_keys.update(record_keys)
    completed_bytes = int(accounting.get("completed_bytes", 0))
    consumed_bytes = int(accounting.get("consumed_bytes", 0))
    return {
        "duplicated_job_count": duplicated_jobs,
        "duplicated_bytes": duplicated_occurrences * bytes_per_key,
        "waste_completed_bytes": max(0, completed_bytes - consumed_bytes),
    }


def _prefill_cuda_terminal_ready(
    request_id: str,
    state: Mapping[str, object],
    *,
    bounded_prefill_cuda_timing: bool,
) -> bool:
    """Wait for worker timing metadata for every prompt-bearing batch."""
    if not bounded_prefill_cuda_timing:
        return True
    expected = {
        int(episode["scheduler_batch_id"])
        for episode in state.get("admission_episodes", [])
        if isinstance(episode, Mapping)
        and isinstance(episode.get("scheduler_batch_id"), int)
        and not isinstance(episode.get("scheduler_batch_id"), bool)
        and isinstance(episode.get("prompt_tokens_locally_executed"), int)
        and not isinstance(episode.get("prompt_tokens_locally_executed"), bool)
        and int(episode["prompt_tokens_locally_executed"]) > 0
    }
    observed = set()
    for record in state.get("prefill_cuda_events", []):
        if not isinstance(record, Mapping):
            continue
        for attribution in record.get("execution_attribution", []):
            if (
                isinstance(attribution, Mapping)
                and attribution.get("request_id") == request_id
                and isinstance(attribution.get("scheduler_batch_id"), int)
                and not isinstance(attribution.get("scheduler_batch_id"), bool)
                and isinstance(attribution.get("prompt_tokens"), int)
                and not isinstance(attribution.get("prompt_tokens"), bool)
                and int(attribution["prompt_tokens"]) > 0
            ):
                observed.add(int(attribution["scheduler_batch_id"]))
    return expected.issubset(observed)


_PREFILL_CUDA_TERMINAL_TIMEOUT_NS = 1_000_000_000


def _prefill_cuda_terminal_status(
    request_id: str,
    state: Mapping[str, object],
    *,
    bounded_prefill_cuda_timing: bool,
    now_ns: int,
) -> str:
    if _prefill_cuda_terminal_ready(
        request_id,
        state,
        bounded_prefill_cuda_timing=bounded_prefill_cuda_timing,
    ):
        return "ready"
    wait_started_ns = state.get("prefill_cuda_terminal_wait_started_ns")
    if not isinstance(wait_started_ns, int) or isinstance(wait_started_ns, bool):
        wait_started_ns = state.get("request_finished_ns")
    if not isinstance(wait_started_ns, int) or isinstance(wait_started_ns, bool):
        wait_started_ns = now_ns
    return (
        "unavailable"
        if now_ns - wait_started_ns >= _PREFILL_CUDA_TERMINAL_TIMEOUT_NS
        else "waiting"
    )


_CANONICAL_EXTERNAL_ID_RE = re.compile(r"^sc\|([^|]+)\|([^|]+)\|([^|]+)\|sample-(\d+)$")
_CANONICAL_INTERNAL_ID_RE = re.compile(
    r"^cmpl-(sc\|[^|]+\|[^|]+\|[^|]+\|sample-\d+)-\d+-([0-9a-f]{8})$"
)


def _parse_request_identity(external_request_id):
    if not isinstance(external_request_id, str):
        return None, None, None, None
    match = _CANONICAL_EXTERNAL_ID_RE.fullmatch(external_request_id)
    if match is None:
        return None, None, None, None
    return match.group(1), match.group(2), external_request_id, int(match.group(4))


def _derive_external_request_id(request_id) -> str | None:
    """Recover the exact client ID from vLLM's canonical completion ID.

    vLLM may leave ``external_req_id`` unset on the scheduler-side Request
    while retaining it in ``cmpl-<external>-<parallel-index>-<hash>``. This
    parser accepts only the documented ``sc|...|sample-N`` identity shape;
    it never invents an ID for ordinary or needle requests.
    """
    if not isinstance(request_id, str):
        return None
    match = _CANONICAL_INTERNAL_ID_RE.fullmatch(request_id)
    if match is None:
        return None
    external_request_id = match.group(1)
    return (
        external_request_id if _parse_request_identity(external_request_id)[0] else None
    )


def _prompt_token_digest(request: Request) -> str | None:
    """Return a stable digest of server-side prompt tokens when available."""
    tokens = getattr(request, "prompt_token_ids", None)
    if not isinstance(tokens, (list, tuple)) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in tokens
    ):
        return None
    payload = json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _serialize_key_position(record: Mapping[str, object]) -> dict[str, object]:
    serialized = dict(record)
    key = serialized.get("key")
    if key is not None:
        serialized["key"] = _event_key(key)
    return serialized


def _serialize_key_positions(records) -> list[dict[str, object]]:
    return [_serialize_key_position(record) for record in records]


def _serialize_lookup_episodes(episodes):
    serialized = []
    for episode in episodes:
        raw_groups = episode.get("groups", {})
        groups = {}
        for group_idx, group in raw_groups.items():
            groups[group_idx] = {
                name: [
                    _event_key(key)
                    for key in sorted(group.get(name, []), key=_event_key_sort)
                ]
                for name in (
                    "eligible",
                    "queried",
                    "hits",
                    "missing",
                    "pending",
                    "retry",
                )
            }
            groups[group_idx]["eligible_ordered"] = [
                _event_key(key) for key in group.get("eligible", [])
            ]
            groups[group_idx]["ordered_probes"] = [
                _serialize_key_position(probe)
                for probe in group.get("ordered_probes", [])
            ]
            groups[group_idx]["observed"] = bool(group.get("observed", False))
            groups[group_idx]["no_eligible"] = (
                not any(
                    groups[group_idx][name]
                    for name in (
                        "eligible",
                        "queried",
                        "hits",
                        "missing",
                        "pending",
                        "retry",
                    )
                )
                and groups[group_idx]["observed"]
            )
        serialized.append(
            {
                "lookup_id": episode["lookup_id"],
                "episode_id": episode.get("episode_id"),
                "started_at_ns": episode.get("started_at_ns"),
                "completed_at_ns": episode.get("completed_at_ns"),
                "groups": groups,
            }
        )
    return serialized


def _lookup_metadata_complete(episodes) -> bool:
    if not episodes:
        return False
    return all(
        bool(episode.get("groups"))
        and all(group.get("observed") is True for group in episode["groups"].values())
        for episode in episodes
    )


def _physical_block_refs(spec) -> list[tuple[int, int]]:
    """Qualify physical block IDs by KV group to avoid hybrid collisions."""
    raw_block_ids = getattr(spec, "block_ids", None)
    raw_group_sizes = getattr(spec, "group_sizes", None)
    block_ids = (
        [] if raw_block_ids is None else [int(block_id) for block_id in raw_block_ids]
    )
    group_sizes = [] if raw_group_sizes is None else list(raw_group_sizes)
    if isinstance(spec, CPULoadStoreSpec):
        # CPU slot zero contains ordinary KV. Only GPU block zero is null.
        return [(0, block_id) for block_id in block_ids]
    if not group_sizes:
        return [(0, block_id) for block_id in block_ids if block_id]
    refs: list[tuple[int, int]] = []
    offset = 0
    for group_idx, group_size in enumerate(group_sizes):
        for block_id in block_ids[offset : offset + int(group_size)]:
            if block_id:
                refs.append((group_idx, block_id))
        offset += int(group_size)
    return refs


def _physical_block_refs_complete(spec) -> bool:
    raw_block_ids = getattr(spec, "block_ids", None)
    raw_group_sizes = getattr(spec, "group_sizes", None)
    block_ids = [] if raw_block_ids is None else list(raw_block_ids)
    group_sizes = [] if raw_group_sizes is None else list(raw_group_sizes)
    return bool(group_sizes) and sum(int(size) for size in group_sizes) == len(
        block_ids
    )


def _score_request_count(scores: RelevanceScores) -> int:
    return sum(len(requests) for requests in scores.values())


def _reset_prompt_accounting_after_preemption(state: dict) -> None:
    """Advance the episode without replaying a prompt already fully accounted."""
    prompt_complete = int(state["episode_prompt_accounted"]) >= int(
        state["required_tokens"]
    )
    state["episode_id"] = int(state["episode_id"]) + 1
    state["episode_external_tokens"] = 0
    state["readmission_pending"] = True
    if not prompt_complete:
        state["episode_prompt_accounted"] = 0
        state["episode_local_reused_tokens"] = 0


def _reconcile_episode_prompt_cursor_after_admission(
    state: dict,
    local_tokens: int,
    external_tokens: int,
    *,
    first_readmission: bool = False,
) -> None:
    """Lower a stale completed cursor when re-admission restored less KV."""
    required_tokens = int(state["required_tokens"])
    restored_prefix = min(
        required_tokens, max(0, int(local_tokens)) + max(0, int(external_tokens))
    )
    restored_external_tokens = min(required_tokens, max(0, int(external_tokens)))
    current_cursor = int(state["episode_prompt_accounted"])
    if (
        first_readmission
        and current_cursor >= required_tokens
        and restored_prefix < required_tokens
    ):
        state["episode_prompt_accounted"] = restored_prefix
    else:
        state["episode_prompt_accounted"] = max(current_cursor, restored_prefix)
    restored_local_tokens = min(required_tokens, max(0, int(local_tokens)))
    if first_readmission:
        state["episode_local_reused_tokens"] = restored_local_tokens
        state["episode_external_tokens"] = restored_external_tokens
    else:
        state["episode_local_reused_tokens"] = max(
            int(state["episode_local_reused_tokens"]), restored_local_tokens
        )
        state["episode_external_tokens"] = max(
            int(state["episode_external_tokens"]), restored_external_tokens
        )


# Step 1.5 (semantic prefetch, see .claude/docs/semantic-eviction-plan.md and
# issues log entries #23-#24). Bound on total GPU blocks any request's
# speculative prefetch reservation is allowed to hold, summed across all
# concurrently-preempted requests -- the plan's own "≤5% of GPU blocks"
# text, so a burst of preemptions can't starve real scheduling of blocks.
# Disabled by default until the DGX audit shows an aggregate TTFT win without
# increasing preemptions. The feature remains available as a controlled knob.
PREFETCH_BUDGET_FRACTION = 0.0
_PREFETCH_SELECTORS = ("semantic", "request_order", "useful_on_time")


def _ready_resident_prefix(manager, candidate_keys) -> list[OffloadKey]:
    """Return the maximal ready CPU prefix without lookup/touch side effects."""
    resident: list[OffloadKey] = []
    for key in candidate_keys:
        block = manager._policy.get(key)
        if block is None or not block.is_ready:
            break
        resident.append(key)
    return resident


def _local_gpu_prefix(block_pool, candidate_keys) -> tuple[int, list[int], bool]:
    """Inspect the native prefix-cache index without allocating or touching."""
    get_cached_block = getattr(block_pool, "get_cached_block", None)
    if not callable(get_cached_block):
        return 0, [], False
    block_ids: list[int] = []
    for key in candidate_keys:
        cached = get_cached_block(
            get_offload_block_hash(key), [get_offload_group_idx(key)]
        )
        if not cached:
            break
        block_ids.append(int(cached[0].block_id))
    return len(block_ids), block_ids, True


def _any_local_gpu_key(block_pool, keys) -> bool | None:
    """Inspect every exact key without taking refs, including prefix holes.

    None means coverage is unavailable; full-prefix staging fails closed then.
    """
    lookup = getattr(block_pool, "get_cached_block", None)
    if not callable(lookup):
        return None
    return any(
        bool(lookup(get_offload_block_hash(key), [get_offload_group_idx(key)]))
        for key in keys
    )


def _select_prefetch_keys(
    manager,
    candidate_keys: list[OffloadKey],
    budget: int,
    selector: str,
    *,
    selection_context: Mapping[str, object] | None = None,
) -> list[OffloadKey]:
    """Select exact request keys for a bounded prefetch reservation."""
    if budget <= 0:
        return []
    if selector == "request_order":
        return list(candidate_keys[:budget])
    if selector == "semantic":
        context = selection_context or {}
        return manager.top_relevant_keys(
            candidate_keys,
            budget,
            max_age_ns=context.get("score_max_age_ns"),
            now_ns=context.get("now_ns"),
        )
    if selector == "useful_on_time":
        context = selection_context or {}
        result = select_useful_on_time(
            list(candidate_keys),
            budget,
            remaining_wait_ns=context.get("remaining_wait_ns"),
            bytes_per_key=context.get("bytes_per_key"),
            estimated_transfer_ns_per_byte=context.get(
                "estimated_transfer_ns_per_byte"
            ),
            estimated_splice_ns_per_byte=context.get("estimated_splice_ns_per_byte"),
        )
        return list(result.selected_keys)
    raise ValueError(f"Unknown prefetch selector {selector!r}")


@dataclass
class _PrefetchState:
    """Scheduler-side record of one request's in-flight or completed
    speculative prefetch, dispatched through the same real `load_jobs`
    channel as a normal load (verified safe in issues log entry #23) and
    ALSO registered in `req_status.transfer_jobs` like a normal load is --
    see the registration site in `_attempt_prefetch_reservation` for why
    (issues log entry #26: not registering there crashes the engine, and
    registering it is safe because `get_num_new_matched_tokens` already
    defers re-admission while it's non-empty). This struct is our own
    lightweight index into that job (which keys, which GPU block ids) for
    the splice step -- it doesn't duplicate or replace the real
    `TransferJobStatus`/`req_status.transfer_jobs` bookkeeping."""

    job_id: int
    keys: list[OffloadKey]
    gpu_block_ids: list[int]
    attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ObservedPrefix:
    run_id: str
    cell_id: str
    session_id: str
    generation: int
    observed_prefix_id: str
    boundary_tokens: int
    keys: tuple[OffloadKey, ...]
    observed_ns: int
    source_request_id: str


@dataclass(slots=True)
class _WaitPrefetchState:
    lease: WaitHintLease
    source: _ObservedPrefix
    req_context: ReqContext
    status: str = "pending"
    job_id: int | None = None
    keys: tuple[OffloadKey, ...] = ()
    gpu_block_ids: tuple[int, ...] = ()
    pending_contributors: int = 0
    bound_request_id: str | None = None
    source_released: bool = False
    destination_released: bool = False
    publication_candidate_keys: tuple[OffloadKey, ...] = ()
    publication_eligible_keys: tuple[OffloadKey, ...] = ()
    publication_local_gpu_prefix_count: int = 0
    publication_local_gpu_block_ids: tuple[int, ...] = ()
    publication_local_gpu_coverage_available: bool = False
    selection_evidence: dict[str, object] = field(default_factory=dict)
    transfer_record: dict[str, object] | None = None
    transfer_evidence_bound: bool = False
    protection_binding_id: str | None = None
    protection_released: bool = False
    coalescing_accepted_ns: int | None = None
    coalescing_started_ns: int | None = None
    coalescing_complete: bool = False
    coalescing_ended_ns: int | None = None
    coalescing_end_reason: str | None = None
    coalescing_distinct_eligible_sources: int = 0


@dataclass
class SemanticOffloadingConnectorMetadata(OffloadingConnectorMetadata):
    scoring_context: dict = field(default_factory=dict)
    scoring_requests: dict[str, list[OffloadKey]] = field(default_factory=dict)
    prefix_scoring_request: dict = field(default_factory=dict)
    scoring_residency_versions: dict = field(default_factory=dict)
    scoring_lease_ids: dict = field(default_factory=dict)
    store_job_keys: dict[int, list[OffloadKey]] = field(default_factory=dict)
    # Exact key -> ordered source GPU blocks. Unlike store_job_keys, this
    # preserves identity after the base scheduler stores keys in a set.
    store_job_blocks: dict[int, list[tuple[OffloadKey, list[int]]]] = field(
        default_factory=dict
    )
    # splice_id -> (src GPU block ids holding prefetched data, dst GPU block
    # ids the scheduler just officially allocated for the same keys).
    splice_jobs: dict[int, tuple[list[int], list[int]]] = field(default_factory=dict)
    # Real evictions the manager's CachePolicy made this step (issues log
    # entries #62-64) -- lets the worker drop exactly these keys from
    # durable_summaries instead of approximating with a FIFO cap.
    evicted_keys: list[OffloadKey] = field(default_factory=list)
    # issues log entry #27: prefetch loads are dispatched through a SEPARATE
    # channel from `load_jobs`, not merged into it. The stock worker's
    # start_kv_transfers() registers every job in `load_jobs` into
    # self._load_jobs, which makes get_finished() report its completion via
    # `finished_recving` -- a channel the core scheduler only understands
    # for two cases (a request WAITING_FOR_REMOTE_KVS, or already finished).
    # A prefetch's request is PREEMPTED, neither of those, so merging it
    # into `load_jobs` crashes the engine (`assert
    # RequestStatus.is_finished(req.status)`) the moment the prefetch
    # completes -- reproduced on the real server. Keyed by job id, same
    # shape as `load_jobs`; SemanticOffloadingConnectorWorker.start_kv_transfers
    # submits these for real execution without registering them in
    # self._load_jobs, so their completion is still reported (unconditionally,
    # via completed_jobs) but never through finished_recving.
    prefetch_load_jobs: dict[int, TransferJob] = field(default_factory=dict)
    # SchedulerOutput's authoritative preemption set, forwarded to workers so
    # their execution-entry callback can record the first forward *after* a
    # preemption rather than the request's first forward ever.
    preempted_req_ids: tuple[str, ...] = ()
    reset_semantic_state: bool = False
    # One scheduler batch's exact request/token attribution. The worker uses
    # this only for the opt-in bounded CUDA-event measurement below; it does
    # not alter scheduling or transfer behavior.
    execution_attribution: list[dict[str, object]] = field(default_factory=list)


@dataclass
class SemanticWorkerMetadata(OffloadingWorkerMetadata):
    pending_scores: RelevanceScores = field(default_factory=dict)
    failed_jobs: dict[int, int] = field(default_factory=dict)
    relevance_updates: ComposedRelevanceUpdates = field(default_factory=dict)
    latest_relevance: LatestRelevance = field(default_factory=dict)
    score_head_counts: ScoreHeadCounts = field(default_factory=dict)
    score_worker_counts: ScoreWorkerCounts = field(default_factory=dict)
    score_reduction: str | None = None
    score_group_size: int | None = None
    score_contributors: int = 0
    rank_ema_mode: str = "default"
    ema_rank_weight_floor: float = 0.0
    # splice_id -> number of workers that have completed the GPU copy.
    completed_splices: dict[int, int] = field(default_factory=dict)
    completed_splice_evidence: dict[int, list[dict[str, object]]] = field(
        default_factory=dict
    )
    # Per-job transfer contributions, aggregated across tensor-parallel
    # workers. The scheduler joins these with TransferJobStatus.req_id.
    transfer_events: PreservationTransferEvents = field(default_factory=dict)
    # Worker-clock model-runner execution-entry boundaries keyed by request.
    # These are host-side boundaries, not device-kernel timestamps. They are
    # retained for general execution diagnostics; preempted_forward_times_ns
    # is the attribution-safe post-preemption stream.
    first_forward_times_ns: dict[str, int] = field(default_factory=dict)
    # First host model-runner execution-entry boundary after a scheduler
    # preemption notification, keyed by request.
    preempted_forward_times_ns: dict[str, int] = field(default_factory=dict)
    # Bounded worker-side CUDA allocator counter peaks. These are allocator
    # observations, not a proof of physical fragmentation.
    allocator_peak_stats: dict[str, int] = field(default_factory=dict)
    async_scoring_events: list[dict[str, object]] = field(default_factory=list)
    async_scoring_status: dict[str, object] = field(default_factory=dict)
    async_score_observations: list[dict[str, object]] = field(default_factory=list)
    prefix_score_observations: list[dict[str, object]] = field(default_factory=list)
    # CUDA elapsed intervals around complete model-runner executions. Only
    # records explicitly marked attribution_eligible may be interpreted as
    # request-level recomputation time.
    prefill_cuda_events: list[dict[str, object]] = field(default_factory=list)

    def _validate_score_representation(self) -> None:
        if self.pending_scores and self.relevance_updates:
            raise ValueError(
                "Semantic worker metadata cannot contain raw and compact scores"
            )
        if self.async_score_observations and (
            self.pending_scores or self.relevance_updates
        ):
            raise ValueError(
                "Async score observations cannot coexist with legacy score metadata"
            )

    def complete_scores(self) -> RelevanceScores:
        """Return only requests scored by the complete worker group."""
        self._validate_score_representation()
        if (
            self.score_group_size is None
            or self.score_contributors != self.score_group_size
        ):
            return {}
        complete: RelevanceScores = {}
        for method, reqs in self.pending_scores.items():
            complete_reqs = {
                req_id: ranked
                for req_id, ranked in reqs.items()
                if self.score_worker_counts.get(method, {}).get(req_id, 0)
                == self.score_group_size
            }
            if complete_reqs:
                complete[method] = complete_reqs
        return complete

    def complete_relevance_updates(self) -> ComposedRelevanceUpdates:
        """Return compact updates only from a complete single-worker group."""
        self._validate_score_representation()
        if (
            self.score_group_size != 1
            or self.score_contributors != self.score_group_size
        ):
            return {}
        return self.relevance_updates

    def complete_latest_relevance(self) -> LatestRelevance:
        """Return compact raw scores only from a complete single-worker group."""
        self._validate_score_representation()
        if (
            self.score_group_size != 1
            or self.score_contributors != self.score_group_size
        ):
            return {}
        return self.latest_relevance

    def complete_async_score_observations(self) -> list[dict[str, object]]:
        """Return ordered async observations only for the supported TP=1 path."""
        self._validate_score_representation()
        if (
            self.score_group_size != 1
            or self.score_contributors != self.score_group_size
        ):
            return []
        return list(self.async_score_observations)

    def complete_prefix_score_observations(self):
        if self.score_group_size != 1 or self.score_contributors != 1:
            return []
        return list(self.prefix_score_observations)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, OffloadingWorkerMetadata)
        base = super().aggregate(other)
        merged_scores: RelevanceScores = {
            method: {req_id: list(ranked) for req_id, ranked in reqs.items()}
            for method, reqs in self.pending_scores.items()
        }
        merged_updates: ComposedRelevanceUpdates = {
            method: dict(updates) for method, updates in self.relevance_updates.items()
        }
        merged_latest: LatestRelevance = {
            method: dict(scores) for method, scores in self.latest_relevance.items()
        }
        merged_counts: ScoreHeadCounts = {
            method: dict(reqs) for method, reqs in self.score_head_counts.items()
        }
        merged_worker_counts: ScoreWorkerCounts = {
            method: dict(reqs) for method, reqs in self.score_worker_counts.items()
        }
        other_scores = getattr(other, "pending_scores", {})
        other_updates = getattr(other, "relevance_updates", {})
        other_latest = getattr(other, "latest_relevance", {})
        rank_ema_mode = getattr(self, "rank_ema_mode", "default")
        other_rank_ema_mode = getattr(other, "rank_ema_mode", "default")
        if rank_ema_mode != other_rank_ema_mode:
            raise ValueError(
                "Cannot aggregate semantic scores with different "
                f"rank_ema_mode values: {rank_ema_mode!r} and "
                f"{other_rank_ema_mode!r}"
            )
        ema_rank_weight_floor = getattr(self, "ema_rank_weight_floor", 0.0)
        other_ema_rank_weight_floor = getattr(other, "ema_rank_weight_floor", 0.0)
        if ema_rank_weight_floor != other_ema_rank_weight_floor:
            raise ValueError(
                "Cannot aggregate semantic scores with different "
                "ema_rank_weight_floor values: "
                f"{ema_rank_weight_floor!r} and {other_ema_rank_weight_floor!r}"
            )
        if merged_updates and other_updates:
            raise ValueError("Cannot aggregate multiple compact relevance updates")
        if (merged_updates and other_scores) or (other_updates and merged_scores):
            raise ValueError("Cannot aggregate compact and raw relevance metadata")
        if merged_latest and other_latest:
            raise ValueError("Cannot aggregate multiple compact raw score metadata")
        if other_updates:
            merged_updates = {
                method: dict(updates) for method, updates in other_updates.items()
            }
        if other_latest:
            merged_latest = {
                method: dict(scores) for method, scores in other_latest.items()
            }
        other_head_counts = getattr(other, "score_head_counts", {})
        other_worker_counts = getattr(other, "score_worker_counts", {})
        other_reduction = getattr(other, "score_reduction", None)
        other_group_size = getattr(other, "score_group_size", None)
        group_sizes = {
            size
            for size in (self.score_group_size, other_group_size)
            if size is not None
        }
        if len(group_sizes) > 1:
            raise ValueError(
                f"Cannot aggregate semantic score groups of different sizes: "
                f"{sorted(group_sizes)}"
            )
        score_group_size = next(iter(group_sizes), None)
        score_contributors = self.score_contributors + getattr(
            other, "score_contributors", 0
        )
        if self.pending_scores:
            score_reduction = self.score_reduction or "mean"
        elif other_scores:
            score_reduction = other_reduction or "mean"
        else:
            score_reduction = self.score_reduction or other_reduction
        if self.pending_scores and other_scores:
            left_reduction = self.score_reduction or "mean"
            right_reduction = other_reduction or "mean"
            if left_reduction != right_reduction:
                raise ValueError(
                    "Cannot aggregate semantic scores with different head "
                    f"reductions: {left_reduction!r} and {right_reduction!r}"
                )
            score_reduction = left_reduction

        for method, reqs in other_scores.items():
            method_scores = merged_scores.setdefault(method, {})
            method_counts = merged_counts.setdefault(method, {})
            method_worker_counts = merged_worker_counts.setdefault(method, {})
            for req_id, ranked in reqs.items():
                right_count = other_head_counts.get(method, {}).get(req_id, 1)
                right_worker_count = other_worker_counts.get(method, {}).get(req_id, 1)
                if req_id not in method_scores:
                    method_scores[req_id] = list(ranked)
                    method_counts[req_id] = right_count
                    method_worker_counts[req_id] = right_worker_count
                    continue

                left_count = method_counts.get(req_id, 1)
                left_worker_count = method_worker_counts.get(req_id, 1)
                left_scores = dict(method_scores[req_id])
                right_scores = dict(ranked)
                common_keys = left_scores.keys() & right_scores.keys()
                reduction = score_reduction or "mean"
                if reduction == "mean":
                    total_count = left_count + right_count
                    combined = {
                        key: (
                            left_scores[key] * left_count
                            + right_scores[key] * right_count
                        )
                        / total_count
                        for key in common_keys
                    }
                elif reduction == "max":
                    total_count = left_count + right_count
                    combined = {
                        key: max(left_scores[key], right_scores[key])
                        for key in common_keys
                    }
                else:
                    raise ValueError(f"Unknown score reduction: {reduction!r}")
                method_scores[req_id] = sorted(
                    combined.items(), key=lambda item: (-item[1], item[0])
                )
                method_counts[req_id] = total_count
                method_worker_counts[req_id] = left_worker_count + right_worker_count
        completed_splices = dict(self.completed_splices)
        for splice_id, count in getattr(other, "completed_splices", {}).items():
            completed_splices[splice_id] = completed_splices.get(splice_id, 0) + count
        completed_splice_evidence = {
            splice_id: [dict(event) for event in events]
            for splice_id, events in self.completed_splice_evidence.items()
        }
        for splice_id, events in getattr(
            other, "completed_splice_evidence", {}
        ).items():
            completed_splice_evidence.setdefault(splice_id, []).extend(
                dict(event) for event in events
            )
        first_forward_times_ns = dict(self.first_forward_times_ns)
        for req_id, timestamp_ns in getattr(
            other, "first_forward_times_ns", {}
        ).items():
            prior = first_forward_times_ns.get(req_id)
            first_forward_times_ns[req_id] = (
                timestamp_ns if prior is None else min(prior, timestamp_ns)
            )
        preempted_forward_times_ns = dict(
            getattr(self, "preempted_forward_times_ns", {})
        )
        for req_id, timestamp_ns in getattr(
            other, "preempted_forward_times_ns", {}
        ).items():
            prior = preempted_forward_times_ns.get(req_id)
            preempted_forward_times_ns[req_id] = (
                timestamp_ns if prior is None else min(prior, timestamp_ns)
            )
        allocator_peak_stats = dict(self.allocator_peak_stats)
        for name, value in getattr(other, "allocator_peak_stats", {}).items():
            allocator_peak_stats[name] = max(allocator_peak_stats.get(name, 0), value)
        async_scoring_events = list(self.async_scoring_events)
        async_scoring_events.extend(getattr(other, "async_scoring_events", []))
        async_score_observations = list(self.async_score_observations)
        other_async_observations = getattr(other, "async_score_observations", [])
        if async_score_observations and other_async_observations:
            raise ValueError("Cannot aggregate async score observations across workers")
        async_score_observations.extend(other_async_observations)
        prefix_score_observations = list(self.prefix_score_observations)
        other_prefix = getattr(other, 'prefix_score_observations', [])
        if prefix_score_observations and other_prefix:
            raise ValueError('Cannot aggregate prefix scores across workers')
        prefix_score_observations.extend(other_prefix)
        async_scoring_status = dict(self.async_scoring_status)
        other_async_status = getattr(other, "async_scoring_status", {})
        if async_scoring_status and other_async_status:
            if async_scoring_status != other_async_status:
                raise ValueError("Cannot aggregate different async scoring statuses")
        elif other_async_status:
            async_scoring_status = dict(other_async_status)
        transfer_events = dict(self.transfer_events)
        for job_id, event in getattr(other, "transfer_events", {}).items():
            previous = transfer_events.get(job_id)
            if previous is None:
                transfer_events[job_id] = event
                continue
            if previous.is_load != event.is_load:
                raise ValueError(
                    "Cannot aggregate load/store transfer event mismatch for job "
                    f"{job_id}"
                )
            transfer_events[job_id] = SemanticTransferCompletion(
                is_load=event.is_load,
                bytes=(
                    previous.bytes + event.bytes
                    if previous.bytes is not None and event.bytes is not None
                    else None
                ),
                time_s=(
                    previous.time_s + event.time_s
                    if previous.time_s is not None and event.time_s is not None
                    else None
                ),
                max_time_s=(
                    max(previous.max_time_s, event.max_time_s)
                    if previous.max_time_s is not None and event.max_time_s is not None
                    else None
                ),
                contributors=previous.contributors + event.contributors,
                success=previous.success and event.success,
                status=(
                    "cancelled"
                    if previous.cancelled or event.cancelled
                    else "success"
                    if previous.success and event.success
                    else "failure"
                ),
                cancelled=previous.cancelled or event.cancelled,
                start_time_s=_earliest_timestamp(
                    previous.start_time_s, event.start_time_s
                ),
                completion_time_s=_latest_timestamp(
                    previous.completion_time_s, event.completion_time_s
                ),
                started_at_ns=(
                    min(
                        value
                        for value in (previous.started_at_ns, event.started_at_ns)
                        if value is not None
                    )
                    if previous.started_at_ns is not None
                    or event.started_at_ns is not None
                    else None
                ),
                completed_at_ns=(
                    max(
                        value
                        for value in (
                            previous.completed_at_ns,
                            event.completed_at_ns,
                        )
                        if value is not None
                    )
                    if previous.completed_at_ns is not None
                    or event.completed_at_ns is not None
                    else None
                ),
                contributor_ids=tuple(
                    sorted(set(previous.contributor_ids) | set(event.contributor_ids))
                ),
            )
        failed_jobs = dict(getattr(self, "failed_jobs", {}))
        for job_id, count in getattr(other, "failed_jobs", {}).items():
            failed_jobs[job_id] = failed_jobs.get(job_id, 0) + count
        completed_jobs = dict(base.completed_jobs)
        prefill_cuda_events = list(self.prefill_cuda_events)
        other_prefill_cuda_events = list(
            getattr(other, "prefill_cuda_events", []) or []
        )
        if prefill_cuda_events and other_prefill_cuda_events:
            raise ValueError("Bounded prefill CUDA attribution currently requires TP=1")
        prefill_cuda_events.extend(other_prefill_cuda_events)
        return SemanticWorkerMetadata(
            completed_jobs=completed_jobs,
            transfer_stats=base.transfer_stats,
            pending_scores=merged_scores,
            failed_jobs=failed_jobs,
            relevance_updates=merged_updates,
            latest_relevance=merged_latest,
            score_head_counts=merged_counts,
            score_worker_counts=merged_worker_counts,
            score_reduction=score_reduction,
            score_group_size=score_group_size,
            score_contributors=score_contributors,
            rank_ema_mode=rank_ema_mode,
            ema_rank_weight_floor=ema_rank_weight_floor,
            completed_splices=completed_splices,
            completed_splice_evidence=completed_splice_evidence,
            transfer_events=transfer_events,
            first_forward_times_ns=first_forward_times_ns,
            preempted_forward_times_ns=preempted_forward_times_ns,
            allocator_peak_stats=allocator_peak_stats,
            async_scoring_events=async_scoring_events,
            async_scoring_status=async_scoring_status,
            async_score_observations=async_score_observations,
            prefix_score_observations=prefix_score_observations,
            prefill_cuda_events=prefill_cuda_events,
        )


class SemanticOffloadingConnectorScheduler(OffloadingConnectorScheduler):
    def __init__(self, spec, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        # OffloadingConnectorScheduler.__init__ gained vllm_config/
        # kv_cache_config params on some vLLM versions (#48150) to build
        # its self.config via SchedulerOffloadConfig.from_spec(...) instead
        # of deriving it from spec alone -- not present on every checkout
        # this project runs against (confirmed: a rented B200 box's vLLM
        # predates this refactor entirely), so construct via the real
        # base __init__'s own actual signature rather than assuming either
        # shape. See _vllm_compat.py.
        construct_scheduler_base(self, spec, vllm_config, kv_cache_config)
        extra_config = dict(getattr(spec, "extra_config", {}) or {})
        from semantic_offload.spec import validate_preservation_audit

        self._preservation_audit_enabled = validate_preservation_audit(extra_config)
        # Never evict generation tombstones: that would let a delayed old
        # generation register itself again. At capacity, new sessions simply
        # lose advice until reset_cache; normal request serving is unaffected.
        self._observed_session_limit = (
            None if self._preservation_audit_enabled
            else extra_config.get("observed_session_registry_size", 4096)
        )
        self._benchmark_telemetry = extra_config.get("benchmark_telemetry", False)
        if type(self._benchmark_telemetry) is not bool:
            raise TypeError("benchmark_telemetry must be a JSON boolean")
        record_relevance_batch(self, 0)
        self._scoring_schedule = extra_config.get("scoring_schedule", "continuous")
        prefix_mean = extra_config.get('prefix_mean_prefetch_scoring', False)
        if not isinstance(prefix_mean, bool):
            raise TypeError('prefix_mean_prefetch_scoring must be a JSON boolean')
        if prefix_mean and (self._scoring_schedule != 'opportunity'
                or extra_config.get('method', 'mean') != 'mean'
                or extra_config.get('head_aggregation', 'mean') != 'mean'
                or extra_config.get('scoring_mode', 'synchronous') != 'synchronous'
                or extra_config.get('prefetch_selector', 'semantic') != 'semantic'):
            raise ValueError('prefix mean requires opportunity/synchronous mean/mean semantic selection')
        if prefix_mean and getattr(self.config, 'num_workers', None) != 1:
            raise ValueError('prefix mean scoring requires exactly one worker')
        self._prefix_mean_prefetch_scoring = prefix_mean
        self._prefix_query_max_age_ns = int(extra_config.get('query_snapshot_max_age_ms', 1000)) * 1_000_000
        self._prefix_mean_score_batch = {}
        self._prefix_mean_scoring_request = {}
        if self._scoring_schedule == "opportunity" and (
            extra_config.get("eviction_policy", "lru") != "arc"
            or not extra_config.get("shared_prefetch_scorer", False)
        ):
            raise ValueError("opportunity scoring is an ARC shared-prefetch experiment")
        self._score_requires_bounded_contract = extra_config.get('scoring_mode') == 'asynchronous'
        self._score_transport_enabled = self._score_requires_bounded_contract or self._scoring_schedule == 'opportunity'
        self._score_source_tokens = {}
        self._score_source_deadlines = {}
        self._score_source_sequence = 0
        self._score_transport_identity = uuid.uuid4().hex
        self._score_transport_generation = 0
        self._score_source_lease_ns = (min(extra_config.get('async_scoring_max_snapshot_age_ms',1000), extra_config.get('query_snapshot_max_age_ms',1000)) if self._scoring_schedule == 'opportunity' else extra_config.get('async_scoring_max_snapshot_age_ms',1000))*1_000_000
        self._decision_scoring_requests = {}
        self._decision_scoring_attempts = {}
        bounded_prefill_cuda_timing = extra_config.get(
            "bounded_prefill_cuda_timing", False
        )
        if not isinstance(bounded_prefill_cuda_timing, bool):
            raise TypeError("bounded_prefill_cuda_timing must be a JSON boolean")
        self._bounded_prefill_cuda_timing = bounded_prefill_cuda_timing
        lifecycle_enabled = extra_config.get("lifecycle_release_enabled", False)
        if not isinstance(lifecycle_enabled, bool):
            raise TypeError("lifecycle_release_enabled must be a JSON boolean")
        lifecycle_release_timing = extra_config.get(
            "lifecycle_release_timing", "inline"
        )
        if lifecycle_release_timing not in ("inline", "scheduler_boundary"):
            raise ValueError(
                "lifecycle_release_timing must be inline or scheduler_boundary"
            )
        self._lifecycle_release_enabled = lifecycle_enabled
        self._lifecycle_release_timing = lifecycle_release_timing
        self._lifecycle_controller = LifecycleReleaseController(
            self.manager, enabled=lifecycle_enabled
        )
        self._lifecycle_request_params: dict[str, object] = {}
        self._pending_lifecycle_releases: dict[str, dict[str, object]] = {}
        self._replicated_layout = bool(getattr(spec, "replicated_layout", False))
        self._pending_job_keys: dict[int, list[OffloadKey]] = {}
        self._pending_job_blocks: dict[int, list[tuple[OffloadKey, list[int]]]] = {}
        self._worker_reset_pending = False
        self._reset_prefetch_releases: dict[int, tuple[list[int], int]] = {}
        fraction = float(
            getattr(spec, "extra_config", {}).get(
                "prefetch_budget_fraction", PREFETCH_BUDGET_FRACTION
            )
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("prefetch_budget_fraction must be between 0 and 1")
        self._prefetch_budget_fraction = fraction
        headroom = int(
            getattr(spec, "extra_config", {}).get("prefetch_headroom_blocks", 0)
        )
        if headroom < 0:
            raise ValueError("prefetch_headroom_blocks must be non-negative")
        self._prefetch_headroom_blocks = headroom
        preserve_gpu_prefix_cache = extra_config.get(
            "prefetch_preserve_gpu_prefix_cache", False
        )
        if not isinstance(preserve_gpu_prefix_cache, bool):
            raise TypeError("prefetch_preserve_gpu_prefix_cache must be a JSON boolean")
        self._prefetch_preserve_gpu_prefix_cache = preserve_gpu_prefix_cache
        selector = str(
            getattr(spec, "extra_config", {}).get("prefetch_selector", "semantic")
        )
        if selector not in _PREFETCH_SELECTORS:
            raise ValueError(
                "prefetch_selector must be one of: " + ", ".join(_PREFETCH_SELECTORS)
            )
        self._prefetch_selector = selector
        score_max_age_ms = getattr(spec, "extra_config", {}).get(
            "prefetch_score_max_age_ms", 1000
        )
        if (
            isinstance(score_max_age_ms, bool)
            or not isinstance(score_max_age_ms, int)
            or score_max_age_ms <= 0
        ):
            raise ValueError("prefetch_score_max_age_ms must be a positive integer")
        self._prefetch_score_max_age_ns = score_max_age_ms * 1_000_000
        fallback_bandwidth_gbps = getattr(spec, "extra_config", {}).get(
            "prefetch_fallback_bandwidth_gbps", 16.0
        )
        if (
            isinstance(fallback_bandwidth_gbps, bool)
            or not isinstance(fallback_bandwidth_gbps, (int, float))
            or not math.isfinite(float(fallback_bandwidth_gbps))
            or not 0.0 < float(fallback_bandwidth_gbps)
        ):
            raise ValueError("prefetch_fallback_bandwidth_gbps must be positive")
        self._prefetch_fallback_ns_per_byte = 1.0 / float(fallback_bandwidth_gbps)
        fallback_splice_bandwidth_gbps = getattr(spec, "extra_config", {}).get(
            "prefetch_fallback_splice_bandwidth_gbps", 16.0
        )
        if (
            isinstance(fallback_splice_bandwidth_gbps, bool)
            or not isinstance(fallback_splice_bandwidth_gbps, (int, float))
            or not math.isfinite(float(fallback_splice_bandwidth_gbps))
            or not 0.0 < float(fallback_splice_bandwidth_gbps)
        ):
            raise ValueError("prefetch_fallback_splice_bandwidth_gbps must be positive")
        self._prefetch_fallback_splice_ns_per_byte = 1.0 / float(
            fallback_splice_bandwidth_gbps
        )
        self._prefetch_transfer_ns_per_byte: float | None = None
        self._prefetch_cost_available_ns: int | None = None
        (
            self._prefetch_bytes_per_key,
            self._prefetch_allocation_bytes_per_key,
        ) = scheduler_key_byte_layout(self.config, spec)
        prefetch_trigger = str(
            getattr(spec, "extra_config", {}).get("prefetch_trigger", "preemption")
        )
        if prefetch_trigger not in {"preemption", "known_prefix_wait_hint"}:
            raise ValueError(
                "prefetch_trigger must be preemption or known_prefix_wait_hint"
            )
        self._prefetch_trigger = prefetch_trigger
        diagnostic_no_dispatch = extra_config.get(
            "wait_hint_diagnostic_no_dispatch", False
        )
        if not isinstance(diagnostic_no_dispatch, bool):
            raise TypeError("wait_hint_diagnostic_no_dispatch must be a JSON boolean")
        self._wait_hint_diagnostic_no_dispatch = diagnostic_no_dispatch
        native_adoption = extra_config.get("wait_hint_native_adoption", False)
        if not isinstance(native_adoption, bool):
            raise TypeError("wait_hint_native_adoption must be a JSON boolean")
        self._wait_hint_native_adoption = native_adoption
        for option in ("wait_hint_coalesce_exact_source", "prefetch_require_full_prefix"):
            value = extra_config.get(option, False)
            if not isinstance(value, bool):
                raise TypeError(f"{option} must be a JSON boolean")
            setattr(self, "_" + option, value)
        groups = getattr(kv_cache_config, "kv_cache_groups", ())
        dense_spec = groups[0].kv_cache_spec if len(groups) == 1 else None
        self._wait_hint_native_adoption_layout_supported = (
            native_adoption
            and type(dense_spec) is FullAttentionSpec
            and dense_spec.dtype in (torch.bfloat16, torch.float16)
            and getattr(getattr(dense_spec, "kv_quant_mode", None), "name", None)
            == "NONE"
            and dense_spec.sliding_window is None
            and dense_spec.attention_chunk_size is None
            and not dense_spec.non_causal
        )
        completed_transcript = extra_config.get("wait_hint_completed_transcript", False)
        if not isinstance(completed_transcript, bool):
            raise TypeError("wait_hint_completed_transcript must be a JSON boolean")
        if completed_transcript and (
            not self._wait_hint_native_adoption_layout_supported
            or self.config.num_workers != 1
            or len(self.config.kv_group_configs) != 1
            or config_blocks_per_chunk(self.config) != 1
            or getattr(self.config.kv_group_configs[0], "is_eagle_group", False)
            or self.config.offload_prompt_only
        ):
            raise ValueError(
                "wait_hint_completed_transcript requires dense TP1 native adoption "
                "and offload_prompt_only=false"
            )
        self._wait_hint_completed_transcript = completed_transcript
        self._published_wait_prefixes: dict[str, dict[str, object]] = {}
        soft_retention = str(
            getattr(spec, "extra_config", {}).get("soft_retention", "off")
        )
        if soft_retention not in {"off", "bounded_hint"}:
            raise ValueError("soft_retention must be off or bounded_hint")
        self._soft_retention = soft_retention
        retain_only = extra_config.get("wait_hint_cpu_retention_only", False)
        if not isinstance(retain_only, bool):
            raise TypeError("wait_hint_cpu_retention_only must be a JSON boolean")
        if retain_only and self._prefetch_trigger != "known_prefix_wait_hint":
            raise ValueError("CPU retention-only requires known-prefix wait hints")
        self._wait_hint_cpu_retention_only = retain_only
        gpu_aware_retention = extra_config.get(
            "wait_hint_cpu_retention_gpu_aware", False
        )
        if not isinstance(gpu_aware_retention, bool):
            raise TypeError(
                "wait_hint_cpu_retention_gpu_aware must be a JSON boolean"
            )
        if gpu_aware_retention and not retain_only:
            raise ValueError("GPU-aware CPU retention requires CPU retention-only")
        gpu_gate = extra_config.get("wait_hint_cpu_retention_gpu_gate", False)
        if not isinstance(gpu_gate, bool):
            raise TypeError(
                "wait_hint_cpu_retention_gpu_gate must be a JSON boolean"
            )
        if gpu_gate and not retain_only:
            raise ValueError("GPU-state gating requires CPU retention-only")
        if gpu_gate and gpu_aware_retention:
            raise ValueError(
                "GPU-state gating cannot be combined with CPU suffix bridging"
            )
        if (gpu_aware_retention or gpu_gate) and (
            not self._wait_hint_native_adoption_layout_supported
            or self.config.num_workers != 1
            or len(self.config.kv_group_configs) != 1
            or config_blocks_per_chunk(self.config) != 1
            or self.config.kv_group_configs[0].group_idx != 0
            or group_config_tokens_per_chunk(self.config.kv_group_configs[0])
            != group_config_tokens_per_block(self.config.kv_group_configs[0])
            or getattr(self.config.kv_group_configs[0], "is_eagle_group", False)
        ):
            feature = "GPU-state gating" if gpu_gate else "GPU-aware CPU retention"
            raise ValueError(
                f"{feature} requires dense TP1 single-block native adoption layout"
            )
        self._wait_hint_cpu_retention_gpu_aware = gpu_aware_retention
        self._wait_hint_cpu_retention_gpu_gate = gpu_gate
        staging_mode = str(
            getattr(spec, "extra_config", {}).get("staging_mode", "dynamic_scratch")
        )
        if staging_mode not in {"dynamic_scratch", "owned_reclaimable"}:
            raise ValueError(
                "staging_mode must be dynamic_scratch or owned_reclaimable"
            )
        self._staging_mode = staging_mode
        wait_hint_ttl_ms = getattr(spec, "extra_config", {}).get(
            "wait_hint_ttl_ms", 2000
        )
        if (
            isinstance(wait_hint_ttl_ms, bool)
            or not isinstance(wait_hint_ttl_ms, int)
            or wait_hint_ttl_ms <= 0
        ):
            raise ValueError("wait_hint_ttl_ms must be a positive integer")
        self._wait_hint_ttl_ns = wait_hint_ttl_ms * 1_000_000
        coalescing_min_sources = getattr(spec, "extra_config", {}).get(
            "wait_hint_coalescing_min_sources", 1
        )
        if (
            isinstance(coalescing_min_sources, bool)
            or not isinstance(coalescing_min_sources, int)
            or coalescing_min_sources < 1
        ):
            raise ValueError(
                "wait_hint_coalescing_min_sources must be a positive integer"
            )
        coalescing_max_ms = getattr(spec, "extra_config", {}).get(
            "wait_hint_coalescing_max_ms", 0
        )
        if (
            isinstance(coalescing_max_ms, bool)
            or not isinstance(coalescing_max_ms, int)
            or not 0 <= coalescing_max_ms <= 1000
        ):
            raise ValueError(
                "wait_hint_coalescing_max_ms must be an integer from 0 to 1000"
            )
        if coalescing_min_sources > 1 and coalescing_max_ms == 0:
            raise ValueError(
                "wait_hint_coalescing_max_ms must be positive when "
                "wait_hint_coalescing_min_sources is greater than 1"
            )
        self._wait_hint_coalescing_min_sources = coalescing_min_sources
        self._wait_hint_coalescing_max_ns = coalescing_max_ms * 1_000_000
        self._wait_hint_coalescing_started_ns: int | None = None
        observed_prefix_limit = getattr(spec, "extra_config", {}).get(
            "observed_prefix_registry_size", 1024
        )
        if (
            isinstance(observed_prefix_limit, bool)
            or not isinstance(observed_prefix_limit, int)
            or observed_prefix_limit <= 0
        ):
            raise ValueError("observed_prefix_registry_size must be a positive integer")
        self._observed_prefix_limit = observed_prefix_limit
        # Step 1.5 (semantic prefetch) needs a live reference to the
        # scheduler's GPU BlockPool to reserve a small budget of blocks
        # ahead of a preempted request's official resumption. Empirically
        # verified safe on the real server (issues log entry #23): calling
        # `gpu_block_pool.get_new_blocks(n)`/`free_blocks(...)` from here
        # correctly moves the shared `free_block_queue` count that
        # `KVCacheManager`'s own admission checks read -- no desync, no
        # corruption, confirmed under real concurrent request load. This
        # reference is the real feature's foundation; the verification
        # probe itself (reserve N, assert count, release, assert restored)
        # was temporary and has been removed after confirming the result.
        self.gpu_block_pool: BlockPool | None = None
        # req_id -> in-flight-or-ready prefetch. See _PrefetchState's
        # docstring for why this is separate from req_status.transfer_jobs.
        self._prefetched: dict[str, _PrefetchState] = {}
        self._observed_prefixes: OrderedDict[
            tuple[str, str, str, int, str, int], _ObservedPrefix
        ] = OrderedDict()
        self._observed_prefix_generations: dict[tuple[str, str, str], int] = {}
        self._wait_prefetch: dict[str, _WaitPrefetchState] = {}
        self._wait_prefetch_jobs: dict[int, _WaitPrefetchState] = {}
        self._wait_hint_results: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._wait_hint_result_limit = 4096
        self._wait_hint_request_bindings: dict[str, str] = {}
        namespace = getattr(spec, "extra_config", {}).get("admission_namespace", {})
        self._wait_hint_namespace = (
            dict(namespace) if isinstance(namespace, dict) else {}
        )
        self._prefetch_reserved_blocks = 0
        self._pending_splice_jobs: dict[int, tuple[list[int], list[int]]] = {}
        self._pending_splice_releases: dict[int, tuple[list[int], int]] = {}
        self._pending_splice_protection_ids: dict[int, str] = {}
        # splice_id -> request awaiting the standard async-load completion
        # signal. A splice replaces the ordinary load job, but core still
        # moves that request to WAITING_FOR_REMOTE_KVS because the connector
        # reported an external hit. Promote it only after every worker has
        # acknowledged the GPU copy fence.
        self._pending_splice_request_ids: dict[int, str] = {}
        self._splice_counter = 0
        # Step 1.5 retry (issues log entry #25): on_request_preempted only
        # fires ONCE, at the instant a request is preempted -- entry #24
        # found that instant structurally has zero free GPU blocks under
        # every sustained-pressure workload tested (vLLM frees just enough
        # blocks for whatever admission triggered the preemption, never
        # more). A request that failed its first reservation attempt stays
        # here and gets retried every subsequent scheduling step (from
        # build_connector_meta, which fires unconditionally every step)
        # until it either succeeds, gets re-admitted (update_state_after_alloc
        # clears it), or finishes/aborts (request_finished clears it) --
        # giving slack that appears *after* the preemption instant (e.g. an
        # unrelated request completing a moment later) a real chance to be
        # used, instead of only the single worst possible instant.
        self._preempted_pending: set[str] = set()
        # Fix A's dispatch-side counterpart: prefetch jobs are batched
        # separately from self._current_batch_load_jobs (which becomes
        # `load_jobs` -- see SemanticOffloadingConnectorMetadata.prefetch_load_jobs
        # for why), reset every step in build_connector_meta the same way.
        self._current_batch_prefetch_load_jobs: dict[int, TransferJob] = {}
        self._preservation_events_enabled = bool(
            getattr(spec, "extra_config", {}).get("preservation_events", False)
        )
        if (
            self._prefetch_trigger == "known_prefix_wait_hint"
            and not self._preservation_events_enabled
        ):
            raise ValueError("known_prefix_wait_hint requires preservation_events")
        self._decision_feature_trace_enabled = bool(
            getattr(spec, "extra_config", {}).get("decision_feature_trace", False)
        )
        self._preservation_requests: dict[str, dict[str, object]] = {}
        self._preservation_finished: dict[str, dict[str, object]] = {}
        self._chain_admission_global_accounting: dict[str, int] = {}
        self._preservation_jobs: dict[int, dict[str, object]] = {}
        self._preservation_job_events: PreservationTransferEvents = {}
        self._preservation_scheduler_batch_id = 0
        self._allocator_peak_stats: dict[str, int] = {}
        self._preservation_known_stored_keys: set[OffloadKey] = set()
        self._preservation_evicted_keys: set[OffloadKey] = set()
        self._preservation_event_sequence = 0
        self._preservation_key_commits: dict[OffloadKey, list[dict[str, object]]] = (
            defaultdict(list)
        )
        self._preservation_key_evictions: dict[OffloadKey, list[dict[str, object]]] = (
            defaultdict(list)
        )
        model_config = getattr(vllm_config, "model_config", None)
        implementation_revision = getattr(spec, "extra_config", {}).get(
            "implementation_revision"
        )
        self._prefetch_provenance = {
            "model": getattr(model_config, "model", None),
            "vllm_version": getattr(__import__("vllm"), "__version__", None),
            "extension": "semantic_offload",
            # The extension is not installed with a runtime revision marker in
            # every checkout. Keep that absence explicit for the harness.
            "extension_version": implementation_revision,
            "config": {
                "extra_config": dict(getattr(spec, "extra_config", {}) or {}),
                "cache_block_size": getattr(
                    getattr(vllm_config, "cache_config", None), "block_size", None
                ),
                "kv_group_count": len(getattr(self.config, "kv_group_configs", ())),
                "blocks_per_chunk": config_blocks_per_chunk(self.config),
                "transfer_bytes_per_key": self._prefetch_bytes_per_key,
                "allocation_bytes_per_key": self._prefetch_allocation_bytes_per_key,
                "kv_groups": [
                    {
                        "group_idx": int(getattr(group, "group_idx", index)),
                        "tokens_per_block": group_config_tokens_per_block(group),
                        "tokens_per_chunk": group_config_tokens_per_chunk(group),
                        "sliding_window_size_in_chunks": getattr(
                            group, "sliding_window_size_in_chunks", None
                        ),
                    }
                    for index, group in enumerate(
                        getattr(self.config, "kv_group_configs", ())
                    )
                ],
            },
            "unavailable_fields": (
                ["full_vllm_config"]
                if implementation_revision
                else ["extension_version", "full_vllm_config"]
            ),
        }

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self.gpu_block_pool = gpu_block_pool

    def _record_wait_hint_event(self, event: dict) -> None:
        if getattr(self, "_preservation_audit_enabled", True):
            record_wait_hint_event(event)

    def _session_registry_full(self, identity: tuple[str, str, str]) -> bool:
        limit = getattr(self, "_observed_session_limit", None)
        return (limit is not None
                and identity not in self._observed_prefix_generations
                and len(self._observed_prefix_generations) >= limit)

    def _store_wait_hint_result(
        self,
        hint: KnownPrefixWaitHint | None,
        *,
        status: str,
        reason: str,
        now_ns: int,
        **fields: object,
    ) -> dict[str, object]:
        event_id = (
            hint.event_id if hint is not None else str(fields.get("event_id", ""))
        )
        result = {
            "status": status,
            "reason": reason,
            "event_id": event_id,
            "observed_at_ns": now_ns,
            **(
                {
                    "run_id": hint.run_id,
                    "cell_id": hint.cell_id,
                    "session_id": hint.session_id,
                    "generation": hint.generation,
                }
                if hint is not None
                else {}
            ),
            **fields,
        }
        if event_id:
            self._wait_hint_results[event_id] = dict(result)
            self._wait_hint_results.move_to_end(event_id)
            while len(self._wait_hint_results) > self._wait_hint_result_limit:
                self._wait_hint_results.popitem(last=False)
        self._record_wait_hint_event(
            {
                "schema": "semantic-known-prefix-wait-hint-event-v1",
                "event": "hint_result",
                **result,
            }
        )
        return result

    @staticmethod
    def _wait_state_identity(state: _WaitPrefetchState) -> dict[str, object]:
        hint = state.lease.hint
        return {
            "run_id": hint.run_id,
            "cell_id": hint.cell_id,
            "session_id": hint.session_id,
            "generation": hint.generation,
        }

    def _retire_wait_prefetch(
        self,
        state: _WaitPrefetchState,
        reason: str,
        *,
        now_ns: int,
    ) -> None:
        """Retire one hint without recycling a live DMA destination."""
        if state.bound_request_id is not None:
            self._wait_hint_request_bindings.pop(state.bound_request_id, None)
            state.bound_request_id = None
        job_live = state.job_id is not None and state.job_id in getattr(
            self, "_wait_prefetch_jobs", {}
        )
        if job_live:
            if state.status != "retired_pending_fence":
                state.status = "retired_pending_fence"
                state.lease.expire()
                self._record_wait_hint_event(
                    {
                        "schema": "semantic-known-prefix-wait-hint-event-v1",
                        "event": "staging_retired_pending_fence",
                        "event_id": state.lease.hint.event_id,
                        **self._wait_state_identity(state),
                        "job_id": state.job_id,
                        "reason": reason,
                        "retired_at_ns": now_ns,
                    }
                )
            return
        self._free_wait_prefetch(state, reason)

    def _retire_superseded_wait_hints(
        self,
        *,
        run_id: str,
        cell_id: str,
        session_id: str,
        generation: int,
        now_ns: int,
    ) -> None:
        for state in list(getattr(self, "_wait_prefetch", {}).values()):
            hint = state.lease.hint
            if (
                hint.run_id == run_id
                and hint.cell_id == cell_id
                and hint.session_id == session_id
                and hint.generation < generation
            ):
                self._retire_wait_prefetch(
                    state, "generation_superseded", now_ns=now_ns
                )

    def _register_observed_prefix(self, request: Request) -> None:
        params = request.kv_transfer_params or {}
        observed_prefix_id = params.get("observed_prefix_id")
        raw_boundary = params.get("observed_prefix_boundary_tokens")
        run_id = params.get("run_id")
        cell_id = params.get("cell_id")
        session_id = params.get("session_id")
        generation = params.get("generation")
        if observed_prefix_id is None and raw_boundary is None:
            return
        if (
            not isinstance(observed_prefix_id, str)
            or not isinstance(raw_boundary, int)
            or isinstance(raw_boundary, bool)
            or raw_boundary <= 0
            or not isinstance(run_id, str)
            or not run_id
            or not isinstance(cell_id, str)
            or not cell_id
            or not isinstance(session_id, str)
            or not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 0
            or len(self.config.kv_group_configs) != 1
        ):
            self._record_wait_hint_event(
                {
                    "schema": "semantic-known-prefix-wait-hint-event-v1",
                    "event": "observed_prefix_rejected",
                    "reason": "invalid_observed_prefix_fields",
                    "request_id": request.request_id,
                }
            )
            return
        group_config = self.config.kv_group_configs[0]
        tokens_per_chunk = group_config_tokens_per_chunk(group_config)
        boundary = raw_boundary // tokens_per_chunk * tokens_per_chunk
        prompt_tokens = list(getattr(request, "prompt_token_ids", ()) or ())
        if boundary <= 0 or boundary > len(prompt_tokens):
            return
        expected_prefix_id = _observed_prefix_digest(
            [int(token) for token in prompt_tokens[:boundary]],
            self._wait_hint_namespace,
        )
        req_status = self._req_status.get(request.request_id)
        group_states = getattr(req_status, "group_states", ()) if req_status else ()
        group_keys = (
            list(getattr(group_states[0], "offload_keys", ()) or ())
            if len(group_states) == 1
            else []
        )
        required_chunks = boundary // tokens_per_chunk
        if (
            observed_prefix_id != expected_prefix_id
            or len(group_keys) < required_chunks
        ):
            self._record_wait_hint_event(
                {
                    "schema": "semantic-known-prefix-wait-hint-event-v1",
                    "event": "observed_prefix_rejected",
                    "reason": "prefix_identity_or_key_coverage_mismatch",
                    "request_id": request.request_id,
                }
            )
            return
        observed_ns = time.monotonic_ns()
        generation_identity = (run_id, cell_id, session_id)
        if self._session_registry_full(generation_identity):
            self._record_wait_hint_event({
                "schema": "semantic-known-prefix-wait-hint-event-v1",
                "event": "observed_prefix_rejected",
                "reason": "session_registry_capacity",
                "request_id": request.request_id,
            })
            return
        current_generation = self._observed_prefix_generations.get(generation_identity)
        if current_generation is not None and generation < current_generation:
            self._record_wait_hint_event(
                {
                    "schema": "semantic-known-prefix-wait-hint-event-v1",
                    "event": "observed_prefix_rejected",
                    "reason": "superseded_generation",
                    "request_id": request.request_id,
                    "run_id": run_id,
                    "cell_id": cell_id,
                    "session_id": session_id,
                    "generation": generation,
                    "current_generation": current_generation,
                    "observed_at_ns": observed_ns,
                }
            )
            return
        if current_generation is None or generation > current_generation:
            for identity in list(self._observed_prefixes):
                if identity[:3] == generation_identity and identity[3] < generation:
                    self._observed_prefixes.pop(identity, None)
            self._retire_superseded_wait_hints(
                run_id=run_id,
                cell_id=cell_id,
                session_id=session_id,
                generation=generation,
                now_ns=observed_ns,
            )
            self._observed_prefix_generations[generation_identity] = generation
        record = _ObservedPrefix(
            run_id=run_id,
            cell_id=cell_id,
            session_id=session_id,
            generation=generation,
            observed_prefix_id=observed_prefix_id,
            boundary_tokens=boundary,
            keys=tuple(group_keys[:required_chunks]),
            observed_ns=observed_ns,
            source_request_id=request.request_id,
        )
        self._save_observed_prefix(record)
        if getattr(self, "_wait_hint_completed_transcript", False):
            self._register_completed_transcript(request, record)

    def _save_observed_prefix(
        self, record: _ObservedPrefix, *, provenance: str | None = None
    ) -> None:
        identity = (
            record.run_id, record.cell_id, record.session_id, record.generation,
            record.observed_prefix_id, record.boundary_tokens,
        )
        self._observed_prefixes[identity] = record
        self._observed_prefixes.move_to_end(identity)
        while len(self._observed_prefixes) > self._observed_prefix_limit:
            self._observed_prefixes.popitem(last=False)
        self._record_wait_hint_event(
            {
                "schema": "semantic-known-prefix-wait-hint-event-v1",
                "event": "observed_prefix_registered",
                "request_id": record.source_request_id,
                "run_id": record.run_id,
                "cell_id": record.cell_id,
                "session_id": record.session_id,
                "generation": record.generation,
                "observed_prefix_id": record.observed_prefix_id,
                "prefix_boundary_tokens": record.boundary_tokens,
                "key_count": len(record.keys),
                "observed_at_ns": record.observed_ns,
                **({"provenance": provenance} if provenance is not None else {}),
            }
        )

    def _register_completed_transcript(
        self, request: Request, declared: _ObservedPrefix
    ) -> None:
        """Add an exact source containing only already computed output KV.

        Called only after the declared input source passes its existing
        identity/generation checks. The final sampled token can be present in
        all_token_ids without ever having been executed; logical token length
        is therefore never used as evidence of computed KV.
        """
        if getattr(getattr(request, "status", None), "name", None) not in {
            "FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED",
        }:
            return
        req_status = self._req_status.get(request.request_id)
        computed = getattr(request, "num_computed_tokens", None)
        if req_status is None or type(computed) is not int or computed < 0:
            return
        group = self.config.kv_group_configs[0]
        chunk_tokens = group_config_tokens_per_chunk(group)
        computed = self._calc_num_offloadable_tokens(req_status, computed)
        boundary = computed // chunk_tokens * chunk_tokens
        if boundary <= request.num_prompt_tokens or boundary > len(request.all_token_ids):
            return
        req_status.update_offload_keys()
        count = boundary // chunk_tokens
        keys = tuple(req_status.group_states[0].offload_keys[:count])
        expected_keys = tuple(make_offload_key(h, 0) for h in request.block_hashes[:count])
        if len(keys) != count or keys != expected_keys or len(set(keys)) != count:
            return
        record = _ObservedPrefix(
            run_id=declared.run_id, cell_id=declared.cell_id,
            session_id=declared.session_id, generation=declared.generation,
            observed_prefix_id=_observed_prefix_digest(
                list(request.all_token_ids[:boundary]), self._wait_hint_namespace,
            ),
            boundary_tokens=boundary, keys=keys, observed_ns=time.monotonic_ns(),
            source_request_id=request.request_id,
        )
        self._save_observed_prefix(record, provenance="completed_computed_transcript")

    def publish_wait_hint(self, raw_hint: dict[str, object]) -> dict[str, object]:
        """Validate and queue a pre-request hint on the scheduler boundary."""
        now_ns = time.monotonic_ns()
        try:
            hint = KnownPrefixWaitHint.from_mapping(raw_hint)
        except (TypeError, ValueError) as exc:
            return self._store_wait_hint_result(
                None,
                status="rejected",
                reason="invalid_hint",
                now_ns=now_ns,
                event_id=(
                    raw_hint.get("event_id", "") if isinstance(raw_hint, dict) else ""
                ),
                detail=str(exc),
            )
        previous = self._wait_hint_results.get(hint.event_id)
        if previous is not None:
            current_generation = self._observed_prefix_generations.get(
                (hint.run_id, hint.cell_id, hint.session_id)
            )
            if current_generation is not None and current_generation != hint.generation:
                return {
                    "status": "rejected",
                    "reason": "superseded_generation",
                    "event_id": hint.event_id,
                    "run_id": hint.run_id,
                    "cell_id": hint.cell_id,
                    "session_id": hint.session_id,
                    "generation": hint.generation,
                    "current_generation": current_generation,
                    "observed_at_ns": now_ns,
                    "idempotent_replay": True,
                }
            return {**previous, "idempotent_replay": True}
        if (
            self._prefetch_trigger != "known_prefix_wait_hint"
            or self._prefetch_budget_fraction == 0
            or _DISABLE_PREFETCH
        ):
            return self._store_wait_hint_result(
                hint,
                status="suppressed",
                reason="wait_hint_prefetch_disabled",
                now_ns=now_ns,
            )
        if hint.namespace_dict() != self._wait_hint_namespace:
            return self._store_wait_hint_result(
                hint,
                status="rejected",
                reason="namespace_mismatch",
                now_ns=now_ns,
            )
        if hint.declared_wait_ms is None:
            return self._store_wait_hint_result(
                hint,
                status="no_action",
                reason="wait_estimate_unavailable",
                now_ns=now_ns,
            )
        if len(self.config.kv_group_configs) != 1:
            return self._store_wait_hint_result(
                hint,
                status="no_action",
                reason="dense_single_group_required",
                now_ns=now_ns,
            )
        boundary = hint.complete_page_boundary(
            group_config_tokens_per_chunk(self.config.kv_group_configs[0])
        )
        if self._session_registry_full((hint.run_id, hint.cell_id, hint.session_id)):
            return self._store_wait_hint_result(
                hint, status="no_action", reason="session_registry_capacity",
                now_ns=now_ns,
            )
        nomination_deadline_ns = (
            hint.input_publication_ns + hint.declared_wait_ms * 1_000_000
        )
        expires_at_ns = nomination_deadline_ns + self._wait_hint_ttl_ns
        current_generation = self._observed_prefix_generations.get(
            (hint.run_id, hint.cell_id, hint.session_id)
        )
        if current_generation is not None and current_generation != hint.generation:
            return self._store_wait_hint_result(
                hint,
                status="rejected",
                reason="superseded_generation",
                now_ns=now_ns,
                current_generation=current_generation,
            )
        source = self._observed_prefixes.get(
            (
                hint.run_id,
                hint.cell_id,
                hint.session_id,
                hint.generation,
                hint.observed_prefix_id,
                boundary,
            )
        )
        if source is None:
            return self._store_wait_hint_result(
                hint,
                status="no_action",
                reason="observed_prefix_unavailable",
                now_ns=now_ns,
                nomination_deadline_ns=nomination_deadline_ns,
                expires_at_ns=expires_at_ns,
            )
        if current_generation != hint.generation:
            return self._store_wait_hint_result(
                hint,
                status="rejected",
                reason="superseded_generation",
                now_ns=now_ns,
                current_generation=current_generation,
            )
        if not (
            source.observed_ns
            <= hint.observation_available_ns
            <= hint.input_publication_ns
            <= now_ns
        ):
            return self._store_wait_hint_result(
                hint,
                status="rejected",
                reason="causal_clock_mismatch",
                now_ns=now_ns,
                source_observed_ns=source.observed_ns,
                decision_ns=now_ns,
            )
        if now_ns >= nomination_deadline_ns:
            return self._store_wait_hint_result(
                hint,
                status="expired",
                reason="nomination_deadline_elapsed",
                now_ns=now_ns,
                nomination_deadline_ns=nomination_deadline_ns,
                expires_at_ns=expires_at_ns,
            )
        if getattr(self, "_wait_hint_cpu_retention_only", False):
            # Exact source/generation/namespace and causal clocks were checked
            # above. Refresh only ready CPU keys reachable by a demand prefix
            # lookup; no lease, GPU reservation, transfer, or future request
            # is created.
            ready = _ready_resident_prefix(self.manager, source.keys)
            gpu_aware = getattr(self, "_wait_hint_cpu_retention_gpu_aware", False)
            bridge_details = ({
                "cpu_only_prefix_keys": len(ready),
                "local_gpu_prefix_keys": None,
                "local_gpu_coverage_available": None,
                "gpu_prefix_lookup_performed": False,
                "additional_ready_cpu_suffix_keys": 0,
            } if gpu_aware else {})
            if gpu_aware and len(ready) < len(source.keys):
                # Native demand lookup can bridge a CPU hole with an exact,
                # contiguous GPU prefix. Preserve the original ready roots,
                # then add only the ready CPU suffix reachable through that
                # prefix. This reads the native index without taking refs,
                # allocating GPU pages, or assuming those pages stay cached.
                cpu_prefix_count = len(ready)
                gpu_prefix_count, _, coverage = _local_gpu_prefix(
                    self.gpu_block_pool, source.keys
                )
                suffix = (
                    _ready_resident_prefix(
                        self.manager, source.keys[gpu_prefix_count:]
                    )
                    if coverage and gpu_prefix_count > cpu_prefix_count else []
                )
                # The ranges are disjoint and ordered by source position.
                # One refresh reverses them together, retaining root priority.
                ready = [*ready, *suffix]
                bridge_details = {
                    "cpu_only_prefix_keys": cpu_prefix_count,
                    "local_gpu_prefix_keys": gpu_prefix_count,
                    "local_gpu_coverage_available": coverage,
                    "gpu_prefix_lookup_performed": True,
                    "additional_ready_cpu_suffix_keys": len(suffix),
                }
            refresh = getattr(self.manager, "refresh_wait_hint_cpu_recency", None)
            gate_details = {}
            skip_cpu_refresh = False
            if getattr(self, "_wait_hint_cpu_retention_gpu_gate", False):
                gpu_count, _, gpu_available = _local_gpu_prefix(
                    self.gpu_block_pool, source.keys
                )
                skip_cpu_refresh = gpu_available and gpu_count == len(source.keys)
                gate_details = {
                    "cpu_retention_gpu_gate_policy":
                        "skip_CPU_advice_while_full_GPU_prefix_cached",
                    "cpu_retention_gpu_index_available": gpu_available,
                    "cpu_retention_gpu_cached_prefix_keys":
                        gpu_count if gpu_available else None,
                    "cpu_retention_source_keys": len(source.keys),
                    "cpu_retention_ready_prefix_keys": len(ready),
                    "cpu_retention_skipped_ready_prefix_keys":
                        len(ready) if skip_cpu_refresh else 0,
                    "cpu_retention_skipped_full_gpu": skip_cpu_refresh,
                }
            # Full GPU coverage only suppresses this advisory touch. It does
            # not promise that either copy will survive until a future return.
            # An unavailable index falls back to the ordinary ready CPU roots.
            if skip_cpu_refresh:
                ready_count = 0
            else:
                ready_count = refresh(ready) if callable(refresh) else 0
            if gpu_aware:
                bridge_details["refreshed_cpu_key_count"] = ready_count
            return self._store_wait_hint_result(
                hint, status="no_action", reason="cpu_prefix_recency_refreshed",
                now_ns=now_ns, nomination_deadline_ns=nomination_deadline_ns,
                expires_at_ns=expires_at_ns, ready_cpu_prefix_keys=ready_count,
                reserved_gpu_blocks=0, speculative_transfer_bytes=0,
                **bridge_details,
                **gate_details,
            )
        if getattr(self, "_wait_hint_coalesce_exact_source", False):
            # Refresh acknowledgements point to the original lease; they never
            # rename it, create aliases, extend TTL or refresh feature clocks.
            for existing in self._wait_prefetch.values():
                old_hint = existing.lease.hint
                if (
                    existing.status not in {"pending", "in_flight", "ready"}
                    or existing.bound_request_id is not None
                    or not existing.lease.active_at(now_ns)
                    or (existing.status == "pending"
                        and now_ns >= existing.lease.nomination_deadline_ns)
                    or old_hint.owner_id != hint.owner_id
                    or old_hint.namespace != hint.namespace
                    or existing.source != source
                    or (old_hint.run_id, old_hint.cell_id, old_hint.session_id,
                        old_hint.generation, old_hint.observed_prefix_id,
                        old_hint.prefix_boundary_tokens)
                    != (hint.run_id, hint.cell_id, hint.session_id,
                        hint.generation, hint.observed_prefix_id,
                        hint.prefix_boundary_tokens)
                ):
                    continue
                return self._store_wait_hint_result(
                    hint, status="accepted", reason="coalesced_exact_source",
                    now_ns=now_ns, canonical_event_id=old_hint.event_id,
                    canonical_state_status=existing.status,
                    canonical_staged_key_count=len(existing.keys),
                    canonical_source_key_count=len(existing.source.keys),
                    nomination_deadline_ns=existing.lease.nomination_deadline_ns,
                    expires_at_ns=existing.lease.expires_at_ns,
                    source_observed_ns=existing.source.observed_ns,
                )
        if (
            getattr(self, "_prefetch_require_full_prefix", False)
            and self._admitted_prefill_in_progress()
        ):
            return self._store_wait_hint_result(
                hint, status="no_action", reason="active_prefill_in_progress",
                now_ns=now_ns, nomination_deadline_ns=nomination_deadline_ns,
                expires_at_ns=expires_at_ns,
            )
        req_context = ReqContext(
            req_id=f"wait-hint:{hint.owner_id}:{hint.event_id}",
            kv_transfer_params={
                "run_id": hint.run_id,
                "cell_id": hint.cell_id,
                "session_id": hint.session_id,
                "generation": hint.generation,
                "observed_prefix_id": hint.observed_prefix_id,
            },
        )
        (
            local_gpu_prefix_count,
            local_gpu_block_ids,
            local_gpu_coverage_available,
        ) = _local_gpu_prefix(self.gpu_block_pool, source.keys)
        candidate_keys = source.keys[local_gpu_prefix_count:]
        resident_prefix = _ready_resident_prefix(self.manager, candidate_keys)
        opportunity = {
            "source_key_count": len(source.keys),
            "local_gpu_prefix_count": local_gpu_prefix_count,
            "local_gpu_block_ids": local_gpu_block_ids,
            "local_gpu_coverage_available": local_gpu_coverage_available,
            "staging_candidate_count": len(candidate_keys),
            "eligible_cpu_staging_prefix_count": len(resident_prefix),
            "positive_opportunity": (
                local_gpu_coverage_available
                and local_gpu_prefix_count < len(source.keys)
                and bool(resident_prefix)
            ),
            "provenance": ("native_block_pool_prefix_index_and_cpu_policy_ready_state"),
        }
        if getattr(self, "_prefetch_require_full_prefix", False) and (
            not source.keys
            or tuple(candidate_keys) != source.keys
            or tuple(resident_prefix) != source.keys
            or _any_local_gpu_key(self.gpu_block_pool, source.keys) is not False
        ):
            return self._store_wait_hint_result(
                hint, status="no_action", reason="full_prefix_unavailable",
                now_ns=now_ns, opportunity=opportunity,
                nomination_deadline_ns=nomination_deadline_ns,
                expires_at_ns=expires_at_ns,
            )
        if getattr(self, "_prefetch_require_full_prefix", False):
            available_budget = min(
                max(0, int(self.gpu_block_pool.num_gpu_blocks
                           * self._prefetch_budget_fraction)
                    - self._prefetch_reserved_blocks),
                max(0, self._prefetch_free_capacity() - self._prefetch_headroom_blocks),
            )
            if available_budget < len(source.keys):
                return self._store_wait_hint_result(
                    hint, status="no_action", reason="full_prefix_budget_unavailable",
                    now_ns=now_ns, opportunity=opportunity,
                    allocation_budget=available_budget,
                    nomination_deadline_ns=nomination_deadline_ns,
                    expires_at_ns=expires_at_ns,
                )
        if local_gpu_coverage_available and not candidate_keys:
            return self._store_wait_hint_result(
                hint,
                status="no_action",
                reason="prefix_already_gpu_resident",
                now_ns=now_ns,
                nomination_deadline_ns=nomination_deadline_ns,
                expires_at_ns=expires_at_ns,
                opportunity=opportunity,
            )
        if not resident_prefix:
            return self._store_wait_hint_result(
                hint,
                status="no_action",
                reason="resident_prefix_unavailable",
                now_ns=now_ns,
                nomination_deadline_ns=nomination_deadline_ns,
                expires_at_ns=expires_at_ns,
                source_key_count=len(source.keys),
                eligible_resident_prefix_count=0,
                opportunity=opportunity,
            )
        lease = WaitHintLease(
            hint=hint,
            nomination_deadline_ns=nomination_deadline_ns,
            expires_at_ns=expires_at_ns,
        )
        state = _WaitPrefetchState(
            lease,
            source,
            req_context,
            publication_candidate_keys=tuple(candidate_keys),
            publication_eligible_keys=tuple(resident_prefix),
            publication_local_gpu_prefix_count=local_gpu_prefix_count,
            publication_local_gpu_block_ids=tuple(local_gpu_block_ids),
            publication_local_gpu_coverage_available=(local_gpu_coverage_available),
            coalescing_accepted_ns=now_ns,
        )
        self._wait_prefetch[hint.event_id] = state
        if self._soft_retention == "bounded_hint":
            self.manager.touch(resident_prefix, req_context)
        return self._store_wait_hint_result(
            hint,
            status="accepted",
            reason="queued_for_scheduler_dispatch",
            now_ns=now_ns,
            nomination_deadline_ns=nomination_deadline_ns,
            expires_at_ns=expires_at_ns,
            prefix_boundary_tokens=boundary,
            source_key_count=len(source.keys),
            eligible_resident_prefix_count=len(resident_prefix),
            opportunity=opportunity,
            source_request_id=source.source_request_id,
            source_observed_ns=source.observed_ns,
            decision_ns=now_ns,
            remaining_wait_ns=hint.remaining_wait_ns(now_ns),
            **({
                "canonical_event_id": hint.event_id,
                "canonical_state_status": state.status,
                "canonical_staged_key_count": 0,
                "canonical_source_key_count": len(source.keys),
            } if getattr(self, "_wait_hint_coalesce_exact_source", False) else {}),
        )

    def _free_wait_prefetch(self, state: _WaitPrefetchState, reason: str) -> None:
        if state.job_id is not None and state.job_id in getattr(
            self, "_wait_prefetch_jobs", {}
        ):
            self._retire_wait_prefetch(state, reason, now_ns=time.monotonic_ns())
            return
        event_id = state.lease.hint.event_id
        self._release_wait_prefetch_protection(state)
        for request_id, prefetch in list(getattr(self, "_prefetched", {}).items()):
            if prefetch.attempt_id != event_id:
                continue
            self._prefetched.pop(request_id, None)
            self._wait_hint_request_bindings.pop(request_id, None)
        if state.bound_request_id is not None:
            self._wait_hint_request_bindings.pop(state.bound_request_id, None)
            state.bound_request_id = None
        released_blocks = 0
        if (
            not state.destination_released
            and state.gpu_block_ids
            and self.gpu_block_pool is not None
        ):
            self._free_prefetch_blocks(list(state.gpu_block_ids))
            self._prefetch_reserved_blocks -= len(state.gpu_block_ids)
            state.destination_released = True
            released_blocks = len(state.gpu_block_ids)
        state.status = "retired"
        state.lease.expire()
        self._wait_prefetch.pop(event_id, None)
        self._record_wait_hint_event(
            {
                "schema": "semantic-known-prefix-wait-hint-event-v1",
                "event": "staging_retired",
                "event_id": event_id,
                **self._wait_state_identity(state),
                "reason": reason,
                "released_blocks": released_blocks,
                "released_bytes": (released_blocks * self._prefetch_bytes_per_key),
                "retired_at_ns": time.monotonic_ns(),
            }
        )

    def _release_wait_prefetch_protection(self, state: _WaitPrefetchState) -> None:
        binding_id = state.protection_binding_id
        if binding_id is None or state.protection_released:
            return
        self.manager.release_staging_keys(binding_id)
        state.protection_released = True

    def _drain_staging_protection_records(self) -> None:
        drain = getattr(self.manager, "drain_staging_protection_records", None)
        if drain is None:
            return
        for record in drain():
            retained = dict(record)
            retained["event"] = "staging_protection"
            self._record_wait_hint_event(retained)
            binding_id = record.get("binding_id")
            for state in (
                *getattr(self, "_preservation_requests", {}).values(),
                *getattr(self, "_preservation_finished", {}).values(),
            ):
                wait_hint = state.get("wait_hint")
                if isinstance(wait_hint, dict) and binding_id == (
                    f"wait-hint:{wait_hint.get('event_id')}"
                ):
                    state.setdefault("staging_protection_records", []).append(
                        dict(record)
                    )

    def _expire_wait_prefetches(self, now_ns: int) -> None:
        for state in list(getattr(self, "_wait_prefetch", {}).values()):
            if state.status in {
                "bound_ready",
                "splice_in_flight",
                "retired_pending_fence",
            }:
                continue
            if (
                state.status == "pending"
                and now_ns > state.lease.nomination_deadline_ns
            ):
                self._retire_wait_prefetch(
                    state, "nomination_deadline_elapsed", now_ns=now_ns
                )
                continue
            if (
                state.lease.state in {"expired", "cancelled"}
                or now_ns > state.lease.expires_at_ns
            ):
                self._retire_wait_prefetch(state, "retention_expired", now_ns=now_ns)

    def _admitted_prefill_in_progress(self) -> bool:
        """Give known admitted prompt growth priority over optional staging.

        Native Request objects are live: every retry sees current progress and
        terminal status. Waiting requests have not acquired prefill slots yet;
        their ordinary demand admission uses the separate reclamation path.
        No decode growth or maximum sequence length is guessed here.
        """
        for req_status in getattr(self, "_req_status", {}).values():
            request = getattr(req_status, "req", None)
            status = getattr(getattr(request, "status", None), "name", None)
            if status in {"RUNNING", "WAITING_FOR_REMOTE_KVS"} and (
                request.num_computed_tokens < request.num_prompt_tokens
            ):
                return True
        return False

    def _prefetch_free_capacity(self) -> int:
        """Bound optional staging to the native queue's unhashed head.

        Cached reclaimable blocks stay available to ordinary demand allocation.
        Keep native allocation/refcount/metrics and free-queue ordering intact.
        """
        pool = self.gpu_block_pool
        if pool is None:
            return 0
        free = pool.get_num_free_blocks()
        if not getattr(self, "_prefetch_preserve_gpu_prefix_cache", False):
            return free
        if not getattr(pool, "enable_caching", True):
            return free
        queue = getattr(pool, "free_block_queue", None)
        iterate = getattr(queue, "iter_blocks_after", None)
        if not callable(iterate):
            return 0
        limit = min(free, int(pool.num_gpu_blocks * self._prefetch_budget_fraction)
                    + self._prefetch_headroom_blocks)
        aliases = getattr(pool, "cached_block_hashes_by_block", {})
        count = 0
        for block in iterate(None):
            if count >= limit:
                break
            if (block.is_null or block.ref_cnt != 0 or block.block_hash is not None
                    or aliases.get(block.block_id)):
                break
            count += 1
        return count

    @profiled('scheduler.prefetch_submit')
    def _attempt_wait_prefetch(
        self, state: _WaitPrefetchState, *, unscored_reason: str | None = None
    ) -> bool:
        # A privileged matched diagnostic can preserve hint publication and
        # source observation while leaving all restoration to demand.
        if getattr(self, "_wait_hint_diagnostic_no_dispatch", False):
            return False
        if self.gpu_block_pool is None:
            return False
        if (
            getattr(self, "_prefetch_require_full_prefix", False)
            and self._admitted_prefill_in_progress()
        ):
            return False
        decision_ns = time.monotonic_ns()
        remaining_wait_ns = state.lease.hint.remaining_wait_ns(decision_ns)
        if remaining_wait_ns is None or remaining_wait_ns <= 0:
            return False
        total_budget = int(
            self.gpu_block_pool.num_gpu_blocks * self._prefetch_budget_fraction
        )
        remaining_budget = max(0, total_budget - self._prefetch_reserved_blocks)
        if remaining_budget <= 0:
            return False
        free_blocks = self._prefetch_free_capacity()
        allocation_budget = min(
            remaining_budget,
            max(0, free_blocks - self._prefetch_headroom_blocks),
        )
        if allocation_budget <= 0:
            return False
        eligible_resident_prefix = _ready_resident_prefix(
            self.manager, state.publication_candidate_keys
        )
        if getattr(self, "_prefetch_require_full_prefix", False) and (
            not state.source.keys
            or tuple(state.publication_candidate_keys) != state.source.keys
            or tuple(eligible_resident_prefix) != state.source.keys
            or allocation_budget < len(state.source.keys)
            or _any_local_gpu_key(self.gpu_block_pool, state.source.keys) is not False
        ):
            state.selection_evidence = {
                "selector": self._prefetch_selector,
                "status": "full_prefix_unavailable",
                "source_key_count": len(state.source.keys),
                "allocation_budget": allocation_budget,
                "decision_eligible_resident_prefix_count": len(eligible_resident_prefix),
                "decision_ns": decision_ns,
            }
            return False
        if not eligible_resident_prefix:
            state.selection_evidence = {
                "selector": self._prefetch_selector,
                "status": "no_eligible_resident_prefix",
                "source_key_count": len(state.source.keys),
                "staging_candidate_count": len(state.publication_candidate_keys),
                "publication_eligible_resident_prefix_count": len(
                    state.publication_eligible_keys
                ),
                "decision_eligible_resident_prefix_count": 0,
                "eligibility_provenance": "policy_get_ready_contiguous_prefix",
                "decision_ns": decision_ns,
            }
            return False
        selection_started_ns = time.monotonic_ns()
        if self._prefetch_selector == "useful_on_time":
            observed_cost = self._prefetch_transfer_ns_per_byte
            estimated_transfer_cost = (
                observed_cost
                if observed_cost is not None
                else self._prefetch_fallback_ns_per_byte
            )
            selection = select_useful_on_time(
                eligible_resident_prefix,
                allocation_budget,
                remaining_wait_ns=remaining_wait_ns,
                bytes_per_key=self._prefetch_bytes_per_key,
                estimated_transfer_ns_per_byte=estimated_transfer_cost,
                estimated_splice_ns_per_byte=(
                    self._prefetch_fallback_splice_ns_per_byte
                ),
            )
            selected = list(selection.selected_keys)
            state.selection_evidence = {
                "selector": "useful_on_time",
                "formula": (
                    "longest_eligible_chain_prefix_with_cumulative_"
                    "transfer_plus_splice_cost_le_remaining_wait"
                ),
                "status": selection.status,
                "remaining_wait_ns": selection.remaining_wait_ns,
                "bytes_per_key": selection.bytes_per_key,
                "estimated_transfer_ns_per_byte": (
                    selection.estimated_transfer_ns_per_byte
                ),
                "estimated_splice_ns_per_byte": (
                    selection.estimated_splice_ns_per_byte
                ),
                "estimated_selected_ns": selection.estimated_selected_ns,
                "transfer_cost_provenance": (
                    "completed_prefetch_transfer_ema"
                    if observed_cost is not None
                    else "configured_transfer_bandwidth_fallback"
                ),
                "splice_cost_provenance": ("configured_splice_bandwidth_fallback"),
                "cost_available_ns": self._prefetch_cost_available_ns,
                "fallback": selection.fallback,
                "tie_break": "immutable_chain_order",
            }
        elif self._prefetch_selector == "semantic":
            semantic_key_scores = ([] if getattr(self, '_prefix_mean_prefetch_scoring', False)
                                   else self._semantic_wait_key_scores(eligible_resident_prefix, decision_ns))
            prefix_mean_score = (self._prefix_mean_score_for_state(state, eligible_resident_prefix, decision_ns)
                                 if getattr(self, '_prefix_mean_prefetch_scoring', False) else None)
            # Semantic information nominates an exact observed source.  Once
            # nominated, movement remains the immutable contiguous source
            # prefix required by exact prefix-cache reuse; relevance must not
            # turn a source chain into an arbitrary set of blocks.
            selected = (
                list(eligible_resident_prefix[:allocation_budget])
                if semantic_key_scores or prefix_mean_score is not None or unscored_reason is not None
                else []
            )
            state.selection_evidence = {
                "selector": "semantic",
                "status": "selected" if selected else "no_causal_score",
                "remaining_wait_ns": remaining_wait_ns,
                "tie_break": "cross_session_score_then_source_identity",
                "within_source_order": "immutable_contiguous_chain_order",
                "semantic_key_scores": semantic_key_scores,
                "prefix_mean_score": prefix_mean_score,
                "score_status": "scored" if semantic_key_scores or prefix_mean_score is not None else "not_evaluated",
                "unscored_reason": unscored_reason if not semantic_key_scores and prefix_mean_score is None else None,
            }
        else:
            selected = _select_prefetch_keys(
                self.manager,
                eligible_resident_prefix,
                allocation_budget,
                self._prefetch_selector,
                selection_context={
                    "score_max_age_ns": self._prefetch_score_max_age_ns,
                    "now_ns": time.monotonic_ns(),
                },
            )
            state.selection_evidence = {
                "selector": self._prefetch_selector,
                "status": "selected",
                "remaining_wait_ns": remaining_wait_ns,
                "tie_break": "immutable_chain_order",
            }
        state.selection_evidence.update(
            {
                "source_key_count": len(state.source.keys),
                "staging_candidate_count": len(state.publication_candidate_keys),
                "publication_local_gpu_prefix_count": (
                    state.publication_local_gpu_prefix_count
                ),
                "publication_local_gpu_block_ids": list(
                    state.publication_local_gpu_block_ids
                ),
                "publication_local_gpu_coverage_available": (
                    state.publication_local_gpu_coverage_available
                ),
                "publication_eligible_resident_prefix_count": len(
                    state.publication_eligible_keys
                ),
                "decision_eligible_resident_prefix_count": len(
                    eligible_resident_prefix
                ),
                "eligible_resident_prefix_keys": [
                    _event_key(key) for key in eligible_resident_prefix
                ],
                "eligibility_provenance": ("policy_get_ready_contiguous_prefix"),
                "decision_ns": decision_ns,
                "allocation_budget": allocation_budget,
                "hint_input_publication_ns": (state.lease.hint.input_publication_ns),
                "hint_nomination_deadline_ns": (state.lease.nomination_deadline_ns),
            }
        )
        state.selection_evidence["nomination_cpu_ns"] = (
            time.monotonic_ns() - selection_started_ns
        )
        count = len(selected)
        if count <= 0:
            return False
        if (getattr(self, "_prefetch_require_full_prefix", False)
                and tuple(selected) != state.source.keys):
            state.selection_evidence["status"] = "full_prefix_selection_incomplete"
            return False
        resident = selected[:count]
        reserved = []
        prepared_load = False
        job_id: int | None = None
        try:
            reserved = self.gpu_block_pool.get_new_blocks(count)
            gpu_block_ids = tuple(int(block.block_id) for block in reserved)
            self._current_batch_allocated_block_ids.update(
                block_id for block_id in gpu_block_ids if block_id != 0
            )
            src_spec = self.manager.prepare_load(resident, state.req_context)
            prepared_load = True
            dst_spec = GPULoadStoreSpec(
                list(gpu_block_ids), group_sizes=[count], block_indices=[0]
            )
            job_id = self._generate_job_id()
            self._current_batch_prefetch_load_jobs[job_id] = TransferJob(
                req_id=state.req_context.req_id,
                src_spec=src_spec,
                dst_spec=dst_spec,
            )
            state.status = "in_flight"
            state.lease.state = "in_flight"
            state.job_id = job_id
            state.keys = tuple(resident)
            state.gpu_block_ids = gpu_block_ids
            state.pending_contributors = int(self.config.num_workers)
            self._wait_prefetch_jobs[job_id] = state
            self._prefetch_reserved_blocks += count
        except BaseException:
            if job_id is not None:
                self._current_batch_prefetch_load_jobs.pop(job_id, None)
                self._wait_prefetch_jobs.pop(job_id, None)
            if prepared_load:
                self.manager.complete_load(resident, state.req_context)
            if reserved:
                self.gpu_block_pool.free_blocks(reserved)
            raise
        self._record_wait_hint_event(
            {
                "schema": "semantic-known-prefix-wait-hint-event-v1",
                "event": "staging_submitted",
                "event_id": state.lease.hint.event_id,
                **self._wait_state_identity(state),
                "job_id": job_id,
                "selected_keys": [_event_key(key) for key in resident],
                "reserved_block_ids": list(state.gpu_block_ids),
                "bytes_per_key": self._prefetch_bytes_per_key,
                "submitted_bytes": len(resident) * self._prefetch_bytes_per_key,
                "submitted_at_ns": time.monotonic_ns(),
                "nomination_deadline_ns": state.lease.nomination_deadline_ns,
                "expires_at_ns": state.lease.expires_at_ns,
                "remaining_wait_ns": remaining_wait_ns,
                "selection": dict(state.selection_evidence),
            }
        )
        return True

    def _semantic_wait_key_scores(
        self, keys: Sequence[OffloadKey], decision_ns: int
    ) -> list[dict[str, object]]:
        """Return only relevance observations causal and fresh at decision time."""
        ema = getattr(self.manager, "relevance_ema", {}).get("mean", {})
        updated = getattr(self.manager, "_relevance_updated_ns", {}).get("mean", {})
        provenance = getattr(self.manager, "_relevance_observation_provenance", {}).get(
            "mean", {}
        )
        feature_snapshots = getattr(
            self.manager, "_relevance_feature_snapshot_ns", {}
        ).get("mean", {})
        max_age_ns = getattr(self, "_prefetch_score_max_age_ns", None)
        observations: list[dict[str, object]] = []
        for key in keys:
            if key not in ema:
                continue
            observed_ns = updated.get(key)
            if observed_ns is not None and not (
                0 <= decision_ns - observed_ns
                and (max_age_ns is None or decision_ns - observed_ns <= max_age_ns)
            ):
                continue
            if observed_ns is None and max_age_ns is not None:
                continue
            score = float(ema[key])
            if not math.isfinite(score):
                continue
            feature_snapshot_ns = feature_snapshots.get(key)
            if feature_snapshot_ns is not None and max_age_ns is not None and not (
                0 <= decision_ns - feature_snapshot_ns <= max_age_ns
            ):
                # Deferred evaluation must not make an old feature fresh by
                # assigning it a new score-computation timestamp.
                continue
            observations.append(
                {
                    "key": _event_key(key),
                    "score": score,
                    "observation_available_ns": observed_ns,
                    "observation_provenance": provenance.get(
                        key, "manager_relevance_updated_ns"
                    ),
                    "age_ns": (
                        decision_ns - observed_ns if observed_ns is not None else None
                    ),
                    "max_age_ns": max_age_ns,
                    "fresh_at_decision": True,
                    "feature_snapshot_captured_ns": feature_snapshot_ns,
                    "feature_age_ns": (
                        decision_ns - feature_snapshot_ns
                        if feature_snapshot_ns is not None
                        else None
                    ),
                }
            )
        return observations

    def _prefix_mean_descriptor(self, state, keys):
        return {
            'event_id': state.lease.hint.event_id,
            'identity': self._wait_hint_exact_source_identity(state),
            'source_request_id': state.source.source_request_id,
            'keys': tuple(keys),
            'residency_versions': {k: getattr(self.manager._policy.get(k), '_semantic_score_version', None) for k in keys},
        }

    def _prefix_mean_descriptor_valid(self, descriptor, now_ns):
        state = getattr(self, '_wait_prefetch', {}).get(descriptor.get('event_id'))
        if state is None or state.status != 'pending' or not state.lease.active_at(now_ns):
            return False
        keys = tuple(_ready_resident_prefix(self.manager, state.publication_candidate_keys))
        expected = self._prefix_mean_descriptor(state, keys)
        return bool(keys) and descriptor == expected and all(
            v is not None for v in expected['residency_versions'].values())

    def _prefix_mean_batch_valid(self, observation, now_ns):
        if not isinstance(observation, dict) or observation.get('contract') != 'prefix-mean-v1' or observation.get('metric') != 'raw_mean_key_dot_mean_head_common_active_query':
            return False
        context = self._score_context()
        if observation.get('transport_epoch') != context.get('epoch'):
            return False
        captured, available, expires = (observation.get(k) for k in (
            'snapshot_captured_ns', 'observation_available_ns', 'expires_at_ns'))
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (captured, available, expires)):
            return False
        if (not 0 <= captured <= available <= now_ns <= expires <= captured + self._prefix_query_max_age_ns
                or now_ns - captured > getattr(self, '_prefetch_score_max_age_ns', self._prefix_query_max_age_ns)):
            return False
        sources, stamps = observation.get('query_sources'), observation.get('query_capture_ns')
        if not isinstance(sources, dict) or not sources or not isinstance(stamps, dict) or set(sources) != set(stamps):
            return False
        return all(token is not None and token == context.get('sources', {}).get(r)
                   and isinstance(stamps[r], int) and not isinstance(stamps[r], bool)
                   and captured <= stamps[r] <= available for r, token in sources.items())

    def receive_prefix_score_observations(self, observations):
        if not getattr(self, '_prefix_mean_prefetch_scoring', False):
            return
        for observation in observations:
            now_ns = time.monotonic_ns()
            if not self._prefix_mean_batch_valid(observation, now_ns):
                PROFILE.count('scheduler.prefix_scores_rejected_context')
                continue
            sequence = observation.get('observation_sequence')
            prior = getattr(self, '_prefix_mean_score_batch', {})
            if (isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0
                    or (prior.get('transport_epoch') == observation['transport_epoch']
                        and sequence <= prior.get('observation_sequence', 0))):
                continue
            accepted = {}
            for event_id, item in observation.get('candidates', {}).items():
                score = item.get('score')
                descriptor = item.get('descriptor', {})
                if (isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(score) or event_id != descriptor.get('event_id')
                        or not self._prefix_mean_descriptor_valid(descriptor, now_ns)):
                    continue
                accepted[event_id] = item
            if not accepted:
                continue
            self._prefix_mean_score_batch = dict(observation, candidates=accepted)
            stats = self.__dict__.setdefault('_prefix_mean_scoring_stats', {})
            for name, value in [('accepted_observations', 1), ('accepted_candidates', len(accepted)),
                                ('accepted_prefix_key_count', sum(len(x['descriptor']['keys']) for x in accepted.values()))]:
                stats[name] = stats.get(name, 0) + value
            self._record_wait_hint_event({
                'schema': 'semantic-prefix-mean-score-v1', 'event': 'prefix_mean_scores_applied',
                'metric': observation['metric'], 'observation_sequence': sequence,
                'query_sources': observation['query_sources'],
                'query_capture_ns': observation['query_capture_ns'],
                'snapshot_captured_ns': observation['snapshot_captured_ns'],
                'observation_available_ns': observation['observation_available_ns'],
                'expires_at_ns': observation['expires_at_ns'], 'consumer_ns': now_ns,
                'candidate_count': len(accepted),
                'scored_key_count': sum(len(x['descriptor']['keys']) for x in accepted.values()),
                'scores': {eid: x['score'] for eid, x in accepted.items()},
            })
            PROFILE.count('scheduler.prefix_score_apply_candidates', len(accepted))

    def _prefix_mean_score_for_state(self, state, keys, now_ns):
        batch = getattr(self, '_prefix_mean_score_batch', {})
        if not self._prefix_mean_batch_valid(batch, now_ns):
            return None
        item = batch.get('candidates', {}).get(state.lease.hint.event_id)
        if item is None or tuple(keys) != tuple(item['descriptor']['keys']):
            return None
        if not self._prefix_mean_descriptor_valid(item['descriptor'], now_ns):
            return None
        return {'score': item['score'], 'metric': batch['metric'],
                'observation_sequence': batch['observation_sequence'],
                'snapshot_captured_ns': batch['snapshot_captured_ns'],
                'observation_available_ns': batch['observation_available_ns'],
                'expires_at_ns': batch['expires_at_ns'],
                'query_sources': batch['query_sources'],
                'query_capture_ns': batch['query_capture_ns'],
                'scored_key_count': len(keys)}

    def _retry_prefix_mean_prefetches(self, pending, now_ns):
        ranked, descriptors = [], {}
        for state in pending:
            keys = tuple(_ready_resident_prefix(self.manager, state.publication_candidate_keys))
            if not keys or (getattr(self, '_prefetch_require_full_prefix', False) and keys != state.source.keys):
                continue
            descriptor = self._prefix_mean_descriptor(state, keys)
            if any(v is None for v in descriptor['residency_versions'].values()):
                continue
            descriptors[state.lease.hint.event_id] = descriptor
            candidate, score = self._semantic_wait_nomination_candidate(state, now_ns)
            ranked.append((state, candidate, score))
        if not ranked:
            return
        missing = any(score is None for _, _, score in ranked)
        if missing:
            active = sorted(r for r, status in getattr(self, '_req_status', {}).items()
                            if getattr(getattr(getattr(status, 'req', None), 'status', None), 'name', None) == 'RUNNING')
            self._prefix_mean_scoring_request = {'active_query_ids': active, 'candidates': descriptors}
            signature = tuple((eid, tuple(d['keys']), tuple(d['residency_versions'].items()))
                              for eid, d in sorted(descriptors.items()))
            attempt = getattr(self, '_prefix_mean_scoring_attempt', None)
            if attempt is None or attempt[0] != signature:
                attempt = self._prefix_mean_scoring_attempt = (signature, now_ns)
            if now_ns - attempt[1] < 50_000_000:
                return
        # Compute while an active peer exists, even if staging cannot yet act.
        # Actual movement retains every existing prefill/capacity/native check.
        if getattr(self, '_prefetch_require_full_prefix', False) and self._admitted_prefill_in_progress():
            return
        pool = self.gpu_block_pool
        budget = min(max(0, int(pool.num_gpu_blocks * self._prefetch_budget_fraction) - self._prefetch_reserved_blocks),
                     max(0, self._prefetch_free_capacity() - self._prefetch_headroom_blocks)) if pool is not None else 0
        if budget <= 0:
            return
        ranked.sort(key=lambda x: (x[2] is None, -(x[2] if x[2] is not None else 0.),
                                  self._wait_hint_exact_source_identity(x[0]), x[0].lease.hint.event_id))
        chosen = []
        reason = 'fresh_active_query_prefix_score_unavailable_after_bounded_wait' if missing else None
        for state, candidate, score in ranked:
            if self._attempt_wait_prefetch(state, unscored_reason=reason):
                chosen.append({'event_id': candidate['event_id'], 'session_id': candidate['session_id'],
                               'generation': candidate['generation'], 'observed_prefix_id': candidate['observed_prefix_id'],
                               'source_request_id': candidate['source_request_id'],
                               'selected_keys': [_event_key(k) for k in state.keys]})
        eligible_blocks = sum(len(c['eligible_resident_prefix_keys']) for _, c, _ in ranked)
        scored = sum(score is not None for _, _, score in ranked)
        self._record_wait_hint_event({
            'schema': 'semantic-cross-session-prefetch-nomination-v1', 'event': 'nomination_decision',
            'selector': 'semantic', 'scoring_transport': 'prefix-mean-v1',
            'decision_ns': now_ns, 'shared_allocation_budget_blocks': budget,
            'eligible_candidate_blocks': eligible_blocks, 'scored_candidate_count': scored,
            'shared_budget_constrained': scored >= 2 and eligible_blocks > budget,
            'candidate_publication_order_event_ids': [s.lease.hint.event_id for s in pending],
            'tie_break': 'common_raw_mean_prefix_score_desc_then_source_identity',
            'highest_ranked_candidate': ranked[0][1] if ranked[0][2] is not None else None,
            'candidates': [c for _, c, _ in ranked], 'chosen': chosen,
            'selected_blocks': sum(len(c['selected_keys']) for c in chosen),
            'selected_bytes': sum(len(c['selected_keys']) for c in chosen) * self._prefetch_bytes_per_key,
            'unscored_reason': reason,
            'no_action_reason': None if chosen else 'no_causal_candidate_or_shared_budget',
        })

    def _semantic_wait_nomination_candidate(
        self, state: _WaitPrefetchState, decision_ns: int
    ) -> tuple[dict[str, object], float | None]:
        eligible = _ready_resident_prefix(
            self.manager, state.publication_candidate_keys
        )
        scores = ([] if getattr(self, '_prefix_mean_prefetch_scoring', False)
                  else self._semantic_wait_key_scores(eligible, decision_ns))
        aggregate = (
            sum(float(item["score"]) for item in scores) / len(scores)
            if scores
            else None
        )
        score_observation_available_ns = (
            max(
                int(item["observation_available_ns"])
                for item in scores
                if item["observation_available_ns"] is not None
            )
            if scores
            and all(item["observation_available_ns"] is not None for item in scores)
            else None
        )
        prefix_score = (self._prefix_mean_score_for_state(state, eligible, decision_ns)
                        if getattr(self, '_prefix_mean_prefetch_scoring', False) else None)
        if getattr(self, '_prefix_mean_prefetch_scoring', False):
            scores = []
            aggregate = prefix_score['score'] if prefix_score is not None else None
            score_observation_available_ns = prefix_score['observation_available_ns'] if prefix_score else None
        hint = state.lease.hint
        candidate = {
            "event_id": hint.event_id,
            "run_id": hint.run_id,
            "cell_id": hint.cell_id,
            "session_id": hint.session_id,
            "generation": hint.generation,
            "observed_prefix_id": state.source.observed_prefix_id,
            "prefix_boundary_tokens": state.source.boundary_tokens,
            "source_request_id": state.source.source_request_id,
            "source_observed_ns": state.source.observed_ns,
            "namespace": hint.namespace_dict(),
            "hint_observation_available_ns": hint.observation_available_ns,
            "hint_input_publication_ns": hint.input_publication_ns,
            "eligible_resident_prefix_keys": [_event_key(key) for key in eligible],
            "semantic_key_scores": scores,
            "aggregate_score": aggregate,
            "score_observation_available_ns": (score_observation_available_ns),
            "score_aggregation": "mean_over_fresh_scored_ready_prefix_keys",
            "causal_inputs": {
                "source_observed_ns": state.source.observed_ns,
                "hint_observation_available_ns": hint.observation_available_ns,
                "hint_input_publication_ns": hint.input_publication_ns,
                "score_observation_available_ns": (score_observation_available_ns),
                "score_max_age_ns": getattr(self, "_prefetch_score_max_age_ns", None),
                "decision_ns": decision_ns,
            },
        }
        if getattr(self, '_prefix_mean_prefetch_scoring', False):
            candidate.update(prefix_mean_score=prefix_score,
                             score_aggregation='raw_mean_prefix_dot_common_active_query')
        return candidate, aggregate

    def _wait_hint_coalescing_enabled(self) -> bool:
        return bool(
            getattr(self, "_wait_hint_coalescing_min_sources", 1) > 1
            and getattr(self, "_wait_hint_coalescing_max_ns", 0) > 0
        )

    @staticmethod
    def _wait_hint_exact_source_identity(
        state: _WaitPrefetchState,
    ) -> tuple[object, ...]:
        hint = state.lease.hint
        return (
            hint.run_id,
            hint.cell_id,
            hint.session_id,
            hint.generation,
            json.dumps(hint.namespace_dict(), sort_keys=True, separators=(",", ":")),
            state.source.observed_prefix_id,
            state.source.boundary_tokens,
        )

    def _coalescing_eligible_sources(
        self, states: Sequence[_WaitPrefetchState], now_ns: int
    ) -> tuple[
        list[_WaitPrefetchState],
        list[_WaitPrefetchState],
        list[dict[str, object]],
    ]:
        representatives: list[_WaitPrefetchState] = []
        duplicates: list[_WaitPrefetchState] = []
        identities: list[dict[str, object]] = []
        seen: set[tuple[object, ...]] = set()
        for state in states:
            if (
                state.status != "pending"
                or state.bound_request_id is not None
                or not state.lease.active_at(now_ns)
                or not _ready_resident_prefix(
                    self.manager, state.publication_candidate_keys
                )
            ):
                continue
            identity = self._wait_hint_exact_source_identity(state)
            if identity in seen:
                duplicates.append(state)
                continue
            seen.add(identity)
            representatives.append(state)
            hint = state.lease.hint
            identities.append(
                {
                    "run_id": hint.run_id,
                    "cell_id": hint.cell_id,
                    "session_id": hint.session_id,
                    "generation": hint.generation,
                    "namespace": hint.namespace_dict(),
                    "observed_prefix_id": state.source.observed_prefix_id,
                    "prefix_boundary_tokens": state.source.boundary_tokens,
                    "source_request_id": state.source.source_request_id,
                    "eligible_resident_prefix_blocks": len(
                        _ready_resident_prefix(
                            self.manager, state.publication_candidate_keys
                        )
                    ),
                }
            )
        return representatives, duplicates, identities

    def _finish_wait_hint_coalescing(
        self,
        states: Sequence[_WaitPrefetchState],
        *,
        now_ns: int,
        reason: str,
        identities: Sequence[dict[str, object]],
        retire_duplicates: bool = True,
    ) -> set[str]:
        start_ns = getattr(self, "_wait_hint_coalescing_started_ns", None)
        if start_ns is None:
            accepted = [
                state.coalescing_accepted_ns
                for state in states
                if state.coalescing_accepted_ns is not None
            ]
            start_ns = min(accepted) if accepted else now_ns
        distinct_count = len(identities)
        distinct_sessions = {
            (
                identity["run_id"],
                identity["cell_id"],
                identity["session_id"],
                identity["generation"],
                json.dumps(
                    identity["namespace"], sort_keys=True, separators=(",", ":")
                ),
            )
            for identity in identities
        }
        dispatch_ids: set[str] = set()
        representatives, duplicates, _ = self._coalescing_eligible_sources(
            states, now_ns
        )
        for state in states:
            if state.status != "pending":
                continue
            state.coalescing_started_ns = start_ns
            state.coalescing_complete = True
            state.coalescing_ended_ns = now_ns
            state.coalescing_end_reason = reason
            state.coalescing_distinct_eligible_sources = distinct_count
        for state in representatives:
            dispatch_ids.add(state.lease.hint.event_id)
        if retire_duplicates:
            for duplicate in duplicates:
                self._free_wait_prefetch(duplicate, "duplicate_exact_source_coalesced")
        deadlines = [
            state.lease.nomination_deadline_ns
            for state in states
            if state.status == "pending"
        ]
        earliest_deadline_ns = min(deadlines) if deadlines else None
        cutoff_ns = min(
            start_ns + self._wait_hint_coalescing_max_ns,
            earliest_deadline_ns
            if earliest_deadline_ns is not None
            else start_ns + self._wait_hint_coalescing_max_ns,
        )
        self._record_wait_hint_event(
            {
                "schema": "semantic-wait-hint-coalescing-v1",
                "event": "coalescing_released",
                "selector": self._prefetch_selector,
                "started_at_ns": start_ns,
                "ended_at_ns": now_ns,
                "release_reason": reason,
                "configured_min_sources": self._wait_hint_coalescing_min_sources,
                "configured_max_wait_ns": self._wait_hint_coalescing_max_ns,
                "distinct_eligible_sources": distinct_count,
                "distinct_eligible_sessions": len(distinct_sessions),
                "distinct_eligible_source_identities": list(identities),
                "actual_delay_ns": max(0, now_ns - start_ns),
                "release_cutoff_ns": cutoff_ns,
                "release_cutoff_overshoot_ns": max(0, now_ns - cutoff_ns),
                "earliest_nomination_deadline_ns": earliest_deadline_ns,
                "remaining_lead_ns": (
                    max(0, earliest_deadline_ns - now_ns)
                    if earliest_deadline_ns is not None
                    else None
                ),
                "duplicate_exact_source_hints": len(duplicates),
                "per_hint_delay": [
                    {
                        "event_id": state.lease.hint.event_id,
                        "accepted_at_ns": state.coalescing_accepted_ns,
                        "actual_delay_ns": max(
                            0,
                            now_ns
                            - (
                                state.coalescing_accepted_ns
                                if state.coalescing_accepted_ns is not None
                                else start_ns
                            ),
                        ),
                        "remaining_lead_ns": max(
                            0, state.lease.nomination_deadline_ns - now_ns
                        ),
                    }
                    for state in states
                ],
            }
        )
        self._wait_hint_coalescing_started_ns = None
        return dispatch_ids

    def _wait_hint_coalescing_gate(self, now_ns: int) -> tuple[bool, set[str] | None]:
        if not self._wait_hint_coalescing_enabled():
            return False, None
        already_released = {
            state.lease.hint.event_id
            for state in getattr(self, "_wait_prefetch", {}).values()
            if state.status == "pending"
            and state.coalescing_complete
            and state.lease.active_at(now_ns)
        }
        if already_released:
            return False, already_released
        states = [
            state
            for state in getattr(self, "_wait_prefetch", {}).values()
            if state.status == "pending" and not state.coalescing_complete
        ]
        if not states:
            return False, None
        representatives, _duplicates, identities = self._coalescing_eligible_sources(
            states, now_ns
        )
        distinct_sessions = {
            (
                identity["run_id"],
                identity["cell_id"],
                identity["session_id"],
                identity["generation"],
                json.dumps(
                    identity["namespace"], sort_keys=True, separators=(",", ":")
                ),
            )
            for identity in identities
        }
        if self._wait_hint_coalescing_started_ns is None:
            accepted = [
                state.coalescing_accepted_ns
                for state in states
                if state.coalescing_accepted_ns is not None
            ]
            start_ns = min(accepted) if accepted else now_ns
            self._wait_hint_coalescing_started_ns = start_ns
            for state in states:
                state.coalescing_started_ns = start_ns
            self._record_wait_hint_event(
                {
                    "schema": "semantic-wait-hint-coalescing-v1",
                    "event": "coalescing_started",
                    "selector": self._prefetch_selector,
                    "started_at_ns": start_ns,
                    "configured_min_sources": (self._wait_hint_coalescing_min_sources),
                    "configured_max_wait_ns": (self._wait_hint_coalescing_max_ns),
                    "distinct_eligible_sources": len(identities),
                    "distinct_eligible_sessions": len(distinct_sessions),
                    "distinct_eligible_source_identities": identities,
                }
            )
        start_ns = self._wait_hint_coalescing_started_ns
        assert start_ns is not None
        deadlines = [state.lease.nomination_deadline_ns for state in states]
        earliest_deadline_ns = min(deadlines)
        if any(state.lease.state == "cancelled" for state in states):
            reason = "cancelled"
        elif now_ns >= earliest_deadline_ns:
            reason = "earliest_nomination_deadline"
        elif not representatives:
            reason = "no_eligible_source"
        elif self.gpu_block_pool is None:
            reason = "no_free_allocation"
        else:
            total_budget = int(
                self.gpu_block_pool.num_gpu_blocks * self._prefetch_budget_fraction
            )
            allocation_budget = min(
                max(0, total_budget - self._prefetch_reserved_blocks),
                max(
                    0,
                    self._prefetch_free_capacity()
                    - self._prefetch_headroom_blocks,
                ),
            )
            if allocation_budget <= 0:
                reason = "no_free_allocation"
            elif len(distinct_sessions) >= self._wait_hint_coalescing_min_sources:
                reason = "min_sources_reached"
            elif now_ns >= start_ns + self._wait_hint_coalescing_max_ns:
                reason = "max_wait_elapsed"
            else:
                return True, set()
        return False, self._finish_wait_hint_coalescing(
            states,
            now_ns=now_ns,
            reason=reason,
            identities=identities,
        )

    @profiled('scheduler.prefetch_decision')
    def _retry_wait_prefetches(self) -> None:
        now_ns = time.monotonic_ns()
        should_wait, dispatch_ids = self._wait_hint_coalescing_gate(now_ns)
        self._expire_wait_prefetches(now_ns)
        if should_wait:
            return
        pending = [
            state
            for state in getattr(self, "_wait_prefetch", {}).values()
            if state.status == "pending"
            and state.lease.active_at(now_ns)
            and (dispatch_ids is None or state.lease.hint.event_id in dispatch_ids)
        ]
        # Ready/in-flight hints still need expiry processing above, but there
        # is no nomination decision to build or serialize without candidates.
        # This path also runs on every ordinary decode step with no hints.
        if not pending:
            return
        if getattr(self, '_prefix_mean_prefetch_scoring', False):
            self._retry_prefix_mean_prefetches(pending, now_ns)
            return
        if (
            getattr(self, "_prefetch_require_full_prefix", False)
            and self._admitted_prefill_in_progress()
        ):
            PROFILE.count("scheduler.full_prefix_active_prefill_retries")
            return
        publication_order_event_ids = [state.lease.hint.event_id for state in pending]
        if self._prefetch_selector != "semantic":
            for state in pending:
                self._attempt_wait_prefetch(state)
            return

        if getattr(self, "_prefetch_require_full_prefix", False):
            # Full-prefix staging cannot act when no complete source fits.
            # Check current capacity on every retry, after expiry/coalescing,
            # before per-key score evidence or score requests are produced.
            # No source-readiness scan or cached capacity can hide a later
            # opportunity; the ordinary allocator checks still run on submit.
            pool = self.gpu_block_pool
            allocation_budget = (
                min(
                    max(0, int(pool.num_gpu_blocks * self._prefetch_budget_fraction)
                        - self._prefetch_reserved_blocks),
                    max(0, self._prefetch_free_capacity() - self._prefetch_headroom_blocks),
                ) if pool is not None else 0
            )
            if not any(0 < len(state.source.keys) <= allocation_budget for state in pending):
                PROFILE.count("scheduler.full_prefix_no_fit_retries")
                return

        nomination_started_ns = time.monotonic_ns()
        ranked: list[tuple[_WaitPrefetchState, dict[str, object], float | None]] = []
        for state in pending:
            candidate, aggregate = self._semantic_wait_nomination_candidate(
                state, now_ns
            )
            ranked.append((state, candidate, aggregate))
        ranked.sort(
            key=lambda item: (
                item[2] is None,
                -(item[2] if item[2] is not None else 0.0),
                item[0].source.run_id,
                item[0].source.cell_id,
                item[0].source.session_id,
                item[0].source.generation,
                item[0].source.observed_prefix_id,
                item[0].lease.hint.event_id,
            )
        )
        initial_reserved = self._prefetch_reserved_blocks
        total_budget = (
            int(self.gpu_block_pool.num_gpu_blocks * self._prefetch_budget_fraction)
            if self.gpu_block_pool is not None
            else 0
        )
        initial_free = (
            self._prefetch_free_capacity()
            if self.gpu_block_pool is not None
            else 0
        )
        shared_allocation_budget = min(
            max(0, total_budget - initial_reserved),
            max(0, initial_free - self._prefetch_headroom_blocks),
        )
        unscored_reason = None
        if getattr(self, "_scoring_schedule", "continuous") == "opportunity":
            eligible = [(state, candidate) for state, candidate, _ in ranked
                        if candidate["eligible_resident_prefix_keys"]]
            # All candidates fit, or only one can be selected: semantic
            # ranking cannot improve this allocation decision.
            needs_ranking = len(eligible) >= 2 and sum(
                len(candidate["eligible_resident_prefix_keys"]) for _, candidate in eligible
            ) > shared_allocation_budget > 0
            if needs_ranking:
                deferred = False
                attempts = self._decision_scoring_attempts
                live_signatures = set()
                for state, _candidate in eligible:
                    ready_keys = tuple(_ready_resident_prefix(
                        self.manager, state.publication_candidate_keys))
                    signature = (state.source.source_request_id, ready_keys)
                    live_signatures.add(signature)
                    # Only request missing/freshness-expired observations.
                    # A cached query itself expires; failures fall back after
                    # at most 50 ms without delaying demand serving.
                    if len(self._semantic_wait_key_scores(ready_keys, now_ns)) == len(ready_keys):
                        continue
                    requested_ns = attempts.get(signature)
                    if requested_ns is None:
                        attempts[signature] = now_ns
                        self._decision_scoring_requests[state.source.source_request_id] = list(ready_keys)
                        deferred = True
                    elif now_ns - requested_ns < 50_000_000:
                        deferred = True
                self._decision_scoring_attempts = {
                    key: value for key, value in attempts.items() if key in live_signatures}
                if deferred:
                    PROFILE.count("scheduler.semantic_decisions_deferred")
                    return
                PROFILE.count("scheduler.semantic_competing_decisions")
                unscored_reason = "score_unavailable_after_bounded_wait"
            else:
                PROFILE.count("scheduler.semantic_ranking_bypassed")
                unscored_reason = "ranking_cannot_change_allocation"
        eligible_candidate_blocks = sum(
            len(candidate["eligible_resident_prefix_keys"])
            for _, candidate, _ in ranked
        )
        scored_candidate_count = sum(
            aggregate is not None for _, _, aggregate in ranked
        )
        shared_budget_constrained = bool(
            scored_candidate_count >= 2
            and eligible_candidate_blocks > shared_allocation_budget
        )
        chosen: list[dict[str, object]] = []
        for state, candidate, _aggregate in ranked:
            submitted = (
                self._attempt_wait_prefetch(state)
                if unscored_reason is None
                else self._attempt_wait_prefetch(state, unscored_reason=unscored_reason)
            )
            if submitted:
                chosen.append(
                    {
                        "event_id": candidate["event_id"],
                        "session_id": candidate["session_id"],
                        "generation": candidate["generation"],
                        "observed_prefix_id": candidate["observed_prefix_id"],
                        "source_request_id": candidate["source_request_id"],
                        "selected_keys": [_event_key(key) for key in state.keys],
                    }
                )
        self._record_wait_hint_event(
            {
                "schema": "semantic-cross-session-prefetch-nomination-v1",
                "event": "nomination_decision",
                "selector": "semantic",
                "decision_ns": now_ns,
                "nomination_cpu_ns": time.monotonic_ns() - nomination_started_ns,
                "shared_allocation_budget_blocks": shared_allocation_budget,
                "shared_reserved_blocks": initial_reserved,
                "eligible_candidate_blocks": eligible_candidate_blocks,
                "scored_candidate_count": scored_candidate_count,
                "shared_budget_constrained": shared_budget_constrained,
                "candidate_publication_order_event_ids": (publication_order_event_ids),
                "tie_break": "aggregate_score_desc_then_source_identity",
                "highest_ranked_candidate": (
                    {
                        "event_id": ranked[0][1]["event_id"],
                        "session_id": ranked[0][1]["session_id"],
                        "observed_prefix_id": ranked[0][1]["observed_prefix_id"],
                        "source_request_id": ranked[0][1]["source_request_id"],
                        "aggregate_score": ranked[0][1]["aggregate_score"],
                    }
                    if ranked and ranked[0][2] is not None
                    else None
                ),
                "candidates": [candidate for _, candidate, _ in ranked],
                "chosen": chosen,
                "selected_blocks": sum(len(item["selected_keys"]) for item in chosen),
                "selected_bytes": sum(
                    len(item["selected_keys"]) * self._prefetch_bytes_per_key
                    for item in chosen
                ),
                "no_action_reason": (
                    None if chosen else "no_causal_candidate_or_shared_budget"
                ),
            }
        )

    def _attach_ready_wait_prefetch(self, state: _WaitPrefetchState) -> None:
        request_id = state.bound_request_id
        if request_id is None or request_id not in self._req_status:
            return
        if self._try_publish_wait_prefix(state):
            return
        binding_id = state.protection_binding_id or (
            f"wait-hint:{state.lease.hint.event_id}"
        )
        protection = self.manager.protect_staging_keys(binding_id, state.keys)
        if protection.get("outcome") not in {"accepted", "noop"}:
            preservation_state = getattr(self, "_preservation_requests", {}).get(
                request_id
            )
            if preservation_state is not None:
                preservation_state["prefetch_failure_reason"] = (
                    "staging_protection_rejected"
                )
                preservation_state["transfer_complete"] = False
            self._free_wait_prefetch(state, "staging_protection_rejected")
            return
        state.protection_binding_id = binding_id
        self._prefetched[request_id] = _PrefetchState(
            job_id=state.job_id if state.job_id is not None else -1,
            keys=list(state.keys),
            gpu_block_ids=list(state.gpu_block_ids),
            attempt_id=state.lease.hint.event_id,
        )
        state.status = "bound_ready"

    @profiled('scheduler.native_publish')
    def _try_publish_wait_prefix(self, state: _WaitPrefetchState) -> bool:
        """Publish a fenced exact prefix before native scheduler cache lookup."""
        if (
            not getattr(self, "_wait_hint_native_adoption", False)
            or not getattr(self, "_wait_hint_native_adoption_layout_supported", False)
            or self.config.num_workers != 1
            or len(self.config.kv_group_configs) != 1
            or config_blocks_per_chunk(self.config) != 1
            or state.status != "ready"
            or not state.source_released
            or state.pending_contributors != 0
        ):
            return False
        publish = getattr(self.gpu_block_pool, "publish_completed_prefix", None)
        req_status = self._req_status.get(state.bound_request_id)
        request = getattr(req_status, "req", None)
        transfer = state.transfer_record or {}
        if (
            publish is None
            or request is None
            or request.num_computed_tokens != 0
            or transfer.get("success") is not True
            or transfer.get("cancelled") is not False
            or transfer.get("contributors") != 1
            or transfer.get("expected_contributors") != 1
            or not isinstance(transfer.get("completed_at_ns"), int)
            or isinstance(transfer.get("completed_at_ns"), bool)
            or not 0 < transfer["completed_at_ns"] <= time.monotonic_ns()
        ):
            return False
        group = self.config.kv_group_configs[0]
        block_size = group_config_tokens_per_block(group)
        boundary = state.lease.hint.prefix_boundary_tokens
        count = boundary // block_size
        if (
            getattr(group, "group_idx", -1) != 0
            or group_config_tokens_per_chunk(group) != block_size
            or boundary != count * block_size
            or count == 0
            or len(state.keys) != count
            or len(state.gpu_block_ids) != count
            or len(set(state.keys)) != count
            or state.lease.state not in {"ready", "resumed"}
            or time.monotonic_ns() > state.lease.expires_at_ns
            or _observed_prefix_digest(
                (request.prompt_token_ids or [])[:boundary],
                self._wait_hint_namespace,
            ) != state.lease.hint.observed_prefix_id
        ):
            return False
        expected_keys = [make_offload_key(h, 0) for h in request.block_hashes[:count]]
        if (
            expected_keys != list(state.source.keys)
            or set(expected_keys) != set(state.keys)
        ):
            return False
        by_key = dict(zip(state.keys, state.gpu_block_ids))
        ids = [by_key[key] for key in expected_keys]
        if any(
            type(block_id) is not int
            or not 0 < block_id < self.gpu_block_pool.num_gpu_blocks
            for block_id in ids
        ):
            return False
        blocks = [self.gpu_block_pool.blocks[block_id] for block_id in ids]
        if not publish(
            request, blocks, block_size=block_size, kv_cache_group_id=0,
            completion_verified=True,
        ):
            return False
        published_ns = time.monotonic_ns()
        state.destination_released = True
        self._prefetch_reserved_blocks -= count
        state.status = "published"
        self._release_wait_prefetch_protection(state)
        event_id = state.lease.hint.event_id
        self._wait_prefetch.pop(event_id, None)
        self._wait_hint_request_bindings.pop(request.request_id, None)
        self._published_wait_prefixes[request.request_id] = {
            "event_id": event_id, "block_ids": ids,
            "published_at_ns": published_ns, "keys": expected_keys,
        }
        self._record_wait_hint_event({
            "schema": "semantic-known-prefix-wait-hint-event-v1",
            "event": "native_prefix_published", "event_id": event_id,
            **self._wait_state_identity(state), "request_id": request.request_id,
            "block_ids": ids, "published_at_ns": published_ns,
            "ownership": "native_prefix_cache_evictable", "gpu_splice_bytes": 0,
        })
        return True

    def _record_native_prefix_adoption(
        self, request: Request, blocks, num_external_tokens: int
    ) -> None:
        published = getattr(self, "_published_wait_prefixes", {}).pop(
            request.request_id, None
        )
        if published is None:
            return
        ids = published["block_ids"]
        prefix_blocks = blocks.blocks[0][:len(ids)]
        actual = [block.block_id for block in prefix_blocks]
        adopted = (
            num_external_tokens == 0
            and actual == ids
            and all(
                block.block_hash == key and block.ref_cnt > 0
                for block, key in zip(prefix_blocks, published["keys"])
            )
        )
        usable_ns = time.monotonic_ns()
        state = getattr(self, "_preservation_requests", {}).get(request.request_id)
        if state is not None:
            state["prefetch_native_adoption"] = {
                "published_at_ns": published["published_at_ns"],
                "adopted_at_ns": usable_ns if adopted else None,
                "same_physical_blocks": adopted, "published_block_ids": ids,
                "actual_block_ids": actual,
            }
            if adopted:
                state["prefetch_consumed_keys"] += len(ids)
                state["prefetch_consumed_blocks"] += len(ids)
                state["prefetch_splice_bytes"] = 0
                state["prefetch_request_usable_ns"] = usable_ns
                state["prefetch_request_usable_source"] = "native_allocator_cache_hit"
                state["prefetch_cleanup_released"] = True
        self._record_wait_hint_event({
            "schema": "semantic-known-prefix-wait-hint-event-v1",
            "event": "native_prefix_adopted" if adopted else "native_prefix_missed",
            "event_id": published["event_id"], "request_id": request.request_id,
            "published_block_ids": ids, "actual_block_ids": actual,
            "observed_at_ns": usable_ns, "gpu_splice_bytes": 0,
        })

    def _bind_wait_prefetch_transfer_evidence(
        self,
        wait_state: _WaitPrefetchState,
        preservation_state: dict[str, object],
        request_id: str,
    ) -> None:
        if wait_state.transfer_evidence_bound or wait_state.transfer_record is None:
            return
        record = dict(wait_state.transfer_record)
        record["request_id"] = request_id
        transfer_bytes = record.get("bytes")
        transfer_time_s = record.get("time_s")
        if (
            isinstance(transfer_bytes, bool)
            or not isinstance(transfer_bytes, int)
            or transfer_bytes <= 0
        ):
            preservation_state["prefetch_failure_reason"] = (
                "wait_prefetch_transfer_bytes_unavailable"
            )
            return
        preservation_state["prefetch_h2d_bytes"] = transfer_bytes
        if isinstance(transfer_time_s, (int, float)) and transfer_time_s >= 0.0:
            preservation_state["prefetch_time_s"] = transfer_time_s
        started_at_ns = record.get("started_at_ns")
        completed_at_ns = record.get("completed_at_ns")
        if isinstance(started_at_ns, int) and not isinstance(started_at_ns, bool):
            preservation_state["prefetch_t2_ns"] = started_at_ns
        if isinstance(completed_at_ns, int) and not isinstance(completed_at_ns, bool):
            preservation_state["prefetch_t3_ns"] = completed_at_ns
            preservation_state["prefetch_fence_ns"] = completed_at_ns
            preservation_state["prefetch_fence_complete"] = bool(
                record.get("success") is True
            )
        preservation_state.setdefault("prefetch_transfer_contributors", {})[
            str(record["job_id"])
        ] = list(record.get("contributor_ids", []))
        preservation_state.setdefault("speculative_transfer_job_statuses", []).append(
            record
        )
        _prefetch_attribution_state(preservation_state).setdefault(
            "transfer_jobs", []
        ).append(record)
        preservation_state["prefetch_attempted_keys"] = int(
            preservation_state.get("prefetch_attempted_keys", 0)
        ) + len(wait_state.keys)
        preservation_state["prefetch_attempted_blocks"] = int(
            preservation_state.get("prefetch_attempted_blocks", 0)
        ) + len(wait_state.gpu_block_ids)
        preservation_state["prefetch_submitted_jobs"] = (
            int(preservation_state.get("prefetch_submitted_jobs", 0)) + 1
        )
        preservation_state["prefetch_submitted_blocks"] = int(
            preservation_state.get("prefetch_submitted_blocks", 0)
        ) + len(wait_state.gpu_block_ids)
        self._record_block_pool_snapshot(
            preservation_state,
            "staging_ready",
            source="wait_prefetch_transfer_completion",
            job_id=int(record["job_id"]),
        )
        wait_state.transfer_evidence_bound = True

    def _complete_wait_prefetch_jobs(self, meta: SemanticWorkerMetadata) -> None:
        completed = dict(getattr(meta, "completed_jobs", {}) or {})
        failed = dict(getattr(meta, "failed_jobs", {}) or {})
        wait_jobs = getattr(self, "_wait_prefetch_jobs", {})
        for job_id, state in list(wait_jobs.items()):
            observed = int(completed.get(job_id, 0)) + int(failed.get(job_id, 0))
            if observed <= 0:
                continue
            state.pending_contributors -= observed
            if state.pending_contributors > 0:
                continue
            transfer = self._preservation_job_events.pop(job_id, None)
            self.manager.complete_load(state.keys, state.req_context)
            state.source_released = True
            wait_jobs.pop(job_id, None)
            meta.completed_jobs.pop(job_id, None)
            failed_jobs = getattr(meta, "failed_jobs", None)
            if isinstance(failed_jobs, dict):
                failed_jobs.pop(job_id, None)
            succeeded = (
                not failed.get(job_id)
                and transfer is not None
                and transfer.success
                and transfer.contributors == int(self.config.num_workers)
            )
            if transfer is not None:
                state.transfer_record = _transfer_event_record(
                    job_id,
                    {
                        "req_id": state.bound_request_id,
                        "is_load": True,
                        "is_prefetch": True,
                        "keys": list(state.keys),
                        "src_block_ids": None,
                        "dst_block_ids": [
                            (0, block_id) for block_id in state.gpu_block_ids
                        ],
                        "expected_contributors": int(self.config.num_workers),
                    },
                    transfer,
                )
            if (
                transfer is not None
                and transfer.success
                and isinstance(transfer.bytes, int)
                and transfer.bytes > 0
                and isinstance(transfer.time_s, (int, float))
                and transfer.time_s >= 0.0
            ):
                sample_ns_per_byte = transfer.time_s * 1_000_000_000 / transfer.bytes
                previous = self._prefetch_transfer_ns_per_byte
                self._prefetch_transfer_ns_per_byte = (
                    sample_ns_per_byte
                    if previous is None
                    else 0.3 * sample_ns_per_byte + 0.7 * previous
                )
                self._prefetch_cost_available_ns = time.monotonic_ns()
            completion_observed_ns = time.monotonic_ns()
            retention_expired = completion_observed_ns > state.lease.expires_at_ns
            if (
                not succeeded
                or state.status == "retired_pending_fence"
                or retention_expired
            ):
                self._free_wait_prefetch(
                    state,
                    ("transfer_failed" if not succeeded else "expired_after_fence"),
                )
                continue
            state.status = "ready"
            state.lease.state = "ready"
            self._attach_ready_wait_prefetch(state)
            if state.bound_request_id is not None:
                preservation_state = getattr(self, "_preservation_requests", {}).get(
                    state.bound_request_id
                )
                if preservation_state is not None and isinstance(
                    preservation_state.get("wait_hint"), dict
                ):
                    self._bind_wait_prefetch_transfer_evidence(
                        state, preservation_state, state.bound_request_id
                    )
                    preservation_state["wait_hint"]["staging_status"] = state.status
                    preservation_state["wait_hint"]["ready_at_ns"] = time.monotonic_ns()
            self._record_wait_hint_event(
                {
                    "schema": "semantic-known-prefix-wait-hint-event-v1",
                    "event": "staging_ready",
                    "event_id": state.lease.hint.event_id,
                    **self._wait_state_identity(state),
                    "job_id": job_id,
                    "ready_at_ns": completion_observed_ns,
                    "transfer_bytes": transfer.bytes,
                    "transfer_time_s": transfer.time_s,
                    "bound_request_id": state.bound_request_id,
                }
            )

    def has_pending_push_work(self) -> bool:
        return super().has_pending_push_work() or bool(
            getattr(self, "_wait_prefetch", {})
            or getattr(self, "_wait_prefetch_jobs", {})
            or getattr(self, "_pending_splice_releases", {})
            or (
                getattr(self, "_bounded_prefill_cuda_timing", False)
                and any(
                    not _prefill_cuda_terminal_ready(
                        request_id,
                        state,
                        bounded_prefill_cuda_timing=True,
                    )
                    for request_id, state in getattr(
                        self, "_preservation_finished", {}
                    ).items()
                )
            )
        )

    def _bind_wait_hint_request(self, request: Request) -> None:
        params = request.kv_transfer_params or {}
        event_id = params.get("wait_hint_event_id")
        if event_id is None:
            return
        state = (
            getattr(self, "_wait_prefetch", {}).get(event_id)
            if isinstance(event_id, str)
            else None
        )
        if state is None:
            request_identity = {
                field: params.get(field)
                for field in ("run_id", "cell_id", "session_id", "generation")
                if params.get(field) is not None
            }
            self._record_wait_hint_event(
                {
                    "schema": "semantic-known-prefix-wait-hint-event-v1",
                    "event": "resume_binding_rejected",
                    "event_id": event_id,
                    **request_identity,
                    "request_id": request.request_id,
                    "reason": "unknown_hint_event",
                }
            )
            return
        hint = state.lease.hint
        identity_matches = (
            params.get("external_request_id") == hint.owner_id
            and params.get("run_id") == hint.run_id
            and params.get("cell_id") == hint.cell_id
            and params.get("session_id") == hint.session_id
            and params.get("generation") == hint.generation
            and params.get("observed_prefix_id") == hint.observed_prefix_id
            and params.get("observed_prefix_boundary_tokens")
            == hint.prefix_boundary_tokens
            and self._observed_prefix_generations.get(
                (hint.run_id, hint.cell_id, hint.session_id)
            )
            == hint.generation
        )
        now_ns = time.monotonic_ns()
        if not identity_matches or not state.lease.active_at(now_ns):
            self._record_wait_hint_event(
                {
                    "schema": "semantic-known-prefix-wait-hint-event-v1",
                    "event": "resume_binding_rejected",
                    "event_id": event_id,
                    **self._wait_state_identity(state),
                    "request_id": request.request_id,
                    "reason": (
                        "resume_identity_mismatch"
                        if not identity_matches
                        else "hint_expired"
                    ),
                }
            )
            return
        if (
            state.status == "pending"
            and not state.coalescing_complete
            and self._wait_hint_coalescing_enabled()
        ):
            coalescing_states = [
                candidate
                for candidate in self._wait_prefetch.values()
                if candidate.status == "pending" and not candidate.coalescing_complete
            ]
            _representatives, _duplicates, identities = (
                self._coalescing_eligible_sources(coalescing_states, now_ns)
            )
            self._finish_wait_hint_coalescing(
                coalescing_states,
                now_ns=now_ns,
                reason="demand_arrived",
                identities=identities,
                retire_duplicates=False,
            )
        state.bound_request_id = request.request_id
        state.lease.resume()
        self._wait_hint_request_bindings[request.request_id] = event_id
        if state.status == "ready":
            self._attach_ready_wait_prefetch(state)
        preservation_state = getattr(self, "_preservation_requests", {}).get(
            request.request_id
        )
        if preservation_state is not None:
            preservation_state["wait_hint"] = {
                "event_id": hint.event_id,
                "owner_id": hint.owner_id,
                "run_id": hint.run_id,
                "cell_id": hint.cell_id,
                "session_id": hint.session_id,
                "generation": hint.generation,
                "observed_prefix_id": hint.observed_prefix_id,
                "prefix_boundary_tokens": hint.prefix_boundary_tokens,
                "observation_available_ns": hint.observation_available_ns,
                "input_publication_ns": hint.input_publication_ns,
                "nomination_deadline_ns": (state.lease.nomination_deadline_ns),
                "expires_at_ns": state.lease.expires_at_ns,
                "bound_at_ns": now_ns,
                "staging_status": state.status,
                "selection": dict(state.selection_evidence),
            }
            self._bind_wait_prefetch_transfer_evidence(
                state, preservation_state, request.request_id
            )
        self._record_wait_hint_event(
            {
                "schema": "semantic-known-prefix-wait-hint-event-v1",
                "event": "resume_bound",
                "event_id": event_id,
                **self._wait_state_identity(state),
                "request_id": request.request_id,
                "staging_status": state.status,
                "bound_at_ns": now_ns,
            }
        )

    def _reclaim_wait_prefetch_for_demand(
        self,
        request: Request,
        num_computed_tokens: int,
        *,
        protected_event_id: str | None,
    ) -> None:
        if (
            getattr(self, "_staging_mode", "dynamic_scratch") != "owned_reclaimable"
            or self.gpu_block_pool is None
        ):
            return
        prompt_tokens = int(getattr(request, "num_prompt_tokens", 0))
        remaining_tokens = max(0, prompt_tokens - int(num_computed_tokens))
        tokens_per_block = group_config_tokens_per_block(
            self.config.kv_group_configs[0]
        )
        demand_blocks = cdiv(remaining_tokens, tokens_per_block)
        free_blocks = self.gpu_block_pool.get_num_free_blocks()
        shortfall = max(0, demand_blocks - free_blocks)
        if shortfall <= 0:
            return
        reclaimed = 0
        candidates = sorted(
            (
                state
                for event_id, state in getattr(self, "_wait_prefetch", {}).items()
                if event_id != protected_event_id
                and state.bound_request_id is None
                and state.status == "ready"
            ),
            key=lambda state: (
                state.lease.hint.input_publication_ns,
                state.lease.hint.event_id,
            ),
        )
        for state in candidates:
            block_count = len(state.gpu_block_ids)
            self._free_wait_prefetch(state, "demand_reclaim")
            reclaimed += block_count
            if reclaimed >= shortfall:
                break
        # Legacy preemption prefetches own scratch allocations too. Their
        # splice happens only *after* native demand admission allocates its
        # destination blocks. Keeping completed scratch when admission cannot
        # fit can therefore deadlock the requesting request on its own source
        # buffers. Drop that optional copy and let the normal exact CPU load
        # serve demand. Completion fences remain mandatory, including for
        # another request's speculative source; pending splice owners have
        # already left _prefetched and are deliberately not reclaimed here.
        reclaimed_legacy = 0
        for req_id, state in list(getattr(self, "_prefetched", {}).items()):
            if reclaimed >= shortfall:
                break
            # Bound wait-hint sources share this index but carry their own
            # CPU protection and lease lifecycle. Only legacy scratch belongs
            # to this reclamation path; the wait-state path above deliberately
            # excludes bound sources.
            if state.attempt_id in getattr(self, "_wait_prefetch", {}):
                continue
            job = self._jobs.get(state.job_id)
            if job is not None and job.pending_count > 0:
                continue
            if state.attempt_id == protected_event_id and protected_event_id is not None:
                continue
            block_count = len(state.gpu_block_ids)
            self._release_prefetch(req_id)
            reclaimed += block_count
            reclaimed_legacy += block_count
        in_flight_blocks = sum(
            len(state.gpu_block_ids)
            for state in getattr(self, "_wait_prefetch", {}).values()
            if state.status in {"in_flight", "retired_pending_fence"}
        )
        in_flight_blocks += sum(
            len(state.gpu_block_ids)
            for state in getattr(self, "_prefetched", {}).values()
            if (job := self._jobs.get(state.job_id)) is not None
            and job.pending_count > 0
        )
        self._record_wait_hint_event(
            {
                "schema": "semantic-known-prefix-wait-hint-event-v1",
                "event": "demand_reclaim",
                "request_id": request.request_id,
                "demand_blocks": demand_blocks,
                "free_blocks_before": free_blocks,
                "shortfall_blocks": shortfall,
                "reclaimed_ready_blocks": reclaimed,
                "reclaimed_legacy_ready_blocks": reclaimed_legacy,
                "reclaimed_ready_bytes": reclaimed * self._prefetch_bytes_per_key,
                "in_flight_unreclaimable_blocks": in_flight_blocks,
                "in_flight_unreclaimable_bytes": (
                    in_flight_blocks * self._prefetch_bytes_per_key
                ),
                "demand_wait_required": reclaimed < shortfall and in_flight_blocks > 0,
                "observed_at_ns": time.monotonic_ns(),
            }
        )

    def _block_pool_snapshot(
        self,
        phase: str,
        *,
        source: str = "scheduler",
        job_id: int | None = None,
    ) -> dict[str, object]:
        """Capture scheduler-owned block-pool accounting at one phase.

        Speculative reservations are removed from the upstream free queue, so
        a free-block count alone cannot distinguish active ownership from
        speculative ownership. Keep both counters and derive the remaining
        active count explicitly. This is telemetry only: it does not reserve,
        reclaim, or otherwise mutate the pool.
        """
        snapshot: dict[str, object] = {
            "schema": "semantic-block-pool-snapshot-v1",
            "phase": phase,
            "source": source,
            "observed_at_ns": time.monotonic_ns(),
            "available": False,
        }
        if job_id is not None:
            snapshot["job_id"] = int(job_id)
        pool = getattr(self, "gpu_block_pool", None)
        if pool is None:
            snapshot["reason"] = "gpu_block_pool_unbound"
            return snapshot
        null_block = getattr(pool, "null_block", None)
        if null_block is None or getattr(null_block, "is_null", None) is not True:
            snapshot["reason"] = "null_block_identity_unavailable"
            return snapshot
        try:
            total_blocks = int(pool.num_gpu_blocks)
            free_blocks = int(pool.get_num_free_blocks())
            null_block_count = 1
            reserved_blocks = int(getattr(self, "_prefetch_reserved_blocks", 0))
            configured_headroom = int(getattr(self, "_prefetch_headroom_blocks", 0))
        except (AttributeError, TypeError, ValueError):
            snapshot["reason"] = "block_pool_counters_unavailable"
            return snapshot
        if (
            total_blocks < 0
            or free_blocks < 0
            or reserved_blocks < 0
            or configured_headroom < 0
        ):
            snapshot["reason"] = "block_pool_counter_out_of_range"
            return snapshot
        schedulable_capacity = max(0, total_blocks - null_block_count)
        raw_referenced_blocks = schedulable_capacity - free_blocks - reserved_blocks
        staging_block_ids: set[int] = set()
        for wait_state in getattr(self, "_wait_prefetch", {}).values():
            staging_block_ids.update(int(value) for value in wait_state.gpu_block_ids)
        for prefetch_state in getattr(self, "_prefetched", {}).values():
            staging_block_ids.update(
                int(value) for value in prefetch_state.gpu_block_ids
            )
        for block_ids, _remaining in getattr(
            self, "_pending_splice_releases", {}
        ).values():
            staging_block_ids.update(int(value) for value in block_ids)
        for block_ids, _remaining in getattr(
            self, "_reset_prefetch_releases", {}
        ).values():
            staging_block_ids.update(int(value) for value in block_ids)
        staging_block_ids.discard(int(getattr(null_block, "block_id", 0)))
        usable_context_blocks = []
        pool_blocks = getattr(pool, "blocks", None)
        if isinstance(pool_blocks, (list, tuple)):
            for block in pool_blocks:
                block_id = int(block.block_id)
                if block_id in staging_block_ids or block.is_null:
                    continue
                ref_cnt = int(getattr(block, "ref_cnt", 0))
                block_hash = getattr(block, "block_hash", None)
                if ref_cnt <= 0 and block_hash is None:
                    continue
                usable_context_blocks.append(
                    {
                        "block_id": block_id,
                        "ref_cnt": ref_cnt,
                        "block_hash_digest": (
                            hashlib.sha256(repr(block_hash).encode()).hexdigest()
                            if block_hash is not None
                            else None
                        ),
                        "reusable_prefix": block_hash is not None,
                    }
                )
        physical_block_bytes = int(getattr(self, "_prefetch_bytes_per_key", 0))
        snapshot.update(
            {
                "available": True,
                "total_blocks": total_blocks,
                "null_block_count": null_block_count,
                "schedulable_capacity_blocks": schedulable_capacity,
                "free_block_queue_blocks": free_blocks,
                "scheduler_prefetch_reserved_blocks_total": reserved_blocks,
                "estimated_referenced_blocks": max(0, raw_referenced_blocks),
                "accounting_underflow_blocks": min(0, raw_referenced_blocks),
                "accounting_consistent": raw_referenced_blocks >= 0,
                "configured_prefetch_headroom_blocks": configured_headroom,
                "scheduler_free_headroom_blocks": max(
                    0, free_blocks - configured_headroom
                ),
            }
        )
        if isinstance(pool_blocks, (list, tuple)):
            snapshot.update(
                {
                    "scheduler_prefetch_reserved_block_ids": sorted(staging_block_ids),
                    "staging_identity_complete": len(staging_block_ids)
                    == reserved_blocks,
                    "staging_reserved_bytes": len(staging_block_ids)
                    * physical_block_bytes,
                    "gpu_usable_context_blocks": usable_context_blocks,
                    "gpu_usable_context_block_ids": [
                        record["block_id"] for record in usable_context_blocks
                    ],
                    "gpu_usable_context_bytes": len(usable_context_blocks)
                    * physical_block_bytes,
                    "gpu_schedulable_capacity_bytes": schedulable_capacity
                    * physical_block_bytes,
                    "physical_block_bytes": physical_block_bytes,
                }
            )
        return snapshot

    def _record_block_pool_snapshot(
        self,
        state: dict[str, object] | None,
        phase: str,
        *,
        source: str = "scheduler",
        job_id: int | None = None,
    ) -> None:
        """Append an explicit BlockPool observation to a request episode."""
        if state is None:
            return
        snapshots = state.setdefault("prefetch_block_pool_snapshots", [])
        if not isinstance(snapshots, list):
            snapshots = []
            state["prefetch_block_pool_snapshots"] = snapshots
        snapshot = self._block_pool_snapshot(phase, source=source, job_id=job_id)
        snapshot["episode_id"] = int(state.get("episode_id", 0))
        snapshot["preemption_index"] = int(state.get("preemption_count", 0))
        snapshot_count = int(state.get("prefetch_block_pool_snapshot_count", 0))
        state["prefetch_block_pool_snapshot_count"] = snapshot_count + 1
        if len(snapshots) >= 64:
            # Keep the first half and the most recent half so long-lived
            # requests cannot grow event payloads without bound.
            snapshots.pop(32)
            state["prefetch_block_pool_snapshots_truncated"] = True
        snapshots.append(snapshot)
        self._record_cpu_cache_snapshot(state, phase, source=source, job_id=job_id)

    def _cpu_cache_snapshot(
        self,
        phase: str,
        *,
        source: str = "scheduler",
        job_id: int | None = None,
    ) -> dict[str, object]:
        """Capture manager-owned CPU-tier occupancy without mutating it."""
        snapshot: dict[str, object] = {
            "schema": "semantic-cpu-cache-snapshot-v1",
            "phase": phase,
            "source": source,
            "observed_at_ns": time.monotonic_ns(),
            "available": False,
        }
        if job_id is not None:
            snapshot["job_id"] = int(job_id)
        manager = getattr(self, "manager", None)
        if manager is None:
            snapshot["reason"] = "cpu_manager_unavailable"
            return snapshot
        try:
            total_blocks = int(manager._num_blocks)
            allocated_blocks = int(manager._num_allocated_blocks)
            free_list_blocks = len(manager._free_list)
            evictable_blocks = int(manager._num_evictable_cache_blocks)
            write_pending_blocks = int(manager._num_write_pending_blocks)
            policy = manager._policy
        except (AttributeError, TypeError, ValueError):
            snapshot["reason"] = "cpu_cache_counters_unavailable"
            return snapshot
        if any(
            value < 0
            for value in (
                total_blocks,
                allocated_blocks,
                free_list_blocks,
                evictable_blocks,
                write_pending_blocks,
            )
        ):
            snapshot["reason"] = "cpu_cache_counter_out_of_range"
            return snapshot
        policy_impl = getattr(policy, "_lru", policy)
        policy_blocks = getattr(policy_impl, "blocks", None)
        if isinstance(policy_blocks, dict):
            resident_block_map = dict(policy_blocks)
        else:
            resident_block_map = {}
            for name in ("t1", "t2"):
                partition = getattr(policy_impl, name, {})
                if isinstance(partition, dict):
                    resident_block_map.update(partition)
        resident_entries = len(resident_block_map)
        ghost_entries = sum(
            len(getattr(policy_impl, name, {}))
            for name in ("b1", "b2")
            if isinstance(getattr(policy_impl, name, {}), dict)
        )
        raw_non_evictable = allocated_blocks - free_list_blocks - evictable_blocks
        physical_block_bytes = int(getattr(self, "_prefetch_bytes_per_key", 0))
        resident_blocks = [
            {
                "key": _event_key(key),
                "cpu_block_id": int(block.block_id),
                "ready": bool(block.is_ready),
                "ref_cnt": int(block.ref_cnt),
            }
            for key, block in sorted(
                resident_block_map.items(), key=lambda item: _event_key_sort(item[0])
            )
        ]
        ready_resident_blocks = sum(record["ready"] for record in resident_blocks)
        snapshot.update(
            {
                "available": True,
                "total_blocks": total_blocks,
                "allocated_blocks": allocated_blocks,
                "free_list_blocks": free_list_blocks,
                "free_capacity_blocks": (
                    total_blocks - allocated_blocks + free_list_blocks
                ),
                "evictable_blocks": evictable_blocks,
                "write_pending_blocks": write_pending_blocks,
                "estimated_non_evictable_blocks": max(0, raw_non_evictable),
                "accounting_underflow_blocks": min(0, raw_non_evictable),
                "accounting_consistent": raw_non_evictable >= 0,
                "policy_resident_entries": resident_entries,
                "resident_blocks": resident_blocks,
                "resident_bytes": resident_entries * physical_block_bytes,
                "ready_resident_bytes": ready_resident_blocks * physical_block_bytes,
                "cpu_capacity_bytes": total_blocks * physical_block_bytes,
                "actual_offloaded_fraction_numerator_bytes": ready_resident_blocks
                * physical_block_bytes,
                "actual_offloaded_fraction_denominator_bytes": total_blocks
                * physical_block_bytes,
                "actual_offloaded_fraction": (
                    ready_resident_blocks / total_blocks if total_blocks else None
                ),
                "write_pending_bytes": write_pending_blocks * physical_block_bytes,
                "physical_block_bytes": physical_block_bytes,
                "policy_ghost_entries": ghost_entries,
                "policy_class": type(policy).__name__,
            }
        )
        return snapshot

    def _record_cpu_cache_snapshot(
        self,
        state: dict[str, object] | None,
        phase: str,
        *,
        source: str = "scheduler",
        job_id: int | None = None,
    ) -> None:
        """Append a bounded CPU-tier occupancy observation to an episode."""
        if state is None:
            return
        snapshots = state.setdefault("prefetch_cpu_cache_snapshots", [])
        if not isinstance(snapshots, list):
            snapshots = []
            state["prefetch_cpu_cache_snapshots"] = snapshots
        snapshot = self._cpu_cache_snapshot(phase, source=source, job_id=job_id)
        snapshot["episode_id"] = int(state.get("episode_id", 0))
        snapshot["preemption_index"] = int(state.get("preemption_count", 0))
        count = int(state.get("prefetch_cpu_cache_snapshot_count", 0))
        state["prefetch_cpu_cache_snapshot_count"] = count + 1
        if len(snapshots) >= 64:
            snapshots.pop(32)
            state["prefetch_cpu_cache_snapshots_truncated"] = True
        snapshots.append(snapshot)

    def _register_lifecycle_request(self, request: Request) -> None:
        """Register only explicit application lifecycle metadata.

        The application supplies identity and intent, never private
        OffloadKeys. Exact ownership keys are added later from scheduler-owned
        request state when the request finishes.
        """
        if not self._lifecycle_release_enabled:
            return
        params = getattr(request, "kv_transfer_params", None) or {}
        if "semantic_lifecycle" not in params:
            return
        raw = params.get("semantic_lifecycle")
        self._lifecycle_request_params[request.request_id] = raw
        if not isinstance(raw, dict):
            return
        owner_id = raw.get("owner_id")
        generation = raw.get("generation")
        if (
            not isinstance(owner_id, str)
            or not owner_id
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            return
        try:
            self._lifecycle_controller.register_owner(
                owner_id, generation, (), state="active"
            )
        except ValueError:
            # A stale/reused owner identity is surfaced as a rejected hint at
            # finalization; request admission itself must remain fail-closed.
            return

    def _lifecycle_request_keys(self, request_id: str) -> tuple[OffloadKey, ...]:
        req_status = self._req_status.get(request_id)
        if req_status is None:
            return ()
        keys: list[OffloadKey] = []
        for group_state in getattr(req_status, "group_states", ()) or ():
            group_keys = getattr(group_state, "offload_keys", ())
            if isinstance(group_keys, (list, tuple)):
                keys.extend(group_keys)
        return tuple(dict.fromkeys(keys))

    def _serialize_lifecycle_hint_result(
        self,
        result: LifecycleHintResult,
    ) -> dict[str, object]:
        manager_result = result.manager_result
        return {
            "status": result.status,
            "event_id": result.event_id,
            "owner_id": result.owner_id,
            "generation": result.generation,
            "reason": result.reason,
            "released_keys": (
                [_event_key(key) for key in manager_result.released_keys]
                if manager_result is not None
                else []
            ),
            "protected_keys": (
                [_event_key(key) for key in manager_result.protected_keys]
                if manager_result is not None
                else []
            ),
            "affected_owner_ids": (
                list(manager_result.affected_owner_ids)
                if manager_result is not None
                else []
            ),
            "operation_count": (
                manager_result.operation_count if manager_result is not None else 0
            ),
            "key_effects": (
                list(manager_result.key_effects) if manager_result is not None else []
            ),
            "bookkeeping": self._lifecycle_controller.bookkeeping_stats(),
        }

    def _process_lifecycle_request(
        self,
        request: Request | str,
        preservation_state: dict[str, object] | None = None,
        *,
        raw_override: object = _MISSING_LIFECYCLE,
        request_keys_override: tuple[OffloadKey, ...] | None = None,
        req_context_override: ReqContext | None = None,
    ) -> LifecycleHintResult | None:
        """Finalize an explicit application hint using scheduler-owned keys."""
        if not self._lifecycle_release_enabled:
            return None
        request_id = request if isinstance(request, str) else request.request_id
        if raw_override is _MISSING_LIFECYCLE:
            if request_id not in self._lifecycle_request_params:
                return None
            raw = self._lifecycle_request_params.pop(request_id)
        else:
            raw = raw_override
        req_status = self._req_status.get(request_id)
        req_context = req_context_override or getattr(req_status, "req_context", None)
        if not isinstance(req_context, ReqContext):
            req_context = ReqContext(req_id=request_id)
        if isinstance(raw, dict):
            event_id = raw.get("event_id", "")
            owner_id = raw.get("owner_id", "")
            generation = raw.get("generation", -1)
            release_requested = raw.get("release", False)
            requested_owner_state = raw.get(
                "owner_state", "dead" if release_requested else "active"
            )
        else:
            event_id = owner_id = ""
            generation = -1
            release_requested = False
            requested_owner_state = "active"
        valid_identity = (
            isinstance(event_id, str)
            and bool(event_id)
            and isinstance(owner_id, str)
            and bool(owner_id)
            and isinstance(generation, int)
            and not isinstance(generation, bool)
            and generation >= 0
            and isinstance(release_requested, bool)
            and requested_owner_state in ("future", "active", "dead")
        )
        hint_fields_valid = valid_identity
        request_keys = (
            tuple(request_keys_override)
            if request_keys_override is not None
            else self._lifecycle_request_keys(request_id)
        )
        if preservation_state is not None:
            self._record_cpu_cache_snapshot(
                preservation_state,
                "lifecycle_before",
                source="lifecycle_hint_before",
            )
        lifecycle_start_ns = time.monotonic_ns()
        if valid_identity:
            try:
                self._lifecycle_controller.register_owner(
                    owner_id,
                    generation,
                    request_keys,
                    state=requested_owner_state,
                )
                if release_requested and requested_owner_state != "active":
                    self._lifecycle_controller.set_owner_state(
                        owner_id, generation, "dead"
                    )
            except ValueError:
                valid_identity = False
        if valid_identity and release_requested:
            hint = LifecycleReleaseHint(
                event_id,
                owner_id,
                generation,
                request_keys,
            )
            result = self._lifecycle_controller.submit(hint, req_context)
        elif valid_identity:
            result = self._lifecycle_controller.submit(None, req_context)
        else:
            hint = LifecycleReleaseHint(
                str(event_id) if isinstance(event_id, str) else "",
                str(owner_id) if isinstance(owner_id, str) else "",
                generation if isinstance(generation, int) else -1,
                request_keys if hint_fields_valid else (),
            )
            result = self._lifecycle_controller.submit(hint, req_context)
        # The preservation event is the durable per-request audit record. Do
        # not retain a second unbounded controller/manager outcome ledger in a
        # long-lived serving process; seen identities and generation
        # tombstones remain available for idempotence and stale rejection.
        self._lifecycle_controller.drain_ledger()
        drain_manager_ledger = getattr(
            self.manager, "drain_lifecycle_release_ledger", None
        )
        if callable(drain_manager_ledger):
            drain_manager_ledger()
        if preservation_state is not None:
            lifecycle_release = self._serialize_lifecycle_hint_result(result)
            lifecycle_release["release_timing"] = getattr(
                self, "_lifecycle_release_timing", "inline"
            )
            lifecycle_release["requested_owner_state"] = requested_owner_state
            lifecycle_release["processing_time_s"] = (
                time.monotonic_ns() - lifecycle_start_ns
            ) / 1_000_000_000
            operation_count = lifecycle_release.get("operation_count")
            lifecycle_release["requested_key_count"] = (
                max(0, int(operation_count) - 1)
                if isinstance(operation_count, int)
                and not isinstance(operation_count, bool)
                else None
            )
            lifecycle_release["released_key_count"] = len(
                lifecycle_release.get("released_keys", [])
            )
            lifecycle_release["protected_key_count"] = len(
                lifecycle_release.get("protected_keys", [])
            )
            lifecycle_release["affected_owner_count"] = len(
                lifecycle_release.get("affected_owner_ids", [])
            )
            lifecycle_release["preemption_count_at_release"] = int(
                preservation_state.get("preemption_count", 0)
            )
            lifecycle_release["release_before_first_preemption"] = (
                preservation_state.get("preemption_count", 0) == 0
            )
            preservation_state["lifecycle_release"] = lifecycle_release
            self._record_cpu_cache_snapshot(
                preservation_state,
                "lifecycle_after",
                source="lifecycle_hint_after",
            )
            snapshots = preservation_state.get("prefetch_cpu_cache_snapshots", [])
            if isinstance(snapshots, list):
                before = next(
                    (
                        item
                        for item in reversed(snapshots)
                        if isinstance(item, dict)
                        and item.get("phase") == "lifecycle_before"
                    ),
                    None,
                )
                after = next(
                    (
                        item
                        for item in reversed(snapshots)
                        if isinstance(item, dict)
                        and item.get("phase") == "lifecycle_after"
                    ),
                    None,
                )
                if isinstance(before, dict) and isinstance(after, dict):
                    effects: dict[str, object] = {}
                    for field in (
                        "allocated_blocks",
                        "evictable_blocks",
                        "free_capacity_blocks",
                        "free_list_blocks",
                        "policy_resident_entries",
                        "policy_ghost_entries",
                    ):
                        before_value = before.get(field)
                        after_value = after.get(field)
                        if isinstance(before_value, int) and isinstance(
                            after_value, int
                        ):
                            effects[f"{field}_before"] = before_value
                            effects[f"{field}_after"] = after_value
                            effects[f"{field}_delta"] = after_value - before_value
                    lifecycle_release["cpu_cache_effects"] = effects
        return result

    def _defer_lifecycle_request(
        self, request: Request, preservation_state: dict[str, object]
    ) -> None:
        """Queue a terminal hint for the next scheduler metadata boundary."""
        request_id = request.request_id
        if request_id in self._pending_lifecycle_releases:
            return
        if request_id not in self._lifecycle_request_params:
            return
        raw = self._lifecycle_request_params.pop(request_id)
        req_status = self._req_status.get(request_id)
        req_context = getattr(req_status, "req_context", None)
        if not isinstance(req_context, ReqContext):
            req_context = ReqContext(req_id=request_id)
        self._pending_lifecycle_releases[request_id] = {
            "raw": raw,
            "request_keys": self._lifecycle_request_keys(request_id),
            "req_context": req_context,
        }
        preservation_state["lifecycle_release_timing"] = "scheduler_boundary"

    def _process_deferred_lifecycle_requests(self) -> None:
        """Apply queued terminal hints at one scheduler-owned boundary."""
        for request_id, pending in list(self._pending_lifecycle_releases.items()):
            state = self._preservation_finished.get(request_id)
            if state is None:
                # request_finished publishes this state before the base
                # connector can tear down its request status. Do not drop an
                # application fact if callback order changes unexpectedly.
                continue
            self._process_lifecycle_request(
                request_id,
                state,
                raw_override=pending["raw"],
                request_keys_override=tuple(pending["request_keys"]),
                req_context_override=pending["req_context"],
            )
            self._pending_lifecycle_releases.pop(request_id, None)

    def _next_preservation_event_sequence(self) -> int:
        sequence = int(getattr(self, "_preservation_event_sequence", 0))
        self._preservation_event_sequence = sequence + 1
        return sequence

    def _ordered_request_key_positions(
        self, request: Request
    ) -> tuple[list[dict[str, object]], bool]:
        """Read the scheduler-owned ordered key chain without cache access."""
        req_status = self._req_status.get(request.request_id)
        prompt_tokens = getattr(request, "prompt_token_ids", None)
        if req_status is None or not isinstance(prompt_tokens, (list, tuple)):
            return [], False
        return self._request_key_positions(request.request_id, len(prompt_tokens))

    def _request_key_positions(
        self, request_id: str, prompt_token_count: int
    ) -> tuple[list[dict[str, object]], bool]:
        req_status = self._req_status.get(request_id)
        if req_status is None or prompt_token_count < 0:
            return [], False
        group_states = getattr(req_status, "group_states", ())
        group_configs = self.config.kv_group_configs
        if len(group_states) != len(group_configs):
            return [], False
        positions: list[dict[str, object]] = []
        complete = True
        for group_idx, (group_state, group_config) in enumerate(
            zip(group_states, group_configs)
        ):
            keys = getattr(group_state, "offload_keys", None)
            if not isinstance(keys, (list, tuple)):
                return [], False
            tokens_per_chunk = group_config_tokens_per_chunk(group_config)
            if tokens_per_chunk <= 0:
                return [], False
            expected_chunks = prompt_token_count // tokens_per_chunk
            if len(keys) < expected_chunks:
                complete = False
            for chunk_index, key in enumerate(keys):
                token_start = chunk_index * tokens_per_chunk
                if token_start >= prompt_token_count:
                    break
                token_end = min(prompt_token_count, token_start + tokens_per_chunk)
                positions.append(
                    {
                        "key": key,
                        "group_idx": group_idx,
                        "chunk_index": chunk_index,
                        "token_start": token_start,
                        "token_end": token_end,
                        "complete_chunk": token_end - token_start == tokens_per_chunk,
                        "provenance": "scheduler_request_group_state",
                        "state": "planned",
                    }
                )
        return positions, complete

    def _refresh_ordered_key_positions(self, request_id: str) -> None:
        """Capture keys only after native ``update_offload_keys`` populated them."""
        state = self._preservation_requests.get(request_id)
        if state is None:
            return
        prompt_token_count = state.get("prompt_token_count")
        if not isinstance(prompt_token_count, int):
            return
        positions, complete = self._request_key_positions(
            request_id, prompt_token_count
        )
        previous = state.get("ordered_key_positions", [])
        if len(positions) < len(previous):
            return
        state["ordered_key_positions"] = positions
        state["ordered_key_positions_complete"] = complete
        state["ordered_key_positions_captured_at_ns"] = time.monotonic_ns()
        state["ordered_key_positions_capture_provenance"] = (
            "after_native_update_offload_keys"
        )

    @staticmethod
    def _position_records_for_keys(
        state: Mapping[str, object], keys
    ) -> list[dict[str, object]]:
        remaining: dict[object, int] = defaultdict(int)
        for key in keys:
            remaining[key] += 1
        selected: list[dict[str, object]] = []
        for record in state.get("ordered_key_positions", []):
            key = record.get("key")
            if remaining.get(key, 0) <= 0:
                continue
            selected.append(dict(record))
            remaining[key] -= 1
        return selected

    def _store_source_key_positions(
        self, request_id: str, keys, scheduled_tokens: int
    ) -> list[dict[str, object]]:
        """Describe store sources independently of prompt demand positions.

        Use the same native offloadable extent as the transfer plan. Running
        stores are planned after this scheduled forward and become committed
        only after the worker's store fence; finished stores are capped at
        processed tokens, excluding final samples and in-flight work.
        """
        req_status = self._req_status.get(request_id)
        request = getattr(req_status, "req", None)
        computed = getattr(request, "num_computed_tokens", None)
        if (
            req_status is None or type(computed) is not int or computed < 0
            or type(scheduled_tokens) is not int or scheduled_tokens < 0
        ):
            return []
        extent = self._calc_num_offloadable_tokens(
            req_status, computed + (0 if request.is_finished() else scheduled_tokens),
        )
        positions, complete = self._request_key_positions(request_id, extent)
        if not complete:
            return []
        return self._position_records_for_keys(
            {"ordered_key_positions": [record for record in positions
                                      if record["complete_chunk"]]}, keys,
        )

    @staticmethod
    def _complete_positions_in_token_range(
        state: Mapping[str, object], start_token: int, end_token: int
    ) -> list[dict[str, object]]:
        return [
            dict(record)
            for record in state.get("ordered_key_positions", [])
            if record.get("complete_chunk") is True
            and isinstance(record.get("token_start"), int)
            and isinstance(record.get("token_end"), int)
            and int(record["token_start"]) >= start_token
            and int(record["token_end"]) <= end_token
        ]

    def _key_lifecycle_for_state(
        self, state: Mapping[str, object]
    ) -> dict[str, list[dict[str, object]]]:
        keys = {
            record.get("key")
            for record in state.get("ordered_key_positions", [])
            if record.get("key") is not None
        }
        commits = [
            dict(event)
            for key in keys
            for event in getattr(self, "_preservation_key_commits", {}).get(key, ())
        ]
        evictions = [
            dict(event)
            for key in keys
            for event in getattr(self, "_preservation_key_evictions", {}).get(key, ())
        ]
        commits.sort(key=lambda event: int(event["sequence"]))
        evictions.sort(key=lambda event: int(event["sequence"]))
        return {"cpu_commits": commits, "cpu_evictions": evictions}

    def _record_cpu_commit_events(
        self,
        *,
        job_id: int,
        state: dict[str, object],
        key_positions,
        completed_at_ns: int | None,
    ) -> None:
        if not getattr(self, "_preservation_audit_enabled", True):
            return
        completion_observed_at_ns = time.monotonic_ns()
        for record in key_positions:
            commit = {
                **dict(record),
                "state": "committed_cpu_ready",
                "job_id": job_id,
                "source_request_id": state["request_id"],
                "manifest_request_id": state.get("manifest_request_id"),
                "run_id": state.get("run_id"),
                "cell_id": state.get("cell_id"),
                "session_id": state.get("policy_session_id"),
                "generation": state.get("generation"),
                "committed_at_ns": completed_at_ns,
                "completion_observed_at_ns": completion_observed_at_ns,
                "sequence": self._next_preservation_event_sequence(),
                "provenance": "worker_store_completion",
            }
            state["cpu_commit_events"].append(commit)
            self._preservation_key_commits[record["key"]].append(commit)

    def _record_cpu_eviction_events(self, evicted_keys) -> None:
        if not getattr(self, "_preservation_audit_enabled", True):
            return
        for key in evicted_keys:
            observed_at_ns = time.monotonic_ns()
            commits = self._preservation_key_commits.get(key, ())
            commit = commits[-1] if commits else None
            eviction = {
                "key": key,
                "state": "committed_cpu_eviction",
                "observed_at_ns": observed_at_ns,
                "sequence": self._next_preservation_event_sequence(),
                "generation": commit.get("generation") if commit else None,
                "source_request_id": (
                    commit.get("source_request_id") if commit else None
                ),
                "manifest_request_id": (
                    commit.get("manifest_request_id") if commit else None
                ),
                "run_id": commit.get("run_id") if commit else None,
                "cell_id": commit.get("cell_id") if commit else None,
                "session_id": commit.get("session_id") if commit else None,
                "prior_commit_sequence": (commit.get("sequence") if commit else None),
                "prior_commit_ns": (commit.get("committed_at_ns") if commit else None),
                "provenance": "manager_prepare_store_evicted_keys",
            }
            self._preservation_key_evictions[key].append(eviction)

    def _drain_chain_admission_evidence(self) -> None:
        """Bind exact-chain admission evidence across async store steps.

        Prepare-side records are normally drained from build_connector_meta,
        while store completion is produced during update_connector_output.
        Draining at both lifecycle boundaries keeps the completion record on
        the finished request before that request's preservation event is
        emitted. Manager aggregate deltas are accumulated independently; they
        are intentionally not required to equal the records from one step.
        """
        records = []
        if hasattr(self.manager, "drain_chain_admission_records"):
            records = self.manager.drain_chain_admission_records()
        for record in records:
            request_id = record.get("request_id")
            state = self._preservation_requests.get(
                request_id
            ) or self._preservation_finished.get(request_id)
            if state is None:
                # Chain evidence is enabled only for preservation campaigns;
                # an absent lifecycle owner is explicit in the global manager
                # accounting rather than being rebound to another request.
                continue
            state.setdefault("chain_admission_records", []).append(record)
            accounting = state.setdefault("chain_admission_accounting", {})
            action = record.get("action")
            reason = record.get("reason")
            qualified_count = len(record.get("qualified_keys", ()))
            admitted_count = len(record.get("admitted_keys", ()))
            if action in {"observe", "reject", "ignore", "admit", "no_store"}:
                accounting["observations"] = int(accounting.get("observations", 0)) + 1
            if action in {"admit", "no_store"}:
                accounting["matched_observations"] = (
                    int(accounting.get("matched_observations", 0)) + 1
                )
                accounting["qualified_chains"] = (
                    int(accounting.get("qualified_chains", 0)) + 1
                )
                accounting["qualified_keys"] = (
                    int(accounting.get("qualified_keys", 0)) + qualified_count
                )
                accounting["admitted_keys"] = (
                    int(accounting.get("admitted_keys", 0)) + admitted_count
                )
            elif action == "store_complete":
                accounting["stored_keys"] = (
                    int(accounting.get("stored_keys", 0)) + admitted_count
                )
            elif action == "store_failed":
                accounting["failed_store_keys"] = (
                    int(accounting.get("failed_store_keys", 0)) + qualified_count
                )
            if reason == "duplicate_request_access":
                accounting["duplicate_request_observations"] = (
                    int(accounting.get("duplicate_request_observations", 0)) + 1
                )
            elif reason == "incomplete_exact_chain_identity":
                accounting["identity_rejections"] = (
                    int(accounting.get("identity_rejections", 0)) + 1
                )
            elif reason == "exact_chain_prefix_mismatch":
                accounting["prefix_rejections"] = (
                    int(accounting.get("prefix_rejections", 0)) + 1
                )
            accounting["overhead_cpu_time_ns"] = int(
                accounting.get("overhead_cpu_time_ns", 0)
            ) + int(record.get("overhead_cpu_time_ns", 0))
            accounting["overhead_wall_time_ns"] = int(
                accounting.get("overhead_wall_time_ns", 0)
            ) + int(record.get("overhead_wall_time_ns", 0))

        if hasattr(self.manager, "drain_chain_admission_accounting"):
            aggregate = self.manager.drain_chain_admission_accounting()
            global_accounting = getattr(
                self, "_chain_admission_global_accounting", None
            )
            if global_accounting is None:
                global_accounting = self._chain_admission_global_accounting = {}
            for name, value in aggregate.items():
                if (
                    not isinstance(name, str)
                    or isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise RuntimeError("invalid chain admission accounting delta")
                global_accounting[name] = int(global_accounting.get(name, 0)) + value

    def _preservation_state(self, request: Request) -> dict[str, object]:
        params = request.kv_transfer_params or {}
        campaign_provenance = params.get("campaign_provenance", {})
        if not isinstance(campaign_provenance, dict):
            campaign_provenance = {}
        required_tokens = int(getattr(request, "num_prompt_tokens", 0))
        supplied_external_request_id = getattr(request, "external_req_id", None)
        derived_external_request_id = _derive_external_request_id(request.request_id)
        external_request_id = (
            supplied_external_request_id or derived_external_request_id
        )
        identity_valid = (
            derived_external_request_id is not None
            and (
                supplied_external_request_id is None
                or supplied_external_request_id == derived_external_request_id
            )
            and _parse_request_identity(external_request_id)[0] is not None
        )
        parsed_run_id, parsed_cell_id, parsed_sample_id, parsed_sample_index = (
            _parse_request_identity(external_request_id)
        )
        required_chunks = sum(
            cdiv(required_tokens, group.tokens_per_chunk)
            for group in self.config.kv_group_configs
        )
        required_blocks = sum(
            cdiv(required_tokens, group.tokens_per_block)
            for group in self.config.kv_group_configs
        )
        ordered_key_positions, ordered_key_positions_complete = (
            self._ordered_request_key_positions(request)
        )
        observed_wall_before_ns = time.time_ns()
        connector_observed_ns = time.monotonic_ns()
        observed_wall_after_ns = time.time_ns()
        request_arrival_time_s = getattr(request, "arrival_time", None)
        request_arrival_wall_ns = (
            int(float(request_arrival_time_s) * 1_000_000_000)
            if isinstance(request_arrival_time_s, (int, float))
            and not isinstance(request_arrival_time_s, bool)
            and math.isfinite(float(request_arrival_time_s))
            else None
        )
        observed_wall_midpoint_ns = (
            observed_wall_before_ns + observed_wall_after_ns
        ) // 2
        request_arrival_aligned_monotonic_ns = (
            connector_observed_ns
            - (observed_wall_midpoint_ns - request_arrival_wall_ns)
            if request_arrival_wall_ns is not None
            else None
        )
        physical_block_table_lifetime_id = hashlib.sha256(
            json.dumps(
                [
                    request.request_id,
                    external_request_id,
                    params.get("generation"),
                    connector_observed_ns,
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return {
            "request_id": request.request_id,
            "external_request_id": external_request_id,
            "benchmark_client_sequence": params.get("benchmark_client_sequence"),
            "manifest_request_id": campaign_provenance.get("manifest_request_id"),
            "manifest_request_index": campaign_provenance.get("manifest_request_index"),
            "human_cell_id": campaign_provenance.get("human_cell_id"),
            "human_profile_id": campaign_provenance.get("human_profile_id"),
            "schedule_role": campaign_provenance.get("schedule_role"),
            "content_origin": campaign_provenance.get("content_origin"),
            "arrival_origin": campaign_provenance.get("arrival_origin"),
            "wait_origin": campaign_provenance.get("wait_origin"),
            "pressure_origin": campaign_provenance.get("pressure_origin"),
            "source_id": campaign_provenance.get("source_id"),
            "session_id": campaign_provenance.get("session_id"),
            "context_token_span": campaign_provenance.get("context_token_span"),
            "policy_session_id": params.get("session_id"),
            "family": campaign_provenance.get("family"),
            "split": campaign_provenance.get("split"),
            "generation": params.get("generation"),
            "request_arrival_time_s": request_arrival_time_s,
            "request_arrival_wall_ns": request_arrival_wall_ns,
            "request_arrival_aligned_monotonic_ns": (
                request_arrival_aligned_monotonic_ns
            ),
            "request_arrival_alignment_uncertainty_ns": (
                observed_wall_after_ns - observed_wall_before_ns
            ),
            "request_arrival_alignment_source": (
                "paired_time_time_ns_and_monotonic_ns_same_host"
                if request_arrival_wall_ns is not None
                else None
            ),
            "connector_observed_wall_before_ns": observed_wall_before_ns,
            "connector_observed_wall_after_ns": observed_wall_after_ns,
            "connector_observed_ns": connector_observed_ns,
            "first_service_ns": None,
            "first_service_source": None,
            "request_finished_ns": None,
            "request_finished_source": None,
            "identity_valid": identity_valid,
            "run_id": params.get("run_id", params.get("semantic_run_id"))
            or parsed_run_id,
            "cell_id": params.get("cell_id", params.get("semantic_cell_id"))
            or parsed_cell_id,
            "sample_id": params.get("semantic_sample_id", params.get("sample_id"))
            or parsed_sample_id,
            "sample_index": params.get(
                "semantic_sample_index", params.get("sample_index")
            )
            if params.get("semantic_sample_index", params.get("sample_index"))
            is not None
            else parsed_sample_index,
            "required_tokens": required_tokens,
            "prompt_token_count": (
                len(request.prompt_token_ids)
                if isinstance(request.prompt_token_ids, (list, tuple))
                else None
            ),
            "prompt_token_digest": _prompt_token_digest(request),
            "ordered_key_positions": ordered_key_positions,
            "ordered_key_positions_complete": ordered_key_positions_complete,
            "ordered_key_positions_captured_at_ns": time.monotonic_ns(),
            "ordered_key_positions_capture_provenance": (
                "native_on_new_request_initial_state"
            ),
            "cpu_commit_events": [],
            "key_use_events": [],
            "first_admission": None,
            "required_chunks": required_chunks,
            "required_blocks": required_blocks,
            "required_block_ids": set(),
            "local_block_ids": set(),
            "external_block_ids": set(),
            "recomputed_block_ids": set(),
            "scheduled_block_ids": set(),
            "loaded_block_ids": set(),
            "stored_block_ids": set(),
            "physical_block_identity_complete": True,
            "physical_block_table_lifetime_id": physical_block_table_lifetime_id,
            "physical_block_table_generation": 0,
            "physical_block_table": None,
            "physical_block_table_updates": [],
            "physical_block_table_errors": [],
            "local_reused_tokens": 0,
            "loaded_tokens": 0,
            "loaded_chunks": 0,
            "stored_chunks": 0,
            "load_bytes": 0,
            "store_bytes": 0,
            "load_time_s": 0.0,
            "store_time_s": 0.0,
            "load_critical_path_s": 0.0,
            "store_critical_path_s": 0.0,
            "load_time_complete": True,
            "store_time_complete": True,
            "load_time_observed": False,
            "store_time_observed": False,
            "load_contributors": 0,
            "store_contributors": 0,
            "load_expected_contributors": int(self.config.num_workers),
            "store_expected_contributors": (
                1 if self._replicated_layout else int(self.config.num_workers)
            ),
            "load_job_count": 0,
            "store_job_count": 0,
            "load_expected_total": 0,
            "store_expected_total": 0,
            "transfer_job_statuses": [],
            "speculative_transfer_job_statuses": [],
            "lookup_observed": False,
            "lookup_no_eligible": False,
            "lookup_sequence": 0,
            "active_lookup_episode": None,
            "lookup_episodes": [],
            "last_lookup_id": None,
            "lookup_keys": [],
            "prior_stored_keys": [],
            "prior_evicted_keys": [],
            "cold_keys": [],
            "prompt_tokens_locally_executed": 0,
            "prompt_tokens_scheduled": 0,
            "recomputed_kv_blocks": 0,
            "execution_metadata_observed": False,
            "first_forward_ns": None,
            "first_forward_source": None,
            "prefill_cuda_events": [],
            "staging_protection_records": [],
            "episode_prompt_accounted": 0,
            "episode_local_reused_tokens": 0,
            "episode_external_tokens": 0,
            "readmission_pending": False,
            "episode_id": 0,
            "admission_episodes": [],
            "admission_decisions": [],
            "admission_decision_accounting": {
                "admitted": 0,
                "rejected": 0,
                "compulsory_fallbacks": 0,
            },
            "chain_admission_records": [],
            "chain_admission_accounting": {
                "observations": 0,
                "matched_observations": 0,
                "qualified_chains": 0,
                "qualified_keys": 0,
                "admitted_keys": 0,
                "stored_keys": 0,
                "failed_store_keys": 0,
                "duplicate_request_observations": 0,
                "identity_rejections": 0,
                "prefix_rejections": 0,
                "overhead_cpu_time_ns": 0,
                "overhead_wall_time_ns": 0,
            },
            "admitted_local_tokens": 0,
            "admitted_external_tokens": 0,
            "admitted_local_blocks": 0,
            "admitted_external_blocks": 0,
            "loaded_tokens_total": 0,
            "admission_count": 0,
            "recompute_admission_count": 0,
            "transfer_complete": True,
            "preempted": False,
            "preemption_count": 0,
            "prefetch_attempts": 0,
            "prefetch_attempted_keys": 0,
            "prefetch_attempted_blocks": 0,
            "prefetch_submitted_jobs": 0,
            "prefetch_submitted_blocks": 0,
            "prefetch_consumed_keys": 0,
            "prefetch_consumed_blocks": 0,
            "prefetch_discarded_keys": 0,
            "prefetch_discarded_blocks": 0,
            # Opt-in prefetch-attribution metadata.  These fields are kept
            # separate from preservation accounting: a missing producer seam
            # must become an explicit unavailable attribution, never a guessed
            # latency or transfer result.
            "prefetch_seed": params.get("seed"),
            "prefetch_episode_id": None,
            "prefetch_t0_ns": None,
            "prefetch_t1_ns": None,
            "prefetch_t2_ns": None,
            "prefetch_t3_ns": None,
            "prefetch_fence_ns": None,
            "prefetch_splice_fence_ns": None,
            "prefetch_install_submitted_ns": None,
            "prefetch_request_usable_ns": None,
            "prefetch_request_usable_source": None,
            "prefetch_splice_fence_evidence": [],
            "prefetch_t4_ns": None,
            "prefetch_t4_source": None,
            "prefetch_t4_episode_id": None,
            "prefetch_t5_start_ns": None,
            "prefetch_t5_end_ns": None,
            "prefetch_t6_ns": None,
            "prefetch_t6_source": None,
            "prefetch_t6_episode_id": None,
            "prefetch_t6_preemption_index": None,
            "prefetch_required_keys": [],
            "prefetch_cpu_resident_keys": [],
            "prefetch_cpu_ready_keys": [],
            "prefetch_cpu_residency_proof": {},
            "prefetch_selected_keys": [],
            "prefetch_intersection_keys": [],
            "prefetch_remainder_keys": [],
            "prefetch_source_map": {},
            "prefetch_destination_map": {},
            "prefetch_reservation_id": None,
            "prefetch_reservation_blocks": [],
            "prefetch_reservation_history": [],
            "prefetch_block_pool_snapshots": [],
            "prefetch_block_pool_snapshot_count": 0,
            "prefetch_block_pool_snapshots_truncated": False,
            "prefetch_cpu_cache_snapshots": [],
            "prefetch_cpu_cache_snapshot_count": 0,
            "prefetch_cpu_cache_snapshots_truncated": False,
            "prefetch_h2d_bytes": None,
            "prefetch_blocking_h2d_bytes": None,
            "prefetch_splice_bytes": None,
            "prefetch_transfer_contributors": {},
            "prefetch_time_s": None,
            "prefetch_blocking_time_s": None,
            "prefetch_total_time_s": None,
            "prefetch_splice_acknowledged": False,
            "prefetch_splice_ack_id": None,
            "prefetch_source_reuse_before_fence": None,
            "prefetch_failure_reason": None,
            "prefetch_cancelled": False,
            "prefetch_stale": False,
            "prefetch_allocation_failures": 0,
            "prefetch_filler_failures": None,
            "prefetch_unavailable_fields": [],
            "prefetch_provenance": dict(self._prefetch_provenance),
            "prefetch_attribution": _new_prefetch_attribution_state(),
            "wait_hint": None,
            "lifecycle_release": None,
            "terminal_status": None,
            "terminal_reason": None,
        }

    def _prefetch_attribution_event(
        self, state: dict[str, object], outcome: str, reason: str
    ) -> dict[str, object]:
        """Build the opt-in prefetch attribution sidecar carried by the
        authoritative preservation event.

        The scheduler can prove the key and reservation identities, but a
        first-forward boundary is owned by the model runner.  Until that
        upstream/runtime seam exists, this record is explicitly unavailable;
        known fields remain useful for diagnosing the gap and are never
        promoted to a latency or PCIe claim.
        """
        preempted = bool(state.get("preempted", False))
        wait_hint = state.get("wait_hint")
        prefetch_status = "not_pressured" if not preempted else "unavailable"
        unavailable_fields = list(state.get("prefetch_unavailable_fields", []))
        if isinstance(wait_hint, dict):
            if int(state.get("prefetch_consumed_blocks", 0)) > 0:
                prefetch_status = "hint_consumed"
                prefetch_reason = (
                    "native_prefix_cache_adopted"
                    if state.get("prefetch_native_adoption", {}).get(
                        "same_physical_blocks"
                    )
                    else "known_prefix_staging_spliced"
                )
            elif state.get("prefetch_failure_reason"):
                prefetch_status = "hint_failed"
                prefetch_reason = str(state["prefetch_failure_reason"])
            else:
                prefetch_status = "hint_no_action"
                prefetch_reason = "no_staged_prefix_consumed"
        elif preempted and state.get("prefetch_failure_reason"):
            if state.get("prefetch_cancelled"):
                prefetch_status = "cancelled"
            elif state.get("prefetch_stale"):
                prefetch_status = "stale"
            else:
                prefetch_status = "failed"
            prefetch_reason = str(state["prefetch_failure_reason"])
        elif not preempted:
            prefetch_reason = "no_preemption_observed"
        else:
            prefetch_reason = (
                "fence_or_complete_attribution_unavailable"
                if state.get("prefetch_t6_ns") is not None
                else "first_forward_timestamp_unavailable"
            )
            if (
                state.get("prefetch_t6_ns") is None
                and "t6_first_forward" not in unavailable_fields
            ):
                unavailable_fields.append("t6_first_forward")
        speculative_transfer = _speculative_transfer_accounting(
            state, int(getattr(self, "_prefetch_bytes_per_key", 0))
        )
        speculative_transfer_costs = _speculative_transfer_cost_accounting(
            state,
            int(getattr(self, "_prefetch_bytes_per_key", 0)),
            speculative_transfer,
        )
        demand_boundary_ns = state.get("prefetch_t4_ns")
        demand_boundary_source = state.get("prefetch_t4_source")
        if demand_boundary_ns is None:
            candidate_ns = speculative_transfer.get("demand_boundary_ns")
            candidate_source = speculative_transfer.get("demand_boundary_source")
            wait_hint = state.get("wait_hint")
            identity_matches = (
                isinstance(wait_hint, dict)
                and wait_hint.get("owner_id") == state.get("external_request_id")
                and wait_hint.get("run_id") == state.get("run_id")
                and wait_hint.get("cell_id") == state.get("cell_id")
                and wait_hint.get("session_id") == state.get("policy_session_id")
                and wait_hint.get("generation") == state.get("generation")
            )
            if (
                int(state.get("prefetch_consumed_blocks", 0)) > 0
                and identity_matches
                and candidate_source == "hinted_request_connector_observation"
                and isinstance(candidate_ns, int)
                and not isinstance(candidate_ns, bool)
                and candidate_ns == state.get("connector_observed_ns")
            ):
                demand_boundary_ns = candidate_ns
                demand_boundary_source = candidate_source
        event: dict[str, object] = {
            "schema": "semantic-prefetch-attribution-v1",
            "attempts": list(_prefetch_attribution_state(state).get("attempts", [])),
            "event": "prefetch_attribution",
            "run_id": state.get("run_id"),
            "cell_id": state.get("cell_id"),
            "seed": state.get("prefetch_seed"),
            "episode_id": f"episode-{int(state.get('episode_id', 0))}",
            "request_id": state.get("external_request_id"),
            "request_first_forward_ns": state.get("first_forward_ns"),
            "request_finished_ns": state.get("request_finished_ns"),
            "request_finished_source": state.get("request_finished_source"),
            "server_timing": {
                "clock_domain": "host_boot_monotonic_ns",
                "arrival_aligned_monotonic_ns": state.get(
                    "request_arrival_aligned_monotonic_ns"
                ),
                "service_start_ns": state.get("first_service_ns"),
                "first_forward_ns": state.get("first_forward_ns"),
                "completion_ns": state.get("request_finished_ns"),
            },
            "status": prefetch_status,
            "status_reason": prefetch_reason,
            "unavailable_fields": unavailable_fields,
            "provenance": state.get("prefetch_provenance", {}),
            "wait_hint": state.get("wait_hint"),
            "native_adoption": state.get("prefetch_native_adoption"),
            "allocator_peak_stats": dict(getattr(self, "_allocator_peak_stats", {})),
            "block_pool_snapshots": list(
                state.get("prefetch_block_pool_snapshots", [])
            ),
            "block_pool_snapshot_count": int(
                state.get("prefetch_block_pool_snapshot_count", 0)
            ),
            "block_pool_snapshots_truncated": bool(
                state.get("prefetch_block_pool_snapshots_truncated", False)
            ),
            "cpu_cache_snapshots": list(state.get("prefetch_cpu_cache_snapshots", [])),
            "cpu_cache_snapshot_count": int(
                state.get("prefetch_cpu_cache_snapshot_count", 0)
            ),
            "cpu_cache_snapshots_truncated": bool(
                state.get("prefetch_cpu_cache_snapshots_truncated", False)
            ),
            # These are only populated when their exact scheduler/worker
            # owner supplied them. Missing entries remain absent rather than
            # being filled with zero-valued guesses.
            "required_keys": list(state.get("prefetch_required_keys", [])),
            "cpu_resident_keys": list(state.get("prefetch_cpu_resident_keys", [])),
            "cpu_ready_keys": list(state.get("prefetch_cpu_ready_keys", [])),
            "cpu_residency_proof": dict(state.get("prefetch_cpu_residency_proof", {})),
            "prefetched_keys": list(state.get("prefetch_selected_keys", [])),
            "intersection_keys": list(state.get("prefetch_intersection_keys", [])),
            "remainder_keys": list(state.get("prefetch_remainder_keys", [])),
            "source_key_to_gpu_block": dict(state.get("prefetch_source_map", {})),
            "destination_key_to_gpu_block": dict(
                state.get("prefetch_destination_map", {})
            ),
            "reservation_id": state.get("prefetch_reservation_id"),
            "reservation_block_ids": list(state.get("prefetch_reservation_blocks", [])),
            "reservation_history": list(state.get("prefetch_reservation_history", [])),
            "transfer_contributors": dict(
                state.get("prefetch_transfer_contributors", {})
            ),
            "transfer_jobs": list(state.get("speculative_transfer_job_statuses", [])),
            "speculative_transfer": speculative_transfer,
            "speculative_transfer_costs": speculative_transfer_costs,
            "prefetch_h2d_bytes": state.get("prefetch_h2d_bytes"),
            "blocking_h2d_bytes": state.get("prefetch_blocking_h2d_bytes"),
            "total_h2d_bytes": (
                int(state["prefetch_h2d_bytes"])
                + int(state["prefetch_blocking_h2d_bytes"])
                if state.get("prefetch_h2d_bytes") is not None
                and state.get("prefetch_blocking_h2d_bytes") is not None
                else None
            ),
            "gpu_splice_bytes": state.get("prefetch_splice_bytes"),
            "extra_preemptions": max(0, int(state.get("preemption_count", 0)) - 1),
            "allocation_failures": state.get("prefetch_allocation_failures"),
            "filler_failures": state.get("prefetch_filler_failures"),
            "source_reuse_before_fence": state.get(
                "prefetch_source_reuse_before_fence"
            ),
            "terminal_cleanup": {
                "reservation_released": bool(
                    state.get("prefetch_cleanup_released", False)
                ),
                "source_reuse_fenced": bool(
                    state.get("prefetch_fence_complete", False)
                ),
                "request_state_cleared": True,
            },
            "fence_before_first_forward": None,
            "splice_acknowledged": bool(
                state.get("prefetch_splice_acknowledged", False)
            ),
            "splice_ack_id": state.get("prefetch_splice_ack_id"),
            "splice_fence_ns": state.get("prefetch_splice_fence_ns"),
            "splice_fence_evidence": list(
                state.get("prefetch_splice_fence_evidence", [])
            ),
            "t6_source": state.get("prefetch_t6_source"),
            "t4_source": demand_boundary_source,
            "t4_episode_id": state.get("prefetch_t4_episode_id"),
            "t6_episode_id": state.get("prefetch_t6_episode_id"),
            "t6_preemption_index": state.get("prefetch_t6_preemption_index"),
        }
        timestamp_fields = {
            "t0_preemption": state.get("prefetch_t0_ns"),
            "t1_reservation_submit": state.get("prefetch_t1_ns"),
            "t2_prefetch_start": state.get("prefetch_t2_ns"),
            "t3_prefetch_complete": state.get("prefetch_t3_ns"),
            "t3_fence_complete": state.get("prefetch_fence_ns"),
            "t4_readmission_allocation": demand_boundary_ns,
            "t5_blocking_load_start": state.get("prefetch_t5_start_ns"),
            "t5_blocking_load_end": state.get("prefetch_t5_end_ns"),
            "t6_first_forward": state.get("prefetch_t6_ns"),
            "destination_install_submitted": state.get("prefetch_install_submitted_ns"),
            "destination_installed": state.get("prefetch_request_usable_ns"),
            "request_usable": state.get("prefetch_request_usable_ns"),
        }
        timestamp_clock_domains = {
            "t0_preemption": "host_monotonic",
            "t1_reservation_submit": "host_monotonic",
            "t2_prefetch_start": "host_monotonic",
            "t3_prefetch_complete": "host_monotonic",
            "t3_fence_complete": "host_monotonic",
            "t4_readmission_allocation": "host_monotonic",
            "t5_blocking_load_start": "host_monotonic",
            "t5_blocking_load_end": "host_monotonic",
            "t6_first_forward": "host_monotonic",
            "destination_install_submitted": "host_monotonic",
            "destination_installed": "host_monotonic",
            "request_usable": "host_monotonic",
        }
        event["timestamp_clock_domains"] = {
            field: timestamp_clock_domains[field]
            for field, value in timestamp_fields.items()
            if value is not None
        }
        t0 = timestamp_fields["t0_preemption"]
        t6 = timestamp_fields["t6_first_forward"]
        if (
            t0 is not None
            and t6 is not None
            and t6 >= t0
            and event["timestamp_clock_domains"].get("t0_preemption")
            == event["timestamp_clock_domains"].get("t6_first_forward")
        ):
            event["preemption_to_model_runner_entry_s"] = (t6 - t0) / 1_000_000_000
            event["preemption_to_model_runner_entry_source"] = (
                "host_monotonic_model_runner_execute_entry"
            )
        else:
            event["preemption_to_model_runner_entry_unavailable_reason"] = (
                "missing_or_incomparable_t0_t6"
            )
        for timestamp_field, value in timestamp_fields.items():
            if value is not None:
                event[timestamp_field] = value
        t2 = timestamp_fields["t2_prefetch_start"]
        t3 = timestamp_fields["t3_prefetch_complete"]
        if t2 is not None and t3 is not None:
            event["prefetch_time_s"] = (t3 - t2) / 1_000_000_000
        if state.get("prefetch_time_s") is not None:
            event["prefetch_device_time_s"] = state["prefetch_time_s"]
            event["prefetch_device_time_source"] = "cuda_event_elapsed_time"
        t5_start = timestamp_fields["t5_blocking_load_start"]
        t5_end = timestamp_fields["t5_blocking_load_end"]
        if t5_start is not None and t5_end is not None:
            event["blocking_time_s"] = (t5_end - t5_start) / 1_000_000_000
        if state.get("prefetch_blocking_time_s") is not None:
            event["blocking_device_time_s"] = state["prefetch_blocking_time_s"]
            event["blocking_device_time_source"] = "cuda_event_elapsed_time"
        if t2 is not None and t3 is not None:
            transfer_end = t3 if t5_end is None else max(t3, t5_end)
            event["total_transfer_time_s"] = (transfer_end - t2) / 1_000_000_000
        demand_ns = timestamp_fields["t4_readmission_allocation"]
        install_submit_ns = timestamp_fields["destination_install_submitted"]
        request_usable_ns = timestamp_fields["request_usable"]
        event["staging_ready_ns"] = t3
        event["demand_boundary_ns"] = demand_ns
        event["demand_boundary_source"] = demand_boundary_source
        event["demand_boundary_clock_domain"] = (
            "host_monotonic" if demand_ns is not None else None
        )
        event["destination_install_submitted_ns"] = install_submit_ns
        event["destination_installed_ns"] = request_usable_ns
        event["request_usable_ns"] = request_usable_ns
        event["request_usable_source"] = state.get("prefetch_request_usable_source")
        event["request_usable_complete"] = bool(
            request_usable_ns is not None
            and state.get("prefetch_splice_acknowledged", False)
        )
        event["transfer_hidden_before_demand_s"] = (
            max(0, min(demand_ns, t3) - t2) / 1_000_000_000
            if all(value is not None for value in (t2, t3, demand_ns)) and t3 >= t2
            else None
        )
        event["staging_ready_lead_time_s"] = (
            max(0, demand_ns - t3) / 1_000_000_000
            if t3 is not None and demand_ns is not None
            else None
        )
        event["installation_time_s"] = (
            (request_usable_ns - install_submit_ns) / 1_000_000_000
            if install_submit_ns is not None
            and request_usable_ns is not None
            and request_usable_ns >= install_submit_ns
            else None
        )
        event["exposed_blocking_stall_s"] = (
            max(0, request_usable_ns - demand_ns) / 1_000_000_000
            if request_usable_ns is not None and demand_ns is not None
            else None
        )
        if preempted:
            for field, value in timestamp_fields.items():
                if value is None and field not in unavailable_fields:
                    unavailable_fields.append(field)
            for field, value in (
                ("gpu_splice_bytes", state.get("prefetch_splice_bytes")),
                (
                    "source_reuse_before_fence",
                    state.get("prefetch_source_reuse_before_fence"),
                ),
                ("fence_before_first_forward", None),
                ("filler_failures", state.get("prefetch_filler_failures")),
            ):
                if value is None and field not in unavailable_fields:
                    unavailable_fields.append(field)
        return event

    def _emit_finished_preservation_events(self) -> None:
        if not self._preservation_events_enabled:
            return
        for req_id, state in list(self._preservation_finished.items()):
            if req_id in self._req_status:
                continue
            timing_status = _prefill_cuda_terminal_status(
                req_id,
                state,
                bounded_prefill_cuda_timing=bool(
                    getattr(self, "_bounded_prefill_cuda_timing", False)
                ),
                now_ns=time.monotonic_ns(),
            )
            if timing_status == "waiting":
                state.setdefault(
                    "prefill_cuda_terminal_wait_started_ns",
                    int(state.get("request_finished_ns") or time.monotonic_ns()),
                )
                continue
            if timing_status == "unavailable":
                state["prefill_cuda_timing_unavailable_reason"] = (
                    "worker_timing_metadata_timeout"
                )
            if state.get("prefetch_splice_ack_id") and not state.get(
                "prefetch_splice_acknowledged", False
            ):
                continue
            required_tokens = int(state["required_tokens"])
            local_reused_tokens = min(
                required_tokens,
                max(
                    int(state["episode_local_reused_tokens"]),
                    int(state["admitted_local_tokens"]),
                ),
            )
            loaded_tokens = min(
                max(0, required_tokens - local_reused_tokens),
                max(0, int(state["episode_external_tokens"])),
            )
            reused_tokens = local_reused_tokens + loaded_tokens
            cumulative_prompt_tokens = int(state["prompt_tokens_locally_executed"])
            recomputed_tokens: int | None = max(0, required_tokens - reused_tokens)
            # An external load is itself authoritative pressure evidence. A
            # zero-local/zero-load cold request is the only case that must
            # remain distinct from a true evicted-under-pressure miss.
            transfer_available = bool(state["transfer_complete"])
            admission_available = bool(state["execution_metadata_observed"])
            serialized_lookup_episodes = _serialize_lookup_episodes(
                state["lookup_episodes"]
            )
            serialized_groups = [
                group
                for episode in serialized_lookup_episodes
                for group in episode["groups"].values()
            ]
            observed_groups = [
                group for group in serialized_groups if group.get("observed") is True
            ]
            all_groups_no_eligible = (
                bool(observed_groups)
                and len(observed_groups) == len(serialized_groups)
                and all(group.get("no_eligible") is True for group in observed_groups)
            )
            load_evidence = bool(
                state["episode_external_tokens"]
                or state["loaded_block_ids"]
                or state["load_job_count"]
            )
            pressure = bool(state["prior_evicted_keys"]) or load_evidence
            lookup_metadata_available = _lookup_metadata_complete(
                serialized_lookup_episodes
            )
            tp_metadata_available = (
                isinstance(state.get("load_expected_contributors"), int)
                and state["load_expected_contributors"] > 0
            )
            table_contract_present = "physical_block_table" in state
            raw_block_table = state.get("physical_block_table")
            block_table_valid = (
                isinstance(raw_block_table, (list, tuple))
                and len(raw_block_table) == len(self.config.kv_group_configs)
                and all(
                    isinstance(group, (list, tuple))
                    and all(
                        isinstance(block_id, int)
                        and not isinstance(block_id, bool)
                        and block_id >= 0
                        for block_id in group
                    )
                    for group in raw_block_table
                )
                and bool(state.get("physical_block_table_updates"))
                and not state.get("physical_block_table_errors")
                and isinstance(state.get("physical_block_table_lifetime_id"), str)
            )
            if table_contract_present and not block_table_valid:
                state["physical_block_identity_complete"] = False
            available = (
                bool(state["identity_valid"])
                and lookup_metadata_available
                and transfer_available
                and admission_available
                and tp_metadata_available
            )
            if not state["identity_valid"]:
                outcome = "unavailable"
                reason = "request_identity_unavailable"
            elif not lookup_metadata_available:
                outcome = "unavailable"
                reason = "lookup_metadata_unavailable"
            elif not state["transfer_complete"]:
                outcome = "unavailable"
                reason = "transfer_metadata_unavailable"
            elif not admission_available:
                outcome = "unavailable"
                reason = "admission_metadata_unavailable"
            elif not state["physical_block_identity_complete"]:
                outcome = "unavailable"
                reason = "physical_block_metadata_unavailable"
            elif not pressure:
                # A cold/non-eligible request can legitimately compute the
                # whole prompt with zero local and zero load activity. It is
                # not evidence of an evicted-under-pressure miss.
                outcome = "not_pressured"
                reason = "cold_or_not_eligible"
            elif loaded_tokens > 0 and recomputed_tokens == 0:
                outcome = "hit"
                reason = "authoritative_offload_events"
            elif loaded_tokens > 0 and recomputed_tokens > 0:
                outcome = "partial"
                reason = "authoritative_offload_events"
            elif recomputed_tokens > 0:
                outcome = "miss"
                reason = "authoritative_offload_events"
            else:
                outcome = "not_pressured"
                reason = "authoritative_offload_events"
            physical_complete = bool(state["physical_block_identity_complete"])
            speculative_transfer = _speculative_transfer_accounting(
                state, int(getattr(self, "_prefetch_bytes_per_key", 0))
            )
            serialized_block_table = (
                [list(group) for group in raw_block_table]
                if block_table_valid
                else None
            )
            block_table_provenance = (
                {
                    "schema": "semantic-physical-block-table-v1",
                    "request_id": state["request_id"],
                    "external_request_id": state["external_request_id"],
                    "lifetime_id": state.get("physical_block_table_lifetime_id"),
                    "lifetime_observed_at_ns": state.get("connector_observed_ns"),
                    "request_generation": state.get("generation"),
                    "table_generation": state.get("physical_block_table_generation"),
                    "tokens_per_block": [
                        group_config_tokens_per_block(group_config)
                        for group_config in self.config.kv_group_configs
                    ],
                    "complete": block_table_valid and physical_complete,
                    "group_block_ids": serialized_block_table,
                    "table_digest": (
                        hashlib.sha256(
                            json.dumps(
                                serialized_block_table, separators=(",", ":")
                            ).encode()
                        ).hexdigest()
                        if serialized_block_table is not None
                        else None
                    ),
                    "updates": list(state.get("physical_block_table_updates", [])),
                    "errors": list(state.get("physical_block_table_errors", [])),
                }
                if table_contract_present
                else None
            )
            eligible_local_blocks = (
                len(state["local_block_ids"]) if physical_complete else None
            )
            eligible_external_blocks = (
                len(state["external_block_ids"]) if physical_complete else None
            )
            eligible_recomputed_blocks = (
                len(state["recomputed_block_ids"]) if physical_complete else None
            )
            authoritative = available and outcome != "unavailable"
            arrival_ns = state.get("request_arrival_aligned_monotonic_ns")
            service_ns = state.get("first_service_ns")
            first_forward_ns = state.get("first_forward_ns")
            finished_ns = state.get("request_finished_ns")
            ordered_server_timing = (
                all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in (
                        arrival_ns,
                        service_ns,
                        first_forward_ns,
                        finished_ns,
                    )
                )
                and arrival_ns <= service_ns <= first_forward_ns <= finished_ns
            )
            server_timing = {
                "schema": "semantic-server-request-timing-v1",
                "clock_domain": "host_boot_monotonic_ns",
                "arrival_wall_clock_domain": "unix_epoch_ns",
                "arrival_wall_ns": state.get("request_arrival_wall_ns"),
                "arrival_aligned_monotonic_ns": arrival_ns,
                "arrival_alignment_uncertainty_ns": state.get(
                    "request_arrival_alignment_uncertainty_ns"
                ),
                "arrival_alignment_source": state.get(
                    "request_arrival_alignment_source"
                ),
                "connector_observed_ns": state.get("connector_observed_ns"),
                "connector_observed_wall_before_ns": state.get(
                    "connector_observed_wall_before_ns"
                ),
                "connector_observed_wall_after_ns": state.get(
                    "connector_observed_wall_after_ns"
                ),
                "service_start_ns": service_ns,
                "service_start_source": state.get("first_service_source"),
                "first_forward_ns": first_forward_ns,
                "first_forward_source": state.get("first_forward_source"),
                "completion_ns": finished_ns,
                "completion_source": state.get("request_finished_source"),
                "ordering_valid": ordered_server_timing,
                "complete": ordered_server_timing,
                "queue_delay_ns": (
                    service_ns - arrival_ns if ordered_server_timing else None
                ),
                "service_to_first_forward_ns": (
                    first_forward_ns - service_ns if ordered_server_timing else None
                ),
                "service_time_ns": (
                    finished_ns - service_ns if ordered_server_timing else None
                ),
                "completion_after_first_forward_ns": (
                    finished_ns - first_forward_ns if ordered_server_timing else None
                ),
            }
            event = {
                "schema": "semantic-preservation-event-v1",
                "event": "request_preservation",
                "event_instance": 1,
                "expected_event_count": 1,
                "request_id": state["request_id"],
                "request_id_kind": "vllm_internal",
                "external_request_id": state["external_request_id"],
                "benchmark_client_sequence": state.get("benchmark_client_sequence"),
                "manifest_request_id": state.get("manifest_request_id"),
                "manifest_request_index": state.get("manifest_request_index"),
                "human_cell_id": state.get("human_cell_id"),
                "human_profile_id": state.get("human_profile_id"),
                "schedule_role": state.get("schedule_role"),
                "content_origin": state.get("content_origin"),
                "arrival_origin": state.get("arrival_origin"),
                "wait_origin": state.get("wait_origin"),
                "pressure_origin": state.get("pressure_origin"),
                "source_id": state.get("source_id"),
                "session_id": state.get("session_id"),
                "context_token_span": state.get("context_token_span"),
                "policy_session_id": state.get("policy_session_id"),
                "family": state.get("family"),
                "split": state.get("split"),
                "generation": state.get("generation"),
                "request_arrival_time_s": state.get("request_arrival_time_s"),
                "connector_observed_ns": state.get("connector_observed_ns"),
                "request_finished_ns": state.get("request_finished_ns"),
                "request_finished_source": state.get("request_finished_source"),
                "request_identity_valid": state["identity_valid"],
                "run_id": state["run_id"],
                "cell_id": state["cell_id"],
                "sample_id": state["sample_id"],
                "sample_index": state["sample_index"],
                "prompt_token_count": state.get("prompt_token_count"),
                "prompt_token_digest": state.get("prompt_token_digest"),
                "ordered_key_positions_complete": bool(
                    state.get("ordered_key_positions_complete", False)
                ),
                "ordered_key_positions_captured_at_ns": state.get(
                    "ordered_key_positions_captured_at_ns"
                ),
                "ordered_key_positions_capture_provenance": state.get(
                    "ordered_key_positions_capture_provenance"
                ),
                "ordered_key_positions": _serialize_key_positions(
                    state.get("ordered_key_positions", [])
                ),
                "cpu_commit_events": _serialize_key_positions(
                    state.get("cpu_commit_events", [])
                ),
                "key_use_events": _serialize_key_positions(
                    state.get("key_use_events", [])
                ),
                "first_admission": (
                    {
                        **state["first_admission"],
                        "local_keys": _serialize_key_positions(
                            state["first_admission"].get("local_keys", [])
                        ),
                        "planned_load_keys": _serialize_key_positions(
                            state["first_admission"].get("planned_load_keys", [])
                        ),
                        "completed_load_keys": _serialize_key_positions(
                            state["first_admission"].get("completed_load_keys", [])
                        ),
                        "recomputed_keys": _serialize_key_positions(
                            state["first_admission"].get("recomputed_keys", [])
                        ),
                    }
                    if isinstance(state.get("first_admission"), dict)
                    else None
                ),
                "key_lifecycle": {
                    name: _serialize_key_positions(records)
                    for name, records in self._key_lifecycle_for_state(state).items()
                },
                "terminal_status": state["terminal_status"],
                "terminal_reason": state["terminal_reason"],
                "first_forward_ns": state.get("first_forward_ns"),
                "first_forward_source": state.get("first_forward_source"),
                "server_timing": server_timing,
                "prefill_cuda_events": list(state.get("prefill_cuda_events", [])),
                "prefill_cuda_timing_unavailable_reason": state.get(
                    "prefill_cuda_timing_unavailable_reason"
                ),
                "staging_protection_records": list(
                    state.get("staging_protection_records", [])
                ),
                "outcome": outcome,
                "reason": reason,
                "authoritative": authoritative,
                "event_integrity": "complete" if authoritative else "unavailable",
                "prefetch_attribution": self._prefetch_attribution_event(
                    state, outcome, reason
                ),
                "lifecycle_release": state.get("lifecycle_release"),
                "lookup_metadata_complete": lookup_metadata_available,
                "lookup_observed": bool(state["lookup_observed"]),
                "lookup_no_eligible": all_groups_no_eligible,
                "transfer_metadata_complete": transfer_available,
                "admission_metadata_complete": admission_available,
                "physical_block_identity_complete": state[
                    "physical_block_identity_complete"
                ],
                "physical_block_table_provenance": block_table_provenance,
                "transfer_job_statuses": state["transfer_job_statuses"],
                "speculative_transfer_job_statuses": list(
                    state.get("speculative_transfer_job_statuses", [])
                ),
                "speculative_transfer_accounting": speculative_transfer,
                "speculative_transfer_costs": _speculative_transfer_cost_accounting(
                    state,
                    int(getattr(self, "_prefetch_bytes_per_key", 0)),
                    speculative_transfer,
                ),
                "transfer_job_counts": {
                    "load": state["load_job_count"],
                    "store": state["store_job_count"],
                },
                "lookup_keys": [
                    _event_key(key)
                    for key in sorted(state["lookup_keys"], key=_event_key_sort)
                ],
                "prior_stored_keys": [
                    _event_key(key)
                    for key in sorted(state["prior_stored_keys"], key=_event_key_sort)
                ],
                "prior_evicted_keys": [
                    _event_key(key)
                    for key in sorted(state["prior_evicted_keys"], key=_event_key_sort)
                ],
                "cold_keys": [
                    _event_key(key)
                    for key in sorted(state["cold_keys"], key=_event_key_sort)
                ],
                "lookup_episodes": serialized_lookup_episodes,
                "required_tokens": required_tokens,
                "reused_tokens": reused_tokens,
                "loaded_tokens": loaded_tokens,
                "recomputed_tokens": recomputed_tokens,
                "actual_prompt_tokens_locally_executed": cumulative_prompt_tokens,
                "cumulative_prompt_tokens_locally_executed": cumulative_prompt_tokens,
                "prompt_tokens_scheduled": state["prompt_tokens_scheduled"],
                "admission_episodes": state["admission_episodes"],
                "admission_decisions": state.get("admission_decisions", []),
                "admission_decision_accounting": state.get(
                    "admission_decision_accounting",
                    {
                        "admitted": 0,
                        "rejected": 0,
                        "compulsory_fallbacks": 0,
                    },
                ),
                "chain_admission_records": list(
                    state.get("chain_admission_records", [])
                ),
                "chain_admission_accounting": dict(
                    state.get("chain_admission_accounting", {})
                ),
                "chain_admission_global_accounting": dict(
                    getattr(self, "_chain_admission_global_accounting", {})
                ),
                "admitted_local_tokens": state["admitted_local_tokens"],
                "admitted_external_tokens": state["admitted_external_tokens"],
                "admission_count": state["admission_count"],
                "recompute_admission_count": state["recompute_admission_count"],
                "required_kv_blocks": (
                    len(state["required_block_ids"]) if physical_complete else None
                ),
                "eligible_kv_blocks": {
                    "local": eligible_local_blocks,
                    "external": eligible_external_blocks,
                    "recomputed": eligible_recomputed_blocks,
                    "required": (
                        len(state["required_block_ids"]) if physical_complete else None
                    ),
                },
                "physical_block_ids": (
                    {
                        "required": sorted(state["required_block_ids"]),
                        "local": sorted(state["local_block_ids"]),
                        "external": sorted(state["external_block_ids"]),
                        "recomputed": sorted(state["recomputed_block_ids"]),
                        "scheduled": sorted(state["scheduled_block_ids"]),
                        "loaded": sorted(state["loaded_block_ids"]),
                        "stored": sorted(state["stored_block_ids"]),
                    }
                    if physical_complete
                    else None
                ),
                "required_kv_chunks": state["required_chunks"],
                "loaded_kv_chunks": state["loaded_chunks"],
                "recomputed_kv_chunks": None,
                "loaded_kv_blocks": (
                    len(state["loaded_block_ids"]) if physical_complete else None
                ),
                "stored_kv_chunks": state["stored_chunks"],
                "stored_kv_blocks": (
                    len(state["stored_block_ids"]) if physical_complete else None
                ),
                "load_bytes": state["load_bytes"] if transfer_available else None,
                "store_bytes": state["store_bytes"] if transfer_available else None,
                "load_worker_seconds": (
                    state["load_time_s"]
                    if state["load_time_observed"]
                    and state["load_time_complete"]
                    and transfer_available
                    else None
                ),
                "store_worker_seconds": (
                    state["store_time_s"]
                    if state["store_time_observed"]
                    and state["store_time_complete"]
                    and transfer_available
                    else None
                ),
                "load_critical_path_s": (
                    state["load_critical_path_s"]
                    if state["load_time_observed"] and transfer_available
                    else None
                ),
                "store_critical_path_s": (
                    state["store_critical_path_s"]
                    if state["store_time_observed"] and transfer_available
                    else None
                ),
                "load_time_s": (
                    state["load_time_s"]
                    if state["load_time_observed"]
                    and state["load_time_complete"]
                    and transfer_available
                    else None
                ),
                "store_time_s": (
                    state["store_time_s"]
                    if state["store_time_observed"]
                    and state["store_time_complete"]
                    and transfer_available
                    else None
                ),
                "tp_transfer_contributors": {
                    "load": state["load_contributors"],
                    "store": state["store_contributors"],
                    "expected": {
                        "load": state["load_expected_total"],
                        "store": state["store_expected_total"],
                    },
                },
                "tp_transfer_contributor_totals": {
                    "actual": {
                        "load": state["load_contributors"],
                        "store": state["store_contributors"],
                    },
                    "expected": {
                        "load": state["load_expected_total"],
                        "store": state["store_expected_total"],
                    },
                },
                # Do not infer TP identity when preservation metadata is
                # absent. The authoritative preservation validator will fail
                # closed on the resulting null fields.
                "tp_world_size": state.get("load_expected_contributors"),
                "tp_rank_count": state.get("load_expected_contributors"),
                "tp_contributors_complete": state["transfer_complete"],
                "preempted": state["preempted"],
                "preemption_count": state["preemption_count"],
                "pressure": pressure if authoritative else None,
                "recomputed": (
                    recomputed_tokens > 0
                    if recomputed_tokens is not None and authoritative
                    else None
                ),
                "prefetch_attempts": state["prefetch_attempts"],
                "prefetch_counts": {
                    "attempted": {
                        "keys": state["prefetch_attempted_keys"],
                        "blocks": state["prefetch_attempted_blocks"],
                    },
                    "consumed": {
                        "keys": state["prefetch_consumed_keys"],
                        "blocks": state["prefetch_consumed_blocks"],
                    },
                    "discarded": {
                        "keys": state["prefetch_discarded_keys"],
                        "blocks": state["prefetch_discarded_blocks"],
                    },
                },
            }
            record_preservation_event(event)
            del self._preservation_finished[req_id]
            self._preservation_requests.pop(req_id, None)

    def _score_context(self):
        if not getattr(self, '_score_transport_enabled', False):
            return {}
        now = time.monotonic_ns()
        for r, deadline in list(self._score_source_deadlines.items()):
            if now > deadline:
                self._score_source_tokens.pop(r, None)
                self._score_source_deadlines.pop(r, None)
        return {'epoch': self._score_transport_identity+':'+str(self._score_transport_generation),
                'sources': dict(self._score_source_tokens),
                'prefix_mean_prefetch_scoring': getattr(self, '_prefix_mean_prefetch_scoring', False),
                'residencies': {} if getattr(self, '_prefix_mean_prefetch_scoring', False) else self.manager.score_residency_versions()}

    def _score_request_started(self, request_id):
        if getattr(self, '_score_transport_enabled', False):
            self._score_source_sequence += 1
            self._score_source_tokens[request_id] = self._score_source_sequence
            self._score_source_deadlines.pop(request_id, None)

    def _score_request_finished(self, request):
        if not getattr(self, '_score_transport_enabled', False):
            return
        status = str(getattr(request, 'status', '')).lower()
        if any(reason in status for reason in ('abort', 'error')):
            self._score_source_tokens.pop(request.request_id, None)
            self._score_source_deadlines.pop(request.request_id, None)
        else:
            self._score_source_deadlines[request.request_id] = time.monotonic_ns()+self._score_source_lease_ns

    def on_new_request(self, request: Request) -> None:
        self._score_request_started(request.request_id)
        original_params = request.kv_transfer_params
        policy_params = _campaign_policy_params(original_params)
        if policy_params is original_params:
            super().on_new_request(request)
        else:
            request.kv_transfer_params = policy_params
            try:
                super().on_new_request(request)
            finally:
                request.kv_transfer_params = original_params
        self._bind_causal_scheduler_step(request.request_id)
        self._register_lifecycle_request(request)
        if (self._preservation_events_enabled
                and getattr(self, "_preservation_audit_enabled", True)):
            self._preservation_requests[request.request_id] = self._preservation_state(
                request
            )
        self._bind_wait_hint_request(request)

    def _bind_causal_scheduler_step(self, request_id: str) -> None:
        """Bind the current scheduler-owned step to the durable request context."""
        status = self._req_status.get(request_id)
        if status is None:
            return
        req_context = getattr(status, "req_context", None)
        if req_context is None:
            return
        params = dict(req_context.kv_transfer_params or {})
        params["scheduler_step"] = int(
            getattr(self, "_preservation_scheduler_batch_id", 0)
        )
        req_context.kv_transfer_params = params

    def _begin_lookup_episode(self, request: Request) -> None:
        state = self._preservation_requests.get(request.request_id)
        if state is None:
            return
        lookup_id = int(state["lookup_sequence"])
        state["lookup_sequence"] = lookup_id + 1
        state["active_lookup_episode"] = {
            "lookup_id": lookup_id,
            "episode_id": int(state.get("episode_id", 0)),
            "started_at_ns": time.monotonic_ns(),
            "groups": {},
        }

    def _mark_lookup_group_evaluated(self, req_id: str, group_idx: int, keys) -> None:
        state = self._preservation_requests.get(req_id)
        if state is None:
            return
        episode = state.get("active_lookup_episode")
        if episode is None or group_idx < 0:
            return
        group = episode["groups"].setdefault(
            str(group_idx),
            {
                "eligible": [],
                "queried": [],
                "hits": [],
                "missing": [],
                "pending": [],
                "retry": [],
                "ordered_probes": [],
                "observed": True,
            },
        )
        group["observed"] = True
        for key in keys:
            if key not in group["eligible"]:
                group["eligible"].append(key)

    def _lookup_group_index(self, req_id: str, keys) -> int:
        req_status = self._req_status.get(req_id)
        if req_status is not None:
            key_set = set(keys)
            for index, group_state in enumerate(req_status.group_states):
                if key_set and key_set.issubset(set(group_state.offload_keys)):
                    return index
        return -1

    def _record_lookup_probe(
        self, req_id: str, group_idx: int, eligible_keys, key, result: LookupResult
    ) -> None:
        state = self._preservation_requests.get(req_id)
        if state is None:
            return
        episode = state.get("active_lookup_episode")
        if episode is None:
            return
        groups = episode["groups"]
        group = groups.setdefault(
            str(group_idx),
            {
                "eligible": [],
                "queried": [],
                "hits": [],
                "missing": [],
                "pending": [],
                "retry": [],
                "ordered_probes": [],
                "observed": False,
            },
        )
        group["observed"] = True
        for eligible in eligible_keys:
            if eligible not in group["eligible"]:
                group["eligible"].append(eligible)
        if key not in group["queried"]:
            group["queried"].append(key)
        if (
            result in (LookupResult.HIT, LookupResult.HIT_PENDING)
            and key not in group["hits"]
        ):
            group["hits"].append(key)
        if result is LookupResult.MISS and key not in group["missing"]:
            group["missing"].append(key)
        if result is LookupResult.HIT_PENDING and key not in group["pending"]:
            group["pending"].append(key)
        if result is LookupResult.RETRY and key not in group["retry"]:
            group["retry"].append(key)
        key_position = next(
            (
                record
                for record in state.get("ordered_key_positions", [])
                if record.get("key") == key
                and int(record.get("group_idx", -1)) == group_idx
            ),
            None,
        )
        group["ordered_probes"].append(
            {
                **(dict(key_position) if key_position is not None else {}),
                "key": key,
                "eligible_chain_index": next(
                    (
                        index
                        for index, eligible in enumerate(eligible_keys)
                        if eligible == key
                    ),
                    None,
                ),
                "result": getattr(result, "name", str(result)).lower(),
                "observed_at_ns": time.monotonic_ns(),
                "sequence": self._next_preservation_event_sequence(),
            }
        )

    def _finish_lookup_episode(self, request: Request) -> None:
        state = self._preservation_requests.get(request.request_id)
        if state is None:
            return
        episode = state.pop("active_lookup_episode", None)
        if episode is None:
            return
        episode["completed_at_ns"] = time.monotonic_ns()
        if any(group.get("observed") for group in episode["groups"].values()):
            state["lookup_observed"] = True
        for group in episode["groups"].values():
            for key in group["queried"]:
                if key not in state["lookup_keys"]:
                    state["lookup_keys"].append(key)
                if key in self._preservation_evicted_keys:
                    if key not in state["prior_evicted_keys"]:
                        state["prior_evicted_keys"].append(key)
                elif key in self._preservation_known_stored_keys:
                    if key not in state["prior_stored_keys"]:
                        state["prior_stored_keys"].append(key)
                elif key not in state["cold_keys"]:
                    state["cold_keys"].append(key)
        state["lookup_episodes"].append(episode)
        state["last_lookup_id"] = episode["lookup_id"]

    def _record_short_remainder_lookup(
        self, request: Request, num_computed_tokens: int, result
    ) -> None:
        """Record the base lookup's explicit no-probe short-remainder path.

        The upstream offloading lookup returns ``0`` before invoking either
        lookup hook when the remaining request range is smaller than one KV
        chunk.  That is a complete no-eligible lookup, not missing metadata,
        but it is safe to record only when the base state proves the same
        short-remainder condition for every configured group.  Other
        hook-free returns remain unavailable rather than being inferred as
        no-eligible.
        """
        if not isinstance(result, tuple) or not result or result[0] != 0:
            return
        state = self._preservation_requests.get(request.request_id)
        if state is None:
            return
        episode = state.get("active_lookup_episode")
        if not isinstance(episode, dict) or episode["groups"]:
            return
        req_status = self._req_status.get(request.request_id)
        config = getattr(self, "config", None)
        group_configs = getattr(config, "kv_group_configs", ())
        req = getattr(req_status, "req", None)
        request_params = getattr(request, "kv_transfer_params", None)
        if getattr(request, "skip_reading_prefix_cache", None) is True or (
            isinstance(request_params, dict)
            and request_params.get("skip_reading_prefix_cache") is True
        ):
            return
        request_tokens = getattr(req, "num_tokens", None)
        local_tokens = getattr(req_status, "num_locally_computed_tokens", None)
        if not isinstance(request_tokens, int) or request_tokens < 0:
            return
        if (
            not isinstance(local_tokens, int)
            or local_tokens != num_computed_tokens
            or local_tokens > request_tokens
        ):
            return
        if not group_configs:
            return
        group_states = getattr(req_status, "group_states", ())
        lookup_groups = getattr(self, "_lookup_groups", ())
        try:
            configured_group_ids = {int(group.group_idx) for group in group_configs}
            lookup_group_ids = {int(group_idx) for group_idx in lookup_groups}
        except (AttributeError, TypeError, ValueError):
            return
        if (
            not lookup_group_ids
            or lookup_group_ids != configured_group_ids
            or len(group_states) != len(group_configs)
        ):
            return
        for group_config, group_state in zip(group_configs, group_states):
            if getattr(group_config, "sliding_window_size_in_chunks", None) is not None:
                return
            tokens_per_chunk = group_config_tokens_per_chunk(group_config)
            if (
                tokens_per_chunk <= 0
                or request_tokens - local_tokens >= tokens_per_chunk
                or not isinstance(
                    getattr(group_state, "offload_keys", None), (list, tuple)
                )
                or len(group_state.offload_keys) < request_tokens // tokens_per_chunk
            ):
                return
        for group_config in group_configs:
            self._mark_lookup_group_evaluated(
                request.request_id, int(group_config.group_idx), ()
            )

    def _invalidate_physical_block_table(
        self, state: dict[str, object], reason: str, scheduler_batch_id: int
    ) -> None:
        state["physical_block_identity_complete"] = False
        state.setdefault("physical_block_table_errors", []).append(
            {
                "reason": reason,
                "scheduler_batch_id": scheduler_batch_id,
                "observed_at_ns": time.monotonic_ns(),
                "sequence": self._next_preservation_event_sequence(),
            }
        )

    def _update_physical_block_table(
        self,
        state: dict[str, object],
        raw_groups,
        *,
        mode: str,
        start_tokens: object,
        scheduled_tokens: object,
        scheduled_prompt_tokens: object,
        scheduler_batch_id: int,
    ) -> tuple[list[int], ...] | None:
        """Maintain the full native request block table from causal deltas."""
        if (
            isinstance(start_tokens, bool)
            or not isinstance(start_tokens, int)
            or start_tokens < 0
        ):
            self._invalidate_physical_block_table(
                state, "missing_or_invalid_num_computed_tokens", scheduler_batch_id
            )
            return None
        if (
            isinstance(scheduled_tokens, bool)
            or not isinstance(scheduled_tokens, int)
            or scheduled_tokens < 0
            or isinstance(scheduled_prompt_tokens, bool)
            or not isinstance(scheduled_prompt_tokens, int)
            or scheduled_prompt_tokens < 0
            or scheduled_prompt_tokens > scheduled_tokens
        ):
            self._invalidate_physical_block_table(
                state, "invalid_scheduled_token_range", scheduler_batch_id
            )
            return None
        expected_groups = len(self.config.kv_group_configs)
        current = state.get("physical_block_table")
        current_groups = (
            tuple(list(group) for group in current)
            if isinstance(current, (list, tuple))
            and len(current) == expected_groups
            and all(isinstance(group, (list, tuple)) for group in current)
            else None
        )
        groups = None
        if raw_groups is not None:
            if (
                not isinstance(raw_groups, (list, tuple))
                or len(raw_groups) != expected_groups
            ):
                self._invalidate_physical_block_table(
                    state, "invalid_block_group_count", scheduler_batch_id
                )
                return None
            parsed = []
            for group in raw_groups:
                if not isinstance(group, (list, tuple)) or any(
                    isinstance(block_id, bool)
                    or not isinstance(block_id, int)
                    or block_id < 0
                    for block_id in group
                ):
                    self._invalidate_physical_block_table(
                        state, "malformed_block_group", scheduler_batch_id
                    )
                    return None
                if len(group) != len(set(group)):
                    self._invalidate_physical_block_table(
                        state, "duplicate_block_id", scheduler_batch_id
                    )
                    return None
                parsed.append(list(group))
            groups = tuple(parsed)
        before_lengths = (
            [len(group) for group in current_groups]
            if current_groups is not None
            else None
        )
        if mode == "scheduled_new_full":
            if groups is None or current_groups is not None:
                self._invalidate_physical_block_table(
                    state, "invalid_new_full_block_table", scheduler_batch_id
                )
                return None
            updated = groups
            generation = 0
        elif mode == "cached_resumed_replace":
            if groups is None:
                self._invalidate_physical_block_table(
                    state, "missing_resumed_full_block_table", scheduler_batch_id
                )
                return None
            updated = groups
            generation = int(state.get("physical_block_table_generation", 0)) + 1
        elif mode in {"cached_append", "cached_no_new_blocks"}:
            if current_groups is None:
                self._invalidate_physical_block_table(
                    state, "missing_prior_full_block_table", scheduler_batch_id
                )
                return None
            if groups is None:
                updated = current_groups
            else:
                updated = tuple(
                    [*current_group, *new_group]
                    for current_group, new_group in zip(
                        current_groups, groups, strict=True
                    )
                )
                if any(len(group) != len(set(group)) for group in updated):
                    self._invalidate_physical_block_table(
                        state, "appended_block_id_overlap", scheduler_batch_id
                    )
                    return None
            generation = int(state.get("physical_block_table_generation", 0))
        else:
            self._invalidate_physical_block_table(
                state, "unknown_block_table_update_mode", scheduler_batch_id
            )
            return None
        tokens_per_block = [
            group_config_tokens_per_block(group_config)
            for group_config in self.config.kv_group_configs
        ]
        scheduled_block_ids = []
        for group_idx, (group_block_size, group) in enumerate(
            zip(tokens_per_block, updated, strict=True)
        ):
            if len(group) < cdiv(start_tokens, group_block_size):
                self._invalidate_physical_block_table(
                    state,
                    "full_block_table_shorter_than_computed_prefix",
                    scheduler_batch_id,
                )
                return None
            start_block = start_tokens // group_block_size
            end_block = (
                start_block
                if scheduled_prompt_tokens == 0
                else cdiv(start_tokens + scheduled_prompt_tokens, group_block_size)
            )
            if end_block > len(group):
                self._invalidate_physical_block_table(
                    state,
                    "full_block_table_shorter_than_scheduled_range",
                    scheduler_batch_id,
                )
                return None
            scheduled_block_ids.extend(
                [int(state.get("episode_id", 0)), group_idx, int(block_id)]
                for block_id in group[start_block:end_block]
                if int(block_id) != 0
            )
        serialized = [list(group) for group in updated]
        table_digest = hashlib.sha256(
            json.dumps(serialized, separators=(",", ":")).encode()
        ).hexdigest()
        state["physical_block_table"] = tuple(tuple(group) for group in updated)
        state["physical_block_table_generation"] = generation
        state.setdefault("physical_block_table_updates", []).append(
            {
                "mode": mode,
                "scheduler_batch_id": scheduler_batch_id,
                "episode_id": int(state.get("episode_id", 0)),
                "table_generation": generation,
                "start_tokens": start_tokens,
                "scheduled_tokens": scheduled_tokens,
                "scheduled_prompt_tokens": scheduled_prompt_tokens,
                "tokens_per_block": tokens_per_block,
                "partial_tail_tokens": [
                    start_tokens % group_block_size
                    for group_block_size in tokens_per_block
                ],
                "scheduled_block_ids": scheduled_block_ids,
                "group_lengths_before": before_lengths,
                "group_lengths_after": [len(group) for group in updated],
                "group_block_ids": serialized,
                "table_digest": table_digest,
                "observed_at_ns": time.monotonic_ns(),
                "sequence": self._next_preservation_event_sequence(),
            }
        )
        return tuple(list(group) for group in updated)

    def _record_scheduler_execution(self, scheduler_output: SchedulerOutput) -> None:
        """Account prompt tokens scheduled for local execution per episode.

        ``num_scheduled_tokens`` is the scheduler's per-step execution
        contract and includes multistep/post-preemption admissions. It is
        intentionally used instead of Request.num_computed_tokens, which is
        logical progress and includes externally loaded KV.
        """
        if not getattr(self, "_preservation_audit_enabled", True):
            return
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        if not hasattr(scheduled, "items"):
            return
        scheduler_batch_id = int(getattr(self, "_preservation_scheduler_batch_id", 0))
        self._preservation_scheduler_batch_id = scheduler_batch_id + 1
        scheduler_batch_request_ids = [str(req_id) for req_id in scheduled]
        scheduler_batch_positions = {
            req_id: position
            for position, req_id in enumerate(scheduler_batch_request_ids)
        }
        scheduled_accounting = {}
        for req_id, raw_scheduled_tokens in scheduled.items():
            state = self._preservation_requests.get(req_id)
            if state is None:
                continue
            scheduled_tokens = max(0, int(raw_scheduled_tokens))
            remaining_prompt = max(
                0,
                int(state["required_tokens"]) - int(state["episode_prompt_accounted"]),
            )
            scheduled_accounting[req_id] = (
                scheduled_tokens,
                min(scheduled_tokens, remaining_prompt),
            )
        scheduled_block_groups: dict[str, tuple[tuple[list[int], ...], int | None]] = {}
        for request_data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            state = self._preservation_requests.get(request_data.req_id)
            if state is None:
                continue
            schedule = scheduled_accounting.get(request_data.req_id)
            if schedule is None:
                self._invalidate_physical_block_table(
                    state,
                    "block_table_payload_without_scheduled_request",
                    scheduler_batch_id,
                )
                continue
            start_tokens = getattr(request_data, "num_computed_tokens", None)
            full_table = self._update_physical_block_table(
                state,
                getattr(request_data, "block_ids", None),
                mode="scheduled_new_full",
                start_tokens=start_tokens,
                scheduled_tokens=schedule[0],
                scheduled_prompt_tokens=schedule[1],
                scheduler_batch_id=scheduler_batch_id,
            )
            if full_table is not None:
                scheduled_block_groups[request_data.req_id] = (
                    full_table,
                    int(start_tokens),
                )
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None:
            req_ids = list(getattr(cached, "req_ids", ()))
            new_block_ids = list(getattr(cached, "new_block_ids", ()))
            num_computed = list(getattr(cached, "num_computed_tokens", ()))
            resumed_req_ids = set(getattr(cached, "resumed_req_ids", set()))
            for index, req_id in enumerate(req_ids):
                state = self._preservation_requests.get(req_id)
                if state is None:
                    continue
                schedule = scheduled_accounting.get(req_id)
                if schedule is None:
                    self._invalidate_physical_block_table(
                        state,
                        "block_table_payload_without_scheduled_request",
                        scheduler_batch_id,
                    )
                    continue
                if index >= len(new_block_ids) or index >= len(num_computed):
                    self._invalidate_physical_block_table(
                        state, "truncated_cached_request_payload", scheduler_batch_id
                    )
                    continue
                block_groups = new_block_ids[index]
                start_tokens = num_computed[index]
                mode = (
                    "cached_resumed_replace"
                    if req_id in resumed_req_ids
                    else (
                        "cached_append"
                        if block_groups is not None
                        else "cached_no_new_blocks"
                    )
                )
                full_table = self._update_physical_block_table(
                    state,
                    block_groups,
                    mode=mode,
                    start_tokens=start_tokens,
                    scheduled_tokens=schedule[0],
                    scheduled_prompt_tokens=schedule[1],
                    scheduler_batch_id=scheduler_batch_id,
                )
                if full_table is not None:
                    scheduled_block_groups[req_id] = (
                        full_table,
                        int(start_tokens),
                    )
        for req_id, scheduled_tokens in scheduled.items():
            state = self._preservation_requests.get(req_id)
            if state is None:
                continue
            scheduled_tokens, prompt_tokens = scheduled_accounting[req_id]
            if scheduled_tokens > 0 and state.get("first_service_ns") is None:
                state["first_service_ns"] = time.monotonic_ns()
                state["first_service_source"] = (
                    "scheduler_num_scheduled_tokens_first_nonzero"
                )
                self._record_block_pool_snapshot(
                    state,
                    "first_scheduler_execution",
                    source="scheduler_num_scheduled_tokens",
                )
            execution_start_token = int(state["episode_prompt_accounted"])
            execution_end_token = execution_start_token + prompt_tokens
            state["prompt_tokens_scheduled"] = (
                int(state["prompt_tokens_scheduled"]) + scheduled_tokens
            )
            state["prompt_tokens_locally_executed"] = (
                int(state["prompt_tokens_locally_executed"]) + prompt_tokens
            )
            block_groups_and_start = scheduled_block_groups.get(req_id)
            block_ids: set[tuple[int, int]] = set()
            if block_groups_and_start is None:
                if prompt_tokens > 0:
                    state["physical_block_identity_complete"] = False
            else:
                block_groups, start_tokens = block_groups_and_start
                for group_idx, group in enumerate(block_groups):
                    if prompt_tokens == 0:
                        continue
                    if group_idx >= len(self.config.kv_group_configs):
                        state["physical_block_identity_complete"] = False
                        continue
                    tokens_per_block = group_config_tokens_per_block(
                        self.config.kv_group_configs[group_idx]
                    )
                    start_block = start_tokens // tokens_per_block
                    end_block = cdiv(start_tokens + prompt_tokens, tokens_per_block)
                    if end_block > len(group):
                        self._invalidate_physical_block_table(
                            state,
                            "full_block_table_shorter_than_scheduled_range",
                            scheduler_batch_id,
                        )
                    selected = group[start_block : min(end_block, len(group))]
                    block_ids.update(
                        (group_idx, int(block_id))
                        for block_id in selected
                        if int(block_id) != 0
                    )
            state["scheduled_block_ids"].update(
                (int(state["episode_id"]), group_idx, block_id)
                for group_idx, block_id in block_ids
            )
            if prompt_tokens > 0:
                state["recomputed_block_ids"].update(
                    (int(state["episode_id"]), group_idx, block_id)
                    for group_idx, block_id in block_ids
                )
                newly_completed_positions = [
                    {
                        **dict(record),
                        "state": "materialized_by_execution",
                        "episode_id": int(state["episode_id"]),
                        "admission_index": max(
                            0, int(state.get("admission_count", 0)) - 1
                        ),
                        "scheduler_batch_id": scheduler_batch_id,
                        "observed_at_ns": time.monotonic_ns(),
                        "sequence": self._next_preservation_event_sequence(),
                        "provenance": "scheduler_num_scheduled_tokens",
                    }
                    for record in state.get("ordered_key_positions", [])
                    if record.get("complete_chunk") is True
                    and isinstance(record.get("token_end"), int)
                    and execution_start_token
                    < int(record["token_end"])
                    <= execution_end_token
                ]
                state.setdefault("key_use_events", []).extend(newly_completed_positions)
                first_admission = state.get("first_admission")
                if int(state["episode_id"]) == 0 and isinstance(first_admission, dict):
                    existing = {
                        (record.get("group_idx"), record.get("chunk_index"))
                        for record in first_admission["recomputed_keys"]
                    }
                    first_admission["recomputed_keys"].extend(
                        record
                        for record in newly_completed_positions
                        if (
                            record.get("group_idx"),
                            record.get("chunk_index"),
                        )
                        not in existing
                    )
            state["episode_prompt_accounted"] = (
                int(state["episode_prompt_accounted"]) + prompt_tokens
            )
            state["execution_metadata_observed"] = True
            state["admission_episodes"].append(
                {
                    "episode_id": state["episode_id"],
                    "lookup_id": state["last_lookup_id"],
                    "scheduler_batch_id": scheduler_batch_id,
                    "scheduler_batch_request_ids": scheduler_batch_request_ids,
                    "scheduler_batch_position": scheduler_batch_positions.get(
                        str(req_id)
                    ),
                    "scheduler_batch_size": len(scheduler_batch_request_ids),
                    "scheduled_tokens": scheduled_tokens,
                    "prompt_tokens_locally_executed": prompt_tokens,
                }
            )

    def _maximal_prefix_lookup(
        self, keys, req_context, req, group_config, start_chunk_idx
    ) -> int | None:
        """Mirror the base lookup while recording the exact queried sets."""
        hit_count = 0
        defer_lookup = False
        eligible_keys = list(keys)
        self._refresh_ordered_key_positions(req.request_id)
        self._mark_lookup_group_evaluated(
            req.request_id, group_config.group_idx, eligible_keys
        )
        for local_idx, key in enumerate(eligible_keys):
            result = self.manager.lookup(key, req_context)
            self._record_lookup_probe(
                req.request_id,
                group_config.group_idx,
                eligible_keys,
                key,
                result,
            )
            match result:
                case LookupResult.HIT:
                    self._events_tracker.record_lookup(
                        req,
                        group_config,
                        start_chunk_idx + local_idx,
                        key,
                    )
                    hit_count += 1
                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    hit_count += 1
                case LookupResult.RETRY:
                    defer_lookup = True
                case LookupResult.MISS:
                    break
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self, keys, sliding_window_size: int, req_context
    ) -> int | None:
        """Mirror the base suffix lookup while recording exact key sets."""
        defer_lookup = False
        consecutive_hits = 0
        eligible_keys = list(keys)
        self._refresh_ordered_key_positions(req_context.req_id)
        group_idx = self._lookup_group_index(req_context.req_id, eligible_keys)
        self._mark_lookup_group_evaluated(req_context.req_id, group_idx, eligible_keys)
        for idx in range(len(eligible_keys) - 1, -1, -1):
            key = eligible_keys[idx]
            result = self.manager.lookup(key, req_context)
            self._record_lookup_probe(
                req_context.req_id,
                group_idx,
                eligible_keys,
                key,
                result,
            )
            match result:
                case LookupResult.HIT:
                    consecutive_hits += 1
                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    defer_lookup = True
                    consecutive_hits = 0
                case LookupResult.MISS:
                    consecutive_hits = 0
            if consecutive_hits == sliding_window_size:
                return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        self._bind_causal_scheduler_step(request.request_id)
        hint_event_id = getattr(self, "_wait_hint_request_bindings", {}).get(
            request.request_id
        )
        self._reclaim_wait_prefetch_for_demand(
            request,
            num_computed_tokens,
            protected_event_id=hint_event_id,
        )
        hint_state = (
            getattr(self, "_wait_prefetch", {}).get(hint_event_id)
            if hint_event_id is not None
            else None
        )
        if hint_state is not None:
            self._expire_wait_prefetches(time.monotonic_ns())
            hint_state = getattr(self, "_wait_prefetch", {}).get(hint_event_id)
        if hint_state is not None and hint_state.status in {
            "pending",
            "in_flight",
        }:
            return None, False
        if getattr(self, "_preservation_events_enabled", False):
            self._begin_lookup_episode(request)
        try:
            result = super().get_num_new_matched_tokens(request, num_computed_tokens)
            self._refresh_ordered_key_positions(request.request_id)
            if getattr(self, "_preservation_events_enabled", False):
                self._record_short_remainder_lookup(
                    request, num_computed_tokens, result
                )
        finally:
            if getattr(self, "_preservation_events_enabled", False):
                self._finish_lookup_episode(request)
        if not getattr(self, "_preservation_events_enabled", False):
            return result
        state = getattr(self, "_preservation_requests", {}).get(request.request_id)
        if (
            state is not None
            and result[0] is not None
            and isinstance(state.get("local_reused_tokens"), int)
            and isinstance(state.get("required_tokens"), int)
        ):
            state["local_reused_tokens"] = max(
                int(state["local_reused_tokens"]), num_computed_tokens
            )
            state["episode_local_reused_tokens"] = min(
                int(state["required_tokens"]), max(0, int(num_computed_tokens))
            )
        return result

    def on_request_preempted(self, request: Request) -> None:
        """Step 1.5: queue a just-preempted request for a prefetch
        reservation attempt -- the actual attempt always happens from
        _retry_pending_prefetches (issues log entry #27), never
        synchronously here. This is NOT just style: `build_connector_meta`'s
        base class flushes any *store* job for a request appearing in
        `scheduler_output.preempted_req_ids` this same step and asserts
        every flushed job `is_store` (offloading/scheduler.py ~line 1104).
        A prefetch is a *load*. If a reservation succeeded synchronously
        here and the request were still in this step's preempted set by
        the time that flush runs, it would trip that assert and crash the
        engine -- not yet observed live only because the tested workloads
        happened to never have free blocks at the preemption instant
        itself (issues log entry #24), not because it can't happen."""
        if getattr(self, "_preservation_events_enabled", False):
            state = getattr(self, "_preservation_requests", {}).get(request.request_id)
            if state is not None:
                state["preempted"] = True
                if state.get("prefetch_t0_ns") is None:
                    state["prefetch_t0_ns"] = time.monotonic_ns()
                self._record_block_pool_snapshot(
                    state,
                    "t0_preemption",
                    source="on_request_preempted",
                )
        if _DISABLE_PREFETCH or self._prefetch_budget_fraction == 0:
            return
        self._preempted_pending.add(request.request_id)

    def _queue_preempted(self, scheduler_output: SchedulerOutput) -> None:
        """vLLM removed the scheduler-side on_request_preempted hook this
        class used to override (confirmed absent from the whole vllm/ tree
        as of 2026-07-29 -- it was replaced by SchedulerOutput's own
        preempted_req_ids field plus a worker-side handle_preemptions
        callback that this connector doesn't need). Preemptions are now
        only observable from inside build_connector_meta via
        scheduler_output.preempted_req_ids, so this is called from there
        instead, first thing, to keep on_request_preempted's original
        contract: queue only, never reserve synchronously here (same
        crash risk its docstring describes -- _retry_pending_prefetches
        still excludes this step's preempted_req_ids from the attempt
        list, so timing is unchanged from when the hook fired)."""
        if _DISABLE_PREFETCH or self._prefetch_budget_fraction == 0:
            return
        self._preempted_pending.update(scheduler_output.preempted_req_ids or ())

    def _retry_pending_prefetches(self, scheduler_output: SchedulerOutput) -> None:
        """Called every scheduling step (from build_connector_meta), for
        every request queued by _queue_preempted (or on_request_preempted,
        no longer called by vLLM but kept for its own unit tests) or a
        previously-failed retry. Requests preempted THIS SAME step are
        skipped (see on_request_preempted's docstring) -- by the next step
        they're no longer in `scheduler_output.preempted_req_ids`, and
        get_num_new_matched_tokens's existing "defer while transfer_jobs is
        non-empty" guard (offloading/scheduler.py ~line 718) prevents them
        from being re-preempted while a prefetch job is pending, so a
        request can only ever be in `preempted_req_ids` again after its
        prior prefetch has fully completed and cleared."""
        if not self._preempted_pending:
            return
        if _DEBUG_ENABLED:
            free_blocks = (
                self.gpu_block_pool.get_num_free_blocks()
                if self.gpu_block_pool
                else None
            )
            debug_print(
                f"STALL_DEBUG retry_sweep pending={list(self._preempted_pending)} "
                f"free_blocks={free_blocks} "
                f"reserved={self._prefetch_reserved_blocks} "
                f"prefetched_keys={list(self._prefetched.keys())}"
            )
        this_step_preempted = scheduler_output.preempted_req_ids or ()
        succeeded = [
            req_id
            for req_id in self._preempted_pending
            if req_id not in this_step_preempted
            and self._attempt_prefetch_reservation(req_id)
        ]
        self._preempted_pending.difference_update(succeeded)

    def _attempt_prefetch_reservation(self, req_id: str) -> bool:
        """One attempt to reserve GPU blocks and start loading `req_id`'s
        highest-relevance CPU-tier content. Returns True on success (or if
        there was nothing to do -- e.g. the request already has a prefetch,
        or genuinely has no candidate keys at all -- so the caller should
        NOT keep retrying it), False if it should be retried later because
        the only blocker was transient (no free GPU blocks right now)."""
        preservation_state = getattr(self, "_preservation_requests", {}).get(req_id)
        if preservation_state is not None:
            preservation_state["prefetch_attempts"] = (
                int(preservation_state["prefetch_attempts"]) + 1
            )
        selector = getattr(self, "_prefetch_selector", "semantic")
        attempt_id = None
        if preservation_state is not None:
            attempt_id = (
                f"prefetch-{req_id}-episode-{preservation_state.get('episode_id', 0)}"
                f"-attempt-{preservation_state['prefetch_attempts']}"
            )
        if self.gpu_block_pool is None:
            self._record_block_pool_snapshot(
                preservation_state,
                "t1_reservation_unavailable",
                source="prefetch_reservation",
            )
            return True  # nothing to retry towards; not a real "no keys" case
        if req_id in self._prefetched:
            return True  # already have one in flight or ready
        req_status = self._req_status.get(req_id)
        if req_status is None:
            return True  # request is gone; nothing left to retry
        candidate_keys = [
            key
            for group_state in req_status.group_states
            for key in group_state.offload_keys
        ]
        if not candidate_keys:
            return True  # never will have keys to prefetch; stop retrying

        total_budget = int(
            self.gpu_block_pool.num_gpu_blocks * self._prefetch_budget_fraction
        )
        remaining_budget = max(0, total_budget - self._prefetch_reserved_blocks)
        if remaining_budget <= 0:
            self._record_block_pool_snapshot(
                preservation_state,
                "t1_reservation_refused",
                source="prefetch_reservation",
            )
            return False  # budget may free up later (another prefetch splices/releases)
        eligible_resident_prefix = _ready_resident_prefix(self.manager, candidate_keys)
        if not eligible_resident_prefix:
            self._record_block_pool_snapshot(
                preservation_state,
                "t1_residency_unavailable",
                source="prefetch_reservation",
            )
            return False
        top_keys = _select_prefetch_keys(
            self.manager,
            eligible_resident_prefix,
            remaining_budget,
            selector,
            selection_context={
                "score_max_age_ns": getattr(
                    self, "_prefetch_score_max_age_ns", 1_000_000_000
                ),
                "now_ns": time.monotonic_ns(),
                "remaining_wait_ns": None,
                "bytes_per_key": getattr(self, "_prefetch_bytes_per_key", None),
                "estimated_transfer_ns_per_byte": getattr(
                    self, "_prefetch_transfer_ns_per_byte", None
                ),
                "estimated_splice_ns_per_byte": getattr(
                    self, "_prefetch_fallback_splice_ns_per_byte", None
                ),
            },
        )
        resident_keys = top_keys
        if not resident_keys:
            # A scored-but-not-yet-resident or not-yet-scored key may become
            # eligible on a later step (e.g. its own store job finishes, or
            # a later query scores it) -- worth retrying, not permanent.
            self._record_block_pool_snapshot(
                preservation_state,
                "t1_residency_unavailable",
                source="prefetch_reservation",
            )
            return False
        free_blocks = self._prefetch_free_capacity()
        n = min(
            len(resident_keys),
            max(0, free_blocks - getattr(self, "_prefetch_headroom_blocks", 0)),
        )
        if n <= 0:
            # issues log entry #24: this was the consistently-hit case under
            # sustained pressure at the preemption instant itself -- the
            # retry loop exists specifically to give a LATER step (when
            # something else frees blocks) a chance this single-shot
            # version never had.
            self._record_block_pool_snapshot(
                preservation_state,
                "t1_reservation_refused",
                source="prefetch_reservation",
            )
            return False
        resident_keys = resident_keys[:n]

        reserved = []
        gpu_block_ids: list[int] = []
        job_id: int | None = None
        prepared_load = False
        reservation_accounted = False
        try:
            reserved = self.gpu_block_pool.get_new_blocks(n)
            gpu_block_ids = [b.block_id for b in reserved]
            if preservation_state is not None:
                preservation_state["prefetch_required_keys"] = [
                    _event_key(key) for key in candidate_keys
                ]
                preservation_state["prefetch_cpu_resident_keys"] = [
                    _event_key(key) for key in resident_keys
                ]
                preservation_state["prefetch_cpu_ready_keys"] = [
                    _event_key(key) for key in resident_keys
                ]
                preservation_state["prefetch_cpu_residency_proof"] = {
                    _event_key(key): "manager_lookup_hit" for key in resident_keys
                }
                preservation_state["prefetch_selected_keys"] = [
                    _event_key(key) for key in resident_keys
                ]
                preservation_state["prefetch_reservation_blocks"] = [
                    str(block_id) for block_id in gpu_block_ids
                ]
                preservation_state["prefetch_reservation_history"].append(
                    {
                        "phase": "reserved",
                        "action": "reserve",
                        "reserved_blocks": len(gpu_block_ids),
                        "free_blocks": self._prefetch_free_capacity(),
                    }
                )
                preservation_state["prefetch_attempted_keys"] = int(
                    preservation_state["prefetch_attempted_keys"]
                ) + len(resident_keys)
                preservation_state["prefetch_attempted_blocks"] = int(
                    preservation_state["prefetch_attempted_blocks"]
                ) + len(gpu_block_ids)
            # issues log entry #27 (Fix C): without this, a just-reallocated
            # block that used to be a finished request's pending GPU->CPU store
            # source wouldn't be recognized by build_connector_meta's
            # store-flush fence (offloading/scheduler.py ~line 1107-1119, which
            # only checks blocks recorded here) -- the prefetch's CPU->GPU
            # write could then race an in-flight store still reading the same
            # physical block, silently corrupting the offloaded copy. No crash,
            # wrong output -- caught by tracing the fence's real precondition,
            # not by a test failing.
            self._current_batch_allocated_block_ids.update(
                bid for bid in gpu_block_ids if bid != 0
            )
            src_spec = self.manager.prepare_load(resident_keys, req_status.req_context)
            prepared_load = True
            dst_spec = GPULoadStoreSpec(
                gpu_block_ids, group_sizes=[n], block_indices=[0]
            )
            job_id = self._generate_job_id()
            if preservation_state is not None:
                preservation_state["prefetch_reservation_id"] = f"prefetch-{job_id}"
            # issues log entry #27 (Fix A): dispatched via the SEPARATE
            # prefetch_load_jobs channel, not load_jobs -- see
            # SemanticOffloadingConnectorMetadata.prefetch_load_jobs's docstring.
            self._current_batch_prefetch_load_jobs[job_id] = TransferJob(
                req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
            )
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(resident_keys),
                is_store=False,
            )
            # issues log entry #26: MUST register here. update_connector_output's
            # generic per-job completion loop unconditionally does
            # `req_status.transfer_jobs.remove(job_id)` for every job in
            # self._jobs, with no exception for jobs that were never added --
            # skipping this crashes the engine with a KeyError the moment this
            # job completes (confirmed on the real server). The original design
            # deliberately avoided this to dodge update_state_after_alloc's
            # `assert not req_status.transfer_jobs`, but that fear was
            # unfounded: get_num_new_matched_tokens already defers re-admission
            # whenever `req_status.transfer_jobs` is non-empty ("Delaying
            # request ... since it still has in-flight transfers"), so a
            # request can never reach update_state_after_alloc while this job
            # is still pending -- the existing framework already provides
            # exactly the protection the separate-tracking design was trying to
            # invent from scratch.
            req_status.transfer_jobs.add(job_id)
            self._prefetch_reserved_blocks += n
            reservation_accounted = True
            self._prefetched[req_id] = _PrefetchState(
                job_id=job_id,
                keys=resident_keys,
                gpu_block_ids=gpu_block_ids,
                attempt_id=attempt_id,
            )
            if preservation_state is not None:
                preservation_state["prefetch_t1_ns"] = time.monotonic_ns()
                self._record_block_pool_snapshot(
                    preservation_state,
                    "t1_reservation_submit",
                    source="prefetch_reservation",
                    job_id=job_id,
                )
                if attempt_id is not None:
                    _append_prefetch_attempt(
                        preservation_state,
                        attempt_id=attempt_id,
                        episode_id=int(preservation_state.get("episode_id", 0)),
                        selector=selector,
                        candidate_keys=candidate_keys,
                        selected_keys=resident_keys,
                        cpu_resident_ready_keys=resident_keys,
                        reservation_id=f"prefetch-job-{job_id}",
                        reservation_block_ids=gpu_block_ids,
                        reservation_submit_time_s=time.monotonic(),
                    )
        except BaseException:
            # Reservation setup is transactional. Undo CPU-tier references,
            # job publication, allocation tracking, and scratch blocks on any
            # failure before the worker owns the transfer.
            if job_id is not None:
                self._current_batch_prefetch_load_jobs.pop(job_id, None)
                self._jobs.pop(job_id, None)
                req_status.transfer_jobs.discard(job_id)
            if prepared_load:
                self.manager.complete_load(resident_keys, req_status.req_context)
            self._prefetched.pop(req_id, None)
            if reservation_accounted:
                self._prefetch_reserved_blocks -= n
            self._current_batch_allocated_block_ids.difference_update(
                bid for bid in gpu_block_ids if bid != 0
            )
            if gpu_block_ids:
                self.gpu_block_pool.free_blocks(
                    [self.gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )
            if preservation_state is not None:
                preservation_state["prefetch_failure_reason"] = (
                    "reservation_setup_failed"
                )
            raise
        debug_print(
            f"PREFETCH_EFFECT_DEBUG RESERVED req={req_id} n={n} job_id={job_id}"
        )
        return True

    def _release_prefetch(self, req_id: str) -> None:
        """Return a request's reserved-but-unused/no-longer-needed prefetch
        blocks to the free pool. Safe to call for a request with no
        prefetch state (no-op)."""
        state = self._prefetched.pop(req_id, None)
        preservation_state = getattr(self, "_preservation_requests", {}).get(
            req_id
        ) or getattr(self, "_preservation_finished", {}).get(req_id)
        if state is None:
            return
        wait_state = getattr(self, "_wait_prefetch", {}).pop(state.attempt_id, None)
        if wait_state is not None:
            self._wait_hint_request_bindings.pop(req_id, None)
            wait_state.status = "retired"
            wait_state.bound_request_id = None
            wait_state.lease.cancel()
        if preservation_state is not None:
            _set_prefetch_attempt_status(
                preservation_state, state.attempt_id, "abandoned"
            )
            if not preservation_state.get("prefetch_splice_acknowledged", False):
                preservation_state["prefetch_stale"] = True
                preservation_state["prefetch_failure_reason"] = "stale_prefetch"
            preservation_state["prefetch_discarded_keys"] = int(
                preservation_state["prefetch_discarded_keys"]
            ) + len(state.keys)
            preservation_state["prefetch_discarded_blocks"] = int(
                preservation_state["prefetch_discarded_blocks"]
            ) + len(state.gpu_block_ids)
        if self.gpu_block_pool is None:
            return
        free_blocks_before = self.gpu_block_pool.get_num_free_blocks()
        self._prefetch_reserved_blocks -= len(state.gpu_block_ids)
        blocks = [self.gpu_block_pool.blocks[bid] for bid in state.gpu_block_ids]
        self.gpu_block_pool.free_blocks(blocks)
        if wait_state is not None:
            wait_state.destination_released = True
            self._record_wait_hint_event(
                {
                    "schema": "semantic-known-prefix-wait-hint-event-v1",
                    "event": "staging_retired",
                    "event_id": wait_state.lease.hint.event_id,
                    **self._wait_state_identity(wait_state),
                    "reason": "bound_prefetch_released",
                    "released_blocks": len(state.gpu_block_ids),
                    "released_bytes": (
                        len(state.gpu_block_ids) * self._prefetch_bytes_per_key
                    ),
                    "retired_at_ns": time.monotonic_ns(),
                }
            )
        if preservation_state is not None:
            preservation_state["prefetch_cleanup_released"] = True
            preservation_state["prefetch_reservation_history"].append(
                {
                    "phase": "released",
                    "action": "release",
                    "reserved_blocks": 0,
                    "free_blocks": self.gpu_block_pool.get_num_free_blocks(),
                    "free_blocks_before": free_blocks_before,
                }
            )

    def _free_prefetch_blocks(self, block_ids: list[int]) -> None:
        if not block_ids or self.gpu_block_pool is None:
            return
        blocks = [self.gpu_block_pool.blocks[bid] for bid in block_ids]
        self.gpu_block_pool.free_blocks(blocks)

    def request_finished(self, request: Request):
        self._score_request_finished(request)
        getattr(self, "_published_wait_prefixes", {}).pop(request.request_id, None)
        # A request can finish (or abort) while still preempted, in which
        # case a still-pending retry attempt would otherwise loop forever --
        # this is the one lifecycle point that always fires before a
        # request's state is torn down, regardless of how it ends.
        self._register_observed_prefix(request)
        self._preempted_pending.discard(request.request_id)
        if getattr(self, "_preservation_events_enabled", False):
            state = getattr(self, "_preservation_requests", {}).get(request.request_id)
            if state is not None:
                status = getattr(request, "status", None)
                status_name = getattr(status, "name", None) or (
                    str(status) if status is not None else "unknown"
                )
                state["terminal_status"] = status_name
                state["request_finished_ns"] = time.monotonic_ns()
                state["request_finished_source"] = "connector_request_finished"
                self._record_block_pool_snapshot(
                    state,
                    "request_finished",
                    source="connector_request_finished",
                )
                terminal_status_lower = status_name.lower()
                state["terminal_reason"] = (
                    "completed"
                    if (
                        terminal_status_lower in {"finished", "completed", "success"}
                        or terminal_status_lower.startswith("finished_")
                    )
                    and "aborted" not in terminal_status_lower
                    else terminal_status_lower
                )
                if self._lifecycle_release_timing == "scheduler_boundary":
                    self._defer_lifecycle_request(request, state)
                else:
                    self._process_lifecycle_request(request, state)
                self._preservation_finished[request.request_id] = state
        state = self._prefetched.get(request.request_id)
        if state is not None:
            job_status = self._jobs.get(state.job_id)
            if job_status is None or job_status.pending_count == 0:
                # The transfer already finished (or was already cleaned up)
                # -- safe to free these GPU blocks right now, nothing is
                # still writing into them.
                self._release_prefetch(request.request_id)
            # else: the prefetch's load job is still actively transferring
            # data into these reserved blocks. Freeing them now would let
            # something else reuse that GPU memory while the worker is
            # still writing to it -- a real data race, not a cleanup nit.
            # Left in self._prefetched; update_connector_output's orphan
            # sweep below releases it once the job's own completion is
            # observed, mirroring the base class's own discipline of
            # keeping req_status alive while any of its jobs are in flight
            # rather than tearing it down eagerly.
        hint_event_id = getattr(self, "_wait_hint_request_bindings", {}).pop(
            request.request_id, None
        )
        hint_state = (
            getattr(self, "_wait_prefetch", {}).get(hint_event_id)
            if hint_event_id is not None
            else None
        )
        if hint_state is not None:
            hint_state.lease.cancel()
            hint_state.bound_request_id = None
            if hint_state.status in {"in_flight", "retired_pending_fence"}:
                self._retire_wait_prefetch(
                    hint_state,
                    "bound_request_finished",
                    now_ns=time.monotonic_ns(),
                )
            elif hint_state.status == "bound_ready":
                self._release_wait_prefetch_protection(hint_state)
                hint_state.status = "retired"
                self._wait_prefetch.pop(hint_event_id, None)
            else:
                self._free_wait_prefetch(hint_state, "bound_request_finished")
        return super().request_finished(request)

    def _cleanup_prefetch_output(self, connector_output: KVConnectorOutput) -> None:
        if _DEBUG_ENABLED:
            meta = connector_output.kv_connector_worker_meta
            completed = getattr(meta, "completed_jobs", None)
            transfers = {
                rid: list(req_status.transfer_jobs)
                for rid, req_status in self._req_status.items()
                if req_status.transfer_jobs
            }
            debug_print(
                f"STALL_DEBUG update_connector_output completed_jobs={completed} "
                f"live_jobs={list(self._jobs.keys())} "
                f"transfer_jobs_by_req={transfers}"
            )
        completed = dict(
            getattr(connector_output.kv_connector_worker_meta, "completed_jobs", {})
            or {}
        )
        completed_splices = dict(
            getattr(
                connector_output.kv_connector_worker_meta,
                "completed_splices",
                {},
            )
            or {}
        )
        completed_splice_evidence = dict(
            getattr(
                connector_output.kv_connector_worker_meta,
                "completed_splice_evidence",
                {},
            )
            or {}
        )
        for job_id, count in completed.items():
            pending = self._reset_prefetch_releases.get(job_id)
            if pending is None:
                continue
            block_ids, remaining = pending
            remaining -= count
            if remaining <= 0:
                self._free_prefetch_blocks(block_ids)
                self._prefetch_reserved_blocks -= len(block_ids)
                del self._reset_prefetch_releases[job_id]
                getattr(self, "_preservation_job_events", {}).pop(job_id, None)
            else:
                self._reset_prefetch_releases[job_id] = (block_ids, remaining)
        for splice_id, count in completed_splices.items():
            pending = self._pending_splice_releases.get(splice_id)
            if pending is None:
                continue
            block_ids, remaining = pending
            remaining -= count
            if remaining <= 0:
                free_blocks_before = (
                    self.gpu_block_pool.get_num_free_blocks()
                    if self.gpu_block_pool is not None
                    else None
                )
                self._free_prefetch_blocks(block_ids)
                self._prefetch_reserved_blocks -= len(block_ids)
                del self._pending_splice_releases[splice_id]
                protection_id = getattr(self, "_pending_splice_protection_ids", {}).pop(
                    splice_id, None
                )
                if protection_id is not None:
                    self.manager.release_staging_keys(protection_id)
                request_id = getattr(self, "_pending_splice_request_ids", {}).pop(
                    splice_id, None
                )
                if request_id is not None:
                    if connector_output.finished_recving is None:
                        connector_output.finished_recving = set()
                    connector_output.finished_recving.add(request_id)
                for state in (
                    *getattr(self, "_preservation_requests", {}).values(),
                    *getattr(self, "_preservation_finished", {}).values(),
                ):
                    if state.get("prefetch_splice_ack_id") != f"splice-{splice_id}":
                        continue
                    state["prefetch_splice_acknowledged"] = True
                    state["prefetch_splice_fence_ns"] = time.monotonic_ns()
                    state["prefetch_request_usable_ns"] = state[
                        "prefetch_splice_fence_ns"
                    ]
                    state["prefetch_request_usable_source"] = (
                        "all_worker_splice_cuda_fences_acknowledged"
                    )
                    state["prefetch_splice_fence_evidence"] = [
                        dict(event)
                        for event in completed_splice_evidence.get(splice_id, [])
                    ]
                    state["prefetch_fence_complete"] = True
                    state["prefetch_cleanup_released"] = True
                    state["prefetch_fence_ns"] = (
                        state.get("prefetch_fence_ns") or time.monotonic_ns()
                    )
                    state["prefetch_reservation_history"].append(
                        {
                            "phase": "released",
                            "action": "release",
                            "reserved_blocks": 0,
                            "free_blocks": (
                                self.gpu_block_pool.get_num_free_blocks()
                                if self.gpu_block_pool is not None
                                else None
                            ),
                            "free_blocks_before": free_blocks_before,
                        }
                    )
                    self._record_block_pool_snapshot(
                        state,
                        "prefetch_request_usable",
                        source="completed_splice_fences",
                    )
            else:
                self._pending_splice_releases[splice_id] = (block_ids, remaining)
        # A prefetch whose owning request already finished/aborted while
        # its load job was still in flight has no one left to ever splice
        # it in (update_state_after_alloc, the only other place that
        # releases a prefetch, can't fire for a request that's gone).
        # req_status.transfer_jobs.remove()/req_status deletion (both in the
        # super() call just above) only happen once a job's completion is
        # actually processed, so `req_id not in self._req_status` here means
        # "this request is finished AND every one of its jobs, including
        # this prefetch, has genuinely completed" -- not just "the request
        # is gone", so this can't race the still-in-flight transfer the
        # same way freeing it in request_finished directly could have.
        orphaned = [
            req_id for req_id in self._prefetched if req_id not in self._req_status
        ]
        for req_id in orphaned:
            self._release_prefetch(req_id)

    def _compute_load_plan(
        self, request: Request, blocks, num_external_tokens: int
    ) -> tuple[list[OffloadKey], list[int], int, int] | None:
        """Reproduces the base connector's single-group positional
        keys-to-load / destination-block computation (mirrors
        `offloading/scheduler.py`'s `update_state_after_alloc`, ~line
        749-821) so that `_try_splice_prefetch`'s splice partition and its
        remainder reload agree on ordering by construction -- both are
        derived from this one call, never recomputed independently.
        Returns `None` if there's nothing pending to load. Also returns
        `num_locally_computed_gpu_blocks` (needed as the remainder reload's
        `block_indices`) and `num_blocks` (needed for
        `next_stored_block_idx` bookkeeping) alongside the base method's
        two return values, since both callers need them."""
        req_status = self._req_status[request.request_id]
        group_config = self.config.kv_group_configs[0]
        group_state = req_status.group_states[0]
        group_blocks = blocks.blocks[0]
        num_cached_tokens = req_status.num_locally_computed_tokens + num_external_tokens
        gpu_block_size = group_config_tokens_per_block(group_config)
        num_gpu_blocks = cdiv(num_cached_tokens, gpu_block_size)
        num_locally_computed_gpu_blocks = num_gpu_blocks
        for i, block in enumerate(group_blocks[:num_gpu_blocks]):
            if not block.is_null and block.block_hash is None:
                num_locally_computed_gpu_blocks = i
                break
        num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks
        if num_pending_gpu_blocks == 0:
            return None

        offloaded_block_size = group_config_tokens_per_chunk(group_config)
        num_blocks = cdiv(num_cached_tokens, offloaded_block_size)
        start_block_idx = num_locally_computed_gpu_blocks // config_blocks_per_chunk(
            self.config
        )
        keys_to_load = group_state.offload_keys[start_block_idx:num_blocks]
        dst_block_ids = [
            block.block_id
            for block in group_blocks[num_locally_computed_gpu_blocks:num_gpu_blocks]
        ]
        assert len(keys_to_load) == len(dst_block_ids)
        return keys_to_load, dst_block_ids, num_locally_computed_gpu_blocks, num_blocks

    @profiled('scheduler.splice_prepare')
    def _try_splice_prefetch(
        self, request: Request, blocks, num_external_tokens: int
    ) -> bool:
        """Step 1.5: splice whatever fraction of this request's official
        load an already-completed prefetch covers via a fast GPU-to-GPU
        copy, and issue a normal (slower) CPU-tier reload for only the
        complementary remainder in the same step -- no longer all-or-
        nothing (see issues log entries #29-32 for why the original
        exact-match design never actually spliced anything: prefetched
        content is matched to destination blocks by `OffloadKey` identity,
        never by list position, since `keys_to_load` is positionally
        ordered and `prefetch.keys` is relevance-ranked -- two different
        orderings that only coincide by chance). Returns True if it spliced
        anything at all (caller must NOT also call the normal load path for
        this request this step -- the remainder reload, if any, is already
        issued here). Returns False only when the prefetch covers none of
        what's needed, in which case the normal path handles everything.

        Deliberately scoped to the single-KV-group, `block_size_factor==1`
        case (this project's actual test/dev model, Qwen2.5-1.5B-Instruct)
        -- a known, documented simplification, same spirit as
        store_job_keys' job-level granularity. At `block_size_factor > 1` a
        scattered (non-contiguous) remainder reload would compute wrong
        intra-block offsets in the worker's block-skip logic, so those
        requests (and multi-group/sliding-window ones) always fall through
        to the normal path unmodified; their prefetch (if any) is simply
        released as stale by the caller."""
        if (
            len(self.config.kv_group_configs) != 1
            or config_blocks_per_chunk(self.config) != 1
        ):
            return False
        prefetch = self._prefetched.get(request.request_id)
        if prefetch is None:
            return False
        debug_print(
            f"PREFETCH_EFFECT_DEBUG {request.request_id}: prefetch exists at "
            f"re-admission, attempting splice"
        )
        # issues log entry #28: `job_status is None` does NOT mean "not
        # ready" -- it means the OPPOSITE. Reaching this method at all (with
        # num_external_tokens > 0) already implies get_num_new_matched_tokens
        # let re-admission through, which its own unconditional first check
        # (offloading/scheduler.py ~line 717-722: `if req_status.transfer_jobs:
        # return None, False`) would never have done while OUR job's id was
        # still registered there. Since a job's removal from
        # req_status.transfer_jobs happens atomically with `del
        # self._jobs[job_id]` in update_connector_output's completion loop,
        # `self._jobs.get(job_id)` returning None at this point means the
        # job already completed and was cleaned up -- treating that as
        # "not ready" made this method return False on every real call,
        # 9/9 times on the real server, discarding a still-good, already-
        # loaded prefetch every single time. Only a still-registered job
        # with pending_count > 0 is genuinely not ready (defensive case,
        # not expected to occur given the deferral guarantee above, but
        # harmless to keep checking).
        req_status = self._req_status[request.request_id]
        plan = self._compute_load_plan(request, blocks, num_external_tokens)
        if plan is None:
            return False
        keys_to_load, dst_block_ids, num_locally_computed_gpu_blocks, num_blocks = plan
        preservation_state = getattr(self, "_preservation_requests", {}).get(
            request.request_id
        )
        job_status = self._jobs.get(prefetch.job_id)
        if job_status is not None and job_status.pending_count > 0:
            if preservation_state is not None:
                _update_prefetch_attempt_plan(
                    preservation_state,
                    prefetch.attempt_id,
                    keys_to_load,
                    status="in_flight_at_readmission",
                    actual_destination_block_ids=dst_block_ids,
                )
            debug_print(
                f"PREFETCH_EFFECT_DEBUG {request.request_id}: NOT READY "
                f"(job_status={job_status})"
            )
            return False  # prefetch load hasn't completed yet

        if preservation_state is not None:
            _update_prefetch_attempt_plan(
                preservation_state,
                prefetch.attempt_id,
                keys_to_load,
                status="ready_at_readmission",
                actual_destination_block_ids=dst_block_ids,
            )

        # Identity-based partition (issues log entries #31-32; the mandatory
        # correctness move the partial-splice plan doc identified): a
        # per-key FIFO queue, not a plain dict, so duplicate keys (identical
        # block content recurring within one request's chain) each still
        # get their own distinct prefetched block rather than colliding.
        prefetched_blocks_by_key: dict[OffloadKey, list[int]] = defaultdict(list)
        for key, block_id in zip(prefetch.keys, prefetch.gpu_block_ids):
            prefetched_blocks_by_key[key].append(block_id)

        splice_src: list[int] = []
        splice_dst: list[int] = []
        splice_key_map: dict[OffloadKey, tuple[int, int]] = {}
        remainder_keys: list[OffloadKey] = []
        remainder_dst: list[int] = []
        for key, dst in zip(keys_to_load, dst_block_ids):
            bucket = prefetched_blocks_by_key.get(key)
            if bucket:
                source = bucket.pop(0)
                splice_src.append(source)
                splice_dst.append(dst)
                splice_key_map[key] = (source, dst)
            else:
                remainder_keys.append(key)
                remainder_dst.append(dst)

        if not splice_src:
            if preservation_state is not None:
                _set_prefetch_attempt_status(
                    preservation_state, prefetch.attempt_id, "ready_no_intersection"
                )
            debug_print(
                f"PREFETCH_EFFECT_DEBUG {request.request_id}: KEY MISMATCH "
                f"spliced=0 keys_to_load={len(keys_to_load)} "
                f"prefetch.keys={len(prefetch.keys)}"
            )
            return False  # nothing usable here; normal path loads everything

        # Guards against issues log entry #31's Q3-2 hazard (a block ending
        # up in both the splice and the remainder-reload sets, letting an
        # async CPU->GPU DMA race the synchronous splice into the same
        # physical block): the partition above is disjoint by construction
        # (each position goes to exactly one list), asserted here as a
        # cheap, permanent guard rather than trusted implicitly.
        assert set(splice_dst).isdisjoint(remainder_dst)
        if preservation_state is not None:
            _set_prefetch_attempt_status(
                preservation_state, prefetch.attempt_id, "consuming"
            )

        group_state = req_status.group_states[0]
        self._current_batch_allocated_block_ids.update(
            bid for bid in splice_dst if bid != 0
        )
        splice_id = self._splice_counter
        self._splice_counter += 1
        self._pending_splice_jobs[splice_id] = (splice_src, splice_dst)
        if not remainder_keys:
            # With a remainder, its ordinary load job already owns the one
            # finished_recving notification. Full coverage has no such job,
            # so the splice acknowledgement must supply it exactly once.
            self._pending_splice_request_ids[splice_id] = request.request_id
        # Detach the request lookup now, but keep every source reservation
        # quarantined until all workers acknowledge the actual GPU splice.
        detached = self._prefetched.pop(request.request_id)
        wait_state = getattr(self, "_wait_prefetch", {}).pop(detached.attempt_id, None)
        if wait_state is not None:
            self._wait_hint_request_bindings.pop(request.request_id, None)
            wait_state.status = "splice_in_flight"
            wait_state.bound_request_id = None
            if wait_state.protection_binding_id is not None:
                pending_protections = getattr(
                    self, "_pending_splice_protection_ids", None
                )
                if pending_protections is None:
                    pending_protections = self._pending_splice_protection_ids = {}
                pending_protections[splice_id] = wait_state.protection_binding_id
            self._record_wait_hint_event(
                {
                    "schema": "semantic-known-prefix-wait-hint-event-v1",
                    "event": "staging_consumed",
                    "event_id": wait_state.lease.hint.event_id,
                    **self._wait_state_identity(wait_state),
                    "request_id": request.request_id,
                    "splice_id": splice_id,
                    "consumed_keys": len(splice_src),
                    "consumed_bytes": len(splice_src) * self._prefetch_bytes_per_key,
                    "discarded_keys": len(detached.keys) - len(splice_src),
                    "discarded_bytes": (len(detached.keys) - len(splice_src))
                    * self._prefetch_bytes_per_key,
                    "consumed_at_ns": time.monotonic_ns(),
                }
            )
        preservation_state = getattr(self, "_preservation_requests", {}).get(
            request.request_id
        )
        if preservation_state is not None:
            preservation_state["prefetch_install_submitted_ns"] = time.monotonic_ns()
            intersection_keys = [key for key in keys_to_load if key in splice_key_map]
            preservation_state["prefetch_required_keys"] = [
                _event_key(key) for key in keys_to_load
            ]
            preservation_state["prefetch_intersection_keys"] = [
                _event_key(key) for key in intersection_keys
            ]
            preservation_state["prefetch_remainder_keys"] = [
                _event_key(key) for key in remainder_keys
            ]
            preservation_state["prefetch_selected_keys"] = [
                _event_key(key) for key in intersection_keys
            ]
            preservation_state["prefetch_cpu_residency_proof"] = {
                _event_key(key): "manager_lookup_hit" for key in intersection_keys
            }
            preservation_state["prefetch_source_map"] = {
                _event_key(key): str(splice_key_map[key][0])
                for key in intersection_keys
            }
            preservation_state["prefetch_destination_map"] = {
                _event_key(key): str(splice_key_map[key][1])
                for key in intersection_keys
            }
            preservation_state["prefetch_splice_ack_id"] = f"splice-{splice_id}"
            preservation_state["prefetch_splice_bytes"] = (
                len(splice_src) * self._prefetch_bytes_per_key
            )
            preservation_state["prefetch_consumed_keys"] = int(
                preservation_state["prefetch_consumed_keys"]
            ) + len(splice_src)
            preservation_state["prefetch_consumed_blocks"] = int(
                preservation_state["prefetch_consumed_blocks"]
            ) + len(splice_src)
            preservation_state["prefetch_discarded_keys"] = (
                int(preservation_state["prefetch_discarded_keys"])
                + len(detached.keys)
                - len(splice_src)
            )
            preservation_state["prefetch_discarded_blocks"] = (
                int(preservation_state["prefetch_discarded_blocks"])
                + len(detached.gpu_block_ids)
                - len(splice_src)
            )
        self._pending_splice_releases[splice_id] = (
            detached.gpu_block_ids,
            self.config.num_workers,
        )

        if remainder_keys:
            # A reduced reload for only the complementary remainder --
            # mirrors offloading/scheduler.py's update_state_after_alloc
            # (~line 823-845) for the single-group case, since the base
            # method takes no "load only this subset" parameter. Safe to
            # issue alongside the splice above because the splice itself is
            # untracked (no TransferJob, nothing in req_status.transfer_jobs
            # -- a fire-and-forget index_copy_ run in bind_connector_metadata
            # before any load starts, see issues log entry #31's Q2), so
            # this remains exactly one tracked job per request, identical in
            # kind to today's normal full reload.
            self._current_batch_allocated_block_ids.update(
                bid for bid in remainder_dst if bid != 0
            )
            src_spec = self.manager.prepare_load(remainder_keys, req_status.req_context)
            dst_spec = GPULoadStoreSpec(
                remainder_dst,
                group_sizes=[len(remainder_dst)],
                block_indices=[num_locally_computed_gpu_blocks],
            )
            job_id = self._generate_job_id()
            assert not req_status.transfer_jobs
            self._current_batch_load_jobs[job_id] = TransferJob(
                req_id=request.request_id, src_spec=src_spec, dst_spec=dst_spec
            )
            req_status.transfer_jobs.add(job_id)
            self._jobs[job_id] = TransferJobStatus(
                req_id=request.request_id,
                pending_count=self.config.num_workers,
                keys=set(remainder_keys),
                is_store=False,
            )
            being_loaded = scheduler_being_loaded_set(self)
            if being_loaded is not None:
                being_loaded.update(remainder_keys)

        # NOTE: no manual manager.complete_load() call here -- checking
        # job_status.pending_count == 0 above already implies
        # update_connector_output's generic, job-id-keyed completion
        # processing (scheduler.py's update_connector_output) has already
        # called it automatically for the PREFETCH job, regardless of
        # whether that job was ever registered in req_status.transfer_jobs.
        # Calling it again here would double-decrement the CPU-tier blocks'
        # ref_cnt -- caught by tracing the real completion path before
        # writing this, not discovered by testing. The remainder reload job
        # created above (if any) is a completely normal tracked job and
        # will get its own automatic complete_load() call when it finishes,
        # same as any other load.
        if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
            set_next_stored_chunk_idx(group_state, num_blocks)
        covered = len(splice_dst) / (len(splice_dst) + len(remainder_dst))
        debug_print(
            f"PREFETCH_EFFECT_DEBUG {request.request_id}: PARTIAL SPLICE "
            f"spliced={len(splice_dst)} reloaded={len(remainder_dst)} "
            f"covered={covered:.2f}"
        )
        return True

    def _capture_readmission_demand_boundary(self, state: dict[str, object]) -> None:
        """Capture demand at the server allocation callback before splice work."""
        state["prefetch_t4_ns"] = time.monotonic_ns()
        state["prefetch_t4_source"] = "scheduler_update_state_after_alloc_entry"
        state["prefetch_t4_episode_id"] = int(state.get("episode_id", 0))
        self._record_block_pool_snapshot(
            state,
            "t4_readmission_allocation",
            source="scheduler_update_state_after_alloc_entry",
        )

    def _record_admission(
        self,
        request: Request,
        blocks,
        num_external_tokens: int,
        *,
        first_readmission: bool | None = None,
    ) -> None:
        """Record one scheduler admission without using logical progress.

        Prompt execution is accounted separately from this allocation hook by
        ``SchedulerOutput.num_scheduled_tokens`` in build_connector_meta().
        This hook records the source tokens admitted for the episode and is
        called on both the ordinary load path and the prefetch-splice path.
        """
        if not getattr(self, "_preservation_events_enabled", False):
            return
        state = getattr(self, "_preservation_requests", {}).get(request.request_id)
        if state is None:
            return
        external_tokens = max(0, int(num_external_tokens))
        if first_readmission is None:
            first_readmission = bool(state["readmission_pending"])
        if first_readmission:
            if state.get("prefetch_t4_episode_id") != int(state.get("episode_id", 0)):
                self._capture_readmission_demand_boundary(state)
            state["prefetch_episode_id"] = f"episode-{int(state['episode_id'])}"
            state["episode_external_tokens"] = external_tokens
        else:
            state["episode_external_tokens"] = max(
                int(state["episode_external_tokens"]), external_tokens
            )
        state["loaded_tokens"] = max(int(state["loaded_tokens"]), external_tokens)
        state["loaded_tokens_total"] = (
            int(state["loaded_tokens_total"]) + external_tokens
        )
        state["admitted_external_tokens"] = (
            int(state["admitted_external_tokens"]) + external_tokens
        )
        req_status = self._req_status.get(request.request_id)
        local_tokens = 0
        if req_status is not None:
            local_tokens = max(
                0, int(getattr(req_status, "num_locally_computed_tokens", 0))
            )
            if first_readmission:
                state["admitted_local_tokens"] = local_tokens
            else:
                state["admitted_local_tokens"] = max(
                    int(state["admitted_local_tokens"]),
                    local_tokens,
                )
        if req_status is not None and hasattr(blocks, "blocks"):
            if len(blocks.blocks) != len(self.config.kv_group_configs):
                state["physical_block_identity_complete"] = False
            for group_idx, group_blocks in enumerate(blocks.blocks):
                if group_idx >= len(self.config.kv_group_configs):
                    state["physical_block_identity_complete"] = False
                    continue
                tokens_per_block = group_config_tokens_per_block(
                    self.config.kv_group_configs[group_idx]
                )
                local_tokens = int(
                    getattr(req_status, "num_locally_computed_tokens", 0)
                )
                local_end = min(len(group_blocks), cdiv(local_tokens, tokens_per_block))
                cached_end = min(
                    len(group_blocks),
                    cdiv(local_tokens + external_tokens, tokens_per_block),
                )
                if cached_end < cdiv(local_tokens + external_tokens, tokens_per_block):
                    state["physical_block_identity_complete"] = False
                actual_blocks = group_blocks[:cached_end]
                num_local_blocks = min(local_end, len(actual_blocks))
                state["admitted_local_blocks"] = (
                    int(state["admitted_local_blocks"]) + num_local_blocks
                )
                state["admitted_external_blocks"] = int(
                    state["admitted_external_blocks"]
                ) + max(0, len(actual_blocks) - num_local_blocks)
                for index, block in enumerate(actual_blocks):
                    block_id = int(getattr(block, "block_id", 0))
                    if not block_id:
                        continue
                    state["required_block_ids"].add(
                        (int(state["episode_id"]), group_idx, block_id)
                    )
                    if index < num_local_blocks:
                        state["local_block_ids"].add(
                            (int(state["episode_id"]), group_idx, block_id)
                        )
                    else:
                        state["external_block_ids"].add(
                            (int(state["episode_id"]), group_idx, block_id)
                        )
        else:
            state["physical_block_identity_complete"] = False
        if int(state["admission_count"]) == 0:
            admission_ns = time.monotonic_ns()
            state["first_admission"] = {
                "admission_index": 0,
                "episode_id": int(state["episode_id"]),
                "lookup_id": state.get("last_lookup_id"),
                "admitted_at_ns": admission_ns,
                "sequence": self._next_preservation_event_sequence(),
                "local_keys": self._complete_positions_in_token_range(
                    state, 0, local_tokens
                ),
                "planned_load_keys": self._complete_positions_in_token_range(
                    state, local_tokens, local_tokens + external_tokens
                ),
                "completed_load_keys": [],
                "recomputed_keys": [],
                "provenance": "scheduler_update_state_after_alloc",
            }
        _reconcile_episode_prompt_cursor_after_admission(
            state,
            local_tokens,
            external_tokens,
            first_readmission=first_readmission,
        )
        state["readmission_pending"] = False
        state["admission_count"] = int(state["admission_count"]) + 1
        if external_tokens == 0 and state["prior_evicted_keys"]:
            state["recompute_admission_count"] = (
                int(state["recompute_admission_count"]) + 1
            )

    def update_state_after_alloc(
        self, request: Request, blocks, num_external_tokens: int
    ) -> None:
        self._record_native_prefix_adoption(request, blocks, num_external_tokens)
        # Being re-admitted at all means this request is no longer
        # preempted -- stop retrying it regardless of what happens below.
        self._preempted_pending.discard(request.request_id)
        preservation_state = getattr(self, "_preservation_requests", {}).get(
            request.request_id
        )
        first_readmission = bool(
            preservation_state is not None
            and preservation_state.get("readmission_pending", False)
        )
        if first_readmission:
            self._capture_readmission_demand_boundary(preservation_state)
        if num_external_tokens > 0 and self._try_splice_prefetch(
            request, blocks, num_external_tokens
        ):
            self._record_admission(
                request,
                blocks,
                num_external_tokens,
                first_readmission=first_readmission,
            )
            return
        # Prefetch either doesn't exist, isn't ready, or covers none of
        # what's needed this step (a partial cover was already spliced and
        # handled above) -- the normal path below will load everything
        # itself, making the stale prefetch redundant.
        if request.request_id in self._prefetched:
            debug_print(
                f"PREFETCH_EFFECT_DEBUG {request.request_id}: STALE, released "
                f"unspliced, falling back to normal load"
            )
            self._release_prefetch(request.request_id)
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        self._record_admission(
            request,
            blocks,
            num_external_tokens,
            first_readmission=first_readmission,
        )

    def _calc_num_offloadable_tokens(self, req_status, num_computed_tokens: int) -> int:
        if (
            not self.config.offload_prompt_only
            and req_status.req.is_finished()
        ):
            # The native finished-request path supplies req.num_tokens, which
            # may include one sampled output with no computed KV. Incremental
            # running stores retain the native scheduled-forward fence path.
            computed = getattr(req_status.req, "num_computed_tokens", None)
            in_flight = getattr(req_status.req, "num_in_flight_tokens", 0)
            num_computed_tokens = (
                min(num_computed_tokens, max(0, computed - in_flight))
                if type(computed) is int and computed >= 0
                and type(in_flight) is int and in_flight >= 0 else 0
            )
        return super()._calc_num_offloadable_tokens(req_status, num_computed_tokens)

    def _build_store_jobs(self, scheduler_output: SchedulerOutput):
        store_jobs = super()._build_store_jobs(scheduler_output)
        # A probe layer belongs to one KV group. Until worker metadata carries
        # that group explicitly, fail closed for hybrid/multi-group models.
        if len(self.config.kv_group_configs) != 1:
            return store_jobs
        blocks_per_chunk = config_blocks_per_chunk(self.config)
        for job_id, job in store_jobs.items():
            status = self._jobs[job_id]
            key_cpu_blocks: dict[OffloadKey, int] = {}
            for key in status.keys:
                block = self.manager._policy.get(key)
                if block is not None:
                    key_cpu_blocks[key] = block.block_id
            layout = build_store_job_layout(
                key_cpu_blocks,
                job.dst_spec.block_ids,
                job.src_spec.block_ids,
                blocks_per_chunk,
            )
            if layout is None:
                # Sliding-window holes cannot be reconstructed from the
                # flattened source list. Guessing would corrupt identities.
                debug_print(
                    f"SEMANTIC_STORE_LAYOUT_SKIPPED job={job_id} "
                    f"keys={len(key_cpu_blocks)} "
                    f"src_blocks={len(job.src_spec.block_ids)} "
                    f"blocks_per_chunk={blocks_per_chunk}"
                )
                continue
            self._pending_job_blocks[job_id] = layout
        return store_jobs

    def _cleanup_failed_job(self, job_id: int, status: str) -> None:
        job_status = self._jobs.get(job_id)
        info = self._preservation_jobs.pop(job_id, None)
        transfer = self._preservation_job_events.pop(job_id, None)
        if job_status is None or info is None:
            return
        req_status = self._req_status.get(job_status.req_id)
        if req_status is None:
            self._jobs.pop(job_id, None)
            return
        req_status.transfer_jobs.discard(job_id)
        state = self._preservation_requests.get(
            job_status.req_id
        ) or self._preservation_finished.get(job_status.req_id)
        if state is not None:
            if info.get("is_prefetch"):
                state["prefetch_failure_reason"] = status
                state["prefetch_cancelled"] = status == "cancelled"
            else:
                state["transfer_complete"] = False
                if not job_status.is_store:
                    state["prefetch_failure_reason"] = status
                    state["prefetch_allocation_failures"] = (
                        int(state["prefetch_allocation_failures"]) + 1
                    )
            transfer_record = _transfer_event_record(
                job_id,
                info,
                transfer,
                status=status,
                success=False,
                cancelled=status == "cancelled",
            )
            if info.get("is_prefetch"):
                state.setdefault("speculative_transfer_job_statuses", []).append(
                    transfer_record
                )
                _prefetch_attribution_state(state)["transfer_jobs"].append(
                    transfer_record
                )
            else:
                state["transfer_job_statuses"].append(transfer_record)
        if job_status.is_store:
            self.manager.complete_store(
                job_status.keys, req_status.req_context, success=False
            )
            for block_id in (job_status.sliding_window_block_ids or []) + (
                job_status.non_sliding_window_block_ids or []
            ):
                pending = self._block_id_to_pending_jobs.get(block_id)
                if pending is not None:
                    pending.discard(job_id)
                    if not pending:
                        del self._block_id_to_pending_jobs[block_id]
        else:
            self.manager.complete_load(job_status.keys, req_status.req_context)
            if self._chunks_being_loaded:
                self._chunks_being_loaded.difference_update(job_status.keys)
            failed_blocks = set(info.get("dst_block_ids", ()))
            for group_idx, group_state in enumerate(req_status.group_states):
                failed_ids = {
                    block_id
                    for ref_group, block_id in failed_blocks
                    if ref_group == group_idx
                }
                group_state.block_ids[:] = [
                    0 if block_id in failed_ids else block_id
                    for block_id in group_state.block_ids
                ]
            if info.get("is_prefetch"):
                self._release_prefetch(job_status.req_id)
        self._jobs.pop(job_id, None)
        if req_status.finished_signaled and not req_status.transfer_jobs:
            del self._req_status[job_status.req_id]

    @profiled('scheduler.consume_worker')
    def update_connector_output(self, connector_output: KVConnectorOutput):
        if not getattr(self, "_preservation_events_enabled", False):
            result = super().update_connector_output(connector_output)
            self._cleanup_prefetch_output(connector_output)
            return result

        meta = connector_output.kv_connector_worker_meta
        for cuda_event in getattr(meta, "prefill_cuda_events", []) or []:
            if not isinstance(cuda_event, dict):
                continue
            for attribution in cuda_event.get("execution_attribution", []) or []:
                if not isinstance(attribution, dict):
                    continue
                req_id = attribution.get("request_id")
                state = self._preservation_requests.get(
                    req_id
                ) or self._preservation_finished.get(req_id)
                if state is not None:
                    state.setdefault("prefill_cuda_events", []).append(dict(cuda_event))
        for name, value in getattr(meta, "allocator_peak_stats", {}).items():
            self._allocator_peak_stats[name] = max(
                self._allocator_peak_stats.get(name, 0), int(value)
            )
        for req_id, timestamp_ns in getattr(
            meta, "preempted_forward_times_ns", {}
        ).items():
            state = self._preservation_requests.get(
                req_id
            ) or self._preservation_finished.get(req_id)
            if state is None:
                continue
            state["prefetch_t6_ns"] = int(timestamp_ns)
            state["prefetch_t6_source"] = "model_runner_execute_entry_after_preemption"
            state["prefetch_t6_episode_id"] = int(state.get("episode_id", 0))
            state["prefetch_t6_preemption_index"] = int(
                state.get("preemption_count", 0)
            )
        for req_id, timestamp_ns in getattr(meta, "first_forward_times_ns", {}).items():
            state = self._preservation_requests.get(
                req_id
            ) or self._preservation_finished.get(req_id)
            if state is None:
                continue
            timestamp_ns = int(timestamp_ns)
            prior = state.get("first_forward_ns")
            if prior is None or timestamp_ns < int(prior):
                state["first_forward_ns"] = timestamp_ns
                state["first_forward_source"] = "model_runner_execute_entry"
        transfer_events = getattr(meta, "transfer_events", {}) or {}
        for job_id, event in transfer_events.items():
            if (not getattr(self, "_preservation_audit_enabled", True)
                    and job_id not in self._jobs):
                # CPU advice has no speculative jobs; late/unknown metadata
                # must not create history with no operational owner.
                continue
            previous = self._preservation_job_events.get(job_id)
            if previous is None:
                self._preservation_job_events[job_id] = event
                continue
            if previous.is_load != event.is_load:
                raise ValueError(
                    f"preservation transfer direction changed for job {job_id}"
                )
            self._preservation_job_events[job_id] = SemanticTransferCompletion(
                is_load=event.is_load,
                bytes=(
                    previous.bytes + event.bytes
                    if previous.bytes is not None and event.bytes is not None
                    else None
                ),
                time_s=(
                    previous.time_s + event.time_s
                    if previous.time_s is not None and event.time_s is not None
                    else None
                ),
                max_time_s=(
                    max(previous.max_time_s, event.max_time_s)
                    if previous.max_time_s is not None and event.max_time_s is not None
                    else None
                ),
                contributors=previous.contributors + event.contributors,
                success=previous.success and event.success,
                status=(
                    "cancelled"
                    if previous.cancelled or event.cancelled
                    else "success"
                    if previous.success and event.success
                    else "failure"
                ),
                cancelled=previous.cancelled or event.cancelled,
                start_time_s=_earliest_timestamp(
                    previous.start_time_s, event.start_time_s
                ),
                completion_time_s=_latest_timestamp(
                    previous.completion_time_s, event.completion_time_s
                ),
                started_at_ns=(
                    min(
                        value
                        for value in (previous.started_at_ns, event.started_at_ns)
                        if value is not None
                    )
                    if previous.started_at_ns is not None
                    or event.started_at_ns is not None
                    else None
                ),
                completed_at_ns=(
                    max(
                        value
                        for value in (
                            previous.completed_at_ns,
                            event.completed_at_ns,
                        )
                        if value is not None
                    )
                    if previous.completed_at_ns is not None
                    or event.completed_at_ns is not None
                    else None
                ),
                contributor_ids=tuple(
                    sorted(set(previous.contributor_ids) | set(event.contributor_ids))
                ),
            )

        self._complete_wait_prefetch_jobs(meta)
        completed_jobs = dict(getattr(meta, "completed_jobs", {}) or {})
        failed_jobs = dict(getattr(meta, "failed_jobs", {}) or {})
        for job_id, failed_count in failed_jobs.items():
            job_status = self._jobs.get(job_id)
            if job_status is None:
                continue
            completed_count = int(completed_jobs.get(job_id, 0))
            if failed_count + completed_count < job_status.pending_count:
                continue
            transfer = self._preservation_job_events.get(job_id)
            failure_status = (
                transfer.status
                if transfer is not None and transfer.status != "success"
                else "failure"
            )
            self._cleanup_failed_job(job_id, failure_status)
        for job_id, count in completed_jobs.items():
            job_status = self._jobs.get(job_id)
            if job_status is None or job_status.pending_count - count > 0:
                continue
            info = self._preservation_jobs.pop(job_id, None)
            state = (
                self._preservation_requests.get(info["req_id"])
                or self._preservation_finished.get(info["req_id"])
                if info is not None
                else None
            )
            candidate_transfer = self._preservation_job_events.get(job_id)
            if (
                info is not None
                and candidate_transfer is not None
                and candidate_transfer.is_load != bool(info["is_load"])
            ):
                self._preservation_jobs[job_id] = info
                self._cleanup_failed_job(job_id, "direction_mismatch")
                continue
            transfer = self._preservation_job_events.pop(job_id, None)
            if info is None or state is None or transfer is None:
                if state is not None:
                    if info is not None and info.get("is_prefetch"):
                        state["prefetch_failure_reason"] = (
                            "prefetch_transfer_completion_unavailable"
                        )
                    else:
                        state["transfer_complete"] = False
                continue
            if transfer.bytes is None:
                if info.get("is_prefetch"):
                    state["prefetch_failure_reason"] = (
                        "prefetch_transfer_bytes_unavailable"
                    )
                else:
                    state["transfer_complete"] = False
            expected_contributors = int(info["expected_contributors"])
            physical_refs = (
                info["dst_block_ids"] if info["is_load"] else info["src_block_ids"]
            )
            physical_refs_complete = (
                info["dst_block_ids_complete"]
                if info["is_load"]
                else info["src_block_ids_complete"]
            )
            if (
                not info.get("is_prefetch")
                and int(info["chunks"]) > 0
                and (not physical_refs or not physical_refs_complete)
            ):
                state["physical_block_identity_complete"] = False
            if not transfer.success:
                if info.get("is_prefetch"):
                    state["prefetch_failure_reason"] = transfer.status
                    state["prefetch_cancelled"] = transfer.cancelled
                else:
                    state["transfer_complete"] = False
                    if info.get("is_load"):
                        state["prefetch_failure_reason"] = transfer.status
                        state["prefetch_allocation_failures"] = (
                            int(state["prefetch_allocation_failures"]) + 1
                        )
            state.setdefault("prefetch_transfer_contributors", {})[str(job_id)] = list(
                transfer.contributor_ids
            )
            transfer_record = _transfer_event_record(job_id, info, transfer)
            if info.get("is_prefetch"):
                state.setdefault("speculative_transfer_job_statuses", []).append(
                    transfer_record
                )
                _prefetch_attribution_state(state)["transfer_jobs"].append(
                    transfer_record
                )
            else:
                state["transfer_job_statuses"].append(transfer_record)
            if transfer.contributors != expected_contributors and not info.get(
                "is_prefetch"
            ):
                state["transfer_complete"] = False
            if not info.get("is_prefetch"):
                if transfer.is_load:
                    state["load_contributors"] = (
                        int(state["load_contributors"]) + transfer.contributors
                    )
                else:
                    state["store_contributors"] = (
                        int(state["store_contributors"]) + transfer.contributors
                    )
            if transfer.bytes is None:
                continue
            if transfer.is_load:
                if info.get("is_prefetch"):
                    if transfer.started_at_ns is not None:
                        previous_start = state.get("prefetch_t2_ns")
                        state["prefetch_t2_ns"] = (
                            transfer.started_at_ns
                            if previous_start is None
                            else min(previous_start, transfer.started_at_ns)
                        )
                    if transfer.completed_at_ns is not None:
                        previous_complete = state.get("prefetch_t3_ns")
                        state["prefetch_t3_ns"] = (
                            transfer.completed_at_ns
                            if previous_complete is None
                            else max(previous_complete, transfer.completed_at_ns)
                        )
                        state["prefetch_fence_ns"] = time.monotonic_ns()
                        state["prefetch_fence_complete"] = bool(transfer.success)
                    if transfer.bytes is not None:
                        state["prefetch_h2d_bytes"] = int(
                            state.get("prefetch_h2d_bytes") or 0
                        ) + int(transfer.bytes)
                    if transfer.time_s is not None:
                        prior_time = state.get("prefetch_time_s") or 0.0
                        state["prefetch_time_s"] = prior_time + transfer.time_s
                    self._record_block_pool_snapshot(
                        state,
                        "prefetch_transfer_observed",
                        source="worker_metadata_aggregation",
                        job_id=job_id,
                    )
                elif info.get("is_load") and state.get("prefetch_remainder_keys"):
                    if transfer.started_at_ns is not None:
                        previous_start = state.get("prefetch_t5_start_ns")
                        state["prefetch_t5_start_ns"] = (
                            transfer.started_at_ns
                            if previous_start is None
                            else min(previous_start, transfer.started_at_ns)
                        )
                    if transfer.completed_at_ns is not None:
                        previous_complete = state.get("prefetch_t5_end_ns")
                        state["prefetch_t5_end_ns"] = (
                            transfer.completed_at_ns
                            if previous_complete is None
                            else max(previous_complete, transfer.completed_at_ns)
                        )
                    if transfer.bytes is not None:
                        state["prefetch_blocking_h2d_bytes"] = int(
                            state.get("prefetch_blocking_h2d_bytes") or 0
                        ) + int(transfer.bytes)
                    if transfer.time_s is not None:
                        prior_time = state.get("prefetch_blocking_time_s") or 0.0
                        state["prefetch_blocking_time_s"] = prior_time + transfer.time_s
                    self._record_block_pool_snapshot(
                        state,
                        "t5_blocking_load_observed",
                        source="worker_metadata_aggregation",
                        job_id=job_id,
                    )
                if not info.get("is_prefetch"):
                    if transfer.contributors == expected_contributors:
                        state["loaded_block_ids"].update(
                            (job_id, group_idx, block_id)
                            for group_idx, block_id in info["dst_block_ids"]
                        )
                        if int(info.get("admission_index", -1)) == 0:
                            first_admission = state.get("first_admission")
                            if isinstance(first_admission, dict):
                                first_admission["completed_load_keys"].extend(
                                    {
                                        **dict(record),
                                        "state": "materialized_by_cpu_load",
                                        "job_id": job_id,
                                        "completed_at_ns": transfer.completed_at_ns,
                                        "sequence": (
                                            self._next_preservation_event_sequence()
                                        ),
                                        "provenance": "worker_load_completion",
                                    }
                                    for record in info.get("key_positions", [])
                                )
                    state["loaded_chunks"] = int(state["loaded_chunks"]) + int(
                        info["chunks"]
                    )
                    state["load_bytes"] = int(state["load_bytes"]) + transfer.bytes
                    if transfer.time_s is None:
                        state["load_time_complete"] = False
                    else:
                        state["load_time_observed"] = True
                        state["load_time_s"] = (
                            float(state["load_time_s"]) + transfer.time_s
                        )
                    if transfer.max_time_s is not None:
                        state["load_critical_path_s"] = (
                            float(state["load_critical_path_s"]) + transfer.max_time_s
                        )
            else:
                if transfer.contributors == expected_contributors:
                    state["stored_block_ids"].update(
                        (job_id, group_idx, block_id)
                        for group_idx, block_id in info["src_block_ids"]
                    )
                state["stored_chunks"] = int(state["stored_chunks"]) + int(
                    info["chunks"]
                )
                state["store_bytes"] = int(state["store_bytes"]) + transfer.bytes
                if transfer.time_s is None:
                    state["store_time_complete"] = False
                else:
                    state["store_time_observed"] = True
                    state["store_time_s"] = (
                        float(state["store_time_s"]) + transfer.time_s
                    )
                if transfer.max_time_s is not None:
                    state["store_critical_path_s"] = (
                        float(state["store_critical_path_s"]) + transfer.max_time_s
                    )
                if transfer.contributors == int(info["expected_contributors"]):
                    self._record_cpu_commit_events(
                        job_id=job_id,
                        state=state,
                        key_positions=info.get("key_positions", []),
                        completed_at_ns=transfer.completed_at_ns,
                    )
                    for key in info["keys"]:
                        self._preservation_known_stored_keys.add(key)
                        self._preservation_evicted_keys.discard(key)

        # Failed/direction-mismatched jobs were cleaned up above. Do not pass
        # their worker completion counts to the base scheduler, which assumes
        # every completed job still exists and would otherwise double-clean or
        # raise while indexing _jobs.
        meta.completed_jobs = {
            job_id: count
            for job_id, count in meta.completed_jobs.items()
            if job_id in self._jobs or job_id in self._reset_prefetch_releases
        }

        super().update_connector_output(connector_output)
        self._cleanup_prefetch_output(connector_output)
        self._drain_staging_protection_records()
        self._drain_chain_admission_evidence()
        self._emit_finished_preservation_events()

    def reset_cache(self) -> None:
        """Reset semantic and speculative state alongside the base cache."""
        self._score_transport_generation = getattr(self, '_score_transport_generation', 0)+1
        self._prefix_mean_score_batch = {}
        self._prefix_mean_scoring_request = {}
        self._prefix_mean_scoring_attempt = None
        self._prefix_mean_scoring_stats = {}
        getattr(self, '_score_source_tokens', {}).clear()
        getattr(self, '_score_source_deadlines', {}).clear()
        getattr(self, "_published_wait_prefixes", {}).clear()
        wait_states = list(getattr(self, "_wait_prefetch", {}).values())
        for wait_state in wait_states:
            if wait_state.status == "bound_ready":
                # The shared _prefetched entry below owns this reservation.
                continue
            job_id = wait_state.job_id
            if (
                wait_state.status == "in_flight"
                and job_id is not None
                and job_id in self._current_batch_prefetch_load_jobs
            ):
                self._current_batch_prefetch_load_jobs.pop(job_id, None)
                self._wait_prefetch_jobs.pop(job_id, None)
                if not wait_state.source_released:
                    self.manager.complete_load(wait_state.keys, wait_state.req_context)
                    wait_state.source_released = True
                self._free_wait_prefetch(wait_state, "reset_before_worker_dispatch")
            elif wait_state.status in {"in_flight", "retired_pending_fence"}:
                assert job_id is not None
                self._reset_prefetch_releases[job_id] = (
                    list(wait_state.gpu_block_ids),
                    wait_state.pending_contributors,
                )
                # Temporarily register only so the native reset path sends the
                # exact job ID through jobs_to_flush before reusing storage.
                self._jobs[job_id] = TransferJobStatus(
                    req_id=wait_state.req_context.req_id,
                    pending_count=wait_state.pending_contributors,
                    keys=set(wait_state.keys),
                    is_store=False,
                )
                wait_state.source_released = True
            else:
                self._free_wait_prefetch(wait_state, "reset")
        getattr(self, "_wait_prefetch", {}).clear()
        getattr(self, "_wait_prefetch_jobs", {}).clear()
        getattr(self, "_wait_hint_request_bindings", {}).clear()
        getattr(self, "_observed_prefixes", {}).clear()
        getattr(self, "_observed_prefix_generations", {}).clear()
        getattr(self, "_wait_hint_results", {}).clear()
        self._wait_hint_coalescing_started_ns = None
        for state in self._prefetched.values():
            job_status = self._jobs.get(state.job_id)
            if job_status is not None and job_status.pending_count > 0:
                self._reset_prefetch_releases[state.job_id] = (
                    state.gpu_block_ids,
                    job_status.pending_count,
                )
            else:
                self._free_prefetch_blocks(state.gpu_block_ids)
                self._prefetch_reserved_blocks -= len(state.gpu_block_ids)
        self._prefetched.clear()
        self._preempted_pending.clear()
        # Splices not yet sent to workers can be released immediately. Sent
        # splices remain quarantined in _pending_splice_releases until acked.
        for splice_id, (src_ids, _dst_ids) in self._pending_splice_jobs.items():
            pending = self._pending_splice_releases.pop(splice_id, None)
            block_ids = pending[0] if pending is not None else src_ids
            self._free_prefetch_blocks(block_ids)
            self._prefetch_reserved_blocks -= len(block_ids)
        for protection_id in set(
            getattr(self, "_pending_splice_protection_ids", {}).values()
        ):
            self.manager.release_staging_keys(protection_id)
        self._drain_staging_protection_records()
        getattr(self, "_pending_splice_protection_ids", {}).clear()
        self._pending_splice_jobs.clear()
        getattr(self, "_pending_splice_request_ids", {}).clear()
        self._current_batch_prefetch_load_jobs.clear()
        self._pending_job_keys.clear()
        self._pending_job_blocks.clear()
        getattr(self, "_preservation_requests", {}).clear()
        getattr(self, "_preservation_finished", {}).clear()
        getattr(self, "_preservation_jobs", {}).clear()
        getattr(self, "_preservation_job_events", {}).clear()
        getattr(self, "_preservation_known_stored_keys", set()).clear()
        getattr(self, "_preservation_evicted_keys", set()).clear()
        getattr(self, "_preservation_key_commits", {}).clear()
        getattr(self, "_preservation_key_evictions", {}).clear()
        getattr(self, "_chain_admission_global_accounting", {}).clear()
        self._preservation_event_sequence = 0
        lifecycle_params = getattr(self, "_lifecycle_request_params", None)
        if isinstance(lifecycle_params, dict):
            lifecycle_params.clear()
        pending_lifecycle = getattr(self, "_pending_lifecycle_releases", None)
        if isinstance(pending_lifecycle, dict):
            pending_lifecycle.clear()
        lifecycle_controller = getattr(self, "_lifecycle_controller", None)
        if lifecycle_controller is not None:
            lifecycle_controller.drain_ledger()
        self._worker_reset_pending = True
        super().reset_cache()
        for name in (
            "_relevance_observation_provenance",
            "_relevance_feature_snapshot_ns",
        ):
            getattr(self.manager, name, {}).clear()

    @profiled('scheduler.build_metadata')
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # _queue_preempted must run before _retry_pending_prefetches, both
        # before super().build_connector_meta(): the latter reads and
        # resets self._current_batch_load_jobs -- a job a retry adds this
        # step needs to already be in that dict when the base class
        # consumes it, not after. Needs scheduler_output itself (not just
        # its eventual effects) to know which requests were preempted THIS
        # step -- see on_request_preempted's docstring for why those must
        # be skipped here.
        if self._preservation_events_enabled:
            if self._lifecycle_release_timing == "scheduler_boundary":
                self._process_deferred_lifecycle_requests()
            for req_id in scheduler_output.preempted_req_ids or ():
                state = self._preservation_requests.get(req_id)
                if state is not None:
                    # This is the authoritative scheduler signal. It is
                    # recorded even when speculative prefetch is disabled.
                    state["preempted"] = True
                    state["preemption_count"] = int(state["preemption_count"]) + 1
                    if state.get("prefetch_t0_ns") is None:
                        state["prefetch_t0_ns"] = time.monotonic_ns()
                    self._record_block_pool_snapshot(
                        state,
                        "t0_preemption",
                        source="build_connector_meta_preempted_ids",
                    )
                    _reset_prompt_accounting_after_preemption(state)
        self._queue_preempted(scheduler_output)
        self._retry_pending_prefetches(scheduler_output)
        self._retry_wait_prefetches()
        if self._preservation_events_enabled:
            self._record_scheduler_execution(scheduler_output)
        base_meta = super().build_connector_meta(scheduler_output)
        assert isinstance(base_meta, OffloadingConnectorMetadata)
        if self._preservation_events_enabled:
            job_entries = (
                [
                    (job_id, entry, True, False)
                    for job_id, entry in base_meta.load_jobs.items()
                ]
                + [
                    (job_id, entry, False, False)
                    for job_id, entry in base_meta.store_jobs.items()
                ]
                + [
                    (job_id, entry, True, True)
                    for job_id, entry in self._current_batch_prefetch_load_jobs.items()
                ]
            )
            for job_id, entry, is_load, is_prefetch in job_entries:
                if job_id in getattr(self, "_wait_prefetch_jobs", {}):
                    continue
                status = self._jobs.get(job_id)
                state = self._preservation_requests.get(
                    entry.req_id
                ) or self._preservation_finished.get(entry.req_id)
                raw_keys = list(status.keys) if status is not None else []
                key_positions = (
                    (
                        self._position_records_for_keys(state, raw_keys)
                        if is_load else self._store_source_key_positions(
                            entry.req_id, raw_keys,
                            scheduler_output.num_scheduled_tokens.get(entry.req_id, 0),
                        )
                    )
                    if state is not None
                    else []
                )
                ordered_keys = [record["key"] for record in key_positions]
                order_complete = len(ordered_keys) == len(raw_keys)
                self._preservation_jobs[job_id] = {
                    "req_id": entry.req_id,
                    "is_load": is_load,
                    "keys": ordered_keys if order_complete else raw_keys,
                    "key_positions": key_positions,
                    "key_order_complete": order_complete,
                    "chunks": len(raw_keys),
                    "admission_index": (
                        max(0, int(state["admission_count"]) - 1)
                        if state is not None and is_load
                        else None
                    ),
                    "expected_contributors": (
                        int(self.config.num_workers)
                        if is_load
                        else (
                            1
                            if self._replicated_layout
                            else int(self.config.num_workers)
                        )
                    ),
                    "is_prefetch": is_prefetch,
                    "dst_block_ids": _physical_block_refs(entry.dst_spec),
                    "src_block_ids": _physical_block_refs(entry.src_spec),
                    "dst_block_ids_complete": _physical_block_refs_complete(
                        entry.dst_spec
                    ),
                    "src_block_ids_complete": _physical_block_refs_complete(
                        entry.src_spec
                    ),
                }
                if state is not None and not is_prefetch:
                    count_key = "load_job_count" if is_load else "store_job_count"
                    state[count_key] = int(state[count_key]) + 1
                    expected_key = (
                        "load_expected_total" if is_load else "store_expected_total"
                    )
                    state[expected_key] = int(state[expected_key]) + int(
                        self._preservation_jobs[job_id]["expected_contributors"]
                    )
                elif state is not None:
                    state["prefetch_submitted_jobs"] = (
                        int(state.get("prefetch_submitted_jobs", 0)) + 1
                    )
                    state["prefetch_submitted_blocks"] = int(
                        state.get("prefetch_submitted_blocks", 0)
                    ) + int(self._preservation_jobs[job_id]["chunks"])
        job_keys = self._pending_job_keys
        self._pending_job_keys = {}
        job_blocks = self._pending_job_blocks
        self._pending_job_blocks = {}
        splice_jobs = self._pending_splice_jobs
        self._pending_splice_jobs = {}
        prefetch_load_jobs = self._current_batch_prefetch_load_jobs
        self._current_batch_prefetch_load_jobs = {}
        evicted_keys: list[OffloadKey] = []
        if hasattr(self.manager, "drain_evicted_keys"):
            evicted_keys = self.manager.drain_evicted_keys()
        for name in (
            "_relevance_observation_provenance",
            "_relevance_feature_snapshot_ns",
        ):
            values_by_method = getattr(self.manager, name, {})
            for values in values_by_method.values():
                for key in evicted_keys:
                    values.pop(key, None)
        self._record_cpu_eviction_events(evicted_keys)
        if getattr(self, "_preservation_audit_enabled", True):
            for key in evicted_keys:
                self._preservation_evicted_keys.add(key)
                self._preservation_known_stored_keys.discard(key)
        if getattr(self, "_decision_feature_trace_enabled", False) and hasattr(
            self.manager, "drain_decision_feature_records"
        ):
            for record in self.manager.drain_decision_feature_records():
                record_decision_feature(record)
        if hasattr(self.manager, "drain_admission_records"):
            admission_records = self.manager.drain_admission_records()
            for record in admission_records:
                request_id = record.get("request_id")
                state = self._preservation_requests.get(
                    request_id
                ) or self._preservation_finished.get(request_id)
                if state is not None:
                    state["admission_decisions"].append(record)
                    accounting = state["admission_decision_accounting"]
                    if record.get("action") == "reject":
                        accounting["rejected"] += 1
                    else:
                        accounting["admitted"] += 1
                    if (
                        record.get("reason")
                        == "compulsory_fallback_unavailable_features"
                    ):
                        accounting["compulsory_fallbacks"] += 1
            if hasattr(self.manager, "drain_admission_accounting"):
                aggregate = self.manager.drain_admission_accounting()
                observed = {
                    "admitted": sum(
                        record.get("action") != "reject" for record in admission_records
                    ),
                    "rejected": sum(
                        record.get("action") == "reject" for record in admission_records
                    ),
                    "compulsory_fallbacks": sum(
                        record.get("reason")
                        == "compulsory_fallback_unavailable_features"
                        for record in admission_records
                    ),
                }
                if aggregate != observed:
                    raise RuntimeError("admission record accounting diverged")
        self._drain_chain_admission_evidence()
        reset_semantic_state = self._worker_reset_pending
        self._worker_reset_pending = False
        meta = SemanticOffloadingConnectorMetadata(
            scoring_context=self._score_context(),
            scoring_requests=dict(getattr(self, "_decision_scoring_requests", {})),
            prefix_scoring_request=dict(getattr(self, "_prefix_mean_scoring_request", {})),
            scoring_lease_ids={r:tuple(eid for eid,state in self._wait_prefetch.items()
                if state.status=='pending' and state.source.source_request_id==r)
                for r in getattr(self,'_decision_scoring_requests',{})},
            scoring_residency_versions={r:{k:getattr(self.manager._policy.get(k),'_semantic_score_version',None)
                for k in keys} for r,keys in getattr(self,'_decision_scoring_requests',{}).items()},
            load_jobs=base_meta.load_jobs,
            store_jobs=base_meta.store_jobs,
            jobs_to_flush=base_meta.jobs_to_flush,
            store_job_keys=job_keys,
            store_job_blocks=job_blocks,
            splice_jobs=splice_jobs,
            prefetch_load_jobs=prefetch_load_jobs,
            evicted_keys=evicted_keys,
            preempted_req_ids=tuple(sorted(scheduler_output.preempted_req_ids or ())),
            reset_semantic_state=reset_semantic_state,
            execution_attribution=[
                {
                    "request_id": str(req_id),
                    "external_request_id": state.get("external_request_id"),
                    "scheduler_batch_id": int(
                        getattr(self, "_preservation_scheduler_batch_id", 1)
                    )
                    - 1,
                    "scheduled_tokens": int(scheduled_tokens),
                    "prompt_tokens": int(
                        next(
                            (
                                episode.get("prompt_tokens_locally_executed", 0)
                                for episode in reversed(
                                    state.get("admission_episodes", [])
                                )
                                if episode.get("scheduler_batch_id")
                                == int(
                                    getattr(
                                        self,
                                        "_preservation_scheduler_batch_id",
                                        1,
                                    )
                                )
                                - 1
                            ),
                            0,
                        )
                    ),
                }
                for req_id, scheduled_tokens in (
                    getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
                ).items()
                if (state := self._preservation_requests.get(req_id)) is not None
            ],
        )
        self._drain_staging_protection_records()
        self._emit_finished_preservation_events()
        getattr(self, "_decision_scoring_requests", {}).clear()
        self._prefix_mean_scoring_request = {}
        return meta


class SemanticOffloadingConnectorWorker(OffloadingConnectorWorker):
    """issues log entry #27 (Fix A): submits prefetch load jobs for real
    execution via the same `self.worker.submit_load()` the stock class uses
    for normal loads, but WITHOUT registering them in `self._load_jobs`.
    That dict is what drives `get_finished()`'s `finished_recving`
    reporting -- which the core scheduler only knows how to interpret for a
    request that's `WAITING_FOR_REMOTE_KVS` or already finished, neither of
    which a preempted prefetch's request is. Their completion is still
    reported normally through `completed_jobs` (`mark_completed()` in
    `get_finished()` is unconditional, independent of `_load_jobs`), which
    is all `SemanticOffloadingConnectorScheduler`'s own bookkeeping
    (`_jobs[job_id].pending_count`, `_try_splice_prefetch`) needs."""

    def _mark_failed_job(self, job_id: int) -> None:
        failed_jobs = getattr(self._connector_worker_meta, "failed_jobs", None)
        if failed_jobs is None:
            failed_jobs = {}
            self._connector_worker_meta.failed_jobs = failed_jobs
        failed_jobs[job_id] = failed_jobs.get(job_id, 0) + 1

    def _mark_submission_failed_receive(self, req_id: str) -> None:
        pending = getattr(self, "_submission_failed_recving", None)
        if pending is None:
            pending = self._submission_failed_recving = set()
        pending.add(req_id)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata) -> None:
        preserve = (
            self.worker is not None
            and getattr(self.worker, "_preservation_events", False)
            and hasattr(self.worker, "register_transfer_direction")
        )
        if not preserve:
            super().start_kv_transfers(metadata)
        else:
            assert self.worker is not None
            for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
                self.worker.register_transfer_direction(job_id, is_load=False)
                success = self.worker.submit_store(job_id, src_spec, dst_spec)
                if not success:
                    self.worker.record_transfer_failure(job_id, is_load=False)
                    self._mark_failed_job(job_id)
            self._unsubmitted_store_jobs.clear()
            for job_id, entry in metadata.load_jobs.items():
                self._load_jobs[job_id] = entry.req_id
                assert isinstance(entry.dst_spec, GPULoadStoreSpec)
                self.worker.register_transfer_direction(
                    job_id,
                    is_load=True,
                    destination_block_ids=entry.dst_spec.block_ids,
                )
                success = self.worker.submit_load(
                    job_id, entry.src_spec, entry.dst_spec
                )
                if not success:
                    self.worker.record_transfer_failure(job_id, is_load=True)
                    self._mark_failed_job(job_id)
                    self._load_jobs.pop(job_id, None)
                    self._mark_submission_failed_receive(entry.req_id)
        prefetch_load_jobs = getattr(metadata, "prefetch_load_jobs", None) or {}
        for job_id, entry in prefetch_load_jobs.items():
            assert self.worker is not None
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            tracked_prefetch_jobs = getattr(
                self, "_semantic_prefetch_load_job_ids", None
            )
            if tracked_prefetch_jobs is None:
                tracked_prefetch_jobs = self._semantic_prefetch_load_job_ids = set()
            tracked_prefetch_jobs.add(job_id)
            if getattr(self.worker, "_preservation_events", False) and hasattr(
                self.worker, "register_transfer_direction"
            ):
                self.worker.register_transfer_direction(
                    job_id,
                    is_load=True,
                    destination_block_ids=entry.dst_spec.block_ids,
                )
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            if not success:
                self.worker.record_transfer_failure(job_id, is_load=True)
                self._mark_failed_job(job_id)

    def get_finished(self, finished_req_ids: set[str]):
        if self.worker is not None and hasattr(self.worker, "retire_async_requests"):
            self.worker.retire_async_requests(finished_req_ids)
        if not (
            self.worker is not None
            and getattr(self.worker, "_preservation_events", False)
        ):
            return super().get_finished(finished_req_ids)
        finished_recving: set[str] = set()
        finished_recving.update(
            getattr(self, "_submission_failed_recving", set()).copy()
        )
        getattr(self, "_submission_failed_recving", set()).clear()
        for transfer_result in self.worker.get_finished():
            job_id = transfer_result.job_id
            prefetch_load_jobs = getattr(self, "_semantic_prefetch_load_job_ids", set())
            is_prefetch_load = job_id in prefetch_load_jobs
            is_load = job_id in self._load_jobs or is_prefetch_load
            if (
                transfer_result.success
                and transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                stats = (
                    self._connector_worker_meta.transfer_stats.load
                    if is_load
                    else self._connector_worker_meta.transfer_stats.store
                )
                stats.record(
                    transfer_result.transfer_size,
                    transfer_result.transfer_time,
                )
            if transfer_result.success:
                self._connector_worker_meta.mark_completed(job_id)
            else:
                self._mark_failed_job(job_id)
            if is_prefetch_load:
                prefetch_load_jobs.discard(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
        return set(), finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.worker is None or not getattr(
            self.worker, "_preservation_events", False
        ):
            parent_method = getattr(super(), "get_block_ids_with_load_errors", None)
            return parent_method() if parent_method is not None else set()
        return self.worker.pop_failed_load_block_ids()

    def handle_preemptions(self, metadata: OffloadingConnectorMetadata):
        preserve = self.worker is not None and getattr(
            self.worker, "_preservation_events", False
        )
        if not preserve:
            return super().handle_preemptions(metadata)
        assert self.worker is not None
        if metadata.jobs_to_flush:
            for job_id in metadata.jobs_to_flush:
                entry = metadata.store_jobs.pop(job_id, None)
                if entry is None:
                    continue
                if not self._is_store_writer:
                    self._connector_worker_meta.mark_completed(job_id)
                    continue
                assert isinstance(entry.src_spec, GPULoadStoreSpec)
                self._unsubmitted_store_jobs.append(
                    (job_id, entry.src_spec, entry.dst_spec)
                )
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            self.worker.register_transfer_direction(job_id, is_load=False)
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            if not success:
                self.worker.record_transfer_failure(job_id, is_load=False)
                self._mark_failed_job(job_id)
        self._unsubmitted_store_jobs.clear()
        if metadata.jobs_to_flush:
            self.worker.wait(metadata.jobs_to_flush)

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        if (
            self.worker is not None
            and self._is_store_writer
            and getattr(self.worker, "_preservation_events", False)
            and hasattr(self.worker, "register_transfer_direction")
        ):
            for job_id in metadata.store_jobs:
                self.worker.register_transfer_direction(job_id, is_load=False)
        return super().prepare_store_kv(metadata)


class SemanticOffloadingConnector(OffloadingConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        parallel_config = vllm_config.parallel_config
        if parallel_config.pipeline_parallel_size != 1:
            raise ValueError(
                "Semantic offloading currently requires pipeline_parallel_size=1; "
                "pipeline stages do not yet share one global probe-layer identity"
            )
        if getattr(parallel_config, "prefill_context_parallel_size", 1) != 1:
            raise ValueError(
                "Semantic offloading currently requires prefill_context_parallel_size=1"
            )
        if getattr(parallel_config, "decode_context_parallel_size", 1) != 1:
            raise ValueError(
                "Semantic offloading currently requires decode_context_parallel_size=1"
            )
        if (
            getattr(parallel_config, "world_size", parallel_config.tensor_parallel_size)
            != parallel_config.tensor_parallel_size
        ):
            raise ValueError(
                "Semantic offloading worker aggregation requires world_size to "
                "equal tensor_parallel_size"
            )
        if bool(getattr(vllm_config.model_config, "use_mla", False)):
            raise ValueError("Semantic offloading does not yet support MLA KV caches")
        self._score_group_size = parallel_config.tensor_parallel_size
        # Deliberately skip OffloadingConnector.__init__ (calling
        # KVConnectorBase_V1.__init__ directly instead): the base __init__
        # would construct its own OffloadingConnectorScheduler around a spec
        # instance we'd then discard in favor of our own scheduler subclass.
        # OffloadingConnectorScheduler.__init__ eagerly calls
        # spec.get_manager(), so doing that twice means constructing (and
        # throwing away) a whole extra SemanticOffloadingManager -- wasteful,
        # and avoidable by just not calling the redundant base constructor.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        # create_spec()'s real signature varies by vLLM checkout (#48150
        # changed it on some, not others still in use across this
        # project's machines) -- see _vllm_compat.py, which inspects the
        # real installed vLLM rather than assuming one shape.
        spec = create_offloading_spec(vllm_config, kv_cache_config)
        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = SemanticOffloadingConnectorScheduler(
                spec, vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = construct_worker(
                SemanticOffloadingConnectorWorker, spec, kv_cache_config
            )
        raw_bounded_timing = getattr(spec, "extra_config", {}).get(
            "bounded_prefill_cuda_timing", False
        )
        if not isinstance(raw_bounded_timing, bool):
            raise TypeError("bounded_prefill_cuda_timing must be a JSON boolean")
        self._bounded_prefill_cuda_timing = raw_bounded_timing
        if raw_bounded_timing and parallel_config.tensor_parallel_size != 1:
            raise ValueError("bounded_prefill_cuda_timing currently requires TP=1")

    def _install_bounded_prefill_cuda_timer(self) -> None:
        """Install an opt-in, transfer-aware CUDA timer on the existing hook.

        This deliberately synchronizes the end event and is therefore for a
        separate bounded attribution run, not serving-performance cells. A
        record is attributable to request prefill only when the batch is
        singleton and all scheduled tokens are prompt tokens. Transfer and
        splice categories are retained separately: the CUDA interval is the
        model-runner current-stream elapsed time after transfer submission,
        not a pure-kernel interval and not additive with overlapping DMA.
        """
        if not getattr(self, "_bounded_prefill_cuda_timing", False):
            return
        assert self.connector_worker is not None
        worker = self.connector_worker.worker
        if worker is None or getattr(worker, "_bounded_prefill_timer_installed", False):
            return
        handle = getattr(worker, "_query_capture_mode", None)
        if handle is None:
            raise RuntimeError(
                "bounded_prefill_cuda_timing requires the existing model-runner hook"
            )
        original_on_complete = getattr(handle, "_on_complete", None)

        def on_complete(model_runner: object, model_runner_output: object) -> None:
            pending = getattr(worker, "_bounded_prefill_inflight_cuda_timing", None)
            if pending is not None:
                worker._bounded_prefill_inflight_cuda_timing = None
                end_event = pending["end_event"]
                end_event.record()
                end_event.synchronize()
                completed_at_ns = time.monotonic_ns()
                elapsed_s = (
                    float(pending["start_event"].elapsed_time(end_event)) / 1000.0
                )
                attribution = pending["execution_attribution"]
                request_ids = [item["request_id"] for item in attribution]
                singleton = len(request_ids) == 1
                prompt_tokens = int(attribution[0]["prompt_tokens"]) if singleton else 0
                scheduled_tokens = (
                    int(attribution[0]["scheduled_tokens"]) if singleton else 0
                )
                no_transfer = not pending["transfer_categories"]
                eligible = bool(
                    singleton
                    and prompt_tokens > 0
                    and prompt_tokens == scheduled_tokens
                )
                if not singleton:
                    eligibility_reason = "non_singleton_or_unmatched_batch"
                elif prompt_tokens <= 0:
                    eligibility_reason = "decode_only_batch"
                elif prompt_tokens != scheduled_tokens:
                    eligibility_reason = "mixed_prompt_decode_batch"
                elif not no_transfer:
                    eligibility_reason = "eligible_singleton_prompt_batch_with_transfer"
                else:
                    eligibility_reason = "eligible_singleton_prompt_batch"
                event = {
                    "schema": "semantic-bounded-prefill-cuda-v3",
                    "execution_sequence": pending["execution_sequence"],
                    "request_ids": request_ids,
                    "execution_attribution": attribution,
                    "started_at_ns": pending["started_at_ns"],
                    "completed_at_ns": completed_at_ns,
                    "host_clock_domain": "host_boot_monotonic_ns",
                    "device_time_s": elapsed_s,
                    "device_time_source": (
                        "cuda_event_current_model_runner_stream_elapsed"
                    ),
                    "transfer_free": no_transfer,
                    "singleton_request": singleton,
                    "all_scheduled_tokens_are_prompt": bool(
                        singleton
                        and prompt_tokens == scheduled_tokens
                        and prompt_tokens > 0
                    ),
                    "attribution_eligible": eligible,
                    "attribution_reason": eligibility_reason,
                    "transfer_categories": pending["transfer_categories"],
                    "tensor_parallel_size": 1,
                    "synchronization_overhead": (
                        "end_event_synchronize; attribution_cells_only"
                    ),
                }
                events = getattr(worker, "_bounded_prefill_cuda_events", None)
                if events is None:
                    events = worker._bounded_prefill_cuda_events = []
                events.append(event)
            if original_on_complete is not None:
                original_on_complete(model_runner, model_runner_output)

        handle._on_complete = on_complete
        worker._bounded_prefill_timer_installed = True

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)
        self._install_bounded_prefill_cuda_timer()

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend):
        super().register_cross_layers_kv_cache(kv_cache, attn_backend)
        self._install_bounded_prefill_cuda_timer()

    def start_load_kv(self, forward_context, **kwargs) -> None:
        attribution = []
        transfer_categories = []
        if (
            self.connector_worker is not None
            and self.connector_worker.worker is not None
        ):
            worker = self.connector_worker.worker
            metadata = self._connector_metadata
            if isinstance(metadata, SemanticOffloadingConnectorMetadata):
                attribution = list(metadata.execution_attribution)
                worker._bounded_prefill_execution_attribution = attribution
                transfer_categories = [
                    name
                    for name, active in (
                        ("demand_load", metadata.load_jobs),
                        ("speculative_load", metadata.prefetch_load_jobs),
                        ("store", metadata.store_jobs),
                        (
                            "deferred_store",
                            self.connector_worker._unsubmitted_store_jobs,
                        ),
                        ("store_flush", metadata.jobs_to_flush),
                        ("gpu_splice", metadata.splice_jobs),
                    )
                    if bool(active)
                ]
                worker._bounded_prefill_transfer_categories = transfer_categories
                worker._bounded_prefill_no_transfer = not transfer_categories
        super().start_load_kv(forward_context, **kwargs)
        authoritative_batch = bool(
            attribution
            and all(
                isinstance(item, dict)
                and isinstance(item.get("request_id"), str)
                and item.get("request_id")
                and isinstance(item.get("external_request_id"), str)
                and item.get("external_request_id")
                and isinstance(item.get("scheduler_batch_id"), int)
                and not isinstance(item.get("scheduler_batch_id"), bool)
                and item.get("scheduler_batch_id") >= 0
                and isinstance(item.get("scheduled_tokens"), int)
                and not isinstance(item.get("scheduled_tokens"), bool)
                and item.get("scheduled_tokens") >= 0
                and isinstance(item.get("prompt_tokens"), int)
                and not isinstance(item.get("prompt_tokens"), bool)
                and 0 <= item.get("prompt_tokens") <= item.get("scheduled_tokens")
                for item in attribution
            )
            and len({item["request_id"] for item in attribution}) == len(attribution)
            and len({item["scheduler_batch_id"] for item in attribution}) == 1
        )
        if (
            getattr(self, "_bounded_prefill_cuda_timing", False)
            and authoritative_batch
            and self.connector_worker is not None
            and self.connector_worker.worker is not None
            and torch.cuda.is_available()
        ):
            worker = self.connector_worker.worker
            if getattr(worker, "_bounded_prefill_inflight_cuda_timing", None):
                raise RuntimeError("bounded prefill CUDA timing batch overlapped")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            started_at_ns = time.monotonic_ns()
            start_event.record()
            execution_sequence = (
                int(getattr(worker, "_bounded_prefill_execution_sequence", 0)) + 1
            )
            worker._bounded_prefill_execution_sequence = execution_sequence
            worker._bounded_prefill_inflight_cuda_timing = {
                "start_event": start_event,
                "end_event": end_event,
                "started_at_ns": started_at_ns,
                "execution_attribution": attribution,
                "transfer_categories": list(transfer_categories),
                "execution_sequence": execution_sequence,
            }

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        if isinstance(self.connector_scheduler, SemanticOffloadingConnectorScheduler):
            self.connector_scheduler.bind_gpu_block_pool(gpu_block_pool)

    def publish_wait_hint(self, hint: dict[str, object]) -> dict[str, object]:
        if not isinstance(
            self.connector_scheduler, SemanticOffloadingConnectorScheduler
        ):
            return {
                "status": "unsupported",
                "reason": "wait_hints_require_scheduler_role",
            }
        return self.connector_scheduler.publish_wait_hint(hint)

    def on_request_preempted(self, request: Request) -> None:
        # The core scheduler calls this on the top-level connector (this
        # class), not on connector_scheduler directly -- same delegation
        # requirement as bind_gpu_block_pool above. Missing this the first
        # time round meant on_request_preempted silently no-op'd via the
        # inherited KVConnectorBase_V1 default despite real preemptions
        # happening -- caught by seeing zero debug output against a nonzero
        # vllm:num_preemptions_total on the real server, not by reasoning
        # about the code.
        if isinstance(self.connector_scheduler, SemanticOffloadingConnectorScheduler):
            self.connector_scheduler.on_request_preempted(request)

    @profiled('metadata.bind_worker')
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        if self.connector_worker is not None and isinstance(
            connector_metadata, SemanticOffloadingConnectorMetadata
        ):
            worker = self.connector_worker.worker
            if hasattr(worker, "receive_preempted_request_ids"):
                worker.receive_preempted_request_ids(
                    connector_metadata.preempted_req_ids
                )
            if connector_metadata.reset_semantic_state and hasattr(
                worker, "reset_semantic_state"
            ):
                worker.reset_semantic_state()
                worker._bounded_prefill_cuda_events = []
                worker._bounded_prefill_pending_request_ids = []
                worker._bounded_prefill_inflight_cuda_timing = None
            if hasattr(worker, "receive_job_blocks"):
                worker.receive_job_blocks(connector_metadata.store_job_blocks)
            if hasattr(worker, "receive_job_keys"):
                worker.receive_job_keys(connector_metadata.store_job_keys)
            if hasattr(worker, "receive_evicted_keys"):
                worker.receive_evicted_keys(connector_metadata.evicted_keys)
            if hasattr(worker, 'receive_scoring_context'):
                worker.receive_scoring_context(connector_metadata.scoring_context)
            if hasattr(worker, 'receive_prefix_score_requests'):
                worker.receive_prefix_score_requests(connector_metadata.prefix_scoring_request)
            if hasattr(worker, "receive_score_requests"):
                worker.receive_score_requests(connector_metadata.scoring_requests, connector_metadata.scoring_residency_versions, connector_metadata.scoring_lease_ids)
            if hasattr(worker, "splice_gpu_blocks"):
                # Runs before super().bind_connector_metadata() below (which
                # hands off the load_jobs that would otherwise redundantly
                # reload the same keys -- but a spliced request has none,
                # since _try_splice_prefetch substitutes for the normal
                # load path entirely rather than running alongside it).
                for splice_id, (
                    src_ids,
                    dst_ids,
                ) in connector_metadata.splice_jobs.items():
                    worker.splice_gpu_blocks(src_ids, dst_ids, splice_id=splice_id)
        super().bind_connector_metadata(connector_metadata)

    @profiled('metadata.build_worker')
    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        base_meta = super().build_connector_worker_meta()
        assert self.connector_worker is not None
        worker = self.connector_worker.worker
        scores: RelevanceScores = {}
        score_head_counts: ScoreHeadCounts = {}
        score_reduction = None
        if hasattr(worker, "pop_pending_score_metadata"):
            scores, score_head_counts, score_reduction = (
                worker.pop_pending_score_metadata()
            )
        else:
            raise TypeError("Semantic worker does not provide versioned score metadata")
        # Synchronous opportunity scoring moves each request into an ordered
        # observation, leaving empty per-method accumulators behind. Empty
        # method containers carry no legacy request metadata. Canonicalize
        # them here; retain every actual request (even an empty ranked list),
        # so genuinely mixed representations still fail strict validation.
        scores = {method: reqs for method, reqs in scores.items() if reqs}
        score_head_counts = {
            method: {req_id: count for req_id, count in reqs.items()
                     if req_id in scores.get(method, {})}
            for method, reqs in score_head_counts.items() if method in scores
        }
        relevance_updates: ComposedRelevanceUpdates = {}
        latest_relevance: LatestRelevance = {}
        rank_ema_mode = getattr(worker, "_rank_ema_mode", "default")
        ema_rank_weight_floor = getattr(worker, "_ema_rank_weight_floor", 0.0)
        if self._score_group_size == 1 and _score_request_count(scores) > 1:
            relevance_updates = compose_relevance_updates(
                scores, rank_ema_mode, ema_rank_weight_floor
            )
            latest_relevance = compose_latest_relevance(scores)
            scores = {}
            score_head_counts = {}
        score_worker_counts: ScoreWorkerCounts = {
            method: {req_id: 1 for req_id in reqs} for method, reqs in scores.items()
        }
        completed_splices = {}
        if hasattr(worker, "pop_completed_splices"):
            completed_splices = {
                splice_id: 1 for splice_id in worker.pop_completed_splices()
            }
        completed_splice_evidence_raw = (
            worker.pop_completed_splice_evidence()
            if hasattr(worker, "pop_completed_splice_evidence")
            else {}
        )
        worker_rank = getattr(
            getattr(getattr(self, "spec", None), "config", None), "parallel", None
        )
        worker_rank = getattr(worker_rank, "rank", None)
        contributor_ids = (str(worker_rank),) if worker_rank is not None else ()
        completed_splice_evidence = {
            splice_id: [
                {
                    **dict(event),
                    "contributor_id": (contributor_ids[0] if contributor_ids else None),
                }
            ]
            for splice_id, event in completed_splice_evidence_raw.items()
        }
        transfer_events: PreservationTransferEvents = {}
        if hasattr(worker, "pop_transfer_events"):
            transfer_events = {
                # The wrapper's parallel rank is the stable contributor
                # identity available at this process boundary.  It is
                # carried with the job event so the scheduler can reject
                # incomplete or duplicate worker acknowledgements.
                job_id: SemanticTransferCompletion(
                    is_load=bool(event.get("is_load")),
                    bytes=event.get("bytes"),
                    time_s=event.get("time_s"),
                    max_time_s=event.get("max_time_s", event.get("time_s")),
                    contributors=int(event.get("contributors", 1)),
                    success=bool(event.get("success", False)),
                    status=str(event.get("status", "failure")),
                    cancelled=bool(event.get("cancelled", False)),
                    start_time_s=event.get("start_time_s"),
                    completion_time_s=event.get("completion_time_s"),
                    started_at_ns=event.get("started_at_ns"),
                    completed_at_ns=event.get("completed_at_ns"),
                    contributor_ids=contributor_ids,
                )
                for job_id, event in worker.pop_transfer_events().items()
            }
        first_forward_times_ns = {}
        if hasattr(worker, "pop_first_forward_times"):
            first_forward_times_ns = worker.pop_first_forward_times()
        preempted_forward_times_ns = {}
        if hasattr(worker, "pop_preempted_forward_times"):
            preempted_forward_times_ns = worker.pop_preempted_forward_times()
        prefill_cuda_events = list(
            getattr(worker, "_bounded_prefill_cuda_events", []) or []
        )
        worker._bounded_prefill_cuda_events = []
        allocator_peak_stats = dict(getattr(worker, "_allocator_peak_stats", {}))
        async_score_observations = (
            worker.pop_async_score_observations()
            if hasattr(worker, "pop_async_score_observations")
            else []
        )
        async_scoring_events = (
            worker.pop_async_scoring_telemetry()
            if hasattr(worker, "pop_async_scoring_telemetry")
            else []
        )
        prefix_score_observations = worker.pop_prefix_score_observations() if hasattr(worker, 'pop_prefix_score_observations') else []
        async_controller = getattr(worker, "_async_scoring", None)
        async_scoring_status = (
            {
                "schema": "semantic-async-scoring-status-v1",
                **async_controller.stats(),
                "max_snapshot_age_ns": getattr(
                    worker, "_async_scoring_max_snapshot_age_ns", None
                ),
            }
            if async_controller is not None
            else {}
        )
        base_meta = base_meta or OffloadingWorkerMetadata()
        assert isinstance(base_meta, OffloadingWorkerMetadata)
        return SemanticWorkerMetadata(
            completed_jobs=base_meta.completed_jobs,
            transfer_stats=base_meta.transfer_stats,
            pending_scores=scores,
            failed_jobs=dict(getattr(base_meta, "failed_jobs", {}) or {}),
            relevance_updates=relevance_updates,
            latest_relevance=latest_relevance,
            score_head_counts=score_head_counts,
            score_worker_counts=score_worker_counts,
            score_reduction=score_reduction,
            score_group_size=self._score_group_size,
            score_contributors=1,
            rank_ema_mode=rank_ema_mode,
            ema_rank_weight_floor=ema_rank_weight_floor,
            completed_splices=completed_splices,
            completed_splice_evidence=completed_splice_evidence,
            transfer_events=transfer_events,
            first_forward_times_ns=first_forward_times_ns,
            preempted_forward_times_ns=preempted_forward_times_ns,
            allocator_peak_stats=allocator_peak_stats,
            async_scoring_events=async_scoring_events,
            async_scoring_status=async_scoring_status,
            async_score_observations=async_score_observations,
            prefix_score_observations=prefix_score_observations,
            prefill_cuda_events=prefill_cuda_events,
        )

    @profiled('scheduler.consume_worker')
    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        meta = connector_output.kv_connector_worker_meta
        updates: ComposedRelevanceUpdates = {}
        latest_relevance: LatestRelevance = {}
        scores: RelevanceScores = {}
        async_observations: list[dict[str, object]] = []
        rank_ema_mode = "default"
        ema_rank_weight_floor = 0.0
        if isinstance(meta, SemanticWorkerMetadata):
            scheduler = self.connector_scheduler

            def bind_async_event(event: dict[str, object]) -> dict[str, object]:
                enriched = dict(event)
                request_bindings = []
                for request_id in event.get("request_ids", []):
                    state = None
                    if scheduler is not None:
                        state = scheduler._preservation_requests.get(
                            request_id
                        ) or scheduler._preservation_finished.get(request_id)
                    request_bindings.append(
                        {
                            "request_id": request_id,
                            "run_id": state.get("run_id") if state else None,
                            "cell_id": state.get("cell_id") if state else None,
                            "session_id": (
                                state.get("policy_session_id") if state else None
                            ),
                            "generation": state.get("generation") if state else None,
                        }
                    )
                enriched["request_bindings"] = request_bindings
                manager = scheduler.manager if scheduler is not None else None
                enriched["namespace"] = dict(
                    getattr(manager, "_admission_namespace", {})
                )
                return enriched

            for event in meta.async_scoring_events:
                record_async_scoring_event(bind_async_event(event))
            if meta.async_scoring_status:
                # Keep status in native metadata, but log transitions only.
                # Ordinary idle polls otherwise encode/flush the same JSON.
                previous = getattr(self, "_last_logged_async_scoring_status", None)
                if previous != meta.async_scoring_status:
                    record_async_scoring_event(
                        {"event": "status", **meta.async_scoring_status}
                    )
                    self._last_logged_async_scoring_status = dict(
                        meta.async_scoring_status
                    )
            rank_ema_mode = getattr(meta, "rank_ema_mode", "default")
            ema_rank_weight_floor = getattr(meta, "ema_rank_weight_floor", 0.0)
            async_observations = meta.complete_async_score_observations()
            updates = meta.complete_relevance_updates()
            latest_relevance = meta.complete_latest_relevance()
            if not updates:
                scores = meta.complete_scores()
                if _score_request_count(scores) > 1:
                    updates = compose_relevance_updates(
                        scores, rank_ema_mode, ema_rank_weight_floor
                    )
                    latest_relevance = compose_latest_relevance(scores)
                    scores = {}
        super().update_connector_output(connector_output)
        if self.connector_scheduler is None:
            return
        if hasattr(self.connector_scheduler, 'receive_prefix_score_observations'):
            self.connector_scheduler.receive_prefix_score_observations(
                meta.complete_prefix_score_observations() if isinstance(meta, SemanticWorkerMetadata) else [])
        manager = self.connector_scheduler.manager

        def record_observation_availability(
            method_updates: Mapping[
                str, Mapping[OffloadKey, tuple[float, float, float]]
            ],
            available_ns: int,
            provenance: str,
            feature_snapshot_ns: int | None = None,
        ) -> None:
            updated_by_method = getattr(manager, "_relevance_updated_ns", {})
            provenance_by_method = getattr(
                manager, "_relevance_observation_provenance", None
            )
            if provenance_by_method is None:
                provenance_by_method = manager._relevance_observation_provenance = {}
            snapshots_by_method = getattr(
                manager, "_relevance_feature_snapshot_ns", None
            )
            if snapshots_by_method is None:
                snapshots_by_method = manager._relevance_feature_snapshot_ns = {}
            for method, keyed_updates in method_updates.items():
                updated = updated_by_method.get(method, {})
                recorded = provenance_by_method.setdefault(method, {})
                snapshots = snapshots_by_method.setdefault(method, {})
                for key in keyed_updates:
                    if key not in updated:
                        continue
                    updated[key] = available_ns
                    recorded[key] = provenance
                    if feature_snapshot_ns is not None:
                        snapshots[key] = feature_snapshot_ns

        applied_any = False
        for observation in async_observations:
            decision_ns = time.monotonic_ns()
            expires_at_ns = observation.get("expires_at_ns")
            observation_available_ns = observation.get("observation_available_ns")
            snapshot_captured_ns = observation.get("snapshot_captured_ns")
            if (
                not isinstance(observation_available_ns, int)
                or isinstance(observation_available_ns, bool)
                or not 0 <= observation_available_ns <= decision_ns
            ):
                record_async_scoring_event(
                    bind_async_event(
                        {
                            "schema": "semantic-async-scoring-event-v1",
                            "event": "consumption_rejected",
                            "observation_sequence": observation.get(
                                "observation_sequence"
                            ),
                            "request_ids": observation.get("request_ids", []),
                            "reason": "observation_clock_invalid",
                            "observation_available_ns": (observation_available_ns),
                            "consumer_decision_ns": decision_ns,
                        }
                    )
                )
                continue
            if snapshot_captured_ns is not None and (
                not isinstance(snapshot_captured_ns, int)
                or isinstance(snapshot_captured_ns, bool)
                or not 0 <= snapshot_captured_ns <= observation_available_ns
            ):
                record_async_scoring_event(
                    bind_async_event(
                        {
                            "schema": "semantic-async-scoring-event-v1",
                            "event": "consumption_rejected",
                            "observation_sequence": observation.get(
                                "observation_sequence"
                            ),
                            "request_ids": observation.get("request_ids", []),
                            "reason": "snapshot_clock_invalid",
                            "snapshot_captured_ns": snapshot_captured_ns,
                            "observation_available_ns": (observation_available_ns),
                            "consumer_decision_ns": decision_ns,
                        }
                    )
                )
                continue
            if not isinstance(expires_at_ns, int) or decision_ns > expires_at_ns:
                record_async_scoring_event(
                    bind_async_event(
                        {
                            "schema": "semantic-async-scoring-event-v1",
                            "event": "consumption_rejected",
                            "observation_sequence": observation.get(
                                "observation_sequence"
                            ),
                            "request_ids": observation.get("request_ids", []),
                            "reason": "snapshot_expired_at_consumer",
                            "expires_at_ns": expires_at_ns,
                            "consumer_decision_ns": decision_ns,
                        }
                    )
                )
                continue
            observation_scores = observation.get("scores")
            if not isinstance(observation_scores, dict) or not observation_scores:
                continue
            if (getattr(self.connector_scheduler, '_score_requires_bounded_contract', False)
                    and observation.get('async_contract') != 'bounded-v2'):
                continue
            if observation.get('async_contract') == 'bounded-v2':
                if (not isinstance(observation.get('source_tokens'), dict)
                        or not isinstance(observation.get('residency_versions'), dict)):
                    continue
                context = getattr(self.connector_scheduler, '_score_context', lambda: {})()
                epoch = observation.get('transport_epoch')
                if epoch is None or epoch != context.get('epoch'):
                    continue
                sequence = observation.get('observation_sequence')
                watermark = getattr(self, '_score_observation_watermark', (None, -1))
                if watermark[0] != epoch:
                    watermark = (epoch, -1)
                if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= watermark[1]:
                    continue
                self._score_observation_watermark = (epoch, sequence)
                tokens = observation.get('source_tokens', {})
                observation_scores = {method: {r: values for r, values in reqs.items()
                    if tokens.get(r) is not None and tokens[r] == context.get('sources', {}).get(r)}
                    for method, reqs in observation_scores.items()}
                if not any(observation_scores.values()):
                    continue
            leases = observation.get('decision_leases')
            if leases is not None:
                states=getattr(self.connector_scheduler,'_wait_prefetch',{})
                valid_requests={r for r,ids in leases.items() if any(
                    eid in states and states[eid].status=='pending'
                    and states[eid].source.source_request_id==r
                    and decision_ns<=states[eid].lease.expires_at_ns for eid in ids)}
                observation_scores={method:{r:values for r,values in reqs.items() if r in valid_requests}
                                    for method,reqs in observation_scores.items()}
                if not any(reqs for reqs in observation_scores.values()):continue
            residency = observation.get('residency_versions')
            if residency is not None:
                # Worker publication checking does not cover eviction/reinsert during transport.
                valid = {k for k,version in residency.items() if version is not None
                    and (block := manager._policy.get(k)) is not None and block.is_ready
                    and getattr(block,'_semantic_score_version',None) == version}
                observation_scores = {method:{req:[(k,v) for k,v in values if k in valid]
                    for req,values in reqs.items()} for method,reqs in observation_scores.items()}
                if not any(values for reqs in observation_scores.values() for values in reqs.values()):
                    continue
            observation_updates = compose_relevance_updates(
                observation_scores, rank_ema_mode, ema_rank_weight_floor
            )
            observation_latest = compose_latest_relevance(observation_scores)
            if not hasattr(manager, "apply_relevance_updates"):
                record_async_scoring_event(
                    bind_async_event(
                        {
                            "schema": "semantic-async-scoring-event-v1",
                            "event": "consumption_rejected",
                            "observation_sequence": observation.get(
                                "observation_sequence"
                            ),
                            "request_ids": observation.get("request_ids", []),
                            "reason": "manager_lacks_relevance_interface",
                            "consumer_decision_ns": decision_ns,
                        }
                    )
                )
                continue
            _t_ur = time.perf_counter() if _TIMING else 0.0
            manager.apply_relevance_updates(observation_updates, observation_latest)
            record_observation_availability(
                observation_updates,
                observation_available_ns,
                "async_worker_observation_available_ns",
                feature_snapshot_ns=snapshot_captured_ns,
            )
            consumed_ns = time.monotonic_ns()
            applied_any = True
            record_relevance_batch(self.connector_scheduler)
            record_async_scoring_event(
                bind_async_event(
                    {
                        "schema": "semantic-async-scoring-event-v1",
                        "event": "consumed",
                        "observation_sequence": observation.get("observation_sequence"),
                        "request_ids": observation.get("request_ids", []),
                        "snapshot_captured_ns": observation.get("snapshot_captured_ns"),
                        "observation_available_ns": observation.get(
                            "observation_available_ns"
                        ),
                        "expires_at_ns": expires_at_ns,
                        "consumer_decision_ns": decision_ns,
                        "consumer_use_ns": consumed_ns,
                    }
                )
            )
            if _TIMING:
                record_timing("update_relevance", time.perf_counter() - _t_ur)

        if updates or scores:
            if updates and hasattr(manager, "apply_relevance_updates"):
                # Scheduler-side fold is O(pool_size); TP=1 also transports
                # this compact form directly, while TP>1 composes only after
                # its required cross-head score reduction and global ranking.
                _t_ur = time.perf_counter() if _TIMING else 0.0
                manager.apply_relevance_updates(updates, latest_relevance)
            elif scores and hasattr(manager, "update_relevance"):
                _t_ur = time.perf_counter() if _TIMING else 0.0
                manager.update_relevance(scores)
            else:
                return
            applied_any = True
            record_relevance_batch(self.connector_scheduler)
            if _TIMING:
                record_timing("update_relevance", time.perf_counter() - _t_ur)
        if applied_any and _TIMING:
            record_count(
                "scheduler_relevance_entries",
                sum(len(values) for values in manager.relevance_ema.values()),
            )

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.connector_worker is not None and hasattr(
            self.connector_worker, "get_block_ids_with_load_errors"
        ):
            return self.connector_worker.get_block_ids_with_load_errors()
        return set()
