# SPDX-License-Identifier: Apache-2.0
"""SemanticOffloadingManager: CPUOffloadingManager with a SemanticPolicy.

CachePolicy is not pluggable by name in vLLM core -- CPUOffloadingManager
only resolves "lru" and "arc" via its own _CACHE_POLICIES dict (see
vllm/v1/kv_offload/cpu/manager.py). We build via the "lru" path (so the base
class's block-pool/ref-count setup runs untouched) and then swap in our own
policy instance afterward. See .claude/docs/semantic-eviction-plan.md,
Step 1.1.
"""

from semantic_offload.profiling import PROFILE, profiled
import hashlib
import json
import math
import time
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from typing_extensions import override
from vllm.v1.kv_offload.base import (
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from semantic_offload.decision_trace import (
    _request_provenance,
    build_decision_feature_records,
)
from semantic_offload.policy import DEFAULT_POLICY_ALPHA, SemanticPolicy

_EMA_ALPHA = 0.3  # ceiling weight on the newest observation; see update_relevance
_EMA_RANK_POWER = 1.0  # unturned, like DEFAULT_POLICY_ALPHA
# Mean is the strongest current live eviction result (5/6 needle-v2 cells,
# versus 2/6 for minmax) and the strongest offline adversarial signal. Keep
# every method selectable so the DGX audit can continue testing this default.
_DEFAULT_METHOD = "mean"
# A diagnostic run at alpha=0.95 gave an identical outcome to 0.5 (issues log
# entry #10) -- the real bottleneck there was a timing/cold-start issue
# (relevance signal arriving after the block was already evicted), not
# insufficient semantic weighting, so raising the default wouldn't help.

ComposedRelevanceUpdates = dict[str, dict[OffloadKey, tuple[float, float, float]]]
LatestRelevance = dict[str, dict[OffloadKey, float]]
_LIFECYCLE_STATES = ("future", "active", "dead")


def validate_ema_rank_weight_floor(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("ema_rank_weight_floor must be a number")
    floor = float(value)
    if not math.isfinite(floor) or not 0.0 <= floor <= _EMA_ALPHA:
        raise ValueError(
            f"ema_rank_weight_floor must be finite and in [0, {_EMA_ALPHA}]"
        )
    return floor


def _rank_ema_weight(rank: int, count: int, floor: float) -> float:
    denominator = max(count - 1, 1)
    legacy = _EMA_ALPHA * (1.0 - rank / denominator) ** _EMA_RANK_POWER
    return max(legacy, floor)


def _lifecycle_key_label(key: OffloadKey) -> str:
    return key.hex() if isinstance(key, bytes) else repr(key)


@dataclass(frozen=True)
class LifecycleReleaseResult:
    """Outcome of one generation-scoped lifecycle release request."""

    outcome: str
    owner_id: str
    generation: int
    released_keys: tuple[OffloadKey, ...] = ()
    protected_keys: tuple[OffloadKey, ...] = ()
    affected_owner_ids: tuple[str, ...] = ()
    reason: str = ""
    operation_count: int = 0
    key_effects: tuple[dict[str, object], ...] = ()


class StagingProtectionMixin:
    """Hold CPU source blocks while a staged GPU copy is request-bound.

    A completed prefetch load normally drops the cache policy's load reference.
    That is correct for an unbound speculative copy, but not after the copy is
    bound to a queued request: evicting the CPU source also invalidates the
    connector's staging entry.  This contract adds one ordinary policy
    reference per binding and therefore has identical hard-protection semantics
    for semantic, ARC, and LRU policies without changing their rankings.
    """

    def _init_staging_protection_contract(self) -> None:
        self._staging_protection_keys: dict[str, tuple[OffloadKey, ...]] = {}
        self._pending_staging_protection_records: list[dict[str, object]] = []

    def _record_staging_protection(
        self,
        *,
        binding_id: str,
        action: str,
        outcome: str,
        keys: Collection[OffloadKey],
        started_ns: int,
        started_cpu_ns: int,
        reason: str,
    ) -> dict[str, object]:
        completed_ns = time.monotonic_ns()
        completed_cpu_ns = time.thread_time_ns()
        record: dict[str, object] = {
            "schema": "semantic-staging-protection-v1",
            "clock_domain": "host_monotonic",
            "binding_id": binding_id,
            "action": action,
            "outcome": outcome,
            "reason": reason,
            "keys": [_lifecycle_key_label(key) for key in keys],
            "key_count": len(keys),
            "started_ns": started_ns,
            "completed_ns": completed_ns,
            "critical_path_wall_time_ns": completed_ns - started_ns,
            "cpu_time_ns": completed_cpu_ns - started_cpu_ns,
            "synchronization_wait_ns": 0,
            "synchronization_status": "no_explicit_synchronization",
        }
        self._pending_staging_protection_records.append(record)
        return record

    def protect_staging_keys(
        self, binding_id: str, keys: Collection[OffloadKey]
    ) -> dict[str, object]:
        """Atomically acquire one cache-policy reference for a binding."""
        started_ns = time.monotonic_ns()
        started_cpu_ns = time.thread_time_ns()
        if not isinstance(binding_id, str) or not binding_id:
            raise ValueError("staging binding_id must be a non-empty string")
        requested = tuple(dict.fromkeys(keys))
        prior = self._staging_protection_keys.get(binding_id)
        if prior is not None:
            if prior != requested:
                raise ValueError("staging binding cannot change protected keys")
            return self._record_staging_protection(
                binding_id=binding_id,
                action="protect",
                outcome="noop",
                keys=requested,
                started_ns=started_ns,
                started_cpu_ns=started_cpu_ns,
                reason="already_protected",
            )
        unavailable = [
            key
            for key in requested
            if (block := self._policy.get(key)) is None or not block.is_ready
        ]
        if unavailable:
            return self._record_staging_protection(
                binding_id=binding_id,
                action="protect",
                outcome="rejected",
                keys=requested,
                started_ns=started_ns,
                started_cpu_ns=started_cpu_ns,
                reason="source_missing_or_not_ready",
            )
        for key in requested:
            block = self._policy.get(key)
            assert block is not None and block.is_ready
            if block.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_blocks -= 1
                if self._num_evictable_cache_blocks < 0:
                    raise RuntimeError("staging protection corrupted evictable count")
            block.ref_cnt += 1
        self._staging_protection_keys[binding_id] = requested
        return self._record_staging_protection(
            binding_id=binding_id,
            action="protect",
            outcome="accepted",
            keys=requested,
            started_ns=started_ns,
            started_cpu_ns=started_cpu_ns,
            reason="bound_staging_reference",
        )

    def release_staging_keys(self, binding_id: str) -> dict[str, object]:
        """Release a binding reference after consume/discard/cancel/finish."""
        started_ns = time.monotonic_ns()
        started_cpu_ns = time.thread_time_ns()
        keys = self._staging_protection_keys.pop(binding_id, None)
        if keys is None:
            return self._record_staging_protection(
                binding_id=binding_id,
                action="release",
                outcome="noop",
                keys=(),
                started_ns=started_ns,
                started_cpu_ns=started_cpu_ns,
                reason="unknown_or_already_released",
            )
        for key in keys:
            block = self._policy.get(key)
            if block is None or block.ref_cnt <= 0:
                raise RuntimeError("protected staging source disappeared")
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1
                self._policy.mark_evictable(key)
        return self._record_staging_protection(
            binding_id=binding_id,
            action="release",
            outcome="accepted",
            keys=keys,
            started_ns=started_ns,
            started_cpu_ns=started_cpu_ns,
            reason="binding_lifetime_ended",
        )

    def drain_staging_protection_records(self) -> list[dict[str, object]]:
        records = self._pending_staging_protection_records
        self._pending_staging_protection_records = []
        return records

    def staging_protection_stats(self) -> dict[str, int]:
        return {
            "active_bindings": len(self._staging_protection_keys),
            "protected_key_references": sum(
                len(keys) for keys in self._staging_protection_keys.values()
            ),
            "pending_records": len(self._pending_staging_protection_records),
        }

    def _reset_staging_protection_contract(self) -> None:
        self._init_staging_protection_contract()


class LifecycleReleaseMixin:
    """Apply explicit post-request release hints conservatively.

    A release hint is an application fact, not a prediction. Only idle CPU
    blocks are removed; a block with an outstanding load/store reference is
    left under the base policy so the hint cannot invalidate active work.
    """

    def _init_lifecycle_contract(self) -> None:
        self._lifecycle_owner_generations: dict[str, int] = {}
        self._lifecycle_owner_states: dict[str, str] = {}
        self._lifecycle_owner_keys: dict[tuple[str, int], set[OffloadKey]] = {}
        self._lifecycle_owners_by_key: dict[OffloadKey, set[tuple[str, int]]] = {}
        self._lifecycle_release_ledger: list[dict[str, object]] = []

    def register_lifecycle_owner(
        self,
        owner_id: str,
        generation: int,
        keys: Collection[OffloadKey],
        *,
        state: str = "active",
    ) -> None:
        """Register logical ownership without changing physical cache state.

        Registration is idempotent for the same owner generation. A newer
        generation may advance only after the previous generation is dead and
        has no remaining logical keys; otherwise a delayed release would be
        ambiguous. ``future`` owners may reserve a shared key logically before
        their first physical access.
        """
        if state not in _LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle owner state: {state!r}")
        if generation < 0:
            raise ValueError("lifecycle generation must be non-negative")
        current_generation = self._lifecycle_owner_generations.get(owner_id)
        if current_generation is not None and current_generation != generation:
            current_state = self._lifecycle_owner_states.get(owner_id)
            current_ref = (owner_id, current_generation)
            current_keys = self._lifecycle_owner_keys.get(current_ref, set())
            if (
                generation <= current_generation
                or current_state not in (None, "dead")
                or current_keys
            ):
                raise ValueError(
                    f"owner {owner_id!r} cannot advance from generation "
                    f"{current_generation} with state={current_state!r} "
                    f"and {len(current_keys)} owned keys"
                )
            self._lifecycle_owner_keys.pop(current_ref, None)
            self._lifecycle_owner_states.pop(owner_id, None)

        current_state = self._lifecycle_owner_states.get(owner_id)
        if current_generation == generation and current_state is None:
            raise ValueError(f"owner {owner_id!r} generation {generation} is retired")
        if current_state is not None and _LIFECYCLE_STATES.index(
            state
        ) < _LIFECYCLE_STATES.index(current_state):
            raise ValueError(
                f"lifecycle state cannot move backward: {current_state} -> {state}"
            )
        self._lifecycle_owner_generations[owner_id] = generation
        self._lifecycle_owner_states[owner_id] = state
        owner_ref = (owner_id, generation)
        owner_keys = self._lifecycle_owner_keys.setdefault(owner_ref, set())
        for key in dict.fromkeys(keys):
            owner_keys.add(key)
            self._lifecycle_owners_by_key.setdefault(key, set()).add(owner_ref)

    def set_lifecycle_owner_state(
        self, owner_id: str, generation: int, state: str
    ) -> None:
        """Advance a registered owner through future -> active -> dead."""
        if state not in _LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle owner state: {state!r}")
        if self._lifecycle_owner_generations.get(owner_id) != generation:
            raise ValueError("unknown lifecycle owner generation")
        current = self._lifecycle_owner_states[owner_id]
        if _LIFECYCLE_STATES.index(state) < _LIFECYCLE_STATES.index(current):
            raise ValueError(
                f"lifecycle state cannot move backward: {current} -> {state}"
            )
        self._lifecycle_owner_states[owner_id] = state

    def retire_lifecycle_owner(self, owner_id: str, generation: int) -> None:
        """Drop detailed state after a dead generation is fully released.

        The highest generation remains as a tombstone so a late hint cannot
        recreate or mutate retired ownership. A subsequent registration must
        use a strictly newer generation.
        """
        current_generation = self._lifecycle_owner_generations.get(owner_id)
        if current_generation != generation:
            raise ValueError("unknown lifecycle owner generation")
        state = self._lifecycle_owner_states.get(owner_id)
        if state != "dead":
            raise ValueError("lifecycle owner must be dead before retirement")
        owner_ref = (owner_id, generation)
        owner_keys = self._lifecycle_owner_keys.get(owner_ref, set())
        if owner_keys:
            raise ValueError(
                f"lifecycle owner still owns {len(owner_keys)} logical keys"
            )
        self._lifecycle_owner_keys.pop(owner_ref, None)
        self._lifecycle_owner_states.pop(owner_id, None)

    def release_lifecycle_owner(
        self,
        owner_id: str,
        generation: int,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LifecycleReleaseResult:
        """Apply a fail-closed, generation-scoped release hint.

        A dead owner can release only its private, idle keys. Shared keys lose
        one logical owner but remain cached; active blocks reject the request
        without mutating ownership so the caller can retry after the active
        transfer completes. Invalid generations and repeated hints are
        recorded and never alter cache state.
        """
        del req_context  # Reserved for the eventual scheduler-owned callback.
        requested = tuple(dict.fromkeys(keys))
        operation_count = len(requested) + 1
        current_generation = self._lifecycle_owner_generations.get(owner_id)
        if current_generation != generation:
            result = LifecycleReleaseResult(
                "rejected",
                owner_id,
                generation,
                reason="wrong_generation",
                operation_count=operation_count,
            )
            self._record_lifecycle_release(result, requested)
            return result
        state = self._lifecycle_owner_states.get(owner_id)
        if state is None:
            result = LifecycleReleaseResult(
                "rejected",
                owner_id,
                generation,
                reason="owner_retired",
                operation_count=operation_count,
            )
            self._record_lifecycle_release(result, requested)
            return result
        if state != "dead":
            result = LifecycleReleaseResult(
                "rejected",
                owner_id,
                generation,
                protected_keys=requested,
                reason="active_reference" if state == "active" else "owner_not_dead",
                operation_count=operation_count,
            )
            self._record_lifecycle_release(result, requested)
            return result

        owner_ref = (owner_id, generation)
        owner_keys = self._lifecycle_owner_keys[owner_ref]

        def key_effect(
            key: OffloadKey,
            *,
            action: str,
            refs_before: set[tuple[str, int]],
            refs_after: set[tuple[str, int]],
            block_before,
            owner_owned_before: bool,
        ) -> dict[str, object]:
            block_ref_cnt = getattr(block_before, "ref_cnt", None)
            return {
                "key": _lifecycle_key_label(key),
                "owner_ref": [owner_id, generation],
                "action": action,
                "owner_owned_before": owner_owned_before,
                "owner_refs_before": [
                    [ref_owner, ref_generation]
                    for ref_owner, ref_generation in sorted(refs_before)
                ],
                "owner_refs_after": [
                    [ref_owner, ref_generation]
                    for ref_owner, ref_generation in sorted(refs_after)
                ],
                "block_present_before": block_before is not None,
                "block_ref_cnt_before": block_ref_cnt
                if isinstance(block_ref_cnt, int)
                and not isinstance(block_ref_cnt, bool)
                else None,
                "block_present_after": self._policy.get(key) is not None,
            }

        owned_requested = [key for key in requested if key in owner_keys]
        active_keys = [
            key
            for key in owned_requested
            if (block := self._policy.get(key)) is not None and block.ref_cnt != 0
        ]
        if active_keys:
            key_effects = tuple(
                key_effect(
                    key,
                    action="rejected_active_reference",
                    refs_before=set(self._lifecycle_owners_by_key.get(key, set())),
                    refs_after=set(self._lifecycle_owners_by_key.get(key, set())),
                    block_before=self._policy.get(key),
                    owner_owned_before=key in owner_keys,
                )
                for key in requested
            )
            result = LifecycleReleaseResult(
                "rejected",
                owner_id,
                generation,
                protected_keys=tuple(active_keys),
                reason="active_reference",
                operation_count=operation_count,
                key_effects=key_effects,
            )
            self._record_lifecycle_release(result, requested)
            return result
        released: list[OffloadKey] = []
        protected: list[OffloadKey] = []
        affected_owner_ids: set[str] = set()
        key_effects: list[dict[str, object]] = []
        shared = False
        for key in requested:
            refs_before = set(self._lifecycle_owners_by_key.get(key, set()))
            block = self._policy.get(key)
            if key not in owner_keys:
                key_effects.append(
                    key_effect(
                        key,
                        action="not_owned",
                        refs_before=refs_before,
                        refs_after=set(refs_before),
                        block_before=block,
                        owner_owned_before=False,
                    )
                )
                continue
            owner_keys.remove(key)
            refs = self._lifecycle_owners_by_key.get(key, set())
            refs.discard(owner_ref)
            affected_owner_ids.update(ref[0] for ref in refs)
            if refs:
                shared = True
                key_effects.append(
                    key_effect(
                        key,
                        action="retained_shared",
                        refs_before=refs_before,
                        refs_after=set(refs),
                        block_before=block,
                        owner_owned_before=True,
                    )
                )
                continue
            if block is None:
                key_effects.append(
                    key_effect(
                        key,
                        action="released_logical_only",
                        refs_before=refs_before,
                        refs_after=set(refs),
                        block_before=block,
                        owner_owned_before=True,
                    )
                )
                continue
            self._policy.remove(key)
            self._free_block(block)
            resident_costs = getattr(
                self, "_admission_resident_cost_per_byte", None
            )
            if isinstance(resident_costs, dict):
                resident_costs.pop(key, None)
            self._num_evictable_cache_blocks -= 1
            if self._num_evictable_cache_blocks < 0:
                raise RuntimeError("lifecycle release corrupted evictable count")
            released.append(key)
            key_effects.append(
                key_effect(
                    key,
                    action="released_private_idle",
                    refs_before=refs_before,
                    refs_after=set(refs),
                    block_before=block,
                    owner_owned_before=True,
                )
            )

        if protected:
            outcome = "rejected"
            reason = "active_reference"
        elif released:
            outcome = "accepted"
            reason = "private_idle_keys"
        else:
            outcome = "noop"
            reason = "shared_owner" if shared else "already_released"
        result = LifecycleReleaseResult(
            outcome,
            owner_id,
            generation,
            released_keys=tuple(released),
            protected_keys=tuple(protected),
            affected_owner_ids=tuple(sorted(affected_owner_ids)),
            reason=reason,
            operation_count=operation_count,
            key_effects=tuple(key_effects),
        )
        self._record_lifecycle_release(result, requested)
        if released:
            pending = getattr(self, "_pending_evicted_keys", None)
            if isinstance(pending, list):
                pending.extend(released)
        return result

    def _record_lifecycle_release(
        self,
        result: LifecycleReleaseResult,
        requested: Collection[OffloadKey],
    ) -> None:
        self._lifecycle_release_ledger.append(
            {
                "owner_id": result.owner_id,
                "generation": result.generation,
                "outcome": result.outcome,
                "reason": result.reason,
                "requested_keys": [_lifecycle_key_label(key) for key in requested],
                "released_keys": [
                    _lifecycle_key_label(key) for key in result.released_keys
                ],
                "protected_keys": [
                    _lifecycle_key_label(key) for key in result.protected_keys
                ],
                "affected_owner_ids": list(result.affected_owner_ids),
                "operation_count": result.operation_count,
                "key_effects": list(result.key_effects),
            }
        )

    def drain_lifecycle_release_ledger(self) -> list[dict[str, object]]:
        records = self._lifecycle_release_ledger
        self._lifecycle_release_ledger = []
        return records

    def lifecycle_bookkeeping_stats(self) -> dict[str, int]:
        """Expose boundedness counters for lifecycle-owned metadata."""
        return {
            "owner_generation_tombstones": len(self._lifecycle_owner_generations),
            "detailed_owner_states": len(self._lifecycle_owner_states),
            "detailed_owner_generations": len(self._lifecycle_owner_keys),
            "logical_owner_key_references": sum(
                len(keys) for keys in self._lifecycle_owner_keys.values()
            ),
            "pending_manager_ledger_entries": len(self._lifecycle_release_ledger),
        }

    def _reset_lifecycle_contract(self) -> None:
        self._init_lifecycle_contract()

    def release_keys(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> tuple[list[OffloadKey], list[OffloadKey]]:
        released: list[OffloadKey] = []
        skipped: list[OffloadKey] = []
        for key in dict.fromkeys(keys):
            block = self._policy.get(key)
            if block is None or block.ref_cnt != 0:
                skipped.append(key)
                continue
            self._policy.remove(key)
            self._free_block(block)
            resident_costs = getattr(
                self, "_admission_resident_cost_per_byte", None
            )
            if isinstance(resident_costs, dict):
                resident_costs.pop(key, None)
            self._num_evictable_cache_blocks -= 1
            if self._num_evictable_cache_blocks < 0:
                raise RuntimeError("lifecycle release corrupted evictable count")
            released.append(key)
        pending = getattr(self, "_pending_evicted_keys", None)
        if isinstance(pending, list):
            pending.extend(released)
        return released, skipped


def _validate_admission_mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("admission_mode must be a string")
    if value not in (
        "disabled",
        "historical_frequency",
        "historical_frequency_cost_per_byte",
    ):
        raise ValueError(
            "admission_mode must be disabled, historical_frequency, or "
            "historical_frequency_cost_per_byte"
        )
    return value


def _validate_nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _validate_admission_costs(value: object) -> dict[int, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("admission_cost_per_byte must be a JSON object")
    costs: dict[int, float] = {}
    for raw_group, raw_cost in value.items():
        if isinstance(raw_group, bool):
            raise TypeError("admission cost group ids must be non-negative integers")
        try:
            group = int(raw_group)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "admission cost group ids must be non-negative integers"
            ) from exc
        if group < 0 or str(group) != str(raw_group):
            raise ValueError(
                "admission cost group ids must be canonical non-negative integers"
            )
        costs[group] = _validate_nonnegative_finite(
            raw_cost, f"admission_cost_per_byte[{group}]"
        )
    return costs


def _validate_admission_namespace(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("admission_namespace must be a JSON object")
    allowed = {
        "model_revision",
        "tokenizer_revision",
        "adapter_namespace",
        "layout_namespace",
    }
    if set(value) - allowed:
        raise ValueError("admission_namespace contains unknown fields")
    return dict(value)


def _validate_chain_admission_max_prefix_blocks(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("chain_admission_max_prefix_blocks must be an integer")
    if value < 0:
        raise ValueError(
            "chain_admission_max_prefix_blocks must be non-negative"
        )
    return value


class FrequencyCostAdmissionMixin:
    def _stamp_score_residencies(self, result):
        if result is None:
            return
        versions = self.__dict__.setdefault('_score_residencies', {})
        for key in result.evicted_keys:
            versions.pop(key, None)
        for key in result.keys_to_store:
            self._score_residency_sequence = getattr(self, '_score_residency_sequence', 0)+1
            versions[key] = self._score_residency_sequence
            self._policy.get(key)._semantic_score_version = self._score_residency_sequence

    def score_residency_versions(self):
        versions = self.__dict__.setdefault('_score_residencies', {})
        current = {}
        for key, version in list(versions.items()):
            block = self._policy.get(key)
            if block is None:
                versions.pop(key, None)
            elif block.is_ready and getattr(block, '_semantic_score_version', None) == version:
                current[key] = version
        return current

    """Optional TinyLFU-style admission using causal, bounded features only."""

    def _init_admission_contract(
        self,
        *,
        admission_mode: object = "disabled",
        admission_cost_per_byte: object = None,
        admission_default_cost_per_byte: object = 1.0,
        admission_namespace: object = None,
        chain_admission_max_prefix_blocks: object = 0,
        max_tracker_size: int,
    ) -> None:
        self._admission_mode = _validate_admission_mode(admission_mode)
        self._admission_cost_per_byte = _validate_admission_costs(
            admission_cost_per_byte
        )
        self._admission_default_cost_per_byte = _validate_nonnegative_finite(
            admission_default_cost_per_byte, "admission_default_cost_per_byte"
        )
        self._admission_namespace = _validate_admission_namespace(admission_namespace)
        self._chain_admission_max_prefix_blocks = (
            _validate_chain_admission_max_prefix_blocks(
                chain_admission_max_prefix_blocks
            )
        )
        if isinstance(max_tracker_size, bool) or max_tracker_size <= 0:
            raise ValueError("max_tracker_size must be a positive integer")
        self._admission_max_tracker_size = int(max_tracker_size)
        if (
            self._chain_admission_max_prefix_blocks
            > self._admission_max_tracker_size
        ):
            raise ValueError(
                "chain_admission_max_prefix_blocks cannot exceed max_tracker_size"
            )
        self._admission_frequencies: OrderedDict[OffloadKey, int] = OrderedDict()
        self._admission_observations: dict[OffloadKey, tuple[int, int]] = {}
        self._admission_resident_cost_per_byte: dict[
            OffloadKey, tuple[float, str, int, int]
        ] = {}
        self._pending_admission_records: list[dict[str, object]] = []
        self._pending_admission_accounting = {
            "admitted": 0,
            "rejected": 0,
            "compulsory_fallbacks": 0,
        }
        self._admission_decision_sequence = 0
        self._admission_observation_sequence = 0
        self._chain_admission_observations: OrderedDict[
            tuple[object, ...], tuple[str, tuple[OffloadKey, ...], int]
        ] = OrderedDict()
        self._chain_admission_pending_stores: dict[
            str, tuple[OffloadKey, ...]
        ] = {}
        self._pending_chain_admission_records: list[dict[str, object]] = []
        self._pending_chain_admission_accounting = {
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
            "tracker_evictions": 0,
            "overhead_cpu_time_ns": 0,
            "overhead_wall_time_ns": 0,
        }
        config_payload = {
            "mode": self._admission_mode,
            "cost_per_byte": self._admission_cost_per_byte,
            "default_cost_per_byte": self._admission_default_cost_per_byte,
            "store_threshold": self.store_threshold,
            "max_tracker_size": self._admission_max_tracker_size,
            "chain_admission_max_prefix_blocks": (
                self._chain_admission_max_prefix_blocks
            ),
        }
        self._admission_config_hash = hashlib.sha256(
            json.dumps(
                config_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def _chain_admission_identity(
        self, req_context: ReqContext
    ) -> tuple[object, ...] | None:
        params = getattr(req_context, "kv_transfer_params", None) or {}
        namespace = self._admission_namespace
        namespace_fields = (
            "model_revision",
            "tokenizer_revision",
            "adapter_namespace",
            "layout_namespace",
        )
        namespace_values = tuple(namespace.get(field) for field in namespace_fields)
        session_id = params.get("session_id", params.get("semantic_session_id"))
        generation = params.get("generation", params.get("semantic_generation"))
        digest = params.get("key_chain_digest")
        if (
            any(not isinstance(value, str) or not value for value in namespace_values)
            or not isinstance(session_id, str)
            or not session_id
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            return None
        try:
            bytes.fromhex(digest)
        except ValueError:
            return None
        return (*namespace_values, session_id, generation, digest)

    def _record_chain_admission(
        self,
        *,
        req_context: ReqContext,
        action: str,
        reason: str,
        identity: tuple[object, ...] | None,
        observed_accesses: int,
        observed_keys: tuple[OffloadKey, ...],
        qualified_keys: tuple[OffloadKey, ...],
        admitted_keys: tuple[OffloadKey, ...],
        started_ns: int,
        started_cpu_ns: int,
    ) -> None:
        completed_ns = time.monotonic_ns()
        completed_cpu_ns = time.thread_time_ns()
        wall_time_ns = completed_ns - started_ns
        cpu_time_ns = completed_cpu_ns - started_cpu_ns
        params = getattr(req_context, "kv_transfer_params", None) or {}
        self._pending_chain_admission_accounting["overhead_cpu_time_ns"] += (
            cpu_time_ns
        )
        self._pending_chain_admission_accounting["overhead_wall_time_ns"] += (
            wall_time_ns
        )
        self._pending_chain_admission_records.append(
            {
                "schema": "semantic-exact-chain-admission-v1",
                "event": "exact_chain_admission",
                "request_id": req_context.req_id,
                "run_id": params.get("run_id", params.get("semantic_run_id")),
                "session_id": (
                    params.get("session_id", params.get("semantic_session_id"))
                ),
                "generation": params.get(
                    "generation", params.get("semantic_generation")
                ),
                "key_chain_digest": params.get("key_chain_digest"),
                "namespace": dict(self._admission_namespace),
                "identity_complete": identity is not None,
                "store_threshold": self.store_threshold,
                "max_prefix_blocks": self._chain_admission_max_prefix_blocks,
                "observed_accesses": observed_accesses,
                "observed_keys": [
                    {"position": position, "key": _lifecycle_key_label(key)}
                    for position, key in enumerate(observed_keys)
                ],
                "qualified_keys": [
                    {"position": position, "key": _lifecycle_key_label(key)}
                    for position, key in enumerate(qualified_keys)
                ],
                "admitted_keys": [
                    {"position": position, "key": _lifecycle_key_label(key)}
                    for position, key in enumerate(admitted_keys)
                ],
                "action": action,
                "reason": reason,
                "started_ns": started_ns,
                "completed_ns": completed_ns,
                "overhead_wall_time_ns": wall_time_ns,
                "overhead_cpu_time_ns": cpu_time_ns,
            }
        )

    def _prepare_exact_chain_admission(
        self,
        keys: list[OffloadKey],
        req_context: ReqContext,
    ) -> tuple[OffloadKey, ...]:
        """Credit one bounded exact prefix after distinct chain accesses.

        Native threshold tracking sees only the first miss because lookup stops
        at that miss.  When enabled, the ordered store keys are an authoritative
        observation that the request actually computed the whole exact prefix.
        A second *distinct request* with the same fully-bound identity may
        therefore credit those same positions without using semantic scores or
        future-demand labels.  The first request still pays the native miss.
        """
        if (
            self._chain_admission_max_prefix_blocks == 0
            or self.store_threshold != 2
            or self.counts is None
            or not keys
        ):
            return ()
        started_ns = time.monotonic_ns()
        started_cpu_ns = time.thread_time_ns()
        self._pending_chain_admission_accounting["observations"] += 1
        identity = self._chain_admission_identity(req_context)
        observed_keys = tuple(keys[: self._chain_admission_max_prefix_blocks])
        if identity is None:
            self._pending_chain_admission_accounting["identity_rejections"] += 1
            self._record_chain_admission(
                req_context=req_context,
                action="reject",
                reason="incomplete_exact_chain_identity",
                identity=None,
                observed_accesses=0,
                observed_keys=observed_keys,
                qualified_keys=(),
                admitted_keys=(),
                started_ns=started_ns,
                started_cpu_ns=started_cpu_ns,
            )
            return ()

        prior = self._chain_admission_observations.get(identity)
        if prior is None:
            if len(self._chain_admission_observations) >= self._admission_max_tracker_size:
                self._chain_admission_observations.popitem(last=False)
                self._pending_chain_admission_accounting["tracker_evictions"] += 1
            self._chain_admission_observations[identity] = (
                req_context.req_id,
                observed_keys,
                1,
            )
            self._record_chain_admission(
                req_context=req_context,
                action="observe",
                reason="first_exact_chain_access",
                identity=identity,
                observed_accesses=1,
                observed_keys=observed_keys,
                qualified_keys=(),
                admitted_keys=(),
                started_ns=started_ns,
                started_cpu_ns=started_cpu_ns,
            )
            return ()

        prior_request_id, prior_keys, prior_accesses = prior
        self._chain_admission_observations.move_to_end(identity)
        if prior_request_id == req_context.req_id:
            self._pending_chain_admission_accounting[
                "duplicate_request_observations"
            ] += 1
            self._record_chain_admission(
                req_context=req_context,
                action="ignore",
                reason="duplicate_request_access",
                identity=identity,
                observed_accesses=prior_accesses,
                observed_keys=observed_keys,
                qualified_keys=(),
                admitted_keys=(),
                started_ns=started_ns,
                started_cpu_ns=started_cpu_ns,
            )
            return ()
        if observed_keys != prior_keys:
            self._pending_chain_admission_accounting["prefix_rejections"] += 1
            # Do not replace the earlier observation: a caller cannot rebind a
            # digest/session/generation tuple to different position identities.
            self._record_chain_admission(
                req_context=req_context,
                action="reject",
                reason="exact_chain_prefix_mismatch",
                identity=identity,
                observed_accesses=prior_accesses,
                observed_keys=observed_keys,
                qualified_keys=(),
                admitted_keys=(),
                started_ns=started_ns,
                started_cpu_ns=started_cpu_ns,
            )
            return ()

        accesses = prior_accesses + 1
        self._chain_admission_observations[identity] = (
            req_context.req_id,
            prior_keys,
            accesses,
        )
        self._pending_chain_admission_accounting["matched_observations"] += 1
        qualified = tuple(
            key for key in prior_keys if self._policy.get(key) is None
        )
        for key in qualified:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] = max(self.counts[key], self.store_threshold)
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = self.store_threshold
        self._pending_chain_admission_accounting["qualified_chains"] += 1
        self._pending_chain_admission_accounting["qualified_keys"] += len(qualified)
        return qualified

    def _observe_admission_frequency(self, key: OffloadKey) -> None:
        frequencies = self._admission_frequencies
        if key in frequencies:
            frequencies[key] += 1
            frequencies.move_to_end(key)
        else:
            if len(frequencies) >= self._admission_max_tracker_size:
                expired, _ = frequencies.popitem(last=False)
                self._admission_observations.pop(expired, None)
            frequencies[key] = 1
        self._admission_observation_sequence += 1
        self._admission_observations[key] = (
            time.monotonic_ns(),
            self._admission_observation_sequence,
        )

    def _admission_cost(
        self, key: OffloadKey, req_context: ReqContext | None = None
    ) -> tuple[float | None, str, int | None, int | None]:
        if self._admission_mode == "historical_frequency":
            return 1.0, "unit_cost_frequency_control", 0, 0
        params = (
            getattr(req_context, "kv_transfer_params", None) or {}
            if req_context is not None
            else {}
        )
        if "admission_cost_per_byte" in params:
            cost = _validate_nonnegative_finite(
                params["admission_cost_per_byte"], "request admission_cost_per_byte"
            )
            provenance = params.get("admission_cost_provenance")
            available_ns = params.get("admission_cost_observation_available_ns")
            publication_ns = params.get("admission_cost_input_publication_ns")
            if not isinstance(provenance, str) or not provenance:
                raise ValueError("request admission cost requires provenance")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in (available_ns, publication_ns)
            ):
                raise ValueError(
                    "request admission cost requires non-negative availability and "
                    "publication timestamps"
                )
            decision_boundary_ns = time.monotonic_ns()
            if not available_ns <= publication_ns <= decision_boundary_ns:
                return (
                    None,
                    "unavailable_noncausal_cost_sample",
                    available_ns,
                    publication_ns,
                )
            return (
                cost,
                f"historical_request_sample:{provenance}",
                available_ns,
                publication_ns,
            )
        try:
            group = get_offload_group_idx(key)
        except (TypeError, ValueError, OverflowError):
            return None, "unavailable", None, None
        if group in self._admission_cost_per_byte:
            return self._admission_cost_per_byte[group], "configured_group", 0, 0
        return self._admission_default_cost_per_byte, "configured_default", 0, 0

    def _peek_admission_victims(
        self, n: int, protected: set[OffloadKey]
    ) -> list[OffloadKey] | None:
        if n <= 0:
            return []
        policy = self._policy
        if hasattr(policy, "peek_victims"):
            preview = policy.peek_victims(n, protected)
            if preview is None:
                return None
            return list(preview)
        evictable = getattr(policy, "evictable_blocks", None)
        blocks = getattr(policy, "blocks", None)
        if isinstance(evictable, OrderedDict) and isinstance(blocks, dict):
            candidates = [key for key in evictable if key not in protected][:n]
            return candidates if len(candidates) == n else None
        t1 = getattr(policy, "t1", None)
        t2 = getattr(policy, "t2", None)
        target = getattr(policy, "target_t1_size", None)
        if not (
            isinstance(t1, OrderedDict)
            and isinstance(t2, OrderedDict)
            and isinstance(target, (int, float))
        ):
            return None
        candidates: list[OffloadKey] = []
        selected: set[OffloadKey] = set()
        virtual_t1_size = len(t1)
        for _ in range(n):
            candidate = None
            if virtual_t1_size >= int(target):
                candidate = next(
                    (
                        key
                        for key, block in t1.items()
                        if block.ref_cnt == 0
                        and key not in protected
                        and key not in selected
                    ),
                    None,
                )
                if candidate is not None:
                    virtual_t1_size -= 1
            if candidate is None:
                candidate = next(
                    (
                        key
                        for key, block in t2.items()
                        if block.ref_cnt == 0
                        and key not in protected
                        and key not in selected
                    ),
                    None,
                )
            if candidate is None:
                return None
            candidates.append(candidate)
            selected.add(candidate)
        return candidates

    def _record_admission_decision(
        self,
        *,
        req_context: ReqContext,
        candidate: OffloadKey,
        victim: OffloadKey | None,
        action: str,
        reason: str,
        candidate_frequency: int | None,
        candidate_cost: float | None,
        candidate_cost_origin: str,
        candidate_cost_available_ns: int | None,
        candidate_cost_publication_ns: int | None,
        victim_frequency: int | None,
        victim_cost: float | None,
        victim_cost_origin: str,
        victim_cost_available_ns: int | None,
        victim_cost_publication_ns: int | None,
    ) -> None:
        self._admission_decision_sequence += 1
        decision_ns = time.monotonic_ns()
        params = getattr(req_context, "kv_transfer_params", None) or {}
        candidate_observation = self._admission_observations.get(candidate)
        victim_observation = (
            self._admission_observations.get(victim) if victim is not None else None
        )
        observation_times = [
            observation[0]
            for observation in (candidate_observation, victim_observation)
            if observation is not None
        ]
        observation_times.extend(
            timestamp
            for timestamp in (
                candidate_cost_available_ns,
                victim_cost_available_ns,
            )
            if timestamp is not None
        )
        observation_available_ns = max(observation_times) if observation_times else 0
        publication_times = [
            observation[0]
            for observation in (candidate_observation, victim_observation)
            if observation is not None
        ]
        publication_times.extend(
            timestamp
            for timestamp in (
                candidate_cost_publication_ns,
                victim_cost_publication_ns,
            )
            if timestamp is not None
        )
        input_publication_ns = max(publication_times) if publication_times else 0
        observation_sequences = [
            observation[1]
            for observation in (candidate_observation, victim_observation)
            if observation is not None
        ]
        namespace = self._admission_namespace
        causal_identity = {
            "engine_run_id": params.get("run_id", params.get("semantic_run_id")),
            "model_revision": namespace.get("model_revision"),
            "tokenizer_revision": namespace.get("tokenizer_revision"),
            "adapter_namespace": namespace.get("adapter_namespace"),
            "layout_namespace": namespace.get("layout_namespace"),
            "session_id": params.get("session_id", params.get("semantic_session_id")),
            "generation": params.get("generation", params.get("semantic_generation")),
            "scheduler_step": params.get("scheduler_step"),
            "key_chain_digest": params.get("key_chain_digest"),
        }
        record = {
            "schema": "semantic-admission-decision-v1",
            "schema_version": 1,
            "event": "admission_decision",
            "request_id": req_context.req_id,
            **causal_identity,
            "decision_sequence": self._admission_decision_sequence,
            "observation_sequence": (
                max(observation_sequences) if observation_sequences else 0
            ),
            "observation_available_ns": observation_available_ns,
            "input_publication_ns": input_publication_ns,
            "consumer_decision_ns": decision_ns,
            "consumer_use_ns": decision_ns,
            "decision_timestamp_ns": decision_ns,
            "clock_domain": "host_monotonic",
            "mode": self._admission_mode,
            "candidate_config_hash": params.get(
                "candidate_config_hash", self._admission_config_hash
            ),
            "admission_config_hash": self._admission_config_hash,
            "candidate_key": _lifecycle_key_label(candidate),
            "victim_key": (
                _lifecycle_key_label(victim) if victim is not None else None
            ),
            "candidate_frequency": candidate_frequency,
            "candidate_cost_per_byte": candidate_cost,
            "candidate_cost_origin": candidate_cost_origin,
            "candidate_cost_observation_available_ns": candidate_cost_available_ns,
            "candidate_cost_input_publication_ns": candidate_cost_publication_ns,
            "candidate_features_available": (
                candidate_frequency is not None and candidate_cost is not None
            ),
            "victim_frequency": victim_frequency,
            "victim_cost_per_byte": victim_cost,
            "victim_cost_origin": victim_cost_origin,
            "victim_cost_observation_available_ns": victim_cost_available_ns,
            "victim_cost_input_publication_ns": victim_cost_publication_ns,
            "feature_provenance": "bounded_historical_lookup_frequency",
            "cost_sample_provenance": {
                "candidate": candidate_cost_origin,
                "victim": victim_cost_origin,
            },
            "victim_features_available": (
                (victim is None and reason == "free_slot")
                or (victim_frequency is not None and victim_cost is not None)
            ),
            "candidate_utility": (
                candidate_frequency * candidate_cost
                if candidate_frequency is not None and candidate_cost is not None
                else None
            ),
            "victim_utility": (
                victim_frequency * victim_cost
                if victim_frequency is not None and victim_cost is not None
                else None
            ),
            "action": action,
            "reason": reason,
            "causal_binding_complete": (
                observation_available_ns <= input_publication_ns <= decision_ns
                and all(value is not None for value in causal_identity.values())
            ),
        }
        self._pending_admission_records.append(record)
        if action == "reject":
            self._pending_admission_accounting["rejected"] += 1
        else:
            self._pending_admission_accounting["admitted"] += 1
        if reason == "compulsory_fallback_unavailable_features":
            self._pending_admission_accounting["compulsory_fallbacks"] += 1

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext):
        if self._admission_mode != "disabled":
            self._observe_admission_frequency(key)
        return super().lookup(key, req_context)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        original_keys = list(keys)
        chain_qualified = self._prepare_exact_chain_admission(
            original_keys, req_context
        )
        if self._admission_mode == "disabled":
            result = super().prepare_store(original_keys, req_context)
            self._complete_exact_chain_prepare(
                req_context, chain_qualified, result
            )
            self._stamp_score_residencies(result)
            return result

        eligible_new = [
            key
            for key in original_keys
            if self._policy.get(key) is None
            and (
                self.counts is None
                or self.counts.get(key, 0) >= self.store_threshold
            )
        ]
        free_slots = self._get_num_free_blocks()
        pressure_slots = max(0, len(eligible_new) - free_slots)
        threshold_surviving = [
            key
            for key in original_keys
            if self.counts is None
            or self.counts.get(key, 0) >= self.store_threshold
        ]
        preview = self._peek_admission_victims(
            pressure_slots, set(threshold_surviving)
        )
        rejected: set[OffloadKey] = set()
        victim_index = 0
        for candidate in eligible_new:
            candidate_frequency = self._admission_frequencies.get(candidate)
            (
                candidate_cost,
                candidate_cost_origin,
                candidate_cost_available_ns,
                candidate_cost_publication_ns,
            ) = self._admission_cost(candidate, req_context)
            if free_slots > 0:
                free_slots -= 1
                self._record_admission_decision(
                    req_context=req_context,
                    candidate=candidate,
                    victim=None,
                    action="admit",
                    reason="free_slot",
                    candidate_frequency=candidate_frequency,
                    candidate_cost=candidate_cost,
                    candidate_cost_origin=candidate_cost_origin,
                    candidate_cost_available_ns=candidate_cost_available_ns,
                    candidate_cost_publication_ns=candidate_cost_publication_ns,
                    victim_frequency=None,
                    victim_cost=None,
                    victim_cost_origin="not_applicable",
                    victim_cost_available_ns=None,
                    victim_cost_publication_ns=None,
                )
                continue

            victim = (
                preview[victim_index]
                if preview is not None and victim_index < len(preview)
                else None
            )
            victim_frequency = (
                self._admission_frequencies.get(victim)
                if victim is not None
                else None
            )
            victim_cost_record = (
                self._admission_resident_cost_per_byte.get(victim)
                if victim is not None
                else None
            )
            victim_cost = victim_cost_record[0] if victim_cost_record else None
            victim_cost_origin = (
                victim_cost_record[1] if victim_cost_record else "unavailable"
            )
            victim_cost_available_ns = (
                victim_cost_record[2] if victim_cost_record else None
            )
            victim_cost_publication_ns = (
                victim_cost_record[3] if victim_cost_record else None
            )
            candidate_features = (
                candidate_frequency is not None and candidate_cost is not None
            )
            victim_features = (
                victim is not None
                and victim_frequency is not None
                and victim_cost is not None
            )
            if not candidate_features or not victim_features:
                if victim is not None:
                    victim_index += 1
                self._record_admission_decision(
                    req_context=req_context,
                    candidate=candidate,
                    victim=victim,
                    action="admit",
                    reason="compulsory_fallback_unavailable_features",
                    candidate_frequency=candidate_frequency,
                    candidate_cost=candidate_cost,
                    candidate_cost_origin=candidate_cost_origin,
                    candidate_cost_available_ns=candidate_cost_available_ns,
                    candidate_cost_publication_ns=candidate_cost_publication_ns,
                    victim_frequency=victim_frequency,
                    victim_cost=victim_cost,
                    victim_cost_origin=(
                        victim_cost_origin if victim is not None else "unavailable"
                    ),
                    victim_cost_available_ns=victim_cost_available_ns,
                    victim_cost_publication_ns=victim_cost_publication_ns,
                )
                continue
            candidate_utility = candidate_frequency * candidate_cost
            victim_utility = victim_frequency * victim_cost
            if candidate_utility > victim_utility:
                victim_index += 1
                action, reason = "admit", "candidate_utility_greater"
            else:
                rejected.add(candidate)
                action, reason = "reject", "candidate_utility_not_greater"
            self._record_admission_decision(
                req_context=req_context,
                candidate=candidate,
                victim=victim,
                action=action,
                reason=reason,
                candidate_frequency=candidate_frequency,
                candidate_cost=candidate_cost,
                candidate_cost_origin=candidate_cost_origin,
                candidate_cost_available_ns=candidate_cost_available_ns,
                candidate_cost_publication_ns=candidate_cost_publication_ns,
                victim_frequency=victim_frequency,
                victim_cost=victim_cost,
                victim_cost_origin=victim_cost_origin,
                victim_cost_available_ns=victim_cost_available_ns,
                victim_cost_publication_ns=victim_cost_publication_ns,
            )

        base_keys = [key for key in original_keys if key not in rejected]
        result = super().prepare_store(base_keys, req_context)
        if result is not None:
            for key in result.evicted_keys:
                self._admission_resident_cost_per_byte.pop(key, None)
            for key in result.keys_to_store:
                cost, origin, available_ns, publication_ns = self._admission_cost(
                    key, req_context
                )
                if cost is not None:
                    self._admission_resident_cost_per_byte[key] = (
                        cost,
                        origin,
                        available_ns if available_ns is not None else 0,
                        publication_ns if publication_ns is not None else 0,
                    )
        self._complete_exact_chain_prepare(req_context, chain_qualified, result)
        self._stamp_score_residencies(result)
        return result

    def _complete_exact_chain_prepare(
        self,
        req_context: ReqContext,
        qualified_keys: tuple[OffloadKey, ...],
        result: PrepareStoreOutput | None,
    ) -> None:
        if not qualified_keys:
            return
        admitted = (
            tuple(key for key in result.keys_to_store if key in qualified_keys)
            if result is not None
            else ()
        )
        self._pending_chain_admission_accounting["admitted_keys"] += len(admitted)
        if admitted:
            self._chain_admission_pending_stores[req_context.req_id] = admitted
        _, prior_keys, accesses = self._chain_admission_observations[
            self._chain_admission_identity(req_context)
        ]
        started_ns = time.monotonic_ns()
        started_cpu_ns = time.thread_time_ns()
        self._record_chain_admission(
            req_context=req_context,
            action="admit" if admitted else "no_store",
            reason=(
                "bounded_exact_chain_prefix_admitted"
                if admitted
                else "qualified_prefix_not_allocated"
            ),
            identity=self._chain_admission_identity(req_context),
            observed_accesses=accesses,
            observed_keys=prior_keys,
            qualified_keys=qualified_keys,
            admitted_keys=admitted,
            started_ns=started_ns,
            started_cpu_ns=started_cpu_ns,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        super().complete_store(keys, req_context, success)
        pending = self._chain_admission_pending_stores.pop(req_context.req_id, ())
        completed = tuple(key for key in keys if key in pending)
        if success:
            self._pending_chain_admission_accounting["stored_keys"] += len(
                completed
            )
            missing = tuple(key for key in pending if key not in completed)
            self._pending_chain_admission_accounting["failed_store_keys"] += len(
                missing
            )
        else:
            self._pending_chain_admission_accounting["failed_store_keys"] += len(
                pending
            )
            missing = pending
        if pending:
            identity = self._chain_admission_identity(req_context)
            started_ns = time.monotonic_ns()
            started_cpu_ns = time.thread_time_ns()
            self._record_chain_admission(
                req_context=req_context,
                action="store_complete" if success and not missing else "store_failed",
                reason=(
                    "exact_chain_store_completed"
                    if success and not missing
                    else (
                        "partial_exact_chain_store_completion"
                        if success
                        else "exact_chain_store_failed"
                    )
                ),
                identity=identity,
                observed_accesses=(
                    self._chain_admission_observations[identity][2]
                    if identity in self._chain_admission_observations
                    else 0
                ),
                observed_keys=pending,
                qualified_keys=pending,
                admitted_keys=completed,
                started_ns=started_ns,
                started_cpu_ns=started_cpu_ns,
            )
        if not success:
            for key in keys:
                self._admission_resident_cost_per_byte.pop(key, None)
                # Native failed completion removes only pending stores. A
                # ready publication can survive a duplicate/late failure;
                # retain its version instead of invalidating unrelated KV.
                if self._policy.get(key) is None:
                    getattr(self, "_score_residencies", {}).pop(key, None)

    def drain_admission_records(self) -> list[dict[str, object]]:
        records = self._pending_admission_records
        self._pending_admission_records = []
        return records

    def drain_admission_accounting(self) -> dict[str, int]:
        accounting = self._pending_admission_accounting
        self._pending_admission_accounting = {
            "admitted": 0,
            "rejected": 0,
            "compulsory_fallbacks": 0,
        }
        return accounting

    def drain_chain_admission_records(self) -> list[dict[str, object]]:
        records = self._pending_chain_admission_records
        self._pending_chain_admission_records = []
        return records

    def drain_chain_admission_accounting(self) -> dict[str, int]:
        accounting = self._pending_chain_admission_accounting
        self._pending_chain_admission_accounting = {
            key: 0 for key in accounting
        }
        return accounting

    @override
    def reset_cache(self) -> None:
        super().reset_cache()
        getattr(self, "_score_residencies", {}).clear()
        # Keep _score_residency_sequence monotonic across reset. Reusing a
        # version would let a delayed pre-reset score match a new publication
        # of the same key after the cache has been emptied and repopulated.
        self._admission_frequencies.clear()
        self._admission_observations.clear()
        self._admission_resident_cost_per_byte.clear()
        self._pending_admission_records.clear()
        for key in self._pending_admission_accounting:
            self._pending_admission_accounting[key] = 0
        self._admission_decision_sequence = 0
        self._admission_observation_sequence = 0
        self._chain_admission_observations.clear()
        self._chain_admission_pending_stores.clear()
        self._pending_chain_admission_records.clear()
        for key in self._pending_chain_admission_accounting:
            self._pending_chain_admission_accounting[key] = 0


class PreservationOffloadingManager(
    StagingProtectionMixin,
    LifecycleReleaseMixin,
    FrequencyCostAdmissionMixin,
    CPUOffloadingManager,
):
    """Stock LRU/ARC manager with authoritative eviction notifications.

    Preservation events must not silently disappear for the stock policies.
    This keeps CPUOffloadingManager's policy and transfer semantics intact and
    only buffers the exact ``PrepareStoreOutput.evicted_keys`` values that the
    shared connector scheduler already consumes for semantic policy runs.
    """

    def refresh_wait_hint_cpu_recency(self, keys: Collection[OffloadKey]) -> int:
        """Refresh ready ARC residents without promoting speculative accesses.

        Hints are advice, not actual cache hits: preserve queue membership,
        ghost history, the adaptive target, ownership and eviction eligibility.
        Return zero for unsupported policies rather than applying hit semantics.
        """
        from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
        policy = self._policy
        if not isinstance(policy, ARCCachePolicy):
            return 0
        refreshed = 0
        for key in reversed(list(keys)):
            block = policy.get(key)
            if block is None or not block.is_ready:
                continue
            queue = policy.t1 if key in policy.t1 else policy.t2
            queue.move_to_end(key)
            refreshed += 1
        return refreshed

    def __init__(
        self,
        *args,
        admission_mode: object = "disabled",
        admission_cost_per_byte: object = None,
        admission_default_cost_per_byte: object = 1.0,
        admission_namespace: object = None,
        chain_admission_max_prefix_blocks: object = 0,
        shared_prefetch_scorer: bool = False,
        method: str = _DEFAULT_METHOD,
        rank_ema_mode: str = "default",
        ema_rank_weight_floor: float = 0.0,
        decision_feature_trace: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_staging_protection_contract()
        self._init_lifecycle_contract()
        self._init_admission_contract(
            admission_mode=admission_mode,
            admission_cost_per_byte=admission_cost_per_byte,
            admission_default_cost_per_byte=admission_default_cost_per_byte,
            admission_namespace=admission_namespace,
            chain_admission_max_prefix_blocks=chain_admission_max_prefix_blocks,
            max_tracker_size=self.max_tracker_size,
        )
        self._pending_evicted_keys: list[OffloadKey] = []
        self._shared_prefetch_scorer = bool(shared_prefetch_scorer)
        self._method = method
        self._rank_ema_mode = rank_ema_mode
        self._ema_rank_weight_floor = validate_ema_rank_weight_floor(
            ema_rank_weight_floor
        )
        self._decision_feature_trace_enabled = bool(decision_feature_trace)
        self._pending_decision_feature_records: list[dict[str, object]] = []
        self._decision_sequence = 0
        self.relevance_ema: dict[str, dict[OffloadKey, float]] = {}
        self._relevance_updated_ns: dict[str, dict[OffloadKey, int]] = {}
        self._latest_relevance: dict[str, dict[OffloadKey, float]] = {}

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        instrumentation_started_ns = time.monotonic_ns()
        instrumentation_started_cpu_ns = time.thread_time_ns()
        candidates: list[tuple[OffloadKey, object]] = []
        candidate_segments: dict[OffloadKey, str] = {}
        policy_state: dict[str, object] = {}
        if self._decision_feature_trace_enabled:
            protected = set(keys)
            policy = self._policy
            if isinstance(getattr(policy, "evictable_blocks", None), OrderedDict):
                candidates = [
                    (key, policy.blocks[key])
                    for key in policy.evictable_blocks
                    if key not in protected
                ]
                candidate_segments = {key: "lru" for key, _ in candidates}
                policy_state = {"evictable_count": len(candidates)}
            elif isinstance(getattr(policy, "t1", None), OrderedDict):
                for segment in ("t1", "t2"):
                    for key, block in getattr(policy, segment).items():
                        if block.ref_cnt == 0 and key not in protected:
                            candidates.append((key, block))
                            candidate_segments[key] = segment
                policy_state = {
                    "target_t1_size": float(policy.target_t1_size),
                    "t1_size": len(policy.t1),
                    "t2_size": len(policy.t2),
                }
        instrumentation_snapshot_completed_ns = time.monotonic_ns()
        instrumentation_snapshot_completed_cpu_ns = time.thread_time_ns()
        policy_started_ns = time.monotonic_ns()
        policy_started_cpu_ns = time.thread_time_ns()
        result = super().prepare_store(keys, req_context)
        policy_completed_ns = time.monotonic_ns()
        policy_completed_cpu_ns = time.thread_time_ns()
        if result is not None and result.evicted_keys:
            self._pending_evicted_keys.extend(result.evicted_keys)
            if self._decision_feature_trace_enabled:
                selected = set(result.evicted_keys)
                ranked = list(result.evicted_keys) + [
                    key for key, _ in candidates if key not in selected
                ]
                event_started_ns = time.monotonic_ns()
                event_started_cpu_ns = time.thread_time_ns()
                records = build_decision_feature_records(
                    context=_request_provenance(
                        req_context.req_id,
                        getattr(req_context, "kv_transfer_params", None),
                    ),
                    decision_sequence=self._decision_sequence,
                    candidates=candidates,
                    ranked_keys=ranked,
                    selected_keys=selected,
                    relevance={},
                    latest_relevance={},
                    sink=self._pending_decision_feature_records.extend,
                    decision_timestamp_ns=policy_completed_ns,
                    scoring_cpu_time_ns=None,
                    policy_cpu_time_ns=(
                        policy_completed_cpu_ns - policy_started_cpu_ns
                    ),
                    critical_path_wall_time_ns=(
                        policy_completed_ns - policy_started_ns
                    ),
                    synchronization_wait_ns=0,
                    synchronization_status="no_explicit_synchronization",
                    policy=self._policy.__class__.__name__,
                    policy_state=policy_state,
                    candidate_policy_segments=candidate_segments,
                )
                event_completed_ns = time.monotonic_ns()
                event_completed_cpu_ns = time.thread_time_ns()
                instrumentation_cpu_time_ns = (
                    instrumentation_snapshot_completed_cpu_ns
                    - instrumentation_started_cpu_ns
                    + event_completed_cpu_ns
                    - event_started_cpu_ns
                )
                instrumentation_wall_time_ns = (
                    instrumentation_snapshot_completed_ns
                    - instrumentation_started_ns
                    + event_completed_ns
                    - event_started_ns
                )
                for record in records:
                    record["instrumentation_cpu_time_ns"] = (
                        instrumentation_cpu_time_ns
                    )
                    record["instrumentation_wall_time_ns"] = (
                        instrumentation_wall_time_ns
                    )
                    record["critical_path_wall_time_ns"] = (
                        event_completed_ns - instrumentation_started_ns
                    )
                self._decision_sequence += 1
        return result

    def drain_decision_feature_records(self) -> list[dict[str, object]]:
        records = self._pending_decision_feature_records
        self._pending_decision_feature_records = []
        return records

    def drain_evicted_keys(self) -> list[OffloadKey]:
        keys = self._pending_evicted_keys
        self._pending_evicted_keys = []
        if keys:
            for values in self.relevance_ema.values():
                for key in keys:
                    values.pop(key, None)
            for values in self._latest_relevance.values():
                for key in keys:
                    values.pop(key, None)
            for values in self._relevance_updated_ns.values():
                for key in keys:
                    values.pop(key, None)
        return keys

    @profiled('scoring.ema_apply')
    def apply_relevance_updates(
        self,
        updates: Mapping[str, Mapping[OffloadKey, tuple[float, float, float]]],
        latest_relevance: Mapping[str, Mapping[OffloadKey, float]] | None = None,
    ) -> None:
        """Track prefetch scores without changing native ARC/LRU eviction."""
        if not self._shared_prefetch_scorer:
            return
        if self._decision_feature_trace_enabled and latest_relevance is not None:
            for method, values in latest_relevance.items():
                self._latest_relevance.setdefault(method, {}).update(
                    {key: float(value) for key, value in values.items()}
                )
        for method, method_updates in updates.items():
            ema = self.relevance_ema.setdefault(method, {})
            updated = self._relevance_updated_ns.setdefault(method, {})
            observed_ns = time.monotonic_ns()
            for key, (decay, offset, unseen_value) in method_updates.items():
                if self._policy.get(key) is None:
                    continue
                previous = ema.get(key)
                ema[key] = (
                    unseen_value if previous is None else decay * previous + offset
                )
                updated[key] = observed_ns

    def update_relevance(
        self, scores: Mapping[str, Mapping[str, Sequence[tuple[OffloadKey, float]]]]
    ) -> None:
        if not self._shared_prefetch_scorer:
            return
        self.apply_relevance_updates(
            compose_relevance_updates(
                scores, self._rank_ema_mode, self._ema_rank_weight_floor
            ),
            compose_latest_relevance(scores),
        )

    def ranked_keys(self, method: str) -> list[tuple[OffloadKey, float]]:
        return sorted(
            self.relevance_ema.get(method, {}).items(),
            key=lambda item: (-item[1], item[0]),
        )

    def top_relevant_keys(
        self,
        candidate_keys: list[OffloadKey],
        k: int,
        method: str = _DEFAULT_METHOD,
        *,
        max_age_ns: int | None = None,
        now_ns: int | None = None,
    ) -> list[OffloadKey]:
        if not self._shared_prefetch_scorer or k <= 0:
            return []
        ema = self.relevance_ema.get(method, {})
        updated = self._relevance_updated_ns.get(method, {})
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        scored = [
            (key, ema[key])
            for key in candidate_keys
            if key in ema
            and (
                max_age_ns is None
                or (
                    key in updated
                    and 0 <= now_ns - updated[key] <= max_age_ns
                )
            )
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [key for key, _ in scored[:k]]

    def reset_cache(self) -> None:
        super().reset_cache()
        self._reset_staging_protection_contract()
        self._reset_lifecycle_contract()
        self._pending_evicted_keys.clear()
        self._pending_decision_feature_records.clear()
        self._decision_sequence = 0
        self.relevance_ema.clear()
        self._latest_relevance.clear()
        self._relevance_updated_ns.clear()


@profiled('scoring.ema_compose')
def compose_relevance_updates(
    scores: Mapping[str, Mapping[str, Sequence[tuple[OffloadKey, float]]]],
    mode: str = "default",
    ema_rank_weight_floor: float = 0.0,
) -> ComposedRelevanceUpdates:
    """Compose request-ranked EMA observations into one affine update per key.

    Each tuple is ``(decay, offset, unseen_value)``. Existing EMA state becomes
    ``decay * previous + offset``; a key without state takes ``unseen_value``
    to preserve the historical first-observation rule.
    """
    if mode not in ("default", "fast"):
        raise ValueError(f"Unknown rank_ema_mode {mode!r}")
    floor = validate_ema_rank_weight_floor(ema_rank_weight_floor)
    composed: ComposedRelevanceUpdates = {}
    weights_by_length: dict[int, list[float]] = {}
    for method, per_req in scores.items():
        accumulators: dict[OffloadKey, list[float]] = {}
        for ranked in per_req.values():
            n = len(ranked)
            weights = weights_by_length.get(n)
            if weights is None:
                weights = [
                    _rank_ema_weight(rank, n, floor)
                    for rank in range(n)
                ]
                weights_by_length[n] = weights
            if mode == "fast":
                # Keep the same operation order as the legacy loop while
                # avoiding a zip/unpack iterator and repeated global lookups
                # on the saturated single-worker path. The fast worker has
                # already produced a stable score order, so this is the only
                # remaining request-by-request composition work.
                ranked_iter = enumerate(ranked)
                for rank, (key, new_score) in ranked_iter:
                    weight = weights[rank]
                    keep = 1.0 - weight
                    previous = accumulators.get(key)
                    if previous is None:
                        accumulators[key] = [keep, weight * new_score, new_score]
                    else:
                        previous[0] *= keep
                        previous[1] = keep * previous[1] + weight * new_score
                        previous[2] = keep * previous[2] + weight * new_score
                continue
            for (key, new_score), weight in zip(ranked, weights):
                keep = 1.0 - weight
                previous = accumulators.get(key)
                if previous is None:
                    accumulators[key] = [keep, weight * new_score, new_score]
                else:
                    previous[0] *= keep
                    previous[1] = keep * previous[1] + weight * new_score
                    previous[2] = keep * previous[2] + weight * new_score
        if accumulators:
            composed[method] = {
                key: (values[0], values[1], values[2])
                for key, values in accumulators.items()
            }
    return composed


def compose_latest_relevance(
    scores: Mapping[str, Mapping[str, Sequence[tuple[OffloadKey, float]]]],
) -> LatestRelevance:
    """Retain the authoritative raw score last observed for each key."""
    latest: LatestRelevance = {}
    for method, per_req in scores.items():
        method_latest = latest.setdefault(method, {})
        for ranked in per_req.values():
            for key, score in ranked:
                method_latest[key] = float(score)
    return latest


class SemanticOffloadingManager(
    StagingProtectionMixin,
    LifecycleReleaseMixin,
    FrequencyCostAdmissionMixin,
    CPUOffloadingManager,
):
    def __init__(
        self,
        num_blocks: int,
        enable_events: bool = False,
        store_threshold: int = 0,
        max_tracker_size: int = 64_000,
        grace_window_blocks: int = 0,
        eviction_mode: str = "blend",
        chain_aware: bool = False,
        session_aware: bool = False,
        session_bonus_half_life: int = 0,
        prefix_aware: bool = False,
        ema_rank_weight_floor: float = 0.0,
        admission_mode: object = "disabled",
        admission_cost_per_byte: object = None,
        admission_default_cost_per_byte: object = 1.0,
        admission_namespace: object = None,
        chain_admission_max_prefix_blocks: object = 0,
        future_reuse_feedback: bool = False,
        future_reuse_weight: float = 0.25,
        decision_feature_trace: bool = False,
        method: str = _DEFAULT_METHOD,
        alpha: float = DEFAULT_POLICY_ALPHA,
    ) -> None:
        super().__init__(
            num_blocks=num_blocks,
            cache_policy="lru",
            enable_events=enable_events,
            store_threshold=store_threshold,
            max_tracker_size=max_tracker_size,
        )
        self._init_staging_protection_contract()
        self._init_lifecycle_contract()
        self._init_admission_contract(
            admission_mode=admission_mode,
            admission_cost_per_byte=admission_cost_per_byte,
            admission_default_cost_per_byte=admission_default_cost_per_byte,
            admission_namespace=admission_namespace,
            chain_admission_max_prefix_blocks=chain_admission_max_prefix_blocks,
            max_tracker_size=max_tracker_size,
        )
        if not isinstance(decision_feature_trace, bool):
            raise TypeError("decision_feature_trace must be a boolean")
        # method -> OffloadKey -> EMA-smoothed relevance score (Step 1.3).
        # Passed into SemanticPolicy by reference -- update_relevance()
        # mutates this dict in place and evict() reads it live (Step 1.4).
        self.relevance_ema: dict[str, dict[OffloadKey, float]] = {}
        self._relevance_updated_ns: dict[str, dict[OffloadKey, int]] = {}
        # Raw scores are an optional trace feature. Do not allocate or retain
        # this second score history in the default serving path.
        self._latest_relevance: LatestRelevance | None = (
            {} if decision_feature_trace else None
        )
        # This separate utility map is updated only from a later, disjoint
        # future window; the current request's preservation event is never
        # used as its own training label.
        self.future_reuse_ema: dict[OffloadKey, float] = {}
        self._future_reuse_feedback = future_reuse_feedback
        self._future_feedback_windows: set[object] = set()
        self._max_tracker_size = max_tracker_size
        self._ema_rank_weight_floor = validate_ema_rank_weight_floor(
            ema_rank_weight_floor
        )
        # Real eviction notifications for the worker's durable_summaries
        # (issues log entries #62-64) -- see prepare_store() below. Drained
        # by SemanticOffloadingConnectorScheduler.build_connector_meta()
        # each step, the same accumulate-then-drain pattern already used for
        # _pending_job_keys.
        self._pending_evicted_keys: list[OffloadKey] = []
        self._pending_decision_feature_records: list[dict[str, object]] = []
        self._decision_feature_trace_enabled = decision_feature_trace
        self._policy = SemanticPolicy(
            cache_capacity=num_blocks,
            relevance_ema=self.relevance_ema,
            method=method,
            alpha=alpha,
            grace_window_blocks=grace_window_blocks,
            mode=eviction_mode,
            chain_aware=chain_aware,
            session_aware=session_aware,
            session_bonus_half_life=session_bonus_half_life,
            prefix_aware=prefix_aware,
            future_reuse_ema=self.future_reuse_ema,
            future_reuse_feedback=future_reuse_feedback,
            future_reuse_weight=future_reuse_weight,
            latest_relevance=(
                self._latest_relevance if decision_feature_trace else None
            ),
            decision_feature_sink=(
                self._record_decision_feature_records
                if decision_feature_trace
                else None
            ),
        )

    def _record_decision_feature_records(
        self, records: list[dict[str, object]]
    ) -> None:
        self._pending_decision_feature_records.extend(records)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        """Wraps the base CPUOffloadingManager.prepare_store() purely to
        observe `PrepareStoreOutput.evicted_keys` -- the base class already
        computes exactly which keys the real CachePolicy just evicted to
        make room for this store (manager.py's own `evict()` call), it just
        never surfaced that anywhere. Buffered here (scheduler-side) and
        drained by the connector scheduler into per-step metadata, mirroring
        _pending_job_keys' accumulate-then-drain pattern, so the worker can
        learn precisely which blocks are gone instead of approximating with
        a FIFO cap (issues log entries #62-64 -- that cap remains as a
        defensive backstop in worker.py, not the primary mechanism anymore)."""
        if self._decision_feature_trace_enabled:
            self._policy.set_decision_feature_context(
                _request_provenance(
                    req_context.req_id,
                    getattr(req_context, "kv_transfer_params", None),
                )
            )
        try:
            result = super().prepare_store(keys, req_context)
        finally:
            if self._decision_feature_trace_enabled:
                self._policy.set_decision_feature_context(None)
        if result is not None and result.evicted_keys:
            self._pending_evicted_keys.extend(result.evicted_keys)
        return result

    def drain_decision_feature_records(self) -> list[dict[str, object]]:
        """Pop decision-time records for the scheduler's current metadata."""
        records = self._pending_decision_feature_records
        self._pending_decision_feature_records = []
        return records

    def drain_evicted_keys(self) -> list[OffloadKey]:
        """Pop and return all evicted keys accumulated since the last drain
        (per-step, from SemanticOffloadingConnectorScheduler.build_connector_meta)."""
        keys = self._pending_evicted_keys
        self._pending_evicted_keys = []
        for key in keys:
            for values in self.relevance_ema.values():
                values.pop(key, None)
            for values in self._relevance_updated_ns.values():
                values.pop(key, None)
            latest = self._latest_relevance
            if latest is not None:
                for values in latest.values():
                    values.pop(key, None)
        return keys

    @override
    def reset_cache(self) -> None:
        super().reset_cache()
        self._reset_staging_protection_contract()
        self._reset_lifecycle_contract()
        self._pending_evicted_keys.clear()
        self._pending_decision_feature_records.clear()
        self.future_reuse_ema.clear()
        self._relevance_updated_ns.clear()
        self._future_feedback_windows.clear()

    def update_relevance(
        self, scores: dict[str, dict[str, list[tuple[OffloadKey, float]]]]
    ) -> None:
        """Fold in this step's worker-reported scores via an EMA, so one
        noisy step can't thrash the cache (plan's Step 1.3 requirement).
        `scores` is method -> req_id -> ranked [(key, score), ...], already
        sorted highest-score-first per query (worker.py's `_on_query_captured`);
        req_id isn't needed for the EMA itself (relevance is per-block, not
        per-requester) so it's flattened away here.

        Every candidate in `ranked` is touched (not just a top-K slice) --
        an earlier `_TOP_M` cap that scored/updated only the top-8 per query
        was removed because most resident blocks would then never earn any
        EMA entry at all, regardless of pool size (issues log entries
        #29-31). But touching every candidate with the *same* flat alpha
        turned out to have its own failure mode (entry #60): a topically
        unrelated distractor query ranks the needle near the bottom of its
        own candidate list and, at full alpha, actively drags an
        already-high needle score back down -- a handful of irrelevant
        queries erase one strong probe's signal well before real capacity
        pressure ever forces an eviction decision. Fix: scale the per-key
        weight by that key's *rank within this query's own ranked list*
        (not an absolute top-K cutoff, so coverage is preserved) -- a
        candidate this query considers most-relevant gets close to the
        full ceiling weight, one it considers least-relevant gets a weight
        near zero, so it keeps its prior score rather than being pulled
        toward this query's near-irrelevant one."""
        if not getattr(self, "_decision_feature_trace_enabled", False):
            latest_relevance = None
        else:
            latest_relevance = getattr(self, "_latest_relevance", None)
            if latest_relevance is None:
                latest_relevance = self._latest_relevance = {}
        for method, per_req in scores.items():
            ema = self.relevance_ema.setdefault(method, {})
            updated_by_method = getattr(self, "_relevance_updated_ns", None)
            if updated_by_method is None:
                updated_by_method = self._relevance_updated_ns = {}
            updated = updated_by_method.setdefault(method, {})
            latest = (
                latest_relevance.setdefault(method, {})
                if latest_relevance is not None
                else None
            )
            for ranked in per_req.values():
                n = len(ranked)
                for rank, (key, new_score) in enumerate(ranked):
                    # Worker metadata can arrive after a key was evicted or
                    # after reset. Never resurrect stale relevance state.
                    policy = getattr(self, "_policy", None)
                    if policy is not None and policy.get(key) is None:
                        continue
                    if latest is not None:
                        latest[key] = float(new_score)
                    weight = _rank_ema_weight(
                        rank, n, getattr(self, "_ema_rank_weight_floor", 0.0)
                    )
                    prev = ema.get(key)
                    ema[key] = (
                        new_score
                        if prev is None
                        else weight * new_score + (1 - weight) * prev
                    )
                    updated[key] = time.monotonic_ns()

    def update_future_reuse_labels(self, batch) -> None:
        """Fold one immutable, validated future-window batch into utility EMA.

        ``FutureReuseLabelBatch`` is intentionally imported lazily so the
        serving policy package does not depend on the offline extractor. The
        batch itself owns canonical event identity and disjoint train/test
        provenance; this method only applies its already-validated training
        entries atomically.
        """
        if not self._future_reuse_feedback:
            return
        from semantic_offload.future_reuse import FutureReuseLabelBatch

        if type(batch) is not FutureReuseLabelBatch:
            raise TypeError("future-reuse updates require a validated label batch")
        try:
            batch._validate_seal()
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("future-reuse batch seal is invalid") from exc
        application_id = batch.application_id
        if application_id in self._future_feedback_windows:
            raise ValueError(f"future label batch already applied: {application_id}")
        if batch.window_identity in self._future_feedback_windows:
            raise ValueError(
                f"future label window already applied: {batch.window_identity!r}"
            )
        items = list(batch.entries)
        for key, reused in items:
            if not isinstance(reused, bool):
                raise TypeError("future-reuse labels must be boolean")
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError(
                    "future-reuse keys must be hashable OffloadKeys"
                ) from exc
        updates = {}
        for key, reused in items:
            value = 1.0 if reused else 0.0
            previous = self.future_reuse_ema.get(key)
            updates[key] = (
                value
                if previous is None
                else _EMA_ALPHA * value + (1.0 - _EMA_ALPHA) * previous
            )
        self.future_reuse_ema.update(updates)
        self._future_feedback_windows.add(application_id)
        self._future_feedback_windows.add(batch.window_identity)
        excess = len(self.future_reuse_ema) - self._max_tracker_size
        if excess > 0:
            for key in list(self.future_reuse_ema)[:excess]:
                del self.future_reuse_ema[key]

    @profiled('scoring.ema_apply')
    def apply_relevance_updates(
        self,
        updates: Mapping[str, Mapping[OffloadKey, tuple[float, float, float]]],
        latest_relevance: Mapping[str, Mapping[OffloadKey, float]] | None = None,
    ) -> None:
        """Apply precomposed observations and optional raw-score metadata."""
        if (
            getattr(self, "_decision_feature_trace_enabled", False)
            and latest_relevance is not None
        ):
            target = getattr(self, "_latest_relevance", None)
            if target is None:
                target = self._latest_relevance = {}
            for method, method_latest in latest_relevance.items():
                target.setdefault(method, {}).update(
                    {key: float(value) for key, value in method_latest.items()}
                )
        for method, method_updates in updates.items():
            ema = self.relevance_ema.setdefault(method, {})
            updated_by_method = getattr(self, "_relevance_updated_ns", None)
            if updated_by_method is None:
                updated_by_method = self._relevance_updated_ns = {}
            updated = updated_by_method.setdefault(method, {})
            observed_ns = time.monotonic_ns()
            for key, (decay, offset, unseen_value) in method_updates.items():
                policy = getattr(self, "_policy", None)
                if policy is not None and policy.get(key) is None:
                    continue
                previous = ema.get(key)
                ema[key] = (
                    unseen_value if previous is None else decay * previous + offset
                )
                updated[key] = observed_ns

    def ranked_keys(self, method: str) -> list[tuple[OffloadKey, float]]:
        """Current EMA-ranked relevance for a scoring method, highest first.
        Read-only introspection for Step 1.3's accept check; Step 1.4 is
        where SemanticPolicy.evict() actually consumes this."""
        ema = self.relevance_ema.get(method, {})
        return sorted(ema.items(), key=lambda kv: kv[1], reverse=True)

    def top_relevant_keys(
        self,
        candidate_keys: list[OffloadKey],
        k: int,
        method: str = _DEFAULT_METHOD,
        *,
        max_age_ns: int | None = None,
        now_ns: int | None = None,
    ) -> list[OffloadKey]:
        """Step 1.5: among `candidate_keys` (typically one request's own
        chain), the `k` with the highest current relevance score -- unlike
        `ranked_keys`, scoped to a caller-given candidate set rather than
        every key ever scored, and silently skips unscored keys (a
        preempted request's own blocks may not have earned a score yet;
        prefetching an arbitrary unscored block isn't better-informed than
        picking any other block at random, so there's nothing to prefer)."""
        if k <= 0:
            return []
        ema = self.relevance_ema.get(method, {})
        updated = getattr(self, "_relevance_updated_ns", {}).get(method, {})
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        scored = [
            (key, ema[key])
            for key in candidate_keys
            if key in ema
            and (
                max_age_ns is None
                or (
                    key in updated
                    and 0 <= now_ns - updated[key] <= max_age_ns
                )
            )
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [key for key, _ in scored[:k]]
