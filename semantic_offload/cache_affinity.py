# SPDX-License-Identifier: Apache-2.0
"""Opt-in cache-affinity request ordering for vLLM V1.

This module is intentionally an out-of-tree scheduler plugin.  It reorders only
the ordinary FCFS waiting queue using an exact local prefix-cache probe.  KV
policy, connector state, allocation, and preempted-request ordering remain
owned by vLLM's scheduler.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal import MultiModalRegistry
    from vllm.v1.core.kv_cache_interface import KVCacheConfig
    from vllm.v1.structured_output import StructuredOutputManager


CacheProbe = Callable[[Request], int]
_CONFIG_NAMESPACE = "semantic_offload.cache_affinity"
_MAX_WINDOW = 32
_MAX_WAIT_S = 0.2
_MAX_OVERTAKES = 8


@dataclass(frozen=True, slots=True)
class CacheAffinityConfig:
    """Compatibility-stable settings for the out-of-tree scheduler plugin."""

    enabled: bool = True
    window_size: int = 32
    max_wait_s: float = 0.2
    max_overtakes: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if isinstance(self.window_size, bool) or not isinstance(self.window_size, int):
            raise TypeError("window_size must be an integer")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.window_size > _MAX_WINDOW:
            raise ValueError(f"window_size must be <= {_MAX_WINDOW}")
        if (
            isinstance(self.max_wait_s, bool)
            or not isinstance(self.max_wait_s, (int, float))
            or not math.isfinite(float(self.max_wait_s))
        ):
            raise TypeError("max_wait_s must be a finite number")
        if self.max_wait_s < 0:
            raise ValueError("max_wait_s must be non-negative")
        if self.max_wait_s > _MAX_WAIT_S:
            raise ValueError(f"max_wait_s must be <= {_MAX_WAIT_S}")
        if isinstance(self.max_overtakes, bool) or not isinstance(
            self.max_overtakes, int
        ):
            raise TypeError("max_overtakes must be an integer")
        if self.max_overtakes < 1:
            raise ValueError("max_overtakes must be positive")
        if self.max_overtakes > _MAX_OVERTAKES:
            raise ValueError(f"max_overtakes must be <= {_MAX_OVERTAKES}")

    @classmethod
    def from_mapping(cls, raw: object | None) -> CacheAffinityConfig:
        """Parse an additional-config mapping, disabling malformed input."""
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            return cls(enabled=False)
        allowed = {"enabled", "window_size", "max_wait_s", "max_overtakes"}
        if any(key not in allowed for key in raw):
            return cls(enabled=False)
        try:
            return cls(**raw)
        except (TypeError, ValueError):
            return cls(enabled=False)


def _config_from_vllm_config(
    vllm_config: object, default: CacheAffinityConfig
) -> CacheAffinityConfig:
    """Read the package namespace, disabling malformed input."""
    additional = getattr(vllm_config, "additional_config", None)
    if additional in (None, {}):
        return default
    if not isinstance(additional, Mapping):
        return CacheAffinityConfig(enabled=False)
    if _CONFIG_NAMESPACE in additional:
        raw_config = additional[_CONFIG_NAMESPACE]
        return (
            CacheAffinityConfig(enabled=False)
            if raw_config is None
            else CacheAffinityConfig.from_mapping(raw_config)
        )
    package_config = additional.get("semantic_offload")
    if package_config is None:
        return default
    if not isinstance(package_config, Mapping):
        return CacheAffinityConfig(enabled=False)
    if "cache_affinity" not in package_config:
        return default
    raw_config = package_config["cache_affinity"]
    return (
        CacheAffinityConfig(enabled=False)
        if raw_config is None
        else CacheAffinityConfig.from_mapping(raw_config)
    )


class CacheAffinityRequestQueue(RequestQueue):
    """FCFS queue with bounded, cache-aware ordering of new arrivals.

    ``prepend_request`` is the scheduler's preemption path.  Those requests
    live in a sticky deque and are never reordered.  New arrivals remain in a
    sortable deque; only its bounded head window is probed and reordered.
    """

    def __init__(
        self,
        cache_probe: CacheProbe,
        *,
        window_size: int = 32,
        max_wait_s: float = 0.2,
        max_overtakes: int = 8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if max_wait_s < 0:
            raise ValueError("max_wait_s must be non-negative")
        if max_wait_s > _MAX_WAIT_S:
            raise ValueError(f"max_wait_s must be <= {_MAX_WAIT_S}")
        if max_overtakes < 1:
            raise ValueError("max_overtakes must be positive")
        if max_overtakes > _MAX_OVERTAKES:
            raise ValueError(f"max_overtakes must be <= {_MAX_OVERTAKES}")
        self._cache_probe = cache_probe
        self._window_size = window_size
        self._max_wait_s = max_wait_s
        self._max_overtakes = max_overtakes
        self._clock = clock
        self._sticky: deque[Request] = deque()
        self._sortable: deque[Request] = deque()
        self._enqueued_at: dict[str, float] = {}
        self._overtakes: dict[str, int] = {}
        self._dirty = False

    @staticmethod
    def _request_id(request: Request) -> str:
        return str(request.request_id)

    def add_request(self, request: Request) -> None:
        request_id = self._request_id(request)
        if request_id in self._enqueued_at:
            raise ValueError(f"request {request_id!r} is already queued")
        self._sortable.append(request)
        self._enqueued_at[request_id] = self._clock()
        self._overtakes[request_id] = 0
        self._dirty = True

    def prepend_request(self, request: Request) -> None:
        request_id = self._request_id(request)
        if request_id not in self._enqueued_at:
            self._enqueued_at[request_id] = self._clock()
            self._overtakes[request_id] = 0
        self._sticky.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(request)

    def _remove_bookkeeping(self, request: Request) -> None:
        request_id = self._request_id(request)
        self._enqueued_at.pop(request_id, None)
        self._overtakes.pop(request_id, None)

    def remove_request(self, request: Request) -> None:
        try:
            self._sticky.remove(request)
        except ValueError:
            self._sortable.remove(request)
        self._remove_bookkeeping(request)
        self._dirty = True

    def remove_requests(self, requests: Iterable[Request]) -> None:
        doomed = list(requests)
        # The scheduler sends the same abort set to multiple waiting queues.
        # Like native bulk removal, absent/repeated requests are harmless.
        removed = [
            request
            for queue in (self._sticky, self._sortable)
            for request in queue
            if request in doomed
        ]
        if not removed:
            return
        self._sticky = deque(
            request for request in self._sticky if request not in doomed
        )
        self._sortable = deque(
            request for request in self._sortable if request not in doomed
        )
        for request in removed:
            self._remove_bookkeeping(request)
        self._dirty = True

    def _cached_blocks(self, request: Request) -> int:
        try:
            cached_blocks = int(self._cache_probe(request))
        except Exception:  # noqa: BLE001 - optional ordering must fail closed
            # Queue ordering is an optimization.  A malformed or unavailable
            # probe must preserve the ordinary FCFS order.
            return 0
        return max(cached_blocks, 0)

    def reorder(
        self,
        *,
        now: float | None = None,
        force: bool = False,
        window_size: int | None = None,
    ) -> None:
        """Reorder a bounded head window and account for overtakes."""
        if not self._sortable or (not self._dirty and not force):
            return
        if now is None:
            now = self._clock()
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be positive")

        effective_window = min(
            self._window_size,
            window_size if window_size is not None else self._window_size,
        )
        head = list(islice(self._sortable, effective_window))
        raw_scores: list[tuple[Request, int, int]] = []
        for position, request in enumerate(head):
            raw_scores.append((request, self._cached_blocks(request), position))

        if force:
            for request, cached_blocks, _ in raw_scores:
                if any(
                    other_blocks > cached_blocks for _, other_blocks, _ in raw_scores
                ):
                    request_id = self._request_id(request)
                    self._overtakes[request_id] = min(
                        self._max_overtakes,
                        self._overtakes.get(request_id, 0) + 1,
                    )

        scored: list[tuple[Request, int, bool, int]] = []
        for request, cached_blocks, position in raw_scores:
            request_id = self._request_id(request)
            waited = now - self._enqueued_at.get(request_id, now)
            aged = (
                waited >= self._max_wait_s
                or self._overtakes.get(request_id, 0) >= self._max_overtakes
            )
            scored.append((request, cached_blocks, aged, position))

        scored.sort(
            key=lambda item: (
                0 if item[2] else 1,
                0 if item[2] else -item[1],
                item[3],
                self._request_id(item[0]),
            )
        )
        reordered = [item[0] for item in scored]
        for position, request in enumerate(reordered):
            self._sortable[position] = request
        self._dirty = False

    def pop_request(self) -> Request:
        if self._sticky:
            request = self._sticky.popleft()
        else:
            self.reorder()
            request = self._sortable.popleft()
        self._remove_bookkeeping(request)
        return request

    def peek_request(self) -> Request:
        if self._sticky:
            return self._sticky[0]
        self.reorder()
        if not self._sortable:
            raise IndexError("peek from an empty queue")
        return self._sortable[0]

    def __bool__(self) -> bool:
        return bool(self._sticky or self._sortable)

    def __len__(self) -> int:
        return len(self._sticky) + len(self._sortable)

    def __iter__(self) -> Iterator[Request]:
        self.reorder()
        return iter((*self._sticky, *self._sortable))


def _local_cached_blocks(scheduler: Scheduler, request: Request) -> int:
    """Return full local GPU prefix blocks without connector/offload lookup."""
    manager = scheduler.kv_cache_manager
    if not manager.prefix_cache_lookup_enabled(request):
        return 0
    max_cache_hit_length = request.num_tokens - 1
    if max_cache_hit_length <= 0:
        return 0
    # The coordinator lookup is the cache manager's read-only local prefix
    # oracle.  Do not call get_computed_blocks(): it records stats and can emit
    # cache events, which would make queue ordering observable.
    _, num_cached_tokens, _ = manager.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
    return num_cached_tokens // scheduler.block_size


class CacheAffinityScheduler(Scheduler):
    """Opt-in V1 scheduler plugin for local prefix-cache affinity.

    Load with ``--scheduler-cls semantic_offload.cache_affinity.CacheAffinityScheduler``.
    The plugin is deliberately a no-op for PRIORITY scheduling; the stock
    scheduler remains the complete behavioral oracle in that mode.
    """

    AFFINITY_CONFIG = CacheAffinityConfig()

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry | None = None,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        kwargs = {
            "vllm_config": vllm_config,
            "kv_cache_config": kv_cache_config,
            "structured_output_manager": structured_output_manager,
            "block_size": block_size,
            "hash_block_size": hash_block_size,
            "include_finished_set": include_finished_set,
            "log_stats": log_stats,
        }
        if mm_registry is not None:
            kwargs["mm_registry"] = mm_registry
        super().__init__(**kwargs)
        config = _config_from_vllm_config(vllm_config, self.AFFINITY_CONFIG)
        self._cache_affinity_enabled = False
        self._cache_affinity_config = config
        if not config.enabled or self.policy != SchedulingPolicy.FCFS:
            return
        self.waiting = CacheAffinityRequestQueue(
            lambda request: _local_cached_blocks(self, request),
            window_size=config.window_size,
            max_wait_s=config.max_wait_s,
            max_overtakes=config.max_overtakes,
        )
        self._cache_affinity_enabled = True

    def schedule(self, throttle_prefills: bool = False):
        if getattr(self, "_cache_affinity_enabled", False):
            config = getattr(self, "_cache_affinity_config", self.AFFINITY_CONFIG)
            self.waiting.reorder(
                force=True,
                window_size=min(
                    config.window_size,
                    4 * self.scheduler_config.max_num_seqs,
                ),
            )
        return super().schedule(throttle_prefills=throttle_prefills)


__all__ = [
    "CacheAffinityConfig",
    "CacheAffinityRequestQueue",
    "CacheAffinityScheduler",
]
