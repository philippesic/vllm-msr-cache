# SPDX-License-Identifier: Apache-2.0
"""Bounded ownership and ordered publication for opt-in async scoring."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition, Lock
from semantic_offload.profiling import PROFILE
from typing import Generic, Protocol, TypeVar


class ReadinessFence(Protocol):
    def synchronize(self) -> None: ...


_HOST_EVENT_POLL_SECONDS = 0.0005


def wait_for_owned_fence(fence: ReadinessFence) -> None:
    """Cooperate while an owned event is pending; generic fences keep their API.

    Some WSL CUDA event synchronize implementations actively wait even with
    blocking=True. Querying then sleeping bounds host polling without changing
    event ordering. Cancellation/reset must still retain/drain the owned job.
    Query errors fall back to the original fence before propagating the error.
    """
    query = getattr(fence, "query", None)
    if callable(query):
        try:
            while not query():
                time.sleep(_HOST_EVENT_POLL_SECONDS)
        except BaseException:
            fence.synchronize()
            raise
    fence.synchronize()


PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class AsyncScoringSnapshot(Generic[PayloadT]):
    observation_sequence: int
    generation: int
    payload: PayloadT
    snapshot_bytes: int
    key_versions: tuple[tuple[object, object], ...]
    observation_available_ns: int
    readiness_fences: tuple[ReadinessFence, ...] = ()


@dataclass(frozen=True, slots=True)
class AsyncScoringCompletion(Generic[PayloadT, ResultT]):
    snapshot: AsyncScoringSnapshot[PayloadT]
    result: ResultT
    input_ready_ns: int
    scoring_started_ns: int
    scoring_completed_ns: int


@dataclass(slots=True)
class _Pending(Generic[PayloadT, ResultT]):
    snapshot: AsyncScoringSnapshot[PayloadT]
    future: Future[AsyncScoringCompletion[PayloadT, ResultT]]
    submitted_ns: int


class AsyncScoringController(Generic[PayloadT, ResultT]):
    """Run scoring off-path while publishing only an ordered ready prefix."""

    def __init__(
        self,
        scorer: Callable[[PayloadT], ResultT],
        *,
        max_queue_items: int,
        max_snapshot_bytes: int,
        workers: int = 2,
        telemetry_limit: int | None = None,
    ) -> None:
        for name, value in (
            ("max_queue_items", max_queue_items),
            ("max_snapshot_bytes", max_snapshot_bytes),
            ("workers", workers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._scorer = scorer
        self._max_queue_items = max_queue_items
        self._max_snapshot_bytes = max_snapshot_bytes
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="semantic-async-score"
        )
        self._pending: OrderedDict[int, _Pending[PayloadT, ResultT]] = OrderedDict()
        self._pending_snapshot_bytes = 0
        self._next_observation_sequence = 0
        self._closed = False
        self._lock = Lock()
        if telemetry_limit is not None and (
            isinstance(telemetry_limit, bool)
            or not isinstance(telemetry_limit, int)
            or telemetry_limit <= 0
        ):
            raise ValueError("telemetry_limit must be a positive integer")
        limit = telemetry_limit or max(32, max_queue_items * 4)
        self._telemetry: deque[dict[str, object]] = deque(maxlen=limit)
        self._resetting = False
        self._state_changed = Condition(self._lock)

    def _record(self, **fields: object) -> None:
        self._telemetry.append(
            {"schema": "semantic-async-scoring-event-v1", **fields}
        )

    def can_accept(self, snapshot_bytes: int) -> bool:
        with self._lock:
            return (
                not self._closed
                and not self._resetting
                and len(self._pending) < self._max_queue_items
                and self._pending_snapshot_bytes + snapshot_bytes
                <= self._max_snapshot_bytes
            )

    def record_pre_capture_drop(
        self, *, generation: int, snapshot_bytes: int
    ) -> None:
        with self._lock:
            sequence = self._next_observation_sequence
            self._next_observation_sequence += 1
            self._record(
                event="submission_dropped",
                observation_sequence=sequence,
                generation=generation,
                reason="drop_newest_before_snapshot_capture",
                queue_items=len(self._pending),
                pending_snapshot_bytes=self._pending_snapshot_bytes,
                submitted_snapshot_bytes=snapshot_bytes,
            )

    @staticmethod
    def _run(
        scorer: Callable[[PayloadT], ResultT],
        snapshot: AsyncScoringSnapshot[PayloadT],
    ) -> AsyncScoringCompletion[PayloadT, ResultT]:
        with PROFILE.scope("async.input_ready_wait"):
            try:
                for fence in snapshot.readiness_fences:
                    wait_for_owned_fence(fence)
            except BaseException:
                # A query failure must not release other pending owned inputs.
                # Keep the snapshot alive until every original fence was tried.
                fence_error = None
                for fence in snapshot.readiness_fences:
                    try:
                        fence.synchronize()
                    except BaseException as error:
                        fence_error = error
                if fence_error is not None:
                    raise fence_error
                raise
        input_ready_ns = time.monotonic_ns()
        started_ns = time.monotonic_ns()
        with PROFILE.scope("async.executor_total"):
            result = scorer(snapshot.payload)
        completed_ns = time.monotonic_ns()
        return AsyncScoringCompletion(
            snapshot=snapshot,
            result=result,
            input_ready_ns=input_ready_ns,
            scoring_started_ns=started_ns,
            scoring_completed_ns=completed_ns,
        )

    def submit(
        self,
        payload: PayloadT,
        *,
        generation: int,
        snapshot_bytes: int,
        key_versions: Mapping[object, object],
        observation_available_ns: int,
        readiness_fences: Sequence[ReadinessFence] = (),
    ) -> int | None:
        if (
            isinstance(snapshot_bytes, bool)
            or not isinstance(snapshot_bytes, int)
            or snapshot_bytes < 0
        ):
            raise ValueError("snapshot_bytes must be a non-negative integer")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")
        if (
            isinstance(observation_available_ns, bool)
            or not isinstance(observation_available_ns, int)
            or observation_available_ns < 0
        ):
            raise ValueError("observation_available_ns must be a non-negative integer")
        with self._lock:
            sequence = self._next_observation_sequence
            self._next_observation_sequence += 1
            if self._closed or self._resetting:
                self._record(
                    event="submission_rejected",
                    observation_sequence=sequence,
                    reason=(
                        "controller_closed"
                        if self._closed
                        else "controller_resetting"
                    ),
                )
                return None
            if (
                len(self._pending) >= self._max_queue_items
                or self._pending_snapshot_bytes + snapshot_bytes
                > self._max_snapshot_bytes
            ):
                self._record(
                    event="submission_dropped",
                    observation_sequence=sequence,
                    reason="drop_newest_queue_or_byte_bound",
                    queue_items=len(self._pending),
                    pending_snapshot_bytes=self._pending_snapshot_bytes,
                    submitted_snapshot_bytes=snapshot_bytes,
                )
                return None
            snapshot = AsyncScoringSnapshot(
                observation_sequence=sequence,
                generation=generation,
                payload=payload,
                snapshot_bytes=snapshot_bytes,
                key_versions=tuple(key_versions.items()),
                observation_available_ns=observation_available_ns,
                readiness_fences=tuple(readiness_fences),
            )
            submitted_ns = time.monotonic_ns()
            def execute():
                if PROFILE.directory:
                    PROFILE.observe("async.queue_wait.host_wall", time.monotonic_ns()-submitted_ns)
                return self._run(self._scorer, snapshot)
            future = self._executor.submit(execute)
            self._pending[sequence] = _Pending(snapshot, future, submitted_ns)
            self._pending_snapshot_bytes += snapshot_bytes
            self._record(
                event="submitted",
                observation_sequence=sequence,
                generation=generation,
                submitted_ns=submitted_ns,
                snapshot_captured_ns=observation_available_ns,
                snapshot_bytes=snapshot_bytes,
                queue_items=len(self._pending),
                pending_snapshot_bytes=self._pending_snapshot_bytes,
            )
            return sequence

    def drain_ready(
        self,
    ) -> tuple[
        list[AsyncScoringCompletion[PayloadT, ResultT]], list[dict[str, object]]
    ]:
        completions: list[AsyncScoringCompletion[PayloadT, ResultT]] = []
        failures: list[dict[str, object]] = []
        with self._lock:
            while self._pending:
                sequence, pending = next(iter(self._pending.items()))
                if not pending.future.done():
                    break
                self._pending.pop(sequence)
                self._pending_snapshot_bytes -= pending.snapshot.snapshot_bytes
                publication_ns = time.monotonic_ns()
                try:
                    completion = pending.future.result()
                except Exception as exc:  # noqa: BLE001 - scorer failures are evidence
                    failure = {
                        "event": "scoring_failed",
                        "observation_sequence": sequence,
                        "generation": pending.snapshot.generation,
                        "publication_ns": publication_ns,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                    failures.append(failure)
                    self._record(**failure)
                    continue
                completions.append(completion)
                self._record(
                    event="ready_for_publication",
                    observation_sequence=sequence,
                    generation=pending.snapshot.generation,
                    queue_age_ns=publication_ns - pending.submitted_ns,
                    scoring_started_ns=completion.scoring_started_ns,
                    scoring_completed_ns=completion.scoring_completed_ns,
                    observation_available_ns=completion.input_ready_ns,
                    publication_ns=publication_ns,
                    snapshot_bytes=pending.snapshot.snapshot_bytes,
                )
        return completions, failures

    def drain_all(
        self,
    ) -> tuple[
        list[AsyncScoringCompletion[PayloadT, ResultT]], list[dict[str, object]]
    ]:
        """Wait for owned work and return the complete ordered publication prefix."""
        with self._lock:
            pending = [item.future for item in self._pending.values()]
        for future in pending:
            try:
                future.result()
            except Exception:  # noqa: BLE001, S112 - drain_ready records the failure
                continue
        return self.drain_ready()

    def reset(self) -> None:
        """Wait for owned readers, discard results, and retain the executor."""
        with self._lock:
            while self._resetting:
                self._state_changed.wait()
            self._resetting = True
            pending = list(self._pending.values())
        reset_failures: list[tuple[str, str]] = []
        try:
            for item in pending:
                try:
                    item.future.result()
                except Exception as exc:  # noqa: BLE001 - reset must reclaim all work
                    reset_failures.append((type(exc).__name__, str(exc)))
        finally:
            with self._lock:
                discarded = len(self._pending)
                discarded_bytes = self._pending_snapshot_bytes
                self._pending.clear()
                self._pending_snapshot_bytes = 0
                self._resetting = False
                self._record(
                    event="reset",
                    discarded=discarded,
                    discarded_snapshot_bytes=discarded_bytes,
                    discarded_failures=len(reset_failures),
                    discarded_failure_types=[
                        failure_type for failure_type, _ in reset_failures
                    ],
                )
                self._state_changed.notify_all()

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            abandoned = len(self._pending)
            abandoned_bytes = self._pending_snapshot_bytes
            self._pending.clear()
            self._pending_snapshot_bytes = 0
            self._record(
                event="shutdown",
                abandoned=abandoned,
                abandoned_snapshot_bytes=abandoned_bytes,
            )

    def pop_telemetry(self) -> list[dict[str, object]]:
        with self._lock:
            records = list(self._telemetry)
            self._telemetry.clear()
            return records

    def record_event(self, **fields: object) -> None:
        with self._lock:
            self._record(**fields)

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "closed": self._closed,
                "resetting": self._resetting,
                "pending_items": len(self._pending),
                "pending_snapshot_bytes": self._pending_snapshot_bytes,
                "max_queue_items": self._max_queue_items,
                "max_snapshot_bytes": self._max_snapshot_bytes,
                "next_observation_sequence": self._next_observation_sequence,
                "telemetry_items": len(self._telemetry),
                "telemetry_limit": self._telemetry.maxlen or 0,
            }
