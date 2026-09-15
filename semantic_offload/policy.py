# SPDX-License-Identifier: Apache-2.0
"""SemanticPolicy: CachePolicy that blends semantic relevance with recency.

get/insert/remove/touch/clear/mark_evictable/mark_non_evictable all delegate
to an internal LRUCachePolicy unchanged -- LRU's own bookkeeping (evictable
order, ref counts) stays correct and is the source of truth for recency.
Only evict() is overridden: it reads LRUCachePolicy's real recency order
non-destructively (`self._lru.evictable_blocks`, a plain `OrderedDict` --
reading a composed instance's own attribute from our own code, not a vLLM
source edit, same class of technique as reading `layer.kv_cache` directly in
Step 1.2) rather than calling `LRUCachePolicy.evict()` as a "peek", which
would mutate its state as a side effect and corrupt the comparison. This is a
first, minimal slice of the plan's Step 1.4 -- not its full design. See
.claude/docs/semantic-eviction-plan.md Step 1.4 and
issues log entries #10 and #11 (grace period -- temporary eviction immunity
for freshly-inserted, as-yet-unscored blocks).
"""

import math
import time
from collections.abc import Callable, Iterable, Mapping
from itertools import pairwise

from typing_extensions import override
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

from semantic_offload._debug import debug_print
from semantic_offload.decision_trace import build_decision_feature_records

# How often (in touch() calls with session_aware on) to print the
# SEMANTIC_SESSION_DEBUG growth line -- issues log entry #70's open question
# (why session_aware causes a much bigger cache-hit-volume jump for minmax
# than mean/cuboid-mean at real 7B/B200 scale, when dev-box testing at the
# same config shows ~no effect for any method at all -- the mechanism only
# seems to manifest at production scale/duration, which this counter is
# meant to let a real B200 run actually observe).
_SESSION_DEBUG_EVERY = 200
DEFAULT_POLICY_ALPHA = 0.5


class SemanticPolicy(CachePolicy):
    """CachePolicy blending semantic relevance with recency (Step 1.4 slice).

    `score = alpha * relevance_norm + (1 - alpha) * recency_norm`, both
    normalized to [0, 1] over the current evictable set; blocks with no
    relevance score yet fall back to pure recency (plan's Step 1.4 text).
    Blocks with the lowest blended score are evicted first.
    """

    def __init__(
        self,
        cache_capacity: int,
        relevance_ema: dict[str, dict[OffloadKey, float]] | None = None,
        method: str = "mean",
        alpha: float = DEFAULT_POLICY_ALPHA,
        grace_window_blocks: int = 0,
        mode: str = "blend",
        chain_aware: bool = False,
        session_aware: bool = False,
        session_bonus_half_life: int = 0,
        prefix_aware: bool = False,
        future_reuse_ema: dict[OffloadKey, float] | None = None,
        future_reuse_feedback: bool = False,
        future_reuse_weight: float = 0.25,
        latest_relevance: dict[str, dict[OffloadKey, float]] | None = None,
        decision_feature_sink: Callable[[list[dict[str, object]]], None] | None = None,
    ) -> None:
        self._lru = LRUCachePolicy(cache_capacity=cache_capacity)
        # Shared mutable reference into SemanticOffloadingManager.relevance_ema
        # -- updated in place by update_relevance() each step; we just read it.
        self._relevance_ema = relevance_ema if relevance_ema is not None else {}
        self._method = method
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be finite and in [0, 1], got {alpha!r}")
        if mode == "unscored_last" and alpha != DEFAULT_POLICY_ALPHA:
            raise ValueError("alpha is not used by unscored_last eviction mode")
        self._alpha = alpha
        # "blend" (default): score = alpha*relevance + (1-alpha)*recency, one
        # continuous ranking. "unscored_last" (issues log entry #11/#18): a
        # strict two-tier ordering with no timer -- every unscored block is
        # evicted before any scored block is touched at all, recency-ordered
        # within the unscored tier, relevance-ordered (lowest first) within
        # the scored tier. Never fixes entry #10-#12's cold-start MISS (the
        # whole candidate pool is unscored at the critical moment there, so
        # the "unscored tier" degenerates to plain LRU, same root cause as
        # grace's failure) -- its value proposition is different: once *some*
        # content legitimately earns a score, never evict it in favor of
        # never-referenced throwaway content, with no window to tune at all.
        assert mode in ("blend", "unscored_last"), mode
        self._mode = mode
        # Grace (entry #11) is only meaningful combined with "blend" -- it
        # was proven unable to fix the cold-start gap on its own, and
        # combining two already-closed-for-that-purpose ideas is out of
        # scope here; "unscored_last" ignores grace_window_blocks entirely.
        # Grace period (issues log entry #11): a freshly-inserted, as-yet-
        # unscored block is immune to eviction for `grace_window_blocks`
        # *blocks inserted* (not wall-clock -- capacity pressure here is
        # driven by store volume) after its own insertion. 0 disables it
        # entirely (matches entry #10's "regular" no-grace behavior exactly).
        # `insert()` is called once per block by manager.py's prepare_store
        # loop (verified against vllm/v1/kv_offload/cpu/manager.py, not
        # assumed), so a simple call counter is a correct proxy for "blocks
        # inserted so far."
        self._grace_window_blocks = grace_window_blocks
        self._insert_count = 0
        self._grace_expiry: dict[OffloadKey, int] = {}
        # Chain-aware ordering (issues log entry #15; upstream precedent
        # PR #47744, which is what put a real req_context onto touch() in
        # the first place). vllm's offloading scheduler calls
        # manager.touch(offload_keys, req_context) with a request's full
        # ordered block chain every time it's stored or hits
        # (offloading/scheduler.py's _touch) -- so consecutive keys in one
        # touch() call are adjacent chain positions. We remember, per key,
        # the key observed right after it. At eviction time a block whose
        # recorded successor is still resident gets a bonus: this empties a
        # chain tail-first instead of letting an early block get evicted
        # while later blocks of the *same* chain (now useless without it --
        # a broken prefix can't be partially reused) survive it. Only
        # wired into "blend" mode below; "unscored_last" is out of scope
        # for this slice.
        self._chain_aware = chain_aware
        self._chain_successor: dict[OffloadKey, OffloadKey] = {}
        self._chain_predecessors: dict[OffloadKey, set[OffloadKey]] = {}
        # Session-proven priority (issues log entry #19's follow-up, after
        # the per-block chain-successor bonus above was found redundant with
        # LRU's own touch() semantics and closed). Targets a different gap
        # than content scoring or chain ordering: an idle-but-active
        # multi-turn chat session has no content evidence yet (that's the
        # already-closed cold-start gap, entries #10-#12) and no chain-
        # internal ordering problem (closed above) -- but it DOES have a
        # structural signal content scoring can't see: has a *different*
        # request already revisited (touched) this exact chain? Real multi-
        # turn conversations get a new req_id every turn (a fresh HTTP
        # request) while reusing the SAME content-addressed OffloadKeys for
        # the shared prefix -- so a touch() call whose keys were last
        # touched by a *different* req_id is direct, real evidence of cross-
        # request continuation, not a speculative bet on the future (unlike
        # a blanket grace period for all fresh blocks, which entry #11
        # proved mathematically can't work: the window long enough to
        # protect a genuine return visit is also long enough to blanket-
        # protect the flood of equally-fresh one-shot traffic). Once
        # detected, the *entire* current touch's key set (the request's full
        # known chain) is marked proven, persistently -- not time-limited,
        # since it's based on an observed fact, not a countdown.
        self._session_aware = session_aware
        self._last_touch_req_id: dict[OffloadKey, str] = {}
        self._session_proven: set[OffloadKey] = set()
        self._session_touch_calls = 0
        # Recency-weighted variant (issues log entry #21's follow-up): a flat
        # bonus is provably unable to distinguish an actively-used proven
        # session from an idle-but-proven one once eviction is choosing only
        # among proven candidates (adding the same constant to every
        # candidate can't change their relative order) -- entry #21 measured
        # this concretely as active sessions losing protection they'd have
        # had under plain LRU, to equalize with idle-but-proven sessions
        # that no longer need it. `session_bonus_half_life > 0` scales the
        # bonus down by how many touch() *calls* (not wall-clock, matching
        # entry #12's precedent of using event counts as the proxy unit)
        # have passed since this specific key was last touched, so a
        # recently-active proven session keeps close to the full bonus while
        # a long-idle one's bonus decays toward (but never reaches) zero --
        # 0 (the default) preserves entry #20's original flat-bonus
        # behavior unchanged.
        self._session_bonus_half_life = session_bonus_half_life
        self._touch_seq = 0
        self._last_touch_seq: dict[OffloadKey, int] = {}
        if not isinstance(prefix_aware, bool):
            raise TypeError("prefix_aware must be a boolean")
        if not isinstance(future_reuse_feedback, bool):
            raise TypeError("future_reuse_feedback must be a boolean")
        if (
            not math.isfinite(future_reuse_weight)
            or not 0.0 <= future_reuse_weight <= 1.0
        ):
            raise ValueError("future_reuse_weight must be finite and in [0, 1]")
        self._prefix_aware = prefix_aware
        self._prefix_chains: dict[OffloadKey, set[tuple[OffloadKey, ...]]] = {}
        self._future_reuse_ema = (
            future_reuse_ema if future_reuse_ema is not None else {}
        )
        self._future_reuse_feedback = future_reuse_feedback
        self._future_reuse_weight = future_reuse_weight
        self._latest_relevance = (
            latest_relevance if latest_relevance is not None else {}
        )
        self._decision_feature_sink = decision_feature_sink
        self._decision_feature_context: Mapping[str, object] | None = None
        self._decision_sequence = 0

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self._lru.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self._lru.insert(key, block)
        # Counter kept live unconditionally (not just when grace is enabled)
        # -- issues log entry #12 measured pipeline latency in these units
        # even in the no-grace ("regular") configuration, and a future
        # session may want the same "blocks inserted so far" timeline again
        # without re-adding this increment.
        self._insert_count += 1
        if self._grace_window_blocks > 0:
            self._grace_expiry[key] = self._insert_count + self._grace_window_blocks

    @override
    def remove(self, key: OffloadKey) -> None:
        self._lru.remove(key)
        self._forget_key(key)

    def _forget_key(self, key: OffloadKey) -> None:
        """Drop all policy metadata owned by a no-longer-resident key."""
        self._grace_expiry.pop(key, None)
        successor = self._chain_successor.pop(key, None)
        if successor is not None:
            predecessors = self._chain_predecessors.get(successor)
            if predecessors is not None:
                predecessors.discard(key)
                if not predecessors:
                    self._chain_predecessors.pop(successor, None)
        for predecessor in self._chain_predecessors.pop(key, ()):
            if self._chain_successor.get(predecessor) == key:
                self._chain_successor.pop(predecessor, None)
        self._last_touch_req_id.pop(key, None)
        self._session_proven.discard(key)
        self._last_touch_seq.pop(key, None)
        for root, chains in list(self._prefix_chains.items()):
            remaining: set[tuple[OffloadKey, ...]] = set()
            for chain in chains:
                if key not in chain:
                    remaining.add(chain)
                    continue
                prefix = chain[: chain.index(key)]
                if prefix:
                    remaining.add(prefix)
            if remaining:
                self._prefix_chains[root] = remaining
            else:
                self._prefix_chains.pop(root, None)
        for scores in self._relevance_ema.values():
            scores.pop(key, None)
        for scores in self._latest_relevance.values():
            scores.pop(key, None)

    def set_decision_feature_context(
        self, context: Mapping[str, object] | None
    ) -> None:
        """Bind the next eviction decision to its scheduler request."""
        self._decision_feature_context = context

    def _record_decision_features(
        self,
        candidates: list[tuple[OffloadKey, BlockStatus]],
        ranked_keys: list[OffloadKey],
        selected_keys: set[OffloadKey],
        *,
        decision_timestamp_ns: int,
        scoring_cpu_time_ns: int,
        policy_cpu_time_ns: int,
        critical_path_wall_time_ns: int,
        relevance: Mapping[OffloadKey, float],
        latest_relevance: Mapping[OffloadKey, float],
    ) -> list[dict[str, object]]:
        if self._decision_feature_sink is None:
            return []
        context = self._decision_feature_context or {}
        records = build_decision_feature_records(
            context=context,
            decision_sequence=self._decision_sequence,
            candidates=candidates,
            ranked_keys=ranked_keys,
            selected_keys=selected_keys,
            relevance=relevance,
            latest_relevance=latest_relevance,
            sink=self._decision_feature_sink,
            decision_timestamp_ns=decision_timestamp_ns,
            scoring_cpu_time_ns=scoring_cpu_time_ns,
            policy_cpu_time_ns=policy_cpu_time_ns,
            critical_path_wall_time_ns=critical_path_wall_time_ns,
            synchronization_wait_ns=0,
            synchronization_status="no_explicit_synchronization",
            policy=f"semantic-{self._method}",
            policy_state={
                "mode": self._mode,
                "method": self._method,
                "alpha": self._alpha,
                "grace_window_blocks": self._grace_window_blocks,
                "chain_aware": self._chain_aware,
                "session_aware": self._session_aware,
                "prefix_aware": self._prefix_aware,
            },
        )
        self._decision_sequence += 1
        return records

    def _future_keep_score(self, key: OffloadKey, base: float) -> float:
        if self._future_reuse_feedback and key in self._future_reuse_ema:
            utility = min(1.0, max(0.0, float(self._future_reuse_ema[key])))
            return (
                1.0 - self._future_reuse_weight
            ) * base + self._future_reuse_weight * utility
        return base

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        # CachePolicy accepts any iterable, including one-shot generators.
        # Materialize once so the delegate and semantic metadata see the
        # same keys.
        ordered = list(keys)
        self._lru.touch(ordered, req_context)
        if self._chain_aware:
            for prev_key, next_key in pairwise(ordered):
                old_successor = self._chain_successor.get(prev_key)
                if old_successor is not None and old_successor != next_key:
                    predecessors = self._chain_predecessors.get(old_successor)
                    if predecessors is not None:
                        predecessors.discard(prev_key)
                        if not predecessors:
                            self._chain_predecessors.pop(old_successor, None)
                self._chain_successor[prev_key] = next_key
                self._chain_predecessors.setdefault(next_key, set()).add(prev_key)
        if self._session_aware or self._prefix_aware:
            req_id = req_context.req_id
            cross_request_hit = any(
                self._last_touch_req_id.get(k) not in (None, req_id) for k in ordered
            )
            if self._session_aware and cross_request_hit:
                self._session_proven.update(ordered)
            if self._prefix_aware and ordered:
                leading_overlap: list[OffloadKey] = []
                for key in ordered:
                    if self._last_touch_req_id.get(key) in (None, req_id):
                        break
                    leading_overlap.append(key)
                if leading_overlap:
                    self._prefix_chains.setdefault(ordered[0], set()).add(
                        tuple(leading_overlap)
                    )
            for k in ordered:
                self._last_touch_req_id[k] = req_id
            if self._session_aware and self._session_bonus_half_life > 0:
                self._touch_seq += 1
                for k in ordered:
                    self._last_touch_seq[k] = self._touch_seq
            if self._session_aware:
                self._session_touch_calls += 1
            if (
                self._session_aware
                and self._session_touch_calls % _SESSION_DEBUG_EVERY == 0
            ):
                debug_print(
                    f"SEMANTIC_SESSION_DEBUG method={self._method} "
                    f"touch_calls={self._session_touch_calls} "
                    f"cross_request_hit={cross_request_hit} "
                    f"session_proven_size={len(self._session_proven)} "
                    f"resident_size={len(self._lru.blocks)}"
                )

    def _eviction_candidates(
        self, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]]:
        # Non-destructive read of LRU's real recency order (oldest first --
        # touch() moves accessed keys to the end of this OrderedDict).
        return [
            (key, self._lru.blocks[key])
            for key in self._lru.evictable_blocks
            if key not in protected
        ]

    def _prefix_protected_keys(self) -> set[OffloadKey]:
        prefix_protected: set[OffloadKey] = set()
        if self._prefix_aware:
            for chains in self._prefix_chains.values():
                root_prefix_protected: set[OffloadKey] = set()
                longest_prefix_length = 0
                for chain in sorted(chains, key=repr):
                    prefix: list[OffloadKey] = []
                    for key in chain:
                        if key not in self._lru.blocks:
                            break
                        prefix.append(key)
                    if len(prefix) > longest_prefix_length:
                        root_prefix_protected = set(prefix)
                        longest_prefix_length = len(prefix)
                    elif len(prefix) == longest_prefix_length:
                        root_prefix_protected.update(prefix)
                prefix_protected.update(root_prefix_protected)
        return prefix_protected

    def _eviction_order(
        self,
        candidates: list[tuple[OffloadKey, BlockStatus]],
        prefix_protected: set[OffloadKey],
    ) -> list[tuple[OffloadKey, BlockStatus]]:
        if self._mode == "unscored_last":
            chosen, _ = self._evict_unscored_last(
                len(candidates), candidates, prefix_protected
            )
            return chosen

        # Grace period (entry #11): a *soft* bonus, not a hard exclude. A
        # hard exclude-then-fallback-to-oldest-in-grace design was tried
        # first and found to degenerate to plain LRU order under sustained
        # capacity pressure: once enough blocks are resident that the whole
        # evictable set is still within its own grace window (e.g. a large
        # grace_window relative to workload churn), the "fallback" pool *is*
        # the whole candidate set, and picking oldest-first from it is
        # exactly what pure LRU does -- silently defeating the mechanism
        # (see issues log entry #11's diagnostic run). Instead, grace adds a
        # large constant bonus to a still-protected block's keep_score, so
        # it is essentially never evicted while any non-grace candidate
        # exists, but if eviction is ever forced to dip into the grace pool
        # anyway, relative ordering *within* that pool still follows the
        # real relevance+recency blend rather than raw recency.
        if self._grace_window_blocks > 0:
            in_grace = {
                key
                for key, _ in candidates
                if self._grace_expiry.get(key, -1) > self._insert_count
            }
        else:
            in_grace = set()

        relevance = self._relevance_ema.get(self._method, {})
        scored_vals = [relevance[k] for k, _ in candidates if k in relevance]
        lo = min(scored_vals) if scored_vals else 0.0
        hi = max(scored_vals) if scored_vals else 0.0
        span = (hi - lo) or 1.0
        total = len(candidates)
        # keep_score without the bonus is always in [0, 1]; +2.0 guarantees
        # a grace-protected block outranks (survives longer than) any
        # non-grace-protected block, while preserving relative order among
        # grace-protected blocks themselves.
        GRACE_BONUS = 2.0
        # Strictly less than GRACE_BONUS: grace's temporary immunity for any
        # freshly-inserted block should still win a tie-break against chain
        # protection, since grace covers a case (unscored blocks) chain
        # protection doesn't reason about at all. Strictly greater than the
        # base score's [0, 1] range, so a chain-protected block always
        # outranks every non-grace, non-chain candidate.
        CHAIN_BONUS = 1.5
        # Strictly greater than GRACE_BONUS: session-proven status is real,
        # observed cross-request evidence (a second distinct request
        # revisited this exact chain), not a speculative freshness bet --
        # it should outrank grace's temporary immunity for merely-unproven
        # fresh blocks in a direct tie-break.
        SESSION_BONUS = 2.5
        PREFIX_BONUS = 3.0

        def has_resident_successor(key: OffloadKey) -> bool:
            successor = self._chain_successor.get(key)
            return successor is not None and successor in self._lru.blocks

        def session_bonus_for(key: OffloadKey) -> float:
            # Flat (issues log entry #20's original design, still the
            # default at half_life=0): every proven key gets the full
            # bonus regardless of how recently it was actually used. Entry
            # #21 found this can't distinguish an active proven session
            # from an idle one once eviction is choosing only among proven
            # candidates -- adding the same constant to every candidate
            # can't change their relative order -- costing active sessions
            # protection plain LRU would have given them for free. Decayed
            # variant (half_life > 0): scale by touch()-call staleness so an
            # active proven session keeps close to the full bonus while an
            # idle one's decays toward, but never reaches, zero -- it should
            # still edge out an equally-idle NEVER-proven candidate (which
            # gets no bonus at all), just not swamp a currently-active one.
            if self._session_bonus_half_life <= 0:
                return SESSION_BONUS
            staleness = self._touch_seq - self._last_touch_seq.get(key, self._touch_seq)
            decay = 1.0 / (1.0 + staleness / self._session_bonus_half_life)
            return SESSION_BONUS * decay

        def keep_score(idx: int, key: OffloadKey) -> float:
            # idx 0 = LRU's next eviction pick (least recent) -> 0.0;
            # idx (total-1) = most recent -> 1.0.
            recency_norm = idx / (total - 1) if total > 1 else 1.0
            if key not in relevance:
                base = recency_norm  # no signal yet -- pure recency
            else:
                relevance_norm = (relevance[key] - lo) / span
                base = self._alpha * relevance_norm + (1 - self._alpha) * recency_norm
            base = self._future_keep_score(key, base)
            if key in in_grace:
                base += GRACE_BONUS
            if self._chain_aware and has_resident_successor(key):
                base += CHAIN_BONUS
            if self._prefix_aware and key in prefix_protected:
                base += PREFIX_BONUS
            if self._session_aware and key in self._session_proven:
                base += session_bonus_for(key)
            return base

        scored = [
            (keep_score(idx, key), key, block)
            for idx, (key, block) in enumerate(candidates)
        ]
        scored.sort(key=lambda t: t[0])  # lowest keep_score evicted first
        return [(key, block) for _, key, block in scored]

    def peek_victims(
        self, n: int, protected: set[OffloadKey]
    ) -> list[OffloadKey] | None:
        """Return the exact next victim keys without mutating policy state."""
        if n == 0:
            return []
        candidates = self._eviction_candidates(protected)
        if len(candidates) < n:
            return None
        ordered = self._eviction_order(candidates, self._prefix_protected_keys())
        return [key for key, _ in ordered[:n]]

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        policy_started_ns = time.monotonic_ns()
        policy_started_cpu_ns = time.thread_time_ns()
        candidates = self._eviction_candidates(protected)
        if len(candidates) < n:
            return None
        scoring_started_cpu_ns = time.thread_time_ns()
        ordered = self._eviction_order(candidates, self._prefix_protected_keys())
        scoring_completed_cpu_ns = time.thread_time_ns()
        decision_timestamp_ns = time.monotonic_ns()
        chosen = ordered[:n]
        relevance = dict(self._relevance_ema.get(self._method, {}))
        latest_relevance = dict(self._latest_relevance.get(self._method, {}))
        for key, _ in chosen:
            self._lru.remove(key)
            self._forget_key(key)
        policy_completed_ns = time.monotonic_ns()
        policy_completed_cpu_ns = time.thread_time_ns()
        instrumentation_started_ns = time.monotonic_ns()
        instrumentation_started_cpu_ns = time.thread_time_ns()
        records = self._record_decision_features(
            candidates,
            [key for key, _ in ordered],
            {key for key, _ in chosen},
            decision_timestamp_ns=decision_timestamp_ns,
            scoring_cpu_time_ns=(
                scoring_completed_cpu_ns - scoring_started_cpu_ns
            ),
            policy_cpu_time_ns=policy_completed_cpu_ns - policy_started_cpu_ns,
            critical_path_wall_time_ns=policy_completed_ns - policy_started_ns,
            relevance=relevance,
            latest_relevance=latest_relevance,
        )
        instrumentation_completed_ns = time.monotonic_ns()
        instrumentation_completed_cpu_ns = time.thread_time_ns()
        for record in records:
            record["instrumentation_cpu_time_ns"] = (
                instrumentation_completed_cpu_ns - instrumentation_started_cpu_ns
            )
            record["instrumentation_wall_time_ns"] = (
                instrumentation_completed_ns - instrumentation_started_ns
            )
            record["critical_path_wall_time_ns"] = (
                instrumentation_completed_ns - policy_started_ns
            )
        return chosen

    def _evict_unscored_last(
        self,
        n: int,
        candidates: list[tuple[OffloadKey, BlockStatus]],
        prefix_protected: set[OffloadKey],
    ) -> tuple[list[tuple[OffloadKey, BlockStatus]], list[OffloadKey]]:
        """Strict two-tier eviction, no timer (entry #11/#18): every unscored
        block is evicted before any scored block, recency-ordered (oldest
        first) within the unscored tier -- `candidates` already arrives in
        that order from `evictable_blocks`. Within the scored tier, lowest
        relevance is evicted first, since real scores are available there and
        ignoring them would waste the whole point of having a scored tier."""
        relevance = self._relevance_ema.get(self._method, {})
        unscored = [(key, block) for key, block in candidates if key not in relevance]
        rest = [(key, block) for key, block in candidates if key in relevance]
        scored_values = [relevance[key] for key, _ in rest]
        relevance_low = min(scored_values) if scored_values else 0.0
        relevance_span = (
            (max(scored_values) - relevance_low) if scored_values else 0.0
        ) or 1.0
        position = {key: index for index, (key, _) in enumerate(candidates)}
        denominator = max(len(candidates) - 1, 1)
        unscored.sort(
            key=lambda kb: self._future_keep_score(kb[0], position[kb[0]] / denominator)
        )
        rest.sort(
            key=lambda kb: self._future_keep_score(
                kb[0], (relevance[kb[0]] - relevance_low) / relevance_span
            )
        )
        unscored_prefix = [kb for kb in unscored if kb[0] in prefix_protected]
        unscored = [kb for kb in unscored if kb[0] not in prefix_protected]
        rest_prefix = [kb for kb in rest if kb[0] in prefix_protected]
        rest = [kb for kb in rest if kb[0] not in prefix_protected]
        eviction_order = unscored + rest + unscored_prefix + rest_prefix
        chosen = eviction_order[:n]
        return chosen, [key for key, _ in eviction_order]

    @override
    def clear(self) -> None:
        self._lru.clear()
        self._grace_expiry.clear()
        self._chain_successor.clear()
        self._chain_predecessors.clear()
        self._last_touch_req_id.clear()
        self._session_proven.clear()
        self._last_touch_seq.clear()
        self._prefix_chains.clear()
        self._insert_count = 0
        self._touch_seq = 0
        self._session_touch_calls = 0
        self._future_reuse_ema.clear()
        self._latest_relevance.clear()
        self._decision_feature_context = None
        self._decision_sequence = 0
        for scores in self._relevance_ema.values():
            scores.clear()

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self._lru.mark_evictable(key)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self._lru.mark_non_evictable(key)
