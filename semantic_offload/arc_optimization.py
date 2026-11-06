"""Opt-in ARC implementation and CPU-advice experiments.

Advice changes ordering only. It never fabricates demand hits, changes queue
membership/ghosts, reserves a block, or claims knowledge of future requests.
"""
from collections import OrderedDict
from dataclasses import dataclass, fields
import math

from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy


class LinearARCCachePolicy(ARCCachePolicy):
    """The native eviction decisions with one scan per queue, not per victim."""

    def evict(self, n, protected):
        if n == 0:
            return []
        first = iter(self.t1.items())
        second = iter(self.t2.items())
        virtual = len(self.t1)
        candidates = []
        def take(iterator):
            for key, block in iterator:
                if block.ref_cnt == 0 and key not in protected:
                    return key, block
            return None
        for _ in range(n):
            item = take(first) if virtual >= int(self.target_t1_size) else None
            from_t1 = item is not None
            if from_t1:
                virtual -= 1
            else:
                item = take(second)
            if item is None:
                return None  # Atomic failure: no queue or ghost was mutated.
            candidates.append((*item, from_t1))
        for key, block, from_t1 in candidates:
            resident, ghost = (self.t1, self.b1) if from_t1 else (self.t2, self.b2)
            del resident[key]
            ghost[key] = None
        for ghost in (self.b1, self.b2):
            for _ in range(len(ghost) - self.cache_capacity):
                ghost.popitem(last=False)
        return [(key, block) for key, block, _ in candidates]


@dataclass(frozen=True)
class AdviceOptions:
    queue: str = 'both'
    order: str = 'root'
    fraction: float = 1.0
    max_keys: int = 0
    boost: float = 1.0
    min_fill: float = 0.0
    cooldown_ms: int = 0
    session_budget: int = 0
    budget_scope: str = 'session'
    max_state: int = 64000

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict):
            raise TypeError('wait_hint_arc_advice must be an object')
        if set(raw) - {f.name for f in fields(cls)}:
            raise ValueError('Unknown ARC advice option')
        obj = cls(**raw)
        if obj.queue not in ('both', 'recent', 'frequent') or obj.order not in ('root', 'suffix'):
            raise ValueError('Unsupported ARC advice queue/order')
        if obj.budget_scope not in ('session', 'wait'):
            raise ValueError('budget_scope must be session or wait')
        for name in ('fraction', 'boost', 'min_fill'):
            x = getattr(obj, name)
            if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1:
                raise ValueError(name + ' must be a finite fraction')
        for name in ('max_keys', 'cooldown_ms', 'session_budget', 'max_state'):
            x = getattr(obj, name)
            if type(x) is not int or x < 0 or (name == 'max_state' and x == 0):
                raise ValueError(name + ' must be a nonnegative integer (max_state positive)')
        return obj


class ARCAdvice:
    """Bounded optional state; the scheduler invokes this only after validation."""
    def __init__(self, options):
        self.options = AdviceOptions.parse(options)
        self.key_times = OrderedDict()
        self.sessions = OrderedDict()
        self.last = {}

    def clear(self):
        self.key_times.clear()
        self.sessions.clear()
        self.last = {}

    def refresh(self, manager, keys, session, now_ns):
        policy = manager._policy
        o = self.options
        self.last = dict(eligible=len(keys), applied=0, pressure_skipped=False, budget_scope=o.budget_scope)
        if not isinstance(policy, ARCCachePolicy):
            return 0
        if (len(policy.t1) + len(policy.t2)) < o.min_fill * policy.cache_capacity:
            self.last['pressure_skipped'] = True
            return 0
        limit = math.ceil(len(keys) * o.fraction)
        if o.max_keys:
            limit = min(limit, o.max_keys)
        spent = self.sessions.get(session, 0)
        if o.session_budget:
            limit = min(limit, max(0, o.session_budget - spent))
        selected = []
        for key in keys[:limit]:
            queue = policy.t1 if key in policy.t1 else policy.t2
            block = queue.get(key)
            if block is None or not block.is_ready:
                continue
            if (o.queue == 'recent' and queue is not policy.t1) or (o.queue == 'frequent' and queue is not policy.t2):
                continue
            last = self.key_times.get(key)
            if o.cooldown_ms and last is not None and now_ns - last < o.cooldown_ms * 1_000_000:
                continue
            selected.append(key)
        if not o.boost:
            selected = []
        ordered = list(reversed(selected)) if o.order == 'root' else selected
        if o.boost == 1:
            for key in ordered:
                queue = policy.t1 if key in policy.t1 else policy.t2
                queue.move_to_end(key)
        elif selected:
            # Move each selected item a fraction of its distance to MRU.
            # Stable sorting preserves unselected order and block identities.
            chosen = {key: rank for rank, key in enumerate(ordered)}
            for queue in (policy.t1, policy.t2):
                count = len(queue)
                items = list(queue.items())
                ranked = sorted(enumerate(items), key=lambda item: (
                    item[0] + o.boost * (count - 1 - item[0]) if item[1][0] in chosen else item[0],
                    chosen.get(item[1][0], -1)))
                queue.clear()
                queue.update(item for _, item in ranked)
        if o.cooldown_ms:
            for key in selected:
                self.key_times[key] = now_ns
                self.key_times.move_to_end(key)
            while len(self.key_times) > o.max_state:
                self.key_times.popitem(last=False)
        if o.session_budget and selected:
            self.sessions[session] = spent + len(selected)
            self.sessions.move_to_end(session)
            while len(self.sessions) > o.max_state:
                self.sessions.popitem(last=False)
        self.last['applied'] = len(selected)
        return len(selected)
