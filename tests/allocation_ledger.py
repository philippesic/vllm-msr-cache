"""Task-local bounded allocator ledger, identical in every timed arm.

UNVALIDATED preparation: run native CPU transparency checks before freezing.
No extra cache probes, GPU reads, fences or allocator decisions. Hash encoding
and disk writes happen only at existing terminal-drain boundaries. This proves
actual victim membership/order, not the counterfactual hit without prefetch.
"""
import collections
import contextvars
import functools
import json
import os
import time
from pathlib import Path

MODULES = {"vllm.v1.core.block_pool", "vllm.v1.core.kv_cache_manager"}
CONTEXT = contextvars.ContextVar("allocation_ledger_context", default=("native_other", None))
FRAME = contextvars.ContextVar("allocation_ledger_frame", default=None)
COUNTS = collections.Counter()
ROWS = []
LIMIT = 16384
PATCHED = set()


def note(row):
    COUNTS["events"] += 1
    if len(ROWS) < LIMIT:
        ROWS.append(row)
    else:
        COUNTS["overflow"] += 1


def encode(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    return value


def dump(reason):
    directory = os.getenv("ALLOCATION_LEDGER_DIR")
    if not directory:
        return
    start = time.monotonic_ns()
    cpu = time.thread_time_ns()
    target = Path(directory) / ("allocator-ledger-%s.json" % os.getpid())
    target.parent.mkdir(parents=True, exist_ok=True)
    obj = {"schema": "allocator-victims-v1", "reason": reason,
           "counters": dict(COUNTS), "rows": encode(ROWS),
           "limits": ["bounded actual transitions, overflow invalidates complete attribution",
                      "victim overlap does not establish counterfactual cache hit",
                      "no extra lookup, CUDA readback, synchronization or allocation"]}
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(obj))
    temp.replace(target)
    COUNTS["dump_wall_ns"] += time.monotonic_ns() - start
    COUNTS["dump_thread_cpu_ns"] += time.thread_time_ns() - cpu


def context_wrap(cls, method, classify):
    original = getattr(cls, method)

    @functools.wraps(original)
    def call(self, *args, **kwargs):
        token = CONTEXT.set(classify(self, *args, **kwargs))
        try:
            return original(self, *args, **kwargs)
        finally:
            CONTEXT.reset(token)

    setattr(cls, method, call)


def patch(module):
    name = module.__name__
    if name in PATCHED:
        return
    PATCHED.add(name)
    if name == "vllm.v1.core.block_pool":
        cls = module.BlockPool
        original = cls.get_new_blocks
        evict = cls._maybe_evict_cached_block

        @functools.wraps(evict)
        def victim(self, block):
            # Native removes both primary and alias hashes; observe before it.
            hashes = tuple(getattr(self, "cached_block_hashes_by_block", {}).get(block.block_id, ()))
            primary = block.block_hash
            if primary is not None and primary not in hashes:
                hashes += (primary,)
            result = evict(self, block)
            frame = FRAME.get()
            if result:
                COUNTS["cached_victim_blocks"] += 1
                COUNTS["cached_victim_hashes"] += len(hashes)
                record = (block.block_id, hashes)
                if frame is not None:
                    frame["victims"].append(record)
                else:
                    note({"event": "other_cache_invalidation", "at_ns": time.monotonic_ns(),
                          "context": CONTEXT.get(), "victim": record})
            return result

        @functools.wraps(original)
        def allocate(self, num_blocks):
            context = CONTEXT.get()
            COUNTS["allocation_calls:" + context[0]] += 1
            COUNTS["allocation_blocks:" + context[0]] += num_blocks
            row = {"event": "allocate", "at_ns": time.monotonic_ns(),
                   "context": context, "requested": num_blocks,
                   "free_before": self.get_num_free_blocks(), "victims": []}
            token = FRAME.set(row)
            try:
                result = original(self, num_blocks)
                row["allocated_ids"] = tuple(block.block_id for block in result)
                COUNTS["allocated_blocks:" + context[0]] += len(result)
                COUNTS["victim_blocks:" + context[0]] += len(row["victims"])
                return result
            except BaseException as error:
                row["error"] = type(error).__name__
                COUNTS["allocation_errors"] += 1
                raise
            finally:
                row["free_after"] = self.get_num_free_blocks()
                FRAME.reset(token)
                note(row)

        cls._maybe_evict_cached_block = victim
        cls.get_new_blocks = allocate
    elif name == "vllm.v1.core.kv_cache_manager":
        cls = module.KVCacheManager
        context_wrap(cls, "allocate_slots", lambda self, request, *a, **kw:
                     ("demand", request.request_id))
        original = cls.get_computed_blocks

        @functools.wraps(original)
        def computed(self, request):
            result = original(self, request)
            blocks, tokens, boundary = result
            note({"event": "native_prefix_lookup", "at_ns": time.monotonic_ns(),
                  "request_id": request.request_id, "prompt_tokens": request.num_tokens,
                  "hashes": tuple(request.block_hashes), "hit_tokens": tokens,
                  "shared_boundary": boundary,
                  "hit_block_ids": tuple(tuple(block.block_id for block in group)
                                         for group in blocks.blocks)})
            return result

        cls.get_computed_blocks = computed
    elif name == "semantic_offload.connector":
        cls = module.SemanticOffloadingConnectorScheduler
        context_wrap(cls, "_attempt_wait_prefetch", lambda self, state, **kw:
                     ("staging_wait", state.source.source_request_id))
        context_wrap(cls, "_attempt_prefetch_reservation", lambda self, req_id:
                     ("staging_preempted", req_id))
        terminal = cls._emit_finished_preservation_events

        @functools.wraps(terminal)
        def finished(self):
            before = len(self._preservation_finished)
            result = terminal(self)
            if len(self._preservation_finished) < before:
                dump("existing_terminal_drain")
            return result

        cls._emit_finished_preservation_events = finished
