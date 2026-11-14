from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vllm.v1.kv_offload.base import make_offload_key


@dataclass(frozen=True)
class PolicyGroup:
    group_id: int
    tokens_per_chunk: int
    window_chunks: int | None
    window_tokens: int | None
    keys: tuple[bytes, ...]
    boundary_keys: tuple[bytes, ...]

    def all_keys(self) -> tuple[bytes, ...]:
        return self.keys + self.boundary_keys

    def required_keys(
        self, boundary: int, partial: bool, tokens_per_hash: int
    ) -> tuple[bytes, ...]:
        end = (boundary + self.tokens_per_chunk - 1) // self.tokens_per_chunk
        window = self.window_chunks
        if window is not None and self.window_tokens is not None:
            window = max(
                window,
                (
                    self.window_tokens
                    + (-boundary % self.tokens_per_chunk)
                    + self.tokens_per_chunk
                    - 1
                )
                // self.tokens_per_chunk,
            )
        start = 0 if window is None else max(0, end - window)
        if partial:
            return self.keys[start : end - 1] + (
                self.boundary_keys[boundary // tokens_per_hash - 1],
            )
        return self.keys[start:end]


@dataclass(frozen=True)
class PolicyPrefix:
    request_id: str
    complete_tokens: int
    groups: tuple[PolicyGroup, ...]
    tokens_per_hash: int
    alignment_tokens: int
    partial_tail: bool


def prefix_from_context(context: Any) -> PolicyPrefix:
    scheduler = context._scheduler
    state = scheduler._req_status.get(context.req_id)
    if state is None:
        raise ValueError("request state unavailable at policy finish")
    config = scheduler.config
    if any(getattr(config, "eagle_proof_margins", ())):
        raise ValueError("speculative groups are outside this protocol")
    complete = context.num_computed_tokens - context.num_in_flight_tokens
    complete = min(complete, context.num_prompt_tokens)
    if complete < 0:
        raise ValueError("invalid computed token boundary")
    partial = config.supports_partial_tail
    groups = []
    for group, group_state in zip(
        config.kv_group_configs, state.group_states, strict=True
    ):
        params = group.manager_config.params
        window_tokens = (
            params.sliding_window - 1 + params.extra_retained_tokens
            if hasattr(params, "sliding_window")
            else None
        )
        boundaries = (
            tuple(
                make_offload_key(key, group.group_idx)
                for key in state.req.block_hashes[: complete // config.tokens_per_hash]
            )
            if partial
            else ()
        )
        groups.append(
            PolicyGroup(
                group.group_idx,
                group.tokens_per_chunk,
                group.sliding_window_size_in_chunks,
                window_tokens,
                tuple(group_state.offload_keys[: complete // group.tokens_per_chunk]),
                boundaries,
            )
        )
    if not groups or not any(group.window_chunks is None for group in groups):
        raise ValueError("full attention group required")
    alignment = math.lcm(*(group.tokens_per_chunk for group in groups))
    if partial and alignment != config.partial_tail_block_size:
        raise ValueError("unsupported partial-tail alignment")
    return PolicyPrefix(
        context.req_id,
        complete,
        tuple(groups),
        config.tokens_per_hash,
        alignment,
        partial,
    )


def returned_sessions(scheduler: Any, held_generations: dict[str, int]) -> set[str]:
    returned: set[str] = set()
    if not held_generations:
        return returned
    for state in scheduler._req_status.values():
        params = state.req_context.kv_transfer_params
        session = params.get("session") if params else None
        generation = params.get("generation") if params else None
        if (
            isinstance(session, str)
            and type(generation) is int
            and session in held_generations
            and generation > held_generations[session]
        ):
            returned.add(session)
    return returned
