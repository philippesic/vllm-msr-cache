from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyGroup:
    group_id: int
    tokens_per_chunk: int
    window_chunks: int | None
    keys: tuple[bytes, ...]


@dataclass(frozen=True)
class PolicyPrefix:
    request_id: str
    complete_tokens: int
    groups: tuple[PolicyGroup, ...]


def prefix_from_context(context: Any) -> PolicyPrefix:
    scheduler = context._scheduler
    state = scheduler._req_status.get(context.req_id)
    if state is None:
        raise ValueError("request state unavailable at policy finish")
    complete = context.num_computed_tokens - context.num_in_flight_tokens
    complete = min(complete, context.num_prompt_tokens)
    if complete < 0:
        raise ValueError("invalid computed token boundary")
    groups = tuple(
        PolicyGroup(
            group.group_idx,
            group.tokens_per_chunk,
            group.load_window_size_in_chunks(complete),
            tuple(group_state.offload_keys[: complete // group.tokens_per_chunk]),
        )
        for group, group_state in zip(
            scheduler.config.kv_group_configs, state.group_states, strict=True
        )
    )
    if not groups or not any(group.window_chunks is None for group in groups):
        raise ValueError("full attention group required")
    return PolicyPrefix(context.req_id, complete, groups)


def returned_sessions(scheduler: Any, held_generations: dict[str, int]) -> set[str]:
    returned = set()
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
