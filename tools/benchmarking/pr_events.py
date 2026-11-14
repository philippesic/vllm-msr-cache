from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from wait_hint_offload.evidence_events import read_journal


def token_positions(start: int, end: int, limit: int) -> set[int]:
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError("invalid scheduled token range")
    return set(range(min(start, limit), min(end, limit)))


def request_sources(
    request: dict[str, Any],
    eligible: int,
    complete: bool,
    completed: set[str],
    failed: set[str],
) -> dict[str, Any]:
    prompt = request["prompt_tokens"]
    if not 0 <= eligible <= prompt:
        raise ValueError("eligible prefix outside request")
    computed: set[int] = set()
    supported: set[int] = set()
    work = 0
    for interval in request["scheduled"]:
        positions = token_positions(
            interval["start"], interval["start"] + interval["tokens"], prompt
        )
        work += len(positions)
        computed.update(positions)
        supported.update(range(min(interval["start"], prompt)))
    if complete and computed | supported != set(range(prompt)):
        raise ValueError("completed prompt lacks scheduler coverage")
    restored: set[int] = set()
    groups: dict[str, set[int]] = {}
    pending = []
    for allocation in request["allocations"]:
        if not allocation["cpu_tokens"]:
            continue
        jobs = {str(job) for job in allocation["load_jobs"]}
        if len(jobs) != 1:
            raise ValueError("CPU allocation must identify one TP1 load job")
        if jobs & failed:
            continue
        if not jobs <= completed:
            pending.extend(sorted(jobs - completed))
            continue
        start = allocation["supplied_gpu_tokens"]
        restored.update(
            token_positions(start, start + allocation["cpu_tokens"], prompt)
        )
        for region in allocation["load_ranges"]:
            groups.setdefault(str(region["group"]), set()).update(
                token_positions(region["start"], region["end"], prompt)
            )
    if complete and pending:
        raise ValueError("completed request has unacknowledged restore")
    eligible_positions = set(range(eligible))
    recomputed = computed & eligible_positions
    cpu = (restored & supported & eligible_positions) - computed
    gpu = (supported & eligible_positions) - computed - cpu
    unknown = eligible_positions - recomputed - cpu - gpu
    if complete and unknown:
        raise ValueError("complete reusable prefix does not reconcile")
    return {
        "eligible_tokens": eligible,
        "gpu_reused_tokens": len(gpu),
        "cpu_reused_tokens": len(cpu),
        "recomputed_tokens": len(recomputed),
        "unobserved_tokens": len(unknown),
        "prompt_compute_tokens": work,
        "repeated_prompt_compute_tokens": work - len(computed),
        "successful_restored_tokens": len(restored),
        "per_group_restored_tokens": {
            group: len(tokens) for group, tokens in groups.items()
        },
        "pending_load_jobs": pending,
        "scheduled_boundaries": [entry["start"] for entry in request["scheduled"]],
        "complete": complete,
    }


def reconcile_events(directory: Path) -> dict[str, Any]:
    paths = sorted(directory.glob("*.jsonl"))
    streams: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        records = read_journal(path)
        stream = records[0]["stream"]
        if stream in {"scheduler", "worker"}:
            if stream in streams:
                raise ValueError("TP1 evidence requires one scheduler and worker")
            streams[stream] = records
    if set(streams) != {"scheduler", "worker"}:
        raise ValueError("missing native event stream")
    jobs: dict[str, dict[str, str]] = {}
    completed: set[str] = set()
    failed: set[str] = set()
    transfer = {kind: {"bytes": 0, "seconds": 0.0} for kind in ("load", "store")}
    for record in streams["worker"]:
        if record["event"] != "worker_transfers":
            continue
        data = record["data"]
        for job, count in data["completed"].items():
            if count != 1 or str(job) in completed:
                raise ValueError("duplicate or non-TP1 transfer completion")
            completed.add(str(job))
        failed.update(str(job) for job in data["failed"])
        for kind in transfer:
            transfer[kind]["bytes"] += data[kind]["bytes"]
            transfer[kind]["seconds"] += data[kind]["seconds"]
    if not failed <= completed:
        raise ValueError("failed transfer missing completion acknowledgement")
    requests: dict[str, dict[str, Any]] = {}
    external: dict[str, str] = {}
    resident: dict[str, dict[str, Any]] = {}
    ever: set[str] = set()
    evicted: set[str] = set()
    slots: dict[int, str] = {}
    group_config: dict[str, Any] | None = None
    cpu_config: dict[str, Any] | None = None
    store_refusals = store_failures = 0
    occupancy = []
    evictions = []
    for record in streams["scheduler"]:
        event, data = record["event"], record["data"]
        if event == "group_config":
            if group_config is not None:
                raise ValueError("duplicate group geometry")
            group_config = data
        elif event == "cpu_config":
            if cpu_config is not None:
                raise ValueError("duplicate CPU cache geometry")
            cpu_config = data
        elif event == "request_new":
            identity, client_id = data["request_id"], data["external_request_id"]
            if identity in requests or not client_id or client_id in external:
                raise ValueError("request identity missing or reused")
            external[client_id] = identity
            requests[identity] = data | {
                "started_ns": record["monotonic_ns"],
                "scheduled": [],
                "allocations": [],
                "finished": None,
                "initial_keys": {
                    key["key"]: (
                        "ready"
                        if resident.get(key["key"], {}).get("ready")
                        else "pending"
                        if key["key"] in resident
                        else "evicted"
                        if key["key"] in evicted
                        else "previously_stored"
                        if key["key"] in ever
                        else "never_stored"
                    )
                    for group in data["groups"]
                    for key in group
                },
            }
        elif event == "allocated":
            requests[data["request_id"]]["allocations"].append(data)
        elif event == "scheduled":
            for entry in data["requests"]:
                requests[entry["request_id"]]["scheduled"].append(entry)
            for kind in ("load", "store"):
                for job, request_id in data[kind + "_jobs"].items():
                    if job in jobs:
                        raise ValueError("transfer job ID reused")
                    jobs[job] = {"kind": kind, "request_id": request_id}
        elif event == "prepare_store":
            store_refusals += not data["accepted"]
            if not data["accepted"] and (data["stored"] or data["evicted"]):
                raise ValueError("refused store mutated cache")
            for entry in data["evicted"]:
                key = entry["key"]
                if key not in resident or resident[key]["loads"]:
                    raise ValueError("eviction of absent or loading key")
                slots.pop(resident.pop(key)["slot"])
                evicted.add(key)
                evictions.append({"monotonic_ns": record["monotonic_ns"], **entry})
            for entry, slot in zip(data["stored"], data["cpu_block_ids"], strict=True):
                key = entry["key"]
                if key in resident or slot in slots:
                    raise ValueError("store reused an occupied key or CPU slot")
                resident[key] = entry | {"slot": slot, "ready": False, "loads": 0}
                slots[slot] = key
        elif event == "complete_store":
            store_failures += not data["success"]
            for entry in data["keys"]:
                key = entry["key"]
                if key not in resident:
                    raise ValueError("store completion for absent key")
                if data["success"]:
                    resident[key]["ready"] = True
                    ever.add(key)
                    if key not in data["ready"]:
                        raise ValueError("successful store did not become ready")
                else:
                    slots.pop(resident.pop(key)["slot"])
                    if key in data["ready"]:
                        raise ValueError("failed store became ready")
        elif event in {"prepare_load", "complete_load"}:
            for entry in data["keys"]:
                key = entry["key"]
                if key not in resident or not resident[key]["ready"]:
                    raise ValueError("load of absent or unready key")
                resident[key]["loads"] += 1 if event == "prepare_load" else -1
                if resident[key]["loads"] < 0:
                    raise ValueError("load completion without prepare")
            if event == "prepare_load" and data["cpu_block_ids"] != [
                resident[entry["key"]]["slot"] for entry in data["keys"]
            ]:
                raise ValueError("load CPU slots disagree with stores")
        elif event == "request_finished":
            requests[data["request_id"]]["finished"] = data
        elif event == "reset":
            resident.clear()
            slots.clear()
        if event in {"prepare_store", "complete_store", "reset"}:
            occupancy.append(
                {"monotonic_ns": record["monotonic_ns"], "chunks": len(resident)}
            )
            if cpu_config is None or len(resident) > cpu_config["num_chunks"]:
                raise ValueError("CPU cache exceeded declared capacity")
    if group_config is None or cpu_config is None or not completed <= jobs.keys():
        raise ValueError("missing geometry or unknown completed transfer")
    alignment = (
        group_config["tokens_per_hash"]
        if group_config["partial_tail"]
        else math.lcm(*(group["tokens_per_chunk"] for group in group_config["groups"]))
    )
    return {
        "schema": "pr-event-reconciliation-v1",
        "group_config": group_config,
        "cpu_config": cpu_config,
        "eligibility_alignment_tokens": alignment,
        "requests": requests,
        "external_request_ids": external,
        "jobs": jobs,
        "completed_jobs": sorted(completed),
        "failed_jobs": sorted(failed),
        "pending_jobs": sorted(jobs.keys() - completed),
        "transfer": transfer,
        "store_refusals": store_refusals,
        "store_failures": store_failures,
        "occupancy": occupancy,
        "evictions": evictions,
        "closed_streams": {
            name: records[-1]["event"] == "close" for name, records in streams.items()
        },
    }
