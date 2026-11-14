from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.config import check_environment
from tools.process import run_owned
from tools.runtime_verify import snapshot, write_record

UPSTREAM = "e9281bd939d4ade90d6aacf4b2319aa89ea46c39"
SOURCE_PATHS = (
    "vllm/v1/kv_offload/base.py",
    "vllm/v1/kv_offload/cpu/manager.py",
    "vllm/v1/kv_offload/cpu/policies/base.py",
    "vllm/v1/kv_offload/cpu/policies/arc.py",
    "vllm/v1/kv_offload/cpu/policies/lru.py",
    "vllm/v1/kv_offload/cpu/policies/factory.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/state.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/offloading/store_triggers/on_compute.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(
    runtime_python: Path,
    upstream: Path,
    out: Path,
    *,
    private_state: bool = False,
    hold_policy: bool = False,
    bookkeeping: bool = False,
    post_response: bool = False,
) -> dict[str, Any]:
    if sum((private_state, hold_policy, bookkeeping, post_response)) > 1:
        raise ValueError("select one feasibility mode")
    root = Path(__file__).resolve().parents[2]
    out = out.resolve()
    if not out.is_relative_to(root / "artifacts"):
        raise ValueError("attempt must be under artifacts")
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True
    ).strip()
    if revision != UPSTREAM or dirty:
        raise ValueError("requires clean frozen upstream checkout")
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    identity = snapshot(root, source)
    environment = check_environment(source)
    environment["VLLM_PLUGINS"] = ""
    environment["VLLM_LOGGING_LEVEL"] = "ERROR"
    installed = Path(
        subprocess.check_output(
            [str(runtime_python.absolute()), "-c", "import vllm; print(vllm.__file__)"],
            env=environment,
            text=True,
        ).strip()
    ).parent.parent
    paths: tuple[str, ...] = SOURCE_PATHS
    patch = None
    if post_response:
        patch = next(
            record
            for record in json.loads(
                (root / "benchmarks/runtime-patches/manifest.json").read_text()
            )
            if record["name"] == "pr-post-response"
        )
        paths = (*paths, patch["target"])
    upstream_hashes = {path: _sha(upstream / path) for path in paths}
    installed_hashes = {path: _sha(installed / path) for path in paths}
    expected_hashes = dict(upstream_hashes)
    if patch is not None:
        if upstream_hashes[patch["target"]] != patch["original_sha256"]:
            raise ValueError("patch does not match frozen upstream")
        expected_hashes[patch["target"]] = patch["patched_sha256"]
    write_record(
        out / "admission.json",
        {
            "schema": "pr-post-feasibility-admission-v1"
            if post_response
            else "pr-bookkeeping-admission-v1"
            if bookkeeping
            else "pr-policy-feasibility-admission-v1"
            if hold_policy
            else "pr-private-feasibility-admission-v1"
            if private_state
            else "pr-feasibility-admission-v1",
            "upstream_revision": revision,
            "upstream_sha256": upstream_hashes,
            "installed_sha256": installed_hashes,
            "source_sha256": identity,
            "runtime_python": str(runtime_python.absolute()),
            "gpu_exposure_seconds": 0,
            "scope": "CPU policy-interface reproduction, not full runtime qualification",
        },
    )
    if installed_hashes != expected_hashes:
        write_record(out / "terminal.json", {"error": "installed source mismatch"})
        raise ValueError("installed source does not match upstream")
    captured = source / "artifacts/observed.json"
    captured.parent.mkdir()
    command = [
        str(runtime_python.absolute()),
        "-m",
        "tools.benchmark",
        "pr-feasibility-native",
        "--out",
        str(captured),
    ]
    if private_state:
        command.append("--private-state")
    if hold_policy:
        command.append("--hold-policy")
    if bookkeeping:
        command.append("--bookkeeping")
    if post_response:
        command.append("--post-response")
    start = time.monotonic()
    with (out / "execution.log").open("x") as log:
        code = run_owned(command, environment, source, 900 if bookkeeping else 120, log)
    if captured.exists():
        (out / "observed.json").write_bytes(captured.read_bytes())
    write_record(
        out / "terminal.json",
        {
            "schema": "pr-feasibility-terminal-v1",
            "exit_code": code,
            "wall_seconds": time.monotonic() - start,
            "execution_sha256": _sha(out / "execution.log"),
            "observed_sha256": _sha(out / "observed.json")
            if (out / "observed.json").exists()
            else None,
        },
    )
    return verify(out)


def verify(directory: Path) -> dict[str, Any]:
    admission = json.loads((directory / "admission.json").read_text())
    terminal = json.loads((directory / "terminal.json").read_text())
    if set(admission) != {
        "schema",
        "upstream_revision",
        "upstream_sha256",
        "installed_sha256",
        "source_sha256",
        "runtime_python",
        "gpu_exposure_seconds",
        "scope",
    } or set(terminal) != {
        "schema",
        "exit_code",
        "wall_seconds",
        "execution_sha256",
        "observed_sha256",
    }:
        raise ValueError("unknown receipt fields")
    expected_hashes = dict(admission["upstream_sha256"])
    expected_paths = set(SOURCE_PATHS)
    if admission["schema"] == "pr-post-feasibility-admission-v1":
        patch = next(
            record
            for record in json.loads(
                (
                    directory / "source/benchmarks/runtime-patches/manifest.json"
                ).read_text()
            )
            if record["name"] == "pr-post-response"
        )
        expected_paths.add(patch["target"])
        if expected_hashes.get(patch["target"]) != patch["original_sha256"]:
            raise ValueError("patch upstream identity mismatch")
        expected_hashes[patch["target"]] = patch["patched_sha256"]
    if (
        admission.get("schema")
        not in {
            "pr-feasibility-admission-v1",
            "pr-private-feasibility-admission-v1",
            "pr-policy-feasibility-admission-v1",
            "pr-bookkeeping-admission-v1",
            "pr-post-feasibility-admission-v1",
        }
        or terminal.get("schema") != "pr-feasibility-terminal-v1"
        or admission["upstream_revision"] != UPSTREAM
        or set(admission["upstream_sha256"]) != expected_paths
        or expected_hashes != admission["installed_sha256"]
        or admission["gpu_exposure_seconds"] != 0
        or terminal["exit_code"] != 0
    ):
        raise ValueError("unqualified feasibility attempt")
    if not admission["source_sha256"]:
        raise ValueError("missing source identity")
    for path, expected in admission["source_sha256"].items():
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("unsafe source path")
        if _sha(directory / "source" / path) != expected:
            raise ValueError("source snapshot drift")
    if _sha(directory / "execution.log") != terminal["execution_sha256"]:
        raise ValueError("execution drift")
    if _sha(directory / "observed.json") != terminal["observed_sha256"]:
        raise ValueError("observation drift")
    observations = json.loads((directory / "observed.json").read_text())
    if admission["schema"] == "pr-post-feasibility-admission-v1":
        return verify_delivery(observations)
    if admission["schema"] == "pr-bookkeeping-admission-v1":
        return verify_bookkeeping(observations)
    if admission["schema"] == "pr-policy-feasibility-admission-v1":
        expected = {}
        for name in (
            "RequestTTLARC",
            "RequestTTLLRU",
            "PostResponseARC",
            "AdviceOffARC",
            "AdviceOffLRU",
        ):
            expected[name] = {
                "pending_not_held": True,
                "late_ready_hold_keys": 0 if name.startswith("AdviceOff") else 2,
                "duplicate_replayed": True,
                "changed_action_refused": True,
                "impossible_eviction_preserved_holds": True,
                "return_released_before_eviction": True,
                "reset_cleared_metadata": True,
            }
        if observations.get("schema") in {
            "pr-policy-observation-v2",
            "pr-policy-observation-v3",
        }:
            extended = observations.get("extended", {})
            identities = extended.get("identity_sequences", {})
            flags = {
                "gpu_only_shared_prefix_hold",
                "shared_union_counted_once",
                "cancellation_preserves_other_owner",
                "expiry_and_duplicate_do_not_rehold",
                "failed_eviction_preserves_holds",
                "yield_prevents_added_store_failure",
                "failed_store_never_ready_or_held",
            }
            if set(extended) != flags | {"identity_sequences"} or any(
                extended[key] is not True for key in flags
            ):
                raise ValueError("missing extended hold evidence")
            if set(identities) != {"arc", "lru"} or any(
                set(value) != {"disabled", "stock"}
                or len(value["stock"]) != 5
                or value["disabled"] != value["stock"]
                for value in identities.values()
            ):
                raise ValueError("hint-off eviction identity failed")
            expected_observation = {
                "schema": observations["schema"],
                "cases": expected,
                "extended": extended,
            }
        else:
            expected_observation = {
                "schema": "pr-policy-observation-v1",
                "cases": expected,
            }
        if observations.get("schema") == "pr-policy-observation-v3":
            expected_observation["geometry"] = {
                "partial_tail_requires_every_group_ready": True,
                "partial_boundary_matches_native_lookup": True,
                "forks_preserve_shared_anchor_and_distinct_checkpoints": True,
                "window_uses_selected_boundary_not_unaligned_finish": True,
            }
            expected_observation["tiering"] = {
                "primary_tier_preserves_scheduler_context": True,
                "late_store_after_finish_joins_hold": True,
                "return_releases_and_reset_clears": True,
            }
        if observations != expected_observation:
            raise ValueError("hold policy observation changed")
        return {
            "schema": "pr-policy-feasibility-verification-v1",
            "observations": observations,
        }
    if admission["schema"] == "pr-private-feasibility-admission-v1":
        if observations != {
            "schema": "pr-private-context-observation-v1",
            "native_context_class": "_VllmReqContext",
            "native_state_class": "RequestOffloadState",
            "full_gpu_hit_keys_per_group": [2, 2],
            "windows": [None, 1],
            "newer_return_detected_before_policy_callback": True,
            "older_inflight_request_does_not_release_newer_hold": True,
            "unrelated_session_unchanged": True,
            "scope": "real context/state methods with CPU scheduler/request fixtures; no serving or hold qualification",
        }:
            raise ValueError("private context observation changed")
        return {
            "schema": "pr-private-feasibility-verification-v1",
            "observations": observations,
        }
    if set(observations) != {"arc", "lru"}:
        raise ValueError("missing policy observations")
    for observed in observations.values():
        if set(observed) != {
            "callbacks_at_return",
            "gpu_hit_finish_groups",
            "shared_gpu_hit_finish_groups",
            "scheduler_gpu_hit_chunks_skipped",
            "request_level_alternative_start",
            "ready_cpu_keys_present",
            "positions_available_privately",
            "late_store_readiness_callback",
            "observer_identity_evictions",
            "stock_identity_evictions",
            "callback_trace",
        }:
            raise ValueError("unknown observation fields")
        if (
            observed["scheduler_gpu_hit_chunks_skipped"] != 2
            or observed["request_level_alternative_start"] != 0
            or len(observed["observer_identity_evictions"]) != 5
            or observed["observer_identity_evictions"]
            != observed["stock_identity_evictions"]
            or not observed["callback_trace"]
            or observed["callbacks_at_return"] != []
            or observed["gpu_hit_finish_groups"] != []
            or observed["shared_gpu_hit_finish_groups"] != []
            or not observed["late_store_readiness_callback"]
            or len(observed["ready_cpu_keys_present"]) != 2
            or observed["positions_available_privately"] != 2
        ):
            raise ValueError("interface observation changed")
    return {
        "schema": "pr-feasibility-verification-v1",
        "upstream_revision": UPSTREAM,
        "scope": "CPU manager/policy callbacks only; no GPU or efficacy qualification",
        "observations": observations,
    }


def verify_bookkeeping(observations: dict[str, Any]) -> dict[str, Any]:
    if set(observations) != {"schema", "scope", "percentile", "metadata", "cells"}:
        raise ValueError("unknown bookkeeping fields")
    if observations["schema"] != "pr-bookkeeping-observation-v1":
        raise ValueError("unknown bookkeeping version")
    expected = {
        (policy, capacity, sessions)
        for policy in ("RequestTTLARC", "RequestTTLLRU")
        for capacity in (10000, 100000, 1000000)
        for sessions in (0, 100, 1000)
    }
    actual = set()
    for cell in observations["cells"]:
        if set(cell) != {
            "policy",
            "chunks",
            "held_sessions",
            "keys_per_held_session",
            "held_physical_keys",
            "active_requests",
            "evict_n",
            "evict_ns",
            "evict_p50_ns",
            "evict_p99_ns",
            "hold_finish_ns",
            "hold_finish_p50_ns",
            "hold_finish_p99_ns",
            "owned_metadata_bytes",
            "charged_metadata_bytes",
        }:
            raise ValueError("unknown bookkeeping cell fields")
        identity = (cell["policy"], cell["chunks"], cell["held_sessions"])
        if identity in actual or identity not in expected:
            raise ValueError("unexpected bookkeeping cell")
        actual.add(identity)
        if cell["active_requests"] != 16 or cell["evict_n"] != 1:
            raise ValueError("unexpected bookkeeping workload")
        sessions = cell["held_sessions"]
        per_session = min(32, cell["chunks"] // 2 // sessions) if sessions else 0
        if (
            cell["keys_per_held_session"] != per_session
            or cell["held_physical_keys"] != sessions * per_session
        ):
            raise ValueError("hold workload mismatch")
        for metric, count in (("evict", 200), ("hold_finish", sessions)):
            values = cell[metric + "_ns"]
            if len(values) != count or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise ValueError("missing timing samples")
            for suffix, quantile in (("p50_ns", 0.5), ("p99_ns", 0.99)):
                value = (
                    sorted(values)[math.ceil(len(values) * quantile) - 1]
                    if values
                    else None
                )
                if cell[metric + "_" + suffix] != value:
                    raise ValueError("incorrect timing summary")
        if cell["owned_metadata_bytes"] <= 0 or cell["charged_metadata_bytes"] < 0:
            raise ValueError("missing metadata accounting")
    if actual != expected:
        raise ValueError("incomplete bookkeeping matrix")
    return {"schema": "pr-bookkeeping-verification-v1", "observations": observations}


def verify_delivery(observed: dict[str, Any]) -> dict[str, Any]:
    flags = {
        "engine_utility_dispatch_exercised",
        "deadline_origin_is_acceptance",
        "late_ready_joined",
        "duplicate_replayed",
        "closed_schema_refused_without_mutation",
        "cancellation_released",
    }
    if set(observed) != flags | {
        "schema",
        "identity_sequences",
        "control_delivery_ns",
        "control_delivery_p50_ns",
        "control_delivery_p99_ns",
        "scope",
    }:
        raise ValueError("unknown post-response observation fields")
    if observed["schema"] != "pr-delivery-observation-v1" or any(
        observed[flag] is not True for flag in flags
    ):
        raise ValueError("failed post-response lifecycle")
    identities = observed["identity_sequences"]
    if (
        set(identities) != {"A3", "A3p"}
        or len(identities["A3"]) != 5
        or identities["A3"] != identities["A3p"]
    ):
        raise ValueError("patched-control identity failed")
    values = observed["control_delivery_ns"]
    if len(values) != 1000 or any(
        type(value) is not int or value < 0 for value in values
    ):
        raise ValueError("incomplete delivery cost samples")
    for quantile, suffix in ((0.5, "p50_ns"), (0.99, "p99_ns")):
        if (
            observed["control_delivery_" + suffix]
            != sorted(values)[math.ceil(len(values) * quantile) - 1]
        ):
            raise ValueError("delivery cost summary mismatch")
    return {"schema": "pr-delivery-verification-v1", "observations": observed}
