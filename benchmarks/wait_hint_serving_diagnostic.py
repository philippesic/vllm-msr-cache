# SPDX-License-Identifier: Apache-2.0
"""Frozen native-child serving diagnostics for known-prefix wait hints.

This is a correctness/readiness harness, not a performance campaign.  It builds
one source-bound A/B/A request artifact, emits four exact fresh-server cases and
can execute one case on the authorized Runner host.  It never selects inputs
from runtime outcomes.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.eviction_prefetch_campaign import (
    FROZEN_ORIGINAL_SEMANTIC_CONFIG,
    NATIVE_RUNTIME_VARIANTS,
    NATIVE_RUNTIME_VARIANTS_BINDING_SHA256,
    _engine_server_args,
    _expected_runtime_resolution,
    _git_identity,
    _parse_preservation_events,
    _parse_runtime_resolutions,
    _parse_wait_hint_events,
    _wait_for_campaign_drain,
    consume_campaign_sse,
    digest,
    file_digest,
    validate_request_manifest,
)

REQUEST_SCHEMA = "ep-wait-hint-serving-requests-v1"
PLAN_SCHEMA = "ep-wait-hint-serving-plan-v1"
CASE_RESULT_SCHEMA = "ep-wait-hint-serving-case-result-v1"
SUITE_RESULT_SCHEMA = "ep-wait-hint-serving-suite-result-v1"
PLUGIN_DISCOVERY_SCHEMA = "ep-wait-hint-plugin-discovery-v1"
BINARY_DISCOVERY_SCHEMA = "ep-native-binary-inventory-verification-v1"
SOURCE_REQUEST_SHA256 = (
    "cc9e6db8ff1c7c409e0ac7ac107483fe4d1aceb27969ad662321451bc8d35240"
)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_NAMESPACE = {
    "model_revision": MODEL_REVISION,
    "tokenizer_revision": MODEL_REVISION,
    "adapter_namespace": "none",
    "layout_namespace": "dense-tp1-bf16-block16",
}
ENGINE_PROFILE = {
    "gpu": "RTX 5080",
    "total_gpu_blocks": 320,
    "kv_block_size_tokens": 16,
    "max_model_len": 5120,
    "gpu_memory_utilization": 0.5,
    "cpu_bytes": 256 * 1024 * 1024,
    "enforce_eager": True,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 8192,
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "seed": 1103,
    "max_new_tokens": 32,
}


class DiagnosticError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def _bound_validation_path(
    runtime_path: Path, review_path: Path, label: str
) -> Path:
    if os.environ.get("EP_STRICT_STAGED_PATHS") == "1":
        _require(
            runtime_path.is_absolute()
            and str(runtime_path).startswith("/home/")
            and runtime_path.is_file(),
            f"{label} strict staged runtime path",
        )
        return runtime_path
    return review_path if review_path.is_file() else runtime_path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def _record_hash(record: Mapping[str, object]) -> str:
    return digest(record)


def build_request_artifact(
    request_document: Mapping[str, object],
    source_path: Path,
    source_runtime_path: str | None = None,
) -> dict[str, object]:
    _require(file_digest(source_path) == SOURCE_REQUEST_SHA256, "request source hash")
    validate_request_manifest(request_document)
    requests = request_document["requests"]
    screen = [
        request
        for request in requests
        if request["split"] == "screen"
        and request["context_budget"] == 2048
        and request["episode"] == "reusable_prefix"
    ]
    qasper_warm = sorted(
        (
            request
            for request in screen
            if request["family"] == "qasper" and request["phase"] == "warm1"
        ),
        key=lambda request: int(request["request_index"]),
    )
    _require(qasper_warm, "missing screen Qasper warm request")
    a_warm = qasper_warm[0]
    a_resumes = [
        request
        for request in screen
        if request["session_id"] == a_warm["session_id"]
        and request["phase"] == "warm2"
        and request["task_origin"] == "supplied_question_1"
    ]
    _require(len(a_resumes) == 1, "A q1 continuation is not unique")
    a_resume = a_resumes[0]
    pollution = [
        request
        for request in screen
        if request["phase"] == "pollution"
        and request["key_chain_digest"] != a_warm["key_chain_digest"]
    ]
    max_pollution_tokens = max(len(request["prompt_token_ids"]) for request in pollution)
    b_candidates = [
        request
        for request in pollution
        if len(request["prompt_token_ids"]) == max_pollution_tokens
    ]
    _require(len(b_candidates) == 1, "longest pollution request is not unique")
    b_pollution = b_candidates[0]
    context_span = a_warm.get("context_token_span")
    _require(
        isinstance(context_span, list)
        and len(context_span) == 2
        and all(isinstance(value, int) for value in context_span),
        "A context span",
    )
    boundary = context_span[1] // ENGINE_PROFILE["kv_block_size_tokens"] * ENGINE_PROFILE[
        "kv_block_size_tokens"
    ]
    warm_prefix = a_warm["prompt_token_ids"][:boundary]
    resume_prefix = a_resume["prompt_token_ids"][:boundary]
    _require(warm_prefix == resume_prefix, "A q0/q1 immutable prefix mismatch")
    _require(
        len(b_pollution["prompt_token_ids"]) + ENGINE_PROFILE["max_new_tokens"]
        <= ENGINE_PROFILE["max_model_len"],
        "B exceeds the frozen 320-block model limit",
    )
    roles = (
        ("a_warm_q0", a_warm),
        ("b_longest_pollution", b_pollution),
        ("a_resume_q1", a_resume),
    )
    artifact = {
        "schema": REQUEST_SCHEMA,
        "source_request_manifest": {
            "path": source_runtime_path or str(source_path.resolve()),
            "source_review_path": str(source_path.resolve()),
            "sha256": SOURCE_REQUEST_SHA256,
        },
        "selection": {
            "split": "screen",
            "context_budget": 2048,
            "a_rule": "first manifest-order Qasper warm1 target and its unique q1 warm2 continuation",
            "b_rule": "unique maximum prompt-token count among different-key-chain pollution records",
            "a_warm_request_id": a_warm["request_id"],
            "a_resume_request_id": a_resume["request_id"],
            "b_request_id": b_pollution["request_id"],
            "b_prompt_tokens": max_pollution_tokens,
            "known_prefix_boundary_tokens": boundary,
            "known_prefix_token_digest": digest(warm_prefix),
        },
        "model_namespace": dict(MODEL_NAMESPACE),
        "engine_profile": dict(ENGINE_PROFILE),
        "requests": [
            {
                "diagnostic_role": role,
                "original_request_id": record["request_id"],
                "original_record_sha256": _record_hash(record),
                "original_record": copy.deepcopy(record),
            }
            for role, record in roles
        ],
        "sequence": ["a_warm_q0", "b_longest_pollution", "wait_boundary", "a_resume_q1"],
        "interpretation": "Constructed functional diagnostic only; not the 32-request episode and not performance evidence.",
    }
    artifact["artifact_digest"] = digest(artifact)
    return artifact


def validate_request_artifact(document: Mapping[str, object]) -> None:
    _require(document.get("schema") == REQUEST_SCHEMA, "diagnostic request schema")
    expected_digest = document.get("artifact_digest")
    body = dict(document)
    body.pop("artifact_digest", None)
    _require(expected_digest == digest(body), "diagnostic request artifact digest")
    source = document.get("source_request_manifest")
    _require(isinstance(source, Mapping), "diagnostic source binding")
    runtime_source_path = Path(str(source.get("path")))
    review_source_path = Path(
        str(source.get("source_review_path", runtime_source_path))
    )
    source_path = _bound_validation_path(
        runtime_source_path, review_source_path, "diagnostic source request"
    )
    _require(
        source.get("sha256") == SOURCE_REQUEST_SHA256
        and source_path.is_file()
        and file_digest(source_path) == SOURCE_REQUEST_SHA256,
        "diagnostic source request binding",
    )
    with source_path.open(encoding="utf-8") as input_file:
        rebuilt = build_request_artifact(
            json.load(input_file),
            source_path,
            source_runtime_path=str(source.get("path")),
        )
    rebuilt["source_request_manifest"]["source_review_path"] = source.get(
        "source_review_path"
    )
    rebuilt.pop("artifact_digest")
    rebuilt["artifact_digest"] = digest(rebuilt)
    _require(rebuilt == document, "diagnostic request artifact does not rebuild")


def _diagnostic_config(
    *, hint: bool, tool_gap_ms: int, implementation_revision: str
) -> dict[str, object]:
    config = copy.deepcopy(FROZEN_ORIGINAL_SEMANTIC_CONFIG)
    config["policy"] = "arc"
    config["request_controls"]["tool_gap_ms"] = tool_gap_ms
    extra = config["extra_config"]
    extra.update(
        {
            "implementation_revision": implementation_revision,
            "preservation_events": True,
            "admission_namespace": dict(MODEL_NAMESPACE),
            "admission_mode": "disabled",
            "prefetch_budget_fraction": 0.05 if hint else 0.0,
            "prefetch_selector": "request_order",
            "prefetch_trigger": "known_prefix_wait_hint" if hint else "preemption",
            "soft_retention": "bounded_hint" if hint else "off",
            "staging_mode": "owned_reclaimable" if hint else "dynamic_scratch",
            "shared_prefetch_scorer": False,
            "decision_feature_trace": False,
        }
    )
    return config


def build_plan(
    request_artifact_path: Path,
    *,
    request_runtime_path: str,
    wheel_path: str,
    wheel_sha256: str,
    plugin_root: str,
    binary_inventory_path: str,
    child_binary_inventory_path: str,
    child_binary_inventory_sha256: str,
    staged_preflight_manifest_path: str,
    staged_preflight_report_path: str,
    extension_root: str,
    extension_revision: str,
    extension_tree: str,
    output_root: str,
    port_base: int = 19200,
) -> dict[str, object]:
    with request_artifact_path.open(encoding="utf-8") as input_file:
        request_artifact = json.load(input_file)
    validate_request_artifact(request_artifact)
    cases = []
    case_specs = (
        ("original149-no-hint", "original149", False, 250, False),
        ("child-no-hint", "wait_hint_utility_v1", False, 250, False),
        ("child-hint-250", "wait_hint_utility_v1", True, 250, False),
        ("child-hint-0-negative", "wait_hint_utility_v1", True, 0, True),
    )
    for offset, (case_id, runtime_id, hint, gap_ms, negatives) in enumerate(case_specs):
        runtime = NATIVE_RUNTIME_VARIANTS[runtime_id]
        config = _diagnostic_config(
            hint=hint,
            tool_gap_ms=gap_ms,
            implementation_revision=extension_revision,
        )
        output = str(Path(output_root) / case_id / "result.json")
        cases.append(
            {
                "case_id": case_id,
                "runtime_variant": runtime_id,
                "runtime": dict(runtime),
                "endpoint_plugin_enabled": hint,
                "tool_gap_ms": gap_ms,
                "negative_calls_after_primary": negatives,
                "zero_gap_expected_behavior": (
                    "request_order_no_action_no_allocation"
                    if hint and gap_ms == 0
                    else None
                ),
                "config": config,
                "config_hash": digest(config),
                "request_bindings": [
                    {
                        "diagnostic_role": entry["diagnostic_role"],
                        "original_request_id": entry["original_request_id"],
                        "original_record_sha256": entry[
                            "original_record_sha256"
                        ],
                        "prompt_token_digest": entry["original_record"][
                            "prompt_token_digest"
                        ],
                        "prompt_token_count": len(
                            entry["original_record"]["prompt_token_ids"]
                        ),
                        "external_request_id": _external_id(case_id, index),
                    }
                    for index, entry in enumerate(request_artifact["requests"])
                ],
                "port": port_base + offset,
                "output": output,
                "argv": [
                    "/home/philip/semantic-cache-agent/wsl-session-20260810-native/.venv/bin/python",
                    "-m",
                    "benchmarks.wait_hint_serving_diagnostic",
                    "execute-case",
                    "--plan={plan}",
                    f"--case-id={case_id}",
                    f"--output={output}",
                ],
            }
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "source": {
            "extension_revision": extension_revision,
            "extension_tree": extension_tree,
            "extension_root": extension_root,
            "runtime_variants_binding_sha256": NATIVE_RUNTIME_VARIANTS_BINDING_SHA256,
        },
        "request_artifact": {
            "path": request_runtime_path,
            "source_review_path": str(request_artifact_path.resolve()),
            "sha256": file_digest(request_artifact_path),
            "artifact_digest": request_artifact["artifact_digest"],
        },
        "plugin_distribution": {
            "wheel_path": wheel_path,
            "wheel_sha256": wheel_sha256,
            "isolated_plugin_root": plugin_root,
            "distribution": "semantic-offload",
            "version": "0.1.0",
            "entry_point_group": "vllm.endpoint_plugins",
            "entry_point_name": "semantic_wait_hint",
            "entry_point_value": "semantic_offload.wait_hint_endpoint:SemanticWaitHintEndpointPlugin",
        },
        "native_binary_inventory": {
            "path": binary_inventory_path,
            "sha256": NATIVE_RUNTIME_VARIANTS["wait_hint_utility_v1"][
                "native_binary_inventory_sha256"
            ],
            "verification": "hash every declared resolved native file before any case",
        },
        "child_runtime_binary_inventory": {
            "path": child_binary_inventory_path,
            "sha256": child_binary_inventory_sha256,
            "verification": "hash child-resolved source-package binaries and import",
        },
        "preflight": {
            "staged_environment_manifest": staged_preflight_manifest_path,
            "staged_environment_output": staged_preflight_report_path,
            "plugin_discovery_output": str(
                Path(output_root) / "preflight" / "plugin-discovery.json"
            ),
            "binary_verification_output": str(
                Path(output_root) / "preflight" / "binary-verification.json"
            ),
            "argv": [
                [
                    "/home/philip/semantic-cache-agent/wsl-session-20260810-native/.venv/bin/python",
                    "-m",
                    "benchmarks.staged_environment_preflight",
                    f"--manifest={staged_preflight_manifest_path}",
                    f"--output={staged_preflight_report_path}",
                ],
                [
                    "/home/philip/semantic-cache-agent/wsl-session-20260810-native/.venv/bin/python",
                    "-m",
                    "benchmarks.wait_hint_serving_diagnostic",
                    "verify-plugin",
                    "--plan={plan}",
                    f"--output={Path(output_root) / 'preflight' / 'plugin-discovery.json'}",
                ],
                [
                    "/home/philip/semantic-cache-agent/wsl-session-20260810-native/.venv/bin/python",
                    "-m",
                    "benchmarks.wait_hint_serving_diagnostic",
                    "verify-binaries",
                    "--plan={plan}",
                    f"--output={Path(output_root) / 'preflight' / 'binary-verification.json'}",
                ],
            ],
        },
        "engine_profile": dict(ENGINE_PROFILE),
        "execution_environment": {
            "PYTHONPATH": os.pathsep.join(
                [
                    plugin_root,
                    extension_root,
                    str(NATIVE_RUNTIME_VARIANTS["wait_hint_utility_v1"]["path"]),
                ]
            ),
            "VLLM_CLI_MODULE": "vllm.entrypoints.cli.main",
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_BATCH_INVARIANT": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "1103",
            "SEMANTIC_OFFLOAD_DISABLE_PREFETCH": "0",
        },
        "timing": {
            "wait_origin": "monotonic boundary after A and B completions plus two-request drain",
            "eligibility_ns": "wait_origin_ns + tool_gap_ms*1000000",
            "positive_hint_rpc": "launched at wait origin; request release never waits for acknowledgement",
            "zero_gap": "resume task launched at the same wait origin concurrently with hint RPC",
            "negative_calls": "after primary A/B/A sequence; cannot supply the positive transfer",
        },
        "cases": cases,
        "suite_checks": [
            "exact A/B/A output token equality across all four cases",
            "original149 and child default paths have identical extension/config/request/engine fields",
            "positive case requires actual CPU-ready staging suffix and incomplete native GPU prefix",
            "positive case requires submitted, ready and consumed bytes plus splice fence acknowledgement",
            "zero-gap permits bounded no-action but no crash, unready read or delayed release",
            "negative calls allocate zero blocks and are identity/status accounted",
        ],
        "coverage": "functional readiness only; no latency or campaign claim",
    }
    plan["plan_digest"] = digest(plan)
    return plan


def verify_binary_inventory(
    inventory_path: Path, expected_sha256: str
) -> dict[str, object]:
    _require(
        inventory_path.is_file()
        and file_digest(inventory_path) == expected_sha256,
        "native binary inventory artifact hash",
    )
    with inventory_path.open(encoding="utf-8") as input_file:
        inventory = json.load(input_file)
    files = inventory.get("files")
    _require(
        inventory.get("schema") == "ep-native149-binary-inventory-v1"
        and isinstance(files, list)
        and inventory.get("binary_files_count") == len(files),
        "native binary inventory schema/count",
    )
    verified = []
    for entry in files:
        _require(isinstance(entry, Mapping), "native binary inventory entry")
        path = Path(str(entry.get("path")))
        resolved = path.resolve()
        _require(
            path.exists()
            and str(resolved) == entry.get("resolved_path")
            and path.is_symlink() is bool(entry.get("symlink"))
            and resolved.is_file()
            and resolved.stat().st_size == entry.get("bytes")
            and file_digest(resolved) == entry.get("sha256"),
            f"native binary mismatch: {path}",
        )
        verified.append(
            {
                "path": str(path),
                "resolved_path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": entry["sha256"],
                "symlink": path.is_symlink(),
            }
        )
    return {
        "schema": BINARY_DISCOVERY_SCHEMA,
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": expected_sha256,
        "verified_file_count": len(verified),
        "verified_total_bytes": sum(int(entry["bytes"]) for entry in verified),
        "verified_entries_digest": digest(verified),
        "verified": True,
    }


def validate_plan(plan: Mapping[str, object], plan_path: Path) -> None:
    _require(plan.get("schema") == PLAN_SCHEMA, "diagnostic plan schema")
    body = dict(plan)
    plan_digest = body.pop("plan_digest", None)
    _require(plan_digest == digest(body), "diagnostic plan digest")
    request_binding = plan.get("request_artifact")
    _require(isinstance(request_binding, Mapping), "diagnostic request binding")
    runtime_request_path = Path(str(request_binding.get("path")))
    review_request_path = Path(
        str(request_binding.get("source_review_path", runtime_request_path))
    )
    request_path = _bound_validation_path(
        runtime_request_path, review_request_path, "diagnostic request artifact"
    )
    _require(
        request_path.is_file()
        and request_binding.get("sha256") == file_digest(request_path),
        "diagnostic request file hash",
    )
    _require(len(plan.get("cases", [])) == 4, "diagnostic case count")
    binary_binding = plan.get("native_binary_inventory")
    _require(
        isinstance(binary_binding, Mapping)
        and binary_binding.get("sha256")
        == NATIVE_RUNTIME_VARIANTS["wait_hint_utility_v1"][
            "native_binary_inventory_sha256"
        ],
        "diagnostic binary inventory binding",
    )
    child_binary_binding = plan.get("child_runtime_binary_inventory")
    _require(
        isinstance(child_binary_binding, Mapping)
        and isinstance(child_binary_binding.get("path"), str)
        and str(child_binary_binding["path"]).startswith("/home/")
        and isinstance(child_binary_binding.get("sha256"), str)
        and len(str(child_binary_binding["sha256"])) == 64,
        "diagnostic child binary inventory binding",
    )
    staged_preflight = plan.get("preflight")
    _require(
        isinstance(staged_preflight, Mapping)
        and str(staged_preflight.get("staged_environment_manifest", "")).startswith(
            "/home/"
        )
        and str(staged_preflight.get("staged_environment_output", "")).startswith(
            "/home/"
        ),
        "diagnostic staged environment preflight binding",
    )
    _require(plan_path.is_absolute(), "diagnostic plan path must be absolute")


def verify_plugin_discovery(
    plugin_root: Path,
    expected: Mapping[str, object],
    *,
    require_native_loader: bool = True,
) -> dict[str, object]:
    _require(plugin_root.is_dir(), "isolated plugin root is missing")
    distributions = list(importlib.metadata.distributions(path=[str(plugin_root)]))
    matching = [
        distribution
        for distribution in distributions
        if distribution.metadata.get("Name", "").lower() == "semantic-offload"
    ]
    _require(len(matching) == 1, "semantic-offload distribution metadata")
    distribution = matching[0]
    _require(
        distribution.version == expected.get("version", "0.1.0"),
        "plugin distribution version",
    )
    entry_points = [
        entry
        for entry in distribution.entry_points
        if entry.group == expected["entry_point_group"]
        and entry.name == expected["entry_point_name"]
    ]
    _require(len(entry_points) == 1, "wait-hint endpoint entry point discovery")
    entry = entry_points[0]
    _require(entry.value == expected["entry_point_value"], "entry point value")
    old_path = list(sys.path)
    old_dont_write_bytecode = sys.dont_write_bytecode
    old_plugins_env = os.environ.get("VLLM_PLUGINS")
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "semantic_offload" or name.startswith("semantic_offload.")
    }
    try:
        for name in saved_modules:
            sys.modules.pop(name, None)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(plugin_root))
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("semantic_offload.wait_hint_endpoint")
        _require(spec is not None and spec.origin is not None, "plugin module spec")
        origin = Path(spec.origin).resolve()
        _require(plugin_root.resolve() in origin.parents, "plugin imported outside root")
        plugin_class = entry.load()
        plugin = plugin_class()
        disabled_plugins = []
        enabled_plugins = []
        if require_native_loader:
            from vllm.plugins import load_endpoint_plugins

            os.environ["VLLM_PLUGINS"] = ""
            disabled_plugins = load_endpoint_plugins(("generate",))
            os.environ["VLLM_PLUGINS"] = str(expected["entry_point_name"])
            enabled_plugins = load_endpoint_plugins(("generate",))
            _require(not disabled_plugins, "endpoint plugin loaded without allowlist")
            _require(
                len(enabled_plugins) == 1
                and enabled_plugins[0].name == expected["entry_point_name"],
                "native endpoint plugin loader did not register plugin",
            )
    finally:
        for name in list(sys.modules):
            if name == "semantic_offload" or name.startswith("semantic_offload."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.dont_write_bytecode = old_dont_write_bytecode
        sys.path[:] = old_path
        if old_plugins_env is None:
            os.environ.pop("VLLM_PLUGINS", None)
        else:
            os.environ["VLLM_PLUGINS"] = old_plugins_env
    inventory = [
        {
            "path": str(path.relative_to(plugin_root)),
            "sha256": file_digest(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(plugin_root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return {
        "schema": PLUGIN_DISCOVERY_SCHEMA,
        "plugin_root": str(plugin_root.resolve()),
        "distribution_name": distribution.metadata["Name"],
        "distribution_version": distribution.version,
        "entry_point_group": entry.group,
        "entry_point_name": entry.name,
        "entry_point_value": entry.value,
        "module_origin": str(origin),
        "plugin_name": plugin.name,
        "required_tasks": list(plugin.required_tasks),
        "native_loader_disabled_count": len(disabled_plugins),
        "native_loader_enabled_names": [
            loaded.name for loaded in enabled_plugins
        ],
        "native_loader_required": require_native_loader,
        "file_inventory": inventory,
        "file_inventory_digest": digest(inventory),
        "verified": True,
    }


def _external_id(case_id: str, index: int) -> str:
    return f"sc|diag-run-{case_id}|diag-cell-{case_id}|diag|sample-{index}"


def _request_payload(
    entry: Mapping[str, object],
    *,
    case: Mapping[str, object],
    index: int,
    observed_prefix_id: str | None,
    boundary: int,
    hint_event_id: str | None,
) -> dict[str, object]:
    record = entry["original_record"]
    external_id = _external_id(str(case["case_id"]), index)
    params: dict[str, object] = {
        "campaign_schema": "ep-campaign-serving-v1",
        "external_request_id": external_id,
        "semantic_request_id": external_id,
        "semantic_sample_id": external_id,
        "sample_index": index,
        "benchmark_client_sequence": index,
        "seed": ENGINE_PROFILE["seed"],
        "run_id": f"diag-run-{case['case_id']}",
        "cell_id": f"diag-cell-{case['case_id']}",
        "session_id": "diag-session-A" if entry["diagnostic_role"].startswith("a_") else "diag-session-B",
        "generation": 0,
        "key_chain_digest": record["key_chain_digest"],
        "candidate_config_hash": case["config_hash"],
        "campaign_provenance": {
            "manifest_request_id": record["request_id"],
            "manifest_request_index": record["request_index"],
            "human_cell_id": case["case_id"],
            "human_profile_id": "wait-hint-serving-diagnostic",
            "schedule_role": entry["diagnostic_role"],
            "content_origin": record["content_origin"],
            "arrival_origin": "frozen_diagnostic_sequence",
            "wait_origin": "post_a_b_completion_boundary",
            "pressure_origin": "single_longest_screen_pollution_request",
            "source_id": record["source_id"],
            "session_id": record["session_id"],
            "context_token_span": record.get("context_token_span"),
            "family": record["family"],
            "split": record["split"],
        },
    }
    if observed_prefix_id is not None:
        params["observed_prefix_id"] = observed_prefix_id
        params["observed_prefix_boundary_tokens"] = boundary
    if hint_event_id is not None:
        params["wait_hint_event_id"] = hint_event_id
    return {
        "model": MODEL_ID,
        "request_id": external_id,
        "prompt": record["prompt_token_ids"],
        "max_tokens": ENGINE_PROFILE["max_new_tokens"],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "seed": ENGINE_PROFILE["seed"] + index,
        "stream": True,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
        "kv_transfer_params": params,
    }


def _request_result(
    entry: Mapping[str, object],
    payload: Mapping[str, object],
    response: Mapping[str, object],
) -> dict[str, object]:
    record = entry["original_record"]
    return {
        "diagnostic_role": entry["diagnostic_role"],
        "original_request_id": entry["original_request_id"],
        "original_record_sha256": entry["original_record_sha256"],
        "prompt_token_digest": record["prompt_token_digest"],
        "prompt_token_count": len(record["prompt_token_ids"]),
        "input_payload": copy.deepcopy(dict(payload)),
        "input_payload_digest": digest(payload),
        **dict(response),
    }


async def _send_completion(session, base_url: str, payload: Mapping[str, object]) -> dict[str, object]:
    started_ns = time.monotonic_ns()
    error = None
    status = None
    stream_result: dict[str, object] = {
        "output_token_ids": [],
        "output_token_digest": digest([]),
        "token_receipt_ns": [],
        "chunk_receipt_ns": [],
        "finish_reasons": [],
        "done_seen": False,
        "terminal_complete": False,
        "first_token_ns": None,
    }
    try:
        async with session.post(f"{base_url}/v1/completions", json=payload) as response:
            status = response.status
            stream_result = await consume_campaign_sse(response.content)
    except Exception as exc:  # noqa: BLE001 - report request failure in result
        error = repr(exc)
    completed_ns = time.monotonic_ns()
    return {
        "external_request_id": payload["request_id"],
        "request_start_ns": started_ns,
        "request_end_ns": completed_ns,
        "http_status": status,
        "error": error,
        "success": error is None and status == 200 and stream_result["terminal_complete"],
        **stream_result,
    }


async def _post_hint(session, base_url: str, payload: Mapping[str, object], label: str) -> dict[str, object]:
    started_ns = time.monotonic_ns()
    status = None
    response_payload: object = None
    error = None
    try:
        async with session.post(
            f"{base_url}/plugins/semantic-offload/wait-hints", json=payload
        ) as response:
            status = response.status
            response_payload = await response.json()
    except Exception as exc:  # noqa: BLE001 - report request failure in result
        error = repr(exc)
    return {
        "label": label,
        "payload": dict(payload),
        "payload_digest": digest(payload),
        "started_ns": started_ns,
        "completed_ns": time.monotonic_ns(),
        "http_status": status,
        "response": response_payload,
        "error": error,
    }


async def _dispatch_at_release_boundary(
    *,
    wait_origin_ns: int,
    tool_gap_ms: int,
    hint_call,
    resume_call,
) -> dict[str, object]:
    """Start hint work at the boundary and release resume by eligibility only."""
    eligibility_ns = wait_origin_ns + tool_gap_ms * 1_000_000
    hint_task = asyncio.create_task(hint_call()) if hint_call is not None else None
    remaining = (eligibility_ns - time.monotonic_ns()) / 1_000_000_000
    if remaining > 0:
        await asyncio.sleep(remaining)
    dispatch_ns = time.monotonic_ns()
    hint_done_at_dispatch = hint_task.done() if hint_task is not None else None
    resume_result = await resume_call()
    hint_result = await hint_task if hint_task is not None else None
    return {
        "resume_eligibility_ns": eligibility_ns,
        "resume_dispatch_ns": dispatch_ns,
        "resume_release_waited_for_hint_ack": False,
        "hint_ack_complete_before_resume_dispatch": hint_done_at_dispatch,
        "resume_result": resume_result,
        "hint_result": hint_result,
    }


def _hint_payload(
    *,
    case: Mapping[str, object],
    owner_id: str,
    event_id: str,
    observed_prefix_id: str,
    boundary: int,
    wait_ms: int,
    available_ns: int,
    publication_ns: int,
) -> dict[str, object]:
    return {
        "schema": "semantic-known-prefix-wait-hint-v1",
        "event_id": event_id,
        "owner_id": owner_id,
        "run_id": f"diag-run-{case['case_id']}",
        "cell_id": f"diag-cell-{case['case_id']}",
        "session_id": "diag-session-A",
        "generation": 0,
        "observed_prefix_id": observed_prefix_id,
        "prefix_boundary_tokens": boundary,
        "observation_available_ns": available_ns,
        "input_publication_ns": publication_ns,
        "declared_wait_ms": wait_ms,
        "wait_provenance": "driver_declared_tool_wait_budget",
        "namespace": dict(MODEL_NAMESPACE),
    }


async def _negative_calls(
    session,
    base_url: str,
    *,
    case: Mapping[str, object],
    observed_prefix_id: str,
    boundary: int,
) -> list[dict[str, object]]:
    now_ns = time.monotonic_ns()
    expired = _hint_payload(
        case=case,
        owner_id="unknown-owner",
        event_id=digest({"case": case["case_id"], "negative": "unknown-owner-expired"}),
        observed_prefix_id=observed_prefix_id,
        boundary=boundary,
        wait_ms=0,
        available_ns=now_ns,
        publication_ns=now_ns,
    )
    first = await _post_hint(session, base_url, expired, "unknown_owner_expired")
    duplicate = await _post_hint(session, base_url, expired, "duplicate_replay")
    missing_now = time.monotonic_ns()
    missing = _hint_payload(
        case=case,
        owner_id="unknown-owner",
        event_id=digest({"case": case["case_id"], "negative": "unknown-prefix"}),
        observed_prefix_id="f" * 64,
        boundary=boundary,
        wait_ms=250,
        available_ns=missing_now,
        publication_ns=missing_now,
    )
    missing_call = await _post_hint(session, base_url, missing, "unknown_prefix")
    future_now = time.monotonic_ns()
    future = _hint_payload(
        case=case,
        owner_id="unknown-owner",
        event_id=digest({"case": case["case_id"], "negative": "future-clock"}),
        observed_prefix_id=observed_prefix_id,
        boundary=boundary,
        wait_ms=250,
        available_ns=future_now + 10**12,
        publication_ns=future_now + 10**12,
    )
    future_call = await _post_hint(session, base_url, future, "future_clock")
    return [first, duplicate, missing_call, future_call]


def _event_action_counts(events: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(event.get("event")) for event in events).items()))


def validate_case_result(
    result: Mapping[str, object],
    case: Mapping[str, object],
    *,
    plan: Mapping[str, object] | None = None,
    finalized: bool = False,
) -> None:
    _require(result.get("schema") == CASE_RESULT_SCHEMA, "case result schema")
    _require(result.get("case_id") == case["case_id"], "case result identity")
    if plan is not None:
        _require(
            result.get("plan_digest") == plan.get("plan_digest")
            and result.get("case_config_hash") == case.get("config_hash")
            and result.get("extension_identity", {}).get("revision")
            == plan.get("source", {}).get("extension_revision")
            and result.get("extension_identity", {}).get("tree")
            == plan.get("source", {}).get("extension_tree")
            and result.get("runtime_identity", {}).get("revision")
            == case.get("runtime", {}).get("revision")
            and result.get("runtime_identity", {}).get("tree")
            == case.get("runtime", {}).get("tree"),
            "case source/config/runtime envelope",
        )
    requests = result.get("requests")
    _require(
        isinstance(requests, list)
        and [entry.get("diagnostic_role") for entry in requests]
        == ["a_warm_q0", "b_longest_pollution", "a_resume_q1"]
        and all(entry.get("success") is True for entry in requests),
        "case request completion",
    )
    expected_requests = case.get("request_bindings")
    if isinstance(expected_requests, list):
        _require(
            len(expected_requests) == len(requests)
            and all(
                request.get("diagnostic_role") == expected.get("diagnostic_role")
                and request.get("original_request_id")
                == expected.get("original_request_id")
                and request.get("original_record_sha256")
                == expected.get("original_record_sha256")
                and request.get("prompt_token_digest")
                == expected.get("prompt_token_digest")
                and request.get("prompt_token_count")
                == expected.get("prompt_token_count")
                and request.get("external_request_id")
                == expected.get("external_request_id")
                and request.get("input_payload_digest")
                == digest(request.get("input_payload"))
                and isinstance(request.get("input_payload"), Mapping)
                and request["input_payload"].get("request_id")
                == expected.get("external_request_id")
                and request["input_payload"].get("model") == MODEL_ID
                and request["input_payload"].get("max_tokens")
                == ENGINE_PROFILE["max_new_tokens"]
                and isinstance(request["input_payload"].get("prompt"), list)
                and len(request["input_payload"]["prompt"])
                == expected.get("prompt_token_count")
                and digest(request["input_payload"]["prompt"])
                == expected.get("prompt_token_digest")
                and request["input_payload"].get("kv_transfer_params", {}).get(
                    "candidate_config_hash"
                )
                == case.get("config_hash")
                for request, expected in zip(requests, expected_requests)
            ),
            "case request source/input envelope",
        )
    _require(
        all(
            isinstance(entry.get("finish_reasons"), list)
            and len(entry["finish_reasons"]) == 1
            and entry["finish_reasons"][0] in {"stop", "length"}
            and 0 < len(entry.get("output_token_ids", []))
            <= ENGINE_PROFILE["max_new_tokens"]
            for entry in requests
        ),
        "case requests require natural terminal evidence within max_new_tokens",
    )
    _require(
        result.get("drain", {}).get("status") == "complete",
        "case drain",
    )
    hint_events = result.get("wait_hint_events")
    _require(isinstance(hint_events, list), "case wait event evidence")
    _require(result.get("wait_hint_event_counts") == _event_action_counts(hint_events), "case wait event counts")
    if case["endpoint_plugin_enabled"]:
        publication = result.get("primary_hint")
        _require(isinstance(publication, Mapping), "enabled case hint publication")
        if int(case["tool_gap_ms"]) > 0:
            _require(
                publication.get("started_ns")
                <= result.get("resume_eligibility_ns")
                and requests[-1].get("request_start_ns")
                >= result.get("resume_eligibility_ns"),
                "positive-gap hint request release clocks",
            )
        else:
            _require(
                publication.get("started_ns") >= result.get("wait_origin_ns")
                and requests[-1].get("request_start_ns")
                >= result.get("resume_eligibility_ns"),
                "zero-gap hint request release clocks",
            )
    else:
        _require(result.get("primary_hint") is None, "no-hint case published hint")
    if case["case_id"] == "child-hint-250":
        primary = result["primary_hint"]
        primary_payload = primary.get("payload")
        _require(
            isinstance(primary_payload, Mapping)
            and primary.get("payload_digest") == digest(primary_payload)
            and primary_payload.get("event_id")
            == primary.get("response", {}).get("event_id")
            and primary.get("response", {}).get("status") == "accepted",
            "positive case primary hint envelope",
        )
        primary_event_id = primary_payload["event_id"]
        response = result["primary_hint"].get("response")
        opportunity = response.get("opportunity") if isinstance(response, Mapping) else None
        _require(
            isinstance(opportunity, Mapping)
            and opportunity.get("positive_opportunity") is True
            and opportunity.get("local_gpu_coverage_available") is True
            and opportunity.get("eligible_cpu_staging_prefix_count", 0) > 0
            and opportunity.get("local_gpu_prefix_count")
            < opportunity.get("source_key_count"),
            "positive case lacks actual CPU/GPU opportunity",
        )
        submitted = [
            event
            for event in hint_events
            if event.get("event") == "staging_submitted"
            and event.get("event_id") == primary_event_id
        ]
        ready = [
            event
            for event in hint_events
            if event.get("event") == "staging_ready"
            and event.get("event_id") == primary_event_id
        ]
        consumed = [
            event
            for event in hint_events
            if event.get("event") == "staging_consumed"
            and event.get("event_id") == primary_event_id
        ]
        _require(
            len(submitted) == len(ready) == len(consumed) == 1,
            "positive case primary lifecycle cardinality",
        )
        submitted_event = submitted[0]
        ready_event = ready[0]
        consumed_event = consumed[0]
        submitted_bytes = int(submitted_event.get("submitted_bytes", 0))
        bytes_per_key = int(submitted_event.get("bytes_per_key", 0))
        consumed_bytes = int(consumed_event.get("consumed_bytes", 0))
        discarded_bytes = int(consumed_event.get("discarded_bytes", 0))
        consumed_keys = int(consumed_event.get("consumed_keys", 0))
        discarded_keys = int(consumed_event.get("discarded_keys", 0))
        _require(
            isinstance(submitted_event.get("job_id"), int)
            and ready_event.get("job_id") == submitted_event["job_id"]
            and isinstance(consumed_event.get("splice_id"), int)
            and isinstance(submitted_event.get("selected_keys"), list)
            and submitted_event["selected_keys"]
            and bytes_per_key > 0
            and submitted_bytes
            == len(submitted_event["selected_keys"]) * bytes_per_key
            and submitted_bytes > 0
            and ready_event.get("transfer_bytes") == submitted_bytes
            and consumed_bytes > 0
            and consumed_bytes == consumed_keys * bytes_per_key
            and discarded_bytes == discarded_keys * bytes_per_key
            and consumed_keys + discarded_keys
            == len(submitted_event["selected_keys"])
            and consumed_bytes + discarded_bytes == submitted_bytes,
            "positive case primary job/key/byte join",
        )
        resume_events = [
            event
            for event in result.get("preservation_events", [])
            if event.get("schedule_role") == "a_resume_q1"
        ]
        request_bindings = case.get("request_bindings")
        expected_resume_id = (
            request_bindings[-1]["external_request_id"]
            if isinstance(request_bindings, list) and request_bindings
            else resume_events[0].get("external_request_id")
            if resume_events
            else None
        )
        _require(
            len(resume_events) == 1
            and resume_events[0].get("external_request_id")
            == expected_resume_id,
            "positive case resume identity",
        )
        attribution = resume_events[0].get("prefetch_attribution", {})
        wait_hint = attribution.get("wait_hint", {})
        expected_splice_ack = f"splice-{consumed_event['splice_id']}"
        selected_keys = set(submitted_event["selected_keys"])
        prefetched_keys = set(attribution.get("prefetched_keys", []))
        source_map = attribution.get("source_key_to_gpu_block", {})
        destination_map = attribution.get("destination_key_to_gpu_block", {})
        splice_fence_ns = attribution.get("splice_fence_ns")
        splice_fence_evidence = attribution.get("splice_fence_evidence")
        resume_result = requests[-1]
        _require(
            wait_hint.get("event_id") == primary_event_id
            and wait_hint.get("owner_id")
            == expected_resume_id
            and attribution.get("splice_acknowledged") is True
            and attribution.get("splice_ack_id") == expected_splice_ack
            and attribution.get("terminal_cleanup", {}).get(
                "source_reuse_fenced"
            )
            is True
            and attribution.get("gpu_splice_bytes") == consumed_bytes
            and attribution.get("prefetch_h2d_bytes") == submitted_bytes
            and prefetched_keys
            and len(prefetched_keys) == consumed_keys
            and prefetched_keys.issubset(selected_keys)
            and set(attribution.get("intersection_keys", []))
            == prefetched_keys
            and isinstance(source_map, Mapping)
            and set(source_map) == prefetched_keys
            and isinstance(destination_map, Mapping)
            and set(destination_map) == prefetched_keys
            and isinstance(wait_hint.get("input_publication_ns"), int)
            and isinstance(submitted_event.get("submitted_at_ns"), int)
            and isinstance(ready_event.get("ready_at_ns"), int)
            and isinstance(consumed_event.get("consumed_at_ns"), int)
            and isinstance(wait_hint.get("bound_at_ns"), int)
            and isinstance(splice_fence_ns, int)
            and isinstance(splice_fence_evidence, list)
            and len(splice_fence_evidence)
            == ENGINE_PROFILE["tensor_parallel_size"]
            and all(
                evidence.get("splice_id") == consumed_event["splice_id"]
                and evidence.get("source_block_count") == consumed_keys
                and evidence.get("destination_block_count") == consumed_keys
                and isinstance(evidence.get("copied_layer_count"), int)
                and evidence["copied_layer_count"] > 0
                and evidence.get("copied_on_cuda") is True
                and evidence.get("same_default_stream") is True
                and evidence.get("issued_before_base_bind_and_forward") is True
                and evidence.get("ordering_witness")
                == "same_default_stream_copy_enqueued_before_base_bind_and_forward"
                and evidence.get("completion_event_satisfied") is True
                and isinstance(evidence.get("issued_at_ns"), int)
                and isinstance(evidence.get("completion_observed_at_ns"), int)
                and consumed_event["consumed_at_ns"]
                <= evidence["issued_at_ns"]
                <= evidence["completion_observed_at_ns"]
                <= splice_fence_ns
                for evidence in splice_fence_evidence
            )
            and isinstance(resume_result.get("first_token_ns"), int)
            and wait_hint["input_publication_ns"]
            <= submitted_event["submitted_at_ns"]
            <= ready_event["ready_at_ns"]
            <= consumed_event["consumed_at_ns"]
            and resume_result["request_start_ns"]
            <= wait_hint["bound_at_ns"]
            <= consumed_event["consumed_at_ns"]
            <= resume_result["first_token_ns"]
            and consumed_event["consumed_at_ns"]
            <= splice_fence_ns
            <= resume_result["request_end_ns"],
            "positive case hint/job/key/byte/fence/resume join",
        )
    if case["tool_gap_ms"] == 0:
        primary_response = result.get("primary_hint", {}).get("response", {})
        primary_event_id = result.get("primary_hint", {}).get("payload", {}).get(
            "event_id"
        )
        zero_no_action_expected = (
            case.get("zero_gap_expected_behavior")
            == "request_order_no_action_no_allocation"
        )
        _require(
            result.get("resume_release_waited_for_hint_ack") is False
            and result.get("resume_dispatch_ns")
            >= result.get("resume_eligibility_ns"),
            "zero-gap resume release accounting",
        )
        if zero_no_action_expected:
            _require(
                primary_response.get("status") in {"expired", "no_action"}
                and not any(
                    event.get("event_id") == primary_event_id
                    and event.get("event") == "staging_submitted"
                    for event in hint_events
                ),
                "request-order zero-gap action differs from zero-budget contract",
            )
    if case["negative_calls_after_primary"]:
        negatives = result.get("negative_calls")
        _require(isinstance(negatives, list) and len(negatives) == 4, "negative call count")
        negative_ids = {call["payload"]["event_id"] for call in negatives}
        allocated = [
            event
            for event in hint_events
            if event.get("event_id") in negative_ids
            and event.get("event") == "staging_submitted"
        ]
        _require(not allocated, "negative hint allocated staging")
        _require(
            all(call.get("http_status") == 200 and call.get("error") is None for call in negatives)
            and negatives[0].get("response", {}).get("status") == "expired"
            and negatives[0].get("response", {}).get("reason")
            == "nomination_deadline_elapsed"
            and negatives[1].get("response", {}).get("status") == "expired"
            and negatives[2].get("response", {}).get("status") == "no_action"
            and negatives[2].get("response", {}).get("reason")
            == "observed_prefix_unavailable"
            and negatives[3].get("response", {}).get("status") == "rejected"
            and negatives[3].get("response", {}).get("reason")
            == "causal_clock_mismatch",
            "negative hint status accounting",
        )
        _require(
            negatives[1].get("response", {}).get("idempotent_replay") is True,
            "duplicate hint was not idempotent",
        )
    if finalized:
        body = dict(result)
        result_digest = body.pop("result_digest", None)
        _require(
            result.get("status") == "valid"
            and result.get("valid") is True
            and result.get("validation_error") is None
            and result_digest == digest(body),
            "finalized case status/digest",
        )


def validate_suite(results: Sequence[Mapping[str, object]], plan: Mapping[str, object]) -> dict[str, object]:
    cases = {case["case_id"]: case for case in plan["cases"]}
    _require(len(results) == len(cases), "suite result count")
    by_case = {str(result.get("case_id")): result for result in results}
    _require(set(by_case) == set(cases), "suite case identities")
    for case_id, result in by_case.items():
        validate_case_result(
            result, cases[case_id], plan=plan, finalized=True
        )
    output_vectors = {
        case_id: [
            request["output_token_ids"] for request in result["requests"]
        ]
        for case_id, result in by_case.items()
    }
    first = next(iter(output_vectors.values()))
    _require(
        all(vector == first for vector in output_vectors.values()),
        "A/B/A exact outputs differ across matched cases",
    )
    original = cases["original149-no-hint"]
    child = cases["child-no-hint"]
    for field in ("config", "tool_gap_ms"):
        _require(original[field] == child[field], f"default path {field} mismatch")
    result = {
        "schema": SUITE_RESULT_SCHEMA,
        "valid": True,
        "plan_digest": plan["plan_digest"],
        "case_ids": sorted(by_case),
        "output_vector_digest": digest(first),
        "interpretation": "Functional serving readiness only; no performance claim.",
    }
    result["result_digest"] = digest(result)
    return result


async def _execute_case_raw(
    plan: Mapping[str, object], case: Mapping[str, object], output: Path
) -> dict[str, object]:
    import aiohttp

    from harness import metrics as metrics_mod
    from harness import policies as policies_mod
    from harness.server import launch_server

    request_path = Path(str(plan["request_artifact"]["path"]))
    with request_path.open(encoding="utf-8") as input_file:  # noqa: ASYNC230
        request_artifact = json.load(input_file)
    validate_request_artifact(request_artifact)
    staged_preflight_path = Path(
        str(plan["preflight"]["staged_environment_output"])
    )
    _require(staged_preflight_path.is_file(), "staged environment preflight missing")
    with staged_preflight_path.open(encoding="utf-8") as input_file:  # noqa: ASYNC230
        staged_preflight = json.load(input_file)
    _require(
        staged_preflight.get("schema")
        == "ep-staged-environment-preflight-report-v1"
        and staged_preflight.get("valid") is True
        and staged_preflight.get("failed_checks") == []
        and staged_preflight.get("blocked_checks") == [],
        "staged environment preflight invalid",
    )
    binary_preflight_path = Path(
        str(plan["preflight"]["binary_verification_output"])
    )
    _require(binary_preflight_path.is_file(), "binary preflight result missing")
    with binary_preflight_path.open(encoding="utf-8") as input_file:  # noqa: ASYNC230
        binary_preflight = json.load(input_file)
    _require(
        binary_preflight.get("schema") == BINARY_DISCOVERY_SCHEMA
        and binary_preflight.get("verified") is True
        and binary_preflight.get("inventory_sha256")
        == plan["native_binary_inventory"]["sha256"],
        "binary preflight result invalid",
    )
    plugin = plan["plugin_distribution"]
    wheel_path = Path(str(plugin["wheel_path"]))
    _require(wheel_path.is_file() and file_digest(wheel_path) == plugin["wheel_sha256"], "wheel binding")
    plugin_discovery = verify_plugin_discovery(
        Path(str(plugin["isolated_plugin_root"])),
        plugin,
        require_native_loader=bool(case["runtime"]["wait_hint_utility"]),
    )
    extension_root = Path(str(plan["source"]["extension_root"]))
    extension_identity = _git_identity(extension_root)
    _require(
        extension_identity["revision"] == plan["source"]["extension_revision"]
        and extension_identity["tree"] == plan["source"]["extension_tree"]
        and extension_identity["clean"] is True,
        "extension identity",
    )
    runtime_root = Path(str(case["runtime"]["path"]))
    runtime_identity = _git_identity(runtime_root)
    _require(
        runtime_identity["revision"] == case["runtime"]["revision"]
        and runtime_identity["tree"] == case["runtime"]["tree"]
        and runtime_identity["clean"] is True,
        "runtime identity",
    )
    config = case["config"]
    kv_config = policies_mod.kv_transfer_config(
        config["policy"], ENGINE_PROFILE["cpu_bytes"], extra_config=config["extra_config"]
    )
    profile = {
        "model_runtime": {
            "model_revision": MODEL_REVISION,
            "dtype": ENGINE_PROFILE["dtype"],
            "tensor_parallel_size": ENGINE_PROFILE["tensor_parallel_size"],
        },
        "resources": {
            "kv_block_size_tokens": ENGINE_PROFILE["kv_block_size_tokens"],
            "enforce_eager": ENGINE_PROFILE["enforce_eager"],
            "max_num_seqs": ENGINE_PROFILE["max_num_seqs"],
            "max_num_batched_tokens": ENGINE_PROFILE["max_num_batched_tokens"],
            "max_model_len": ENGINE_PROFILE["max_model_len"],
            "gpu_memory_utilization": ENGINE_PROFILE["gpu_memory_utilization"],
            "total_gpu_blocks": ENGINE_PROFILE["total_gpu_blocks"],
        },
        "scheduler_backend": config["scheduler_backend"],
    }
    python_path = os.pathsep.join(
        [
            str(Path(str(plugin["isolated_plugin_root"])).resolve()),
            str(extension_root.resolve()),
            str(runtime_root.resolve()),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    handle = launch_server(
        MODEL_ID,
        int(case["port"]),
        str(output.parent),
        gpu_memory_utilization=ENGINE_PROFILE["gpu_memory_utilization"],
        max_model_len=ENGINE_PROFILE["max_model_len"],
        num_gpu_blocks_override=ENGINE_PROFILE["total_gpu_blocks"],
        kv_transfer_config=kv_config,
        extra_args=_engine_server_args(profile, seed=ENGINE_PROFILE["seed"]),
        extra_env={
            "PYTHONPATH": python_path,
            "VLLM_PLUGINS": "semantic_wait_hint" if case["endpoint_plugin_enabled"] else "",
            "VLLM_CLI_MODULE": "vllm.entrypoints.cli.main",
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_BATCH_INVARIANT": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": str(ENGINE_PROFILE["seed"]),
            "SEMANTIC_OFFLOAD_DISABLE_PREFETCH": "0",
        },
        log_label=str(case["case_id"]),
    )
    entries = {entry["diagnostic_role"]: entry for entry in request_artifact["requests"]}
    boundary = int(request_artifact["selection"]["known_prefix_boundary_tokens"])
    prefix_tokens = entries["a_warm_q0"]["original_record"]["prompt_token_ids"][:boundary]
    observed_prefix_id = digest({"namespace": MODEL_NAMESPACE, "token_ids": prefix_tokens})
    hint_event_id = digest({"schema": "ep-serving-diagnostic-hint-v1", "case_id": case["case_id"]})
    request_results: list[dict[str, object]] = []
    primary_hint = None
    negative_calls: list[dict[str, object]] = []
    before_evidence: dict[str, object] = {}
    after_evidence: dict[str, object] = {}
    drain: dict[str, object] = {"status": "not_started"}
    wait_origin_ns = 0
    resume_eligibility_ns = 0
    resume_dispatch_ns = 0
    hint_ack_complete_before_resume_dispatch: bool | None = None
    execution_error = None
    resolved_runtime: dict[str, object] = {}
    try:
        resolutions = _parse_runtime_resolutions(handle.log_path)
        unique_resolutions = {digest(item): item for item in resolutions}
        _require(len(unique_resolutions) == 1, "runtime resolution")
        resolved_runtime = next(iter(unique_resolutions.values()))
        _require(
            resolved_runtime
            == _expected_runtime_resolution(
                profile, profile["scheduler_backend"]
            ),
            "runtime resolution differs from diagnostic profile",
        )
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            before_evidence = metrics_mod.snapshot_evidence(handle.metrics_url())
            for index, role in enumerate(("a_warm_q0", "b_longest_pollution")):
                payload = _request_payload(
                    entries[role],
                    case=case,
                    index=index,
                    observed_prefix_id=observed_prefix_id if role == "a_warm_q0" else None,
                    boundary=boundary,
                    hint_event_id=None,
                )
                response = await _send_completion(session, handle.base_url(), payload)
                request_results.append(
                    _request_result(entries[role], payload, response)
                )
            pre_hint_drain = await _wait_for_campaign_drain(
                handle,
                expected_events=2,
                metrics_mod=metrics_mod,
                background_mode="synchronous",
            )
            _require(pre_hint_drain.get("status") == "complete", "pre-hint drain")
            wait_origin_ns = time.monotonic_ns()
            hint_call = None
            if case["endpoint_plugin_enabled"]:
                publication_ns = time.monotonic_ns()
                resume_owner = _external_id(str(case["case_id"]), 2)
                hint_payload = _hint_payload(
                    case=case,
                    owner_id=resume_owner,
                    event_id=hint_event_id,
                    observed_prefix_id=observed_prefix_id,
                    boundary=boundary,
                    wait_ms=int(case["tool_gap_ms"]),
                    available_ns=wait_origin_ns,
                    publication_ns=publication_ns,
                )
                async def hint_call():
                    return await _post_hint(
                        session, handle.base_url(), hint_payload, "primary"
                    )

            async def resume_call():
                resume_payload = _request_payload(
                    entries["a_resume_q1"],
                    case=case,
                    index=2,
                    observed_prefix_id=observed_prefix_id,
                    boundary=boundary,
                    hint_event_id=(
                        hint_event_id
                        if case["endpoint_plugin_enabled"]
                        else None
                    ),
                )
                resume_response = await _send_completion(
                    session, handle.base_url(), resume_payload
                )
                return _request_result(
                    entries["a_resume_q1"], resume_payload, resume_response
                )

            release = await _dispatch_at_release_boundary(
                wait_origin_ns=wait_origin_ns,
                tool_gap_ms=int(case["tool_gap_ms"]),
                hint_call=hint_call,
                resume_call=resume_call,
            )
            resume_eligibility_ns = int(release["resume_eligibility_ns"])
            resume_dispatch_ns = int(release["resume_dispatch_ns"])
            hint_ack_complete_before_resume_dispatch = release[
                "hint_ack_complete_before_resume_dispatch"
            ]
            request_results.append(release["resume_result"])
            primary_hint = release["hint_result"]
            drain = await _wait_for_campaign_drain(
                handle,
                expected_events=3,
                metrics_mod=metrics_mod,
                background_mode="synchronous",
            )
            if case["negative_calls_after_primary"]:
                negative_calls = await _negative_calls(
                    session,
                    handle.base_url(),
                    case=case,
                    observed_prefix_id=observed_prefix_id,
                    boundary=boundary,
                )
                await asyncio.sleep(0.1)
            after_evidence = metrics_mod.snapshot_evidence(handle.metrics_url())
    except Exception as exc:  # noqa: BLE001 - preserve diagnostic failure evidence
        execution_error = repr(exc)
    finally:
        handle.shutdown()
    wait_events = _parse_wait_hint_events(handle.log_path)
    preservation_events = _parse_preservation_events(handle.log_path)
    metric_delta, metric_provenance = metrics_mod.diff_with_provenance(
        before_evidence.get("values", {}), after_evidence.get("values", {})
    )
    result = {
        "schema": CASE_RESULT_SCHEMA,
        "case_id": case["case_id"],
        "plan_digest": plan["plan_digest"],
        "case_config_hash": case["config_hash"],
        "extension_identity": extension_identity,
        "runtime_identity": runtime_identity,
        "resolved_runtime": resolved_runtime,
        "plugin_discovery": plugin_discovery,
        "server_command": handle.command,
        "server_log": handle.log_path,
        "wait_origin_ns": wait_origin_ns,
        "resume_eligibility_ns": resume_eligibility_ns,
        "resume_dispatch_ns": resume_dispatch_ns,
        "resume_release_waited_for_hint_ack": False,
        "hint_ack_complete_before_resume_dispatch": (
            hint_ack_complete_before_resume_dispatch
        ),
        "primary_hint": primary_hint,
        "negative_calls": negative_calls,
        "requests": request_results,
        "drain": drain,
        "metrics_before": before_evidence,
        "metrics_after": after_evidence,
        "metric_delta": metric_delta,
        "metric_delta_provenance": metric_provenance,
        "wait_hint_events": wait_events,
        "wait_hint_event_counts": _event_action_counts(wait_events),
        "preservation_events": preservation_events,
        "interpretation": "Functional serving diagnostic only; no performance claim.",
        "execution_error": execution_error,
    }
    return result


async def execute_case(
    plan: Mapping[str, object], case: Mapping[str, object], output: Path
) -> dict[str, object]:
    """Preserve raw evidence before enforcing the fail-closed case gates."""
    try:
        result = await _execute_case_raw(plan, case, output)
    except Exception as exc:
        failure = {
            "schema": CASE_RESULT_SCHEMA,
            "case_id": case.get("case_id"),
            "plan_digest": plan.get("plan_digest"),
            "case_config_hash": case.get("config_hash"),
            "status": "runtime_exception_before_raw_result",
            "valid": False,
            "error": repr(exc),
            "preserved_at_ns": time.monotonic_ns(),
        }
        failure["result_digest"] = digest(failure)
        _write_json(output, failure)
        raise
    result["status"] = "raw_complete_pending_validation"
    result["valid"] = False
    result["result_digest"] = digest(result)
    _write_json(output, result)
    if result.get("execution_error") is not None:
        result["status"] = "runtime_failed_evidence_preserved"
        result["validation_error"] = "execution did not complete"
        result.pop("result_digest", None)
        result["result_digest"] = digest(result)
        _write_json(output, result)
        raise DiagnosticError(str(result["execution_error"]))
    try:
        validate_case_result(result, case, plan=plan)
    except Exception as exc:
        result["status"] = "validation_failed_evidence_preserved"
        result["validation_error"] = repr(exc)
        result.pop("result_digest", None)
        result["result_digest"] = digest(result)
        _write_json(output, result)
        raise
    result["status"] = "valid"
    result["valid"] = True
    result["validation_error"] = None
    result.pop("result_digest", None)
    result["result_digest"] = digest(result)
    _write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-requests")
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--source-runtime-path")
    build.add_argument("--output", required=True, type=Path)
    plan_parser = subparsers.add_parser("build-plan")
    plan_parser.add_argument("--requests", required=True, type=Path)
    plan_parser.add_argument("--request-runtime-path", required=True)
    plan_parser.add_argument("--plan-runtime-path", required=True)
    plan_parser.add_argument("--wheel", required=True)
    plan_parser.add_argument("--wheel-sha256", required=True)
    plan_parser.add_argument("--plugin-root", required=True)
    plan_parser.add_argument("--binary-inventory", required=True)
    plan_parser.add_argument("--child-binary-inventory", required=True)
    plan_parser.add_argument("--child-binary-inventory-sha256", required=True)
    plan_parser.add_argument("--staged-preflight-manifest", required=True)
    plan_parser.add_argument("--staged-preflight-report", required=True)
    plan_parser.add_argument("--extension-root", required=True)
    plan_parser.add_argument("--extension-revision", required=True)
    plan_parser.add_argument("--extension-tree", required=True)
    plan_parser.add_argument("--output-root", required=True)
    plan_parser.add_argument("--port-base", type=int, default=19200)
    plan_parser.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify-plugin")
    verify.add_argument("--plan", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    verify_binaries = subparsers.add_parser("verify-binaries")
    verify_binaries.add_argument("--plan", required=True, type=Path)
    verify_binaries.add_argument("--output", required=True, type=Path)
    execute = subparsers.add_parser("execute-case")
    execute.add_argument("--plan", required=True, type=Path)
    execute.add_argument("--case-id", required=True)
    execute.add_argument("--output", required=True, type=Path)
    suite = subparsers.add_parser("validate-suite")
    suite.add_argument("--plan", required=True, type=Path)
    suite.add_argument("--result", action="append", required=True, type=Path)
    suite.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "build-requests":
        with args.source.open(encoding="utf-8") as input_file:
            artifact = build_request_artifact(
                json.load(input_file),
                args.source,
                source_runtime_path=args.source_runtime_path,
            )
        _write_json(args.output, artifact)
        return 0
    if args.command == "build-plan":
        plan = build_plan(
            args.requests,
            request_runtime_path=args.request_runtime_path,
            wheel_path=args.wheel,
            wheel_sha256=args.wheel_sha256,
            plugin_root=args.plugin_root,
            binary_inventory_path=args.binary_inventory,
            child_binary_inventory_path=args.child_binary_inventory,
            child_binary_inventory_sha256=args.child_binary_inventory_sha256,
            staged_preflight_manifest_path=args.staged_preflight_manifest,
            staged_preflight_report_path=args.staged_preflight_report,
            extension_root=args.extension_root,
            extension_revision=args.extension_revision,
            extension_tree=args.extension_tree,
            output_root=args.output_root,
            port_base=args.port_base,
        )
        for case in plan["cases"]:
            case["argv"] = [
                argument.format(plan=args.plan_runtime_path)
                for argument in case["argv"]
            ]
        plan["preflight"]["argv"] = [
            [argument.format(plan=args.plan_runtime_path) for argument in command]
            for command in plan["preflight"]["argv"]
        ]
        plan["runtime_plan_path"] = args.plan_runtime_path
        plan.pop("plan_digest")
        plan["plan_digest"] = digest(plan)
        _write_json(args.output, plan)
        return 0
    with args.plan.open(encoding="utf-8") as input_file:
        plan = json.load(input_file)
    validate_plan(plan, args.plan.resolve())
    if args.command == "verify-plugin":
        result = verify_plugin_discovery(
            Path(str(plan["plugin_distribution"]["isolated_plugin_root"])),
            plan["plugin_distribution"],
        )
        _write_json(args.output, result)
        return 0
    if args.command == "verify-binaries":
        binding = plan["native_binary_inventory"]
        result = verify_binary_inventory(
            Path(str(binding["path"])), str(binding["sha256"])
        )
        from benchmarks.native_child_runtime import verify_child_inventory

        child_binding = plan["child_runtime_binary_inventory"]
        result["child_runtime"] = verify_child_inventory(
            Path(str(child_binding["path"])),
            str(child_binding["sha256"]),
        )
        _write_json(args.output, result)
        return 0
    if args.command == "execute-case":
        cases = [case for case in plan["cases"] if case["case_id"] == args.case_id]
        _require(len(cases) == 1, "unknown diagnostic case")
        asyncio.run(execute_case(plan, cases[0], args.output))
        return 0
    results = []
    for result_path in args.result:
        with result_path.open(encoding="utf-8") as input_file:
            results.append(json.load(input_file))
    _write_json(args.output, validate_suite(results, plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
