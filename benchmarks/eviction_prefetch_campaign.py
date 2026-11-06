# SPDX-License-Identifier: Apache-2.0
"""Manifest gate and explicit-cell launcher for the E01--E09 campaign.

The launcher deliberately expands only predeclared arms.  It never constructs
a factorial, chooses a treatment from heldout results, or fills missing causal
fields.  ``plan`` is safe on a CPU host; ``run`` additionally requires every
selected arm to be marked ready with an explicit runner argv template.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "ep-campaign-manifest-v1"
REQUEST_SCHEMA = "ep-request-manifest-v1"
PLAN_SCHEMA = "ep-campaign-plan-v1"
CAUSAL_SCHEMA = "ep-causal-online-input-v1"
SOURCE_BASELINE = "0998fdf85ca5446a2ac0284cab81988165a297e2"
RUNTIME_REVISION = "149cb6d864c46a88f38ab8602c7e3071034f8512"
NATIVE_RUNTIME_VARIANTS = {
    "original149": {
        "revision": RUNTIME_REVISION,
        "tree": "adb805d0bd51486cde394f06c36b72a933617f2e",
        "path": "/home/philip/semantic-cache-agent/wsl-session-20260810-native/src/vllm",
        "wait_hint_utility": False,
    },
    "wait_hint_utility_v1": {
        "revision": "f3f036805c886c562d1b5397da40a898f163e93a",
        "tree": "c4b6cd9a94426340256f0957215e7bbb03ff1bc9",
        "path": (
            "/home/philip/semantic-cache-agent/"
            "eviction-prefetch-campaign-20260905/vllm-wait-hint-git"
        ),
        "wait_hint_utility": True,
        "patch_sha256": (
            "09ddfb2cce52343fa65ed60d2508747f64ee651e1d862f668b9f6d21562d042b"
        ),
        "native_binary_inventory_sha256": (
            "2479dd9972bddb97fc5e4e629627f2e5b15641d48275bcd23c3a1149676272ab"
        ),
    },
}
NATIVE_RUNTIME_VARIANTS_BINDING_SHA256 = (
    "702842264de6d590f6c97612c72ad441c2fdaed30edb12d99984936b78f2edb6"
)
FEATURES = tuple(f"E{index:02d}" for index in range(1, 10))
SEEDS = (1103, 2207, 3301, 4409, 5519)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260905
REQUIRED_EFFECTIVE_ENVIRONMENT = {
    "SEMANTIC_OFFLOAD_DISABLE_PREFETCH": "0",
}
REQUIRED_ENGINE_PROFILE = {
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "enforce_eager": True,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 8192,
    "kv_block_size_tokens": 16,
    "gpu_memory_utilization": 0.5,
}
REQUIRED_MAX_MODEL_LEN_BY_GPU_BLOCKS = {320: 5120, 640: 8192}


def _required_engine_profile(total_gpu_blocks: object) -> dict[str, object]:
    _require(
        isinstance(total_gpu_blocks, int)
        and not isinstance(total_gpu_blocks, bool)
        and total_gpu_blocks in REQUIRED_MAX_MODEL_LEN_BY_GPU_BLOCKS,
        "total GPU blocks differ from frozen readiness profiles",
    )
    return {
        **REQUIRED_ENGINE_PROFILE,
        "max_model_len": REQUIRED_MAX_MODEL_LEN_BY_GPU_BLOCKS[total_gpu_blocks],
    }


def _engine_profile_matches(
    actual: Mapping[str, object],
    total_gpu_blocks: object,
    *,
    bounded_prefill_cuda_timing: bool,
) -> bool:
    """Allow singleton scheduling only for explicit CUDA attribution cells."""
    standard = _required_engine_profile(total_gpu_blocks)
    if actual == standard:
        return True
    attribution = {**standard, "max_num_seqs": 1}
    return bounded_prefill_cuda_timing and actual == attribution


FROZEN_ORIGINAL_SEMANTIC_CONFIG = {
    "policy": "semantic-mean",
    "scheduler_backend": "native_sync_fcfs",
    "request_controls": {"max_offload_tokens": None},
    "extra_config": {
        "method": "mean",
        "capture_stride": 1,
        "probe_layer": "middle",
        "head_aggregation": "mean",
        "rank_ema_mode": "default",
        "summary_offload_overlap": False,
        "prefix_aware": False,
        "ema_rank_weight_floor": 0.0,
        "alpha": 0.5,
        "admission_mode": "disabled",
        "store_threshold": 0,
        "max_tracker_size": 64_000,
        "grace_window_blocks": 0,
        "eviction_mode": "blend",
        "chain_aware": False,
        "session_aware": False,
        "session_bonus_half_life": 0,
        "scoring_mode": "synchronous",
        "prefetch_budget_fraction": 0.0,
        "prefetch_headroom_blocks": 0,
        "prefetch_selector": "semantic",
        "prefetch_trigger": "preemption",
        "soft_retention": "off",
        "staging_mode": "dynamic_scratch",
        "lifecycle_release_enabled": False,
        "decision_feature_trace": False,
    },
}
_PROFILE_BOUND_EXTRA_CONFIG = frozenset(
    {
        "implementation_revision",
        "preservation_events",
        "admission_namespace",
        "nvtx_ranges",
    }
)
REQUIRED_METRICS = frozenset(
    {
        "ttft_ms",
        "throughput_tokens_per_s",
        "episode_makespan_ms",
        "recomputed_prompt_tokens",
        "preservation_reused_tokens",
        "load_bytes",
        "store_bytes",
        "preemptions",
    }
)
REQUIRED_NEUTRAL_SETTINGS = {
    "E01": {"prefix_aware": False},
    "E02": {"ema_rank_weight_floor": 0.0},
    "E03": {"scoring_mode": "synchronous"},
    "E04": {
        "native_store_threshold": 0,
        "prefix_cap": "none",
        "admission_mode": "off",
    },
    "E05": {"prefetch_trigger": "preemption", "soft_retention": "off"},
    "E06": {"staging_mode": "dynamic_scratch"},
    "E07": {"prefetch_selector": "request_order"},
    "E08": {"scheduler": "native_sync_fcfs"},
    "E09": {"alpha": 0.5},
}
REQUIRED_SETTINGS = {
    "E01": {"prefix_aware": [False, True]},
    "E02": {"ema_rank_weight_floor": [0, 0.01, 0.03, 0.05, 0.1]},
    "E03": {
        "capture_stride": [1, 4],
        "scoring_mode": ["synchronous", "asynchronous"],
    },
    "E04": {
        "native_store_threshold": [0, 2],
        "prefix_cap": ["none", "known_shared_prefix_tokens"],
        "admission_mode": [
            "off",
            "historical_frequency",
            "historical_frequency_cost_per_byte",
        ],
    },
    "E05": {
        "prefetch_trigger": ["preemption", "known_prefix_wait_hint"],
        "tool_gap_ms": [0, 50, 250],
        "soft_retention": ["off", "bounded_hint"],
    },
    "E06": {
        "staging_mode": ["dynamic_scratch", "owned_reclaimable"],
        "fraction": [0.01, 0.05],
    },
    "E07": {"prefetch_selector": ["request_order", "semantic", "useful_on_time"]},
    "E08": {
        "scheduler": [
            "native_sync_fcfs",
            "cache_affinity",
            "native_async_if_supported",
        ]
    },
    "E09": {"alpha": [0.5, 0.6]},
}
COMBINATION_OBLIGATIONS = frozenset(
    {
        "interaction:E01+E02+E09",
        "interaction:E03+E07",
        "interaction:E04+E05",
        "interaction:E05+E06+E07",
        "interaction:E08+full_compatible_bundle",
        "control:original_semantic_no_prefetch",
        "control:original_semantic_dynamic_prefetch",
        "control:ARC_same_admission_prefetch_staging",
        "control:LRU_same_admission_prefetch_staging",
        "control:best_supported_native_scheduler",
        *(f"leave_one_out:{feature}" for feature in FEATURES),
    }
)
ORIGIN_FIELDS = (
    "content_origin",
    "arrival_origin",
    "wait_origin",
    "pressure_origin",
)
IDENTITY_FIELDS = (
    "source_id",
    "session_id",
    "family",
    "split",
    "generation",
)
ONLINE_TIME_FIELDS = (
    "observation_available_ns",
    "input_publication_ns",
    "consumer_decision_ns",
    "consumer_use_ns",
)


class ManifestError(ValueError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _opaque_token(namespace: str, kind: str, value: object, length: int = 24) -> str:
    return (
        kind
        + hashlib.sha256(
            namespace.encode() + b"\0" + _canonical_bytes(value)
        ).hexdigest()[:length]
    )


def digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_identity(path: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    return {
        "path": str(path.resolve()),
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "clean": git("status", "--porcelain") == "",
    }


def _scheduler_server_args(backend: object) -> list[str]:
    if backend == "native_sync_fcfs":
        return ["--no-async-scheduling"]
    if backend == "native_async_if_supported":
        return ["--async-scheduling"]
    if backend == "cache_affinity":
        return [
            "--no-async-scheduling",
            "--scheduler-cls",
            "semantic_offload.cache_affinity.CacheAffinityScheduler",
        ]
    raise ManifestError(f"unsupported executable scheduler_backend {backend!r}")


def _scheduler_resolution(backend: object) -> dict[str, object]:
    classes = {
        "native_sync_fcfs": "vllm.v1.core.sched.scheduler.Scheduler",
        "native_async_if_supported": (
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
        ),
        "cache_affinity": "semantic_offload.cache_affinity.CacheAffinityScheduler",
    }
    return {
        "backend": backend,
        "scheduler_class": classes.get(str(backend)),
        "async_scheduling": backend == "native_async_if_supported",
        "server_args": _scheduler_server_args(backend),
    }


def _expected_runtime_resolution(
    profile: Mapping[str, object], backend: object
) -> dict[str, object]:
    scheduler = _scheduler_resolution(backend)
    model = profile["model_runtime"]
    resources = profile["resources"]
    return {
        "schema": "semantic-runtime-resolution-v1",
        "available": True,
        "async_scheduling": scheduler["async_scheduling"],
        "scheduler_class": scheduler["scheduler_class"],
        "dtype": model["dtype"],
        "tensor_parallel_size": model["tensor_parallel_size"],
        "enforce_eager": resources["enforce_eager"],
        "max_num_seqs": resources["max_num_seqs"],
        "max_num_batched_tokens": resources["max_num_batched_tokens"],
        "kv_block_size_tokens": resources["kv_block_size_tokens"],
        "max_model_len": resources["max_model_len"],
        "gpu_memory_utilization": resources["gpu_memory_utilization"],
        "num_gpu_blocks_override": resources["total_gpu_blocks"],
    }


def _engine_server_args(profile: Mapping[str, object], *, seed: int) -> list[str]:
    model = profile["model_runtime"]
    resources = profile["resources"]
    _require(resources["enforce_eager"] is True, "campaign requires eager execution")
    return [
        "--seed",
        str(seed),
        "--revision",
        str(model["model_revision"]),
        "--dtype",
        str(model["dtype"]),
        "--tensor-parallel-size",
        str(model["tensor_parallel_size"]),
        "--block-size",
        str(resources["kv_block_size_tokens"]),
        "--enforce-eager",
        "--max-num-seqs",
        str(resources["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(resources["max_num_batched_tokens"]),
        *_scheduler_resolution(profile["scheduler_backend"])["server_args"],
    ]


def _validate_runner_argv_template(argv: Sequence[str]) -> None:
    joined = " ".join(argv)
    _require("execute-cell" in argv, "runner argv must use execute-cell")
    for required in (
        "--manifest={manifest}",
        "--request-manifest={request_manifest}",
        "--cell-id={cell_id}",
        "--profile-id={profile_id}",
        "--port={port}",
        "--output={output_dir}",
    ):
        _require(required in joined, f"runner argv missing exact {required}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _load_bound_file(binding: Mapping[str, object], base: Path, label: str) -> Path:
    raw_path = binding.get("path")
    expected = binding.get("sha256")
    _require(isinstance(raw_path, str) and raw_path, f"{label}.path is required")
    _require(isinstance(expected, str) and len(expected) == 64, f"{label}.sha256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base / path).resolve()
    _require(path.is_file(), f"{label} does not exist: {path}")
    _require(file_digest(path) == expected, f"{label} sha256 mismatch")
    return path


def build_request_manifest(
    contexts_path: Path,
    prepared_manifest_path: Path,
) -> dict[str, object]:
    """Freeze the lead-approved request graph from immutable prepared contexts."""
    with prepared_manifest_path.open(encoding="utf-8") as source:
        prepared = json.load(source)
    _require(prepared.get("schema") == "ep-prepared-contexts-v1", "prepared schema")
    _require(
        prepared.get("output_sha256") == file_digest(contexts_path),
        "prepared context sha256 mismatch",
    )
    tokenizer_path = prepared.get("tokenizer_path")
    _require(isinstance(tokenizer_path, str), "prepared tokenizer path")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, revision=None
    )
    rows: list[dict[str, Any]] = []
    with contexts_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                row = json.loads(line)
                _require(isinstance(row, dict), f"context line {line_number}")
                rows.append(row)
    groups: dict[tuple[str, str, int], list[list[dict[str, Any]]]] = defaultdict(list)
    by_context: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = (
        defaultdict(dict)
    )
    for row in rows:
        profile = (row["family"], row["split"], int(row["context_budget"]))
        context_id = row["context_sha256"]
        context_group = by_context[profile].get(context_id)
        if context_group is None:
            context_group = []
            by_context[profile][context_id] = context_group
            groups[profile].append(context_group)
        context_group.append(row)
    for (family, split, budget), profile_groups in groups.items():
        _require(len(profile_groups) == 8, f"{family}/{split}/{budget}: eight groups")
        expected_cases = 2 if family == "qasper" else 1
        _require(
            all(len(group) == expected_cases for group in profile_groups),
            f"{family}/{split}/{budget}: prepared case cardinality",
        )

    system_prompt = "Use the supplied context to answer the request. Respond concisely."
    prompt_prefix_ids = tokenizer.encode(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nContext:\n",
        add_special_tokens=False,
    )
    request_separator_ids = tokenizer.encode("\n\nRequest:\n", add_special_tokens=False)
    prompt_suffix_ids = tokenizer.encode(
        "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
    )

    def prompt_record(row: Mapping[str, Any]) -> tuple[list[int], tuple[int, int]]:
        context_ids = [int(token) for token in row["selected_context_token_ids"]]
        task_ids = [int(token) for token in row["task_token_ids"]]
        context_start = len(prompt_prefix_ids)
        prompt_ids = [
            *prompt_prefix_ids,
            *context_ids,
            *request_separator_ids,
            *task_ids,
            *prompt_suffix_ids,
        ]
        return prompt_ids, (context_start, context_start + len(context_ids))

    requests: list[dict[str, object]] = []
    episode_order: dict[str, list[str]] = defaultdict(list)

    def append_phase(
        *,
        split: str,
        budget: int,
        episode: str,
        phase: str,
        phase_index: int,
        group_indices: Sequence[int],
        qasper_task: int,
        fresh_server_id: str,
        families: Sequence[str] = ("qasper", "repobench-p"),
    ) -> None:
        phase_ordinal = 0
        for group_index in group_indices:
            for family in families:
                group = groups[(family, split, budget)][group_index]
                task_index = qasper_task if family == "qasper" else 0
                row = group[task_index]
                request_id = (
                    f"{split}-{budget}-{episode}-{phase_index:02d}-"
                    f"{phase}-{phase_ordinal:02d}"
                )
                prompt_ids, context_span = prompt_record(row)
                session_id = f"{split}-{budget}-{episode}-{family}-group-{group_index}"
                task_origin = (
                    f"supplied_question_{task_index}"
                    if family == "qasper"
                    else "supplied_task_exact_repeat"
                )
                record: dict[str, object] = {
                    "request_id": request_id,
                    "request_index": len(episode_order[f"{split}:{budget}:{episode}"]),
                    "phase_index": phase_index,
                    "phase_ordinal": phase_ordinal,
                    "relative_release_ms": phase_ordinal * 125,
                    "depends_on_phase_completion": (
                        None if phase_index == 0 else phase_index - 1
                    ),
                    "fresh_server_id": fresh_server_id,
                    "source_id": row["source_id"],
                    "source_sha256": row["source_sha256"],
                    "context_sha256": row["context_sha256"],
                    "input_sha256": row["input_sha256"],
                    "session_id": session_id,
                    "family": family,
                    "split": split,
                    "context_budget": budget,
                    "generation": 0,
                    "content_origin": "immutable_prepared_longbench_context",
                    "arrival_origin": "fixed_125ms_within_phase_after_barrier",
                    "wait_origin": (
                        "fresh_server_start"
                        if phase_index == 0
                        else "all_prior_phase_requests_completed"
                    ),
                    "pressure_origin": (
                        "predeclared_one_use_pollution"
                        if phase == "pollution"
                        else "predeclared_episode_dependency_graph"
                    ),
                    "prompt_token_ids": prompt_ids,
                    "prompt_token_digest": digest(prompt_ids),
                    "context_token_span": list(context_span),
                    "key_chain_digest": digest(row["selected_context_token_ids"]),
                    "truncation": {
                        "strategy": (
                            "head1024_tail1024" if row["truncated"] else "none"
                        ),
                        "original_token_count": row["original_context_tokens"],
                        "selected_token_count": row["selected_context_tokens"],
                        "token_spans": row["token_spans"],
                    },
                    "suffix_budget_tokens": 32,
                    "temperature": 0.0,
                    "episode": episode,
                    "phase": phase,
                    "task_origin": task_origin,
                    "serving_inputs": {
                        "session_id": session_id,
                        "generation": 0,
                        "key_chain_digest": digest(row["selected_context_token_ids"]),
                    },
                    "runtime_observation_fields": [
                        "eligibility_ns",
                        "send_ns",
                        "completion_ns",
                    ],
                }
                if phase in {"revisit", "b_revisit"}:
                    record["revisit_origin"] = (
                        "supplied_question_0_after_distinct_question"
                        if family == "qasper"
                        else "exact_repeat_constructed_cache_workload"
                    )
                requests.append(record)
                episode_order[f"{split}:{budget}:{episode}"].append(request_id)
                phase_ordinal += 1

    profiles = sorted({(row["split"], int(row["context_budget"])) for row in rows})
    for split, budget in profiles:
        reusable_server = f"{split}-{budget}-reusable-prefix"
        phase_index = 0
        for group_index in range(4):
            for family in ("qasper", "repobench-p"):
                for phase, task in (("warm1", 0), ("warm2", 1)):
                    append_phase(
                        split=split,
                        budget=budget,
                        episode="reusable_prefix",
                        phase=phase,
                        phase_index=phase_index,
                        group_indices=[group_index],
                        qasper_task=task,
                        fresh_server_id=reusable_server,
                        families=[family],
                    )
                    phase_index += 1
        for phase, indices in (("pollution", range(4, 8)), ("revisit", range(4))):
            append_phase(
                split=split,
                budget=budget,
                episode="reusable_prefix",
                phase=phase,
                phase_index=phase_index,
                group_indices=indices,
                qasper_task=0,
                fresh_server_id=reusable_server,
            )
            phase_index += 1
        topic_server = f"{split}-{budget}-distinct-topic-shift"
        phase_index = 0
        for topic, indices in (("a", range(2)), ("b", range(2, 4))):
            for group_index in indices:
                for family in ("qasper", "repobench-p"):
                    for warm, task in (("warm1", 0), ("warm2", 1)):
                        append_phase(
                            split=split,
                            budget=budget,
                            episode="distinct_topic_shift",
                            phase=f"{topic}_{warm}",
                            phase_index=phase_index,
                            group_indices=[group_index],
                            qasper_task=task,
                            fresh_server_id=topic_server,
                            families=[family],
                        )
                        phase_index += 1
        for phase, indices in (("pollution", range(4, 8)), ("b_revisit", range(2, 4))):
            append_phase(
                split=split,
                budget=budget,
                episode="distinct_topic_shift",
                phase=phase,
                phase_index=phase_index,
                group_indices=indices,
                qasper_task=0,
                fresh_server_id=topic_server,
            )
            phase_index += 1
        append_phase(
            split=split,
            budget=budget,
            episode="nonreuse_null",
            phase="one_use",
            phase_index=0,
            group_indices=range(4),
            qasper_task=0,
            fresh_server_id=f"{split}-{budget}-nonreuse-null",
        )
    document = {
        "schema": REQUEST_SCHEMA,
        "prepared_contexts": {
            "path": str(contexts_path.resolve()),
            "sha256": file_digest(contexts_path),
        },
        "prepared_manifest": {
            "path": str(prepared_manifest_path.resolve()),
            "sha256": file_digest(prepared_manifest_path),
        },
        "tokenizer": {
            "path": tokenizer_path,
            "file_sha256": prepared["tokenizer_file_sha256"],
        },
        "prompt_template": {
            "system": system_prompt,
            "renderer": "qwen_im_token_segment_concatenation_v1",
            "prefix_token_ids": prompt_prefix_ids,
            "request_separator_token_ids": request_separator_ids,
            "suffix_token_ids": prompt_suffix_ids,
        },
        "phase_contract": {
            "barrier": "all_prior_phase_requests_completed",
            "target_warm_order": "adjacent_distinct_requests_before_next_target",
            "within_phase_spacing_ms": 125,
            "phase_time_charged_to_episode_makespan": True,
            "identical_dependency_graph_not_realized_wall_clock": True,
        },
        "serving_input_allowlist": [
            "session_id",
            "generation",
            "key_chain_digest",
        ],
        "diagnostic_labels_are_separate": True,
        "episode_states": {
            "reusable_prefix": "ready",
            "distinct_topic_shift": "ready",
            "nonreuse_null": "ready",
            "causal_tool_resume": "pending",
        },
        "episode_order_digests": {
            name: digest(order) for name, order in sorted(episode_order.items())
        },
        "requests": requests,
    }
    validate_request_manifest(document)
    return document


def extend_causal_tool_resume_manifest(
    document: Mapping[str, object],
    *,
    base_sha256: str,
    kv_block_size_tokens: int = 16,
) -> dict[str, object]:
    """Create a separate exact wait-hint episode without mutating the base."""
    validate_request_manifest(document)
    _require(
        document["episode_states"]["causal_tool_resume"] == "pending",
        "base causal_tool_resume episode must be pending",
    )
    _require(
        isinstance(kv_block_size_tokens, int)
        and not isinstance(kv_block_size_tokens, bool)
        and kv_block_size_tokens > 0,
        "causal extension KV block size",
    )
    extended = copy.deepcopy(dict(document))
    base_requests = list(extended["requests"])
    additions: list[dict[str, object]] = []
    profiles = sorted(
        {(request["split"], request["context_budget"]) for request in base_requests}
    )
    for split, context_budget in profiles:
        reusable = sorted(
            (
                request
                for request in base_requests
                if request["split"] == split
                and request["context_budget"] == context_budget
                and request["episode"] == "reusable_prefix"
            ),
            key=lambda request: int(request["request_index"]),
        )
        _require(len(reusable) == 32, "causal extension requires reusable32 source")
        warm2_by_owner = {
            (request["family"], request["session_id"]): request
            for request in reusable[:16]
            if request["phase"] == "warm2"
        }
        causal_order: list[str] = []
        for source_request in reusable:
            record = copy.deepcopy(source_request)
            original_session = str(source_request["session_id"])
            record["fresh_server_id"] = f"{split}-{context_budget}-causal-tool-resume"
            record["session_id"] = "causal-tool-resume:" + digest(original_session)[:24]
            record["episode"] = "causal_tool_resume"
            if source_request["phase"] == "pollution":
                record["phase_index"] = 16
                record["phase_ordinal"] = int(record["request_index"]) - 16
                record["relative_release_ms"] = 125 * int(record["phase_ordinal"])
                record["depends_on_phase_completion"] = 15
            if source_request["phase"] == "revisit":
                warm2 = warm2_by_owner[(source_request["family"], original_session)]
                for field in (
                    "prompt_token_ids",
                    "prompt_token_digest",
                    "input_sha256",
                    "task_origin",
                ):
                    if field in warm2:
                        record[field] = copy.deepcopy(warm2[field])
                record["phase"] = "resume"
                record["phase_index"] = 16
                record["phase_ordinal"] = int(record["request_index"]) - 24
                record["relative_release_ms"] = 250
                record["depends_on_phase_completion"] = 15
                record["wait_origin"] = "common_250ms_tool_wait_after_warm_barrier"
                record["arrival_origin"] = "common_resume_release_at_wait_deadline"
                record["revisit_origin"] = (
                    "supplied_question_1_after_tool_wait"
                    if record["family"] == "qasper"
                    else "exact_repeat_after_tool_wait"
                )
            record["request_id"] = (
                f"{split}-{context_budget}-causal-tool-resume-"
                f"{int(record['request_index']):02d}-{record['phase']}-"
                f"{record['family']}"
            )
            serving_inputs = copy.deepcopy(dict(record["serving_inputs"]))
            serving_inputs["session_id"] = record["session_id"]
            context_boundary = int(record["context_token_span"][1])
            boundary = context_boundary // kv_block_size_tokens * kv_block_size_tokens
            _require(boundary > 0, "known prefix has no complete KV page")
            known_prefix_tokens = record["prompt_token_ids"][:boundary]
            record["known_prefix_boundary_tokens"] = boundary
            record["known_prefix_token_digest"] = digest(known_prefix_tokens)
            serving_inputs["known_prefix_boundary_tokens"] = boundary
            serving_inputs["known_prefix_token_digest"] = record[
                "known_prefix_token_digest"
            ]
            record["serving_inputs"] = serving_inputs
            additions.append(record)
            causal_order.append(record["request_id"])
        extended["episode_order_digests"][
            f"{split}:{context_budget}:causal_tool_resume"
        ] = digest(causal_order)
    extended["requests"].extend(additions)
    extended["episode_states"]["causal_tool_resume"] = "ready"
    extended["causal_tool_resume_extension"] = {
        "schema": "ep-causal-tool-resume-extension-v1",
        "base_request_manifest_sha256": base_sha256,
        "base_request_count": len(base_requests),
        "added_request_count": len(additions),
        "kv_block_size_tokens": kv_block_size_tokens,
        "hint_boundary": "final_warm2_completion",
        "hint_eligibility_offset_ms": 0,
        "pollution_release_offsets_ms": [125 * index for index in range(8)],
        "resume_release_offset_ms": 250,
        "release_equations": {
            "pollution": "W + 125ms*phase_ordinal",
            "resume": "W + 250ms",
        },
        "activity_phase_index": 16,
        "activity_completion_dependency": 15,
        "activity_completion_rule": "all_16_responses",
        "hint_acknowledgement_gates_release": False,
        "policy_visible_wait": "fixed_250ms_base_wait_only",
        "future_suffix_withheld_until_release": True,
    }
    allowlist = list(extended.get("serving_input_allowlist", []))
    for field in (
        "known_prefix_boundary_tokens",
        "known_prefix_token_digest",
    ):
        if field not in allowlist:
            allowlist.append(field)
    extended["serving_input_allowlist"] = allowlist
    validate_request_manifest(extended)
    return extended


def validate_request_manifest(document: Mapping[str, object]) -> dict[str, object]:
    _require(document.get("schema") == REQUEST_SCHEMA, "request manifest schema")
    requests = document.get("requests")
    _require(isinstance(requests, list) and requests, "requests must be non-empty")
    seen_ids: set[str] = set()
    counts: Counter[tuple[str, str, int, str, str]] = Counter()
    episode_requests: dict[tuple[str, int, str], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    episode_states = document.get("episode_states")
    _require(isinstance(episode_states, Mapping), "episode_states is required")
    causal_extension = document.get("causal_tool_resume_extension", {})
    causal_block_size = (
        causal_extension.get("kv_block_size_tokens", 16)
        if isinstance(causal_extension, Mapping)
        else 16
    )
    activation_training = (
        causal_extension.get("activation_training")
        if isinstance(causal_extension, Mapping)
        else None
    )
    training_profile = (
        (
            activation_training.get("split"),
            activation_training.get("context_budget"),
        )
        if isinstance(activation_training, Mapping)
        else None
    )
    for required in (
        "reusable_prefix",
        "distinct_topic_shift",
        "nonreuse_null",
        "causal_tool_resume",
    ):
        _require(
            episode_states.get(required) in {"ready", "pending"},
            f"episode_states.{required} must be ready or pending",
        )
    if episode_states.get("causal_tool_resume") == "ready":
        _require(
            isinstance(causal_block_size, int)
            and not isinstance(causal_block_size, bool)
            and causal_block_size > 0,
            "causal tool-resume KV block size",
        )
    for index, request in enumerate(requests):
        _require(isinstance(request, Mapping), f"request {index} must be an object")
        request_id = request.get("request_id")
        _require(isinstance(request_id, str) and request_id, f"request {index} id")
        _require(request_id not in seen_ids, f"duplicate request_id {request_id}")
        seen_ids.add(request_id)
        for field in ORIGIN_FIELDS + IDENTITY_FIELDS:
            _require(request.get(field) not in (None, ""), f"{request_id}: {field}")
        _require(request["family"] in {"qasper", "repobench-p"}, "unknown family")
        _require(request["split"] in {"screen", "heldout"}, "unknown split")
        _require(
            request.get("context_budget") in {2048, 4096},
            f"{request_id}: context_budget",
        )
        _require(
            isinstance(request["generation"], int) and request["generation"] >= 0,
            f"{request_id}: generation",
        )
        tokens = request.get("prompt_token_ids")
        _require(
            isinstance(tokens, list)
            and tokens
            and all(isinstance(token, int) and token >= 0 for token in tokens),
            f"{request_id}: prompt_token_ids",
        )
        _require(
            request.get("prompt_token_digest") == digest(tokens),
            f"{request_id}: prompt token digest mismatch",
        )
        _require(
            isinstance(request.get("key_chain_digest"), str)
            and re.fullmatch(r"[0-9a-f]{64}", request["key_chain_digest"]) is not None,
            f"{request_id}: key chain digest",
        )
        truncation = request.get("truncation")
        _require(isinstance(truncation, Mapping), f"{request_id}: truncation")
        _require(
            truncation.get("strategy") in {"none", "head1024_tail1024"},
            f"{request_id}: truncation strategy",
        )
        _require(
            isinstance(truncation.get("original_token_count"), int)
            and truncation["original_token_count"]
            >= truncation.get("selected_token_count", len(tokens)),
            f"{request_id}: original token count",
        )
        _require(
            isinstance(request.get("suffix_budget_tokens"), int)
            and request["suffix_budget_tokens"] >= 0,
            f"{request_id}: suffix budget",
        )
        episode = request.get("episode")
        phase = request.get("phase")
        _require(
            episode
            in {
                "reusable_prefix",
                "distinct_topic_shift",
                "nonreuse_null",
                "causal_tool_resume",
            },
            f"{request_id}: episode",
        )
        if episode == "reusable_prefix":
            _require(
                phase in {"warm1", "warm2", "pollution", "revisit"},
                f"{request_id}: reusable phase",
            )
            expected = (
                ("supplied_question_1" if phase == "warm2" else "supplied_question_0")
                if request["family"] == "qasper"
                else "supplied_task_exact_repeat"
            )
            _require(
                request.get("task_origin") == expected,
                f"{request_id}: task_origin must be {expected}",
            )
            if phase == "revisit":
                revisit_origin = (
                    "supplied_question_0_after_distinct_question"
                    if request["family"] == "qasper"
                    else "exact_repeat_constructed_cache_workload"
                )
                _require(
                    request.get("revisit_origin") == revisit_origin,
                    f"{request_id}: revisit_origin must be {revisit_origin}",
                )
        elif episode == "distinct_topic_shift":
            _require(
                phase
                in {
                    "a_warm1",
                    "a_warm2",
                    "b_warm1",
                    "b_warm2",
                    "pollution",
                    "b_revisit",
                },
                f"{request_id}: topic-shift phase",
            )
        elif episode == "nonreuse_null":
            _require(phase == "one_use", f"{request_id}: null phase")
        elif episode == "causal_tool_resume":
            training_row = (
                training_profile == (request["split"], request["context_budget"])
                and phase == "activation_train"
            )
            extended_activity_row = training_profile == (
                request["split"],
                request["context_budget"],
            ) and phase in {"load", "recovery"}
            _require(
                phase in {"warm1", "warm2", "pollution", "resume"}
                or training_row
                or extended_activity_row,
                f"{request_id}: causal tool-resume phase",
            )
            boundary = request.get("known_prefix_boundary_tokens")
            context_span = request.get("context_token_span")
            _require(
                isinstance(boundary, int)
                and not isinstance(boundary, bool)
                and isinstance(context_span, list)
                and len(context_span) == 2
                and boundary <= context_span[1]
                and context_span[1] - boundary < causal_block_size
                and boundary % causal_block_size == 0
                and 0 < boundary <= len(tokens)
                and request.get("known_prefix_token_digest")
                == digest(tokens[:boundary]),
                f"{request_id}: known prefix boundary/digest mismatch",
            )
            if phase == "resume":
                expected_task = (
                    "supplied_question_1"
                    if request["family"] == "qasper"
                    else "supplied_task_exact_repeat"
                )
                _require(
                    request.get("task_origin") == expected_task,
                    f"{request_id}: resume task was not previously observed",
                )
        serving_inputs = request.get("serving_inputs")
        _require(isinstance(serving_inputs, Mapping), f"{request_id}: serving_inputs")
        _require(
            not {"episode", "phase", "revisit_origin", "future_reuse"}.intersection(
                serving_inputs
            ),
            f"{request_id}: future phase labels cannot be serving inputs",
        )
        _require(
            isinstance(request.get("request_index"), int)
            and request["request_index"] >= 0,
            f"{request_id}: request_index",
        )
        _require(
            isinstance(request.get("phase_index"), int) and request["phase_index"] >= 0,
            f"{request_id}: phase_index",
        )
        _require(
            isinstance(request.get("relative_release_ms"), int)
            and request["relative_release_ms"] >= 0,
            f"{request_id}: relative_release_ms",
        )
        episode_requests[
            (request["split"], request["context_budget"], str(episode))
        ].append(request)
        counts[
            (
                request["family"],
                request["split"],
                request["context_budget"],
                str(episode),
                str(phase),
            )
        ] += 1

    profiles = {(request["split"], request["context_budget"]) for request in requests}
    for split, context_budget in profiles:
        if episode_states["reusable_prefix"] == "ready":
            for phase in ("warm1", "warm2", "pollution", "revisit"):
                count = sum(
                    counts[
                        (
                            family,
                            split,
                            context_budget,
                            "reusable_prefix",
                            phase,
                        )
                    ]
                    for family in ("qasper", "repobench-p")
                )
                _require(
                    count == 8,
                    f"{split}/{context_budget} reusable_prefix/{phase} must have 8 requests",
                )
        if episode_states["distinct_topic_shift"] == "ready":
            expected_counts = {
                "a_warm1": 4,
                "a_warm2": 4,
                "b_warm1": 4,
                "b_warm2": 4,
                "pollution": 8,
                "b_revisit": 4,
            }
            for phase, expected in expected_counts.items():
                count = sum(
                    counts[
                        (
                            family,
                            split,
                            context_budget,
                            "distinct_topic_shift",
                            phase,
                        )
                    ]
                    for family in ("qasper", "repobench-p")
                )
                _require(
                    count == expected,
                    f"{split}/{context_budget} topic-shift/{phase} must have {expected} requests",
                )
        if episode_states["nonreuse_null"] == "ready":
            count = sum(
                counts[(family, split, context_budget, "nonreuse_null", "one_use")]
                for family in ("qasper", "repobench-p")
            )
            _require(
                count == 8,
                f"{split}/{context_budget} nonreuse_null must have 8 requests",
            )
        causal = sorted(
            episode_requests[(split, context_budget, "causal_tool_resume")],
            key=lambda item: item["request_index"],
        )
        if episode_states["causal_tool_resume"] == "ready":
            if training_profile == (split, context_budget):
                training_count = activation_training.get("training_request_count")
                activity_phase_index = activation_training.get("activity_phase_index")
                post_load_count = int(
                    activation_training.get("post_activity_load_request_count", 0)
                )
                recovery_count = int(
                    activation_training.get("post_drain_recovery_request_count", 0)
                )
                _require(
                    activation_training.get("schema")
                    == "msr-activation-training-extension-v1"
                    and isinstance(training_count, int)
                    and training_count >= 1
                    and activity_phase_index == training_count
                    and isinstance(activation_training.get("target_sequence"), list)
                    and len(activation_training["target_sequence"]) == training_count
                    and isinstance(
                        activation_training.get("serialize_within_phase", False), bool
                    )
                    and len(causal)
                    == training_count + 16 + post_load_count + recovery_count
                    and [item["request_index"] for item in causal]
                    == list(
                        range(training_count + 16 + post_load_count + recovery_count)
                    ),
                    "activation training request/index contract",
                )
                training_rows = causal[:training_count]
                pollution = causal[training_count : training_count + 8]
                resumes = causal[training_count + 8 : training_count + 16]
                post_load = causal[
                    training_count + 16 : training_count + 16 + post_load_count
                ]
                recovery = causal[training_count + 16 + post_load_count :]
                pollution_offsets = activation_training.get(
                    "pollution_release_offsets_ms",
                    [125 * index for index in range(8)],
                )
                score_refresh_request_id = activation_training.get(
                    "score_refresh_request_id"
                )
                score_refresh_idle_ms = activation_training.get(
                    "context_age_idle_before_score_refresh_ms", 0
                )
                _require(
                    all(
                        item["phase"] == "activation_train"
                        and item["phase_index"] == index
                        and item.get("depends_on_phase_completion")
                        == (index - 1 if index else None)
                        and item["relative_release_ms"]
                        == (
                            score_refresh_idle_ms
                            if item["request_id"] == score_refresh_request_id
                            else 0
                        )
                        for index, item in enumerate(training_rows)
                    )
                    and (
                        score_refresh_request_id is None
                        or (
                            isinstance(score_refresh_idle_ms, int)
                            and not isinstance(score_refresh_idle_ms, bool)
                            and score_refresh_idle_ms in {2_000, 5_000, 15_000}
                            and training_rows[-1]["request_id"]
                            == score_refresh_request_id
                        )
                    )
                    and [item["phase"] for item in pollution] == ["pollution"] * 8
                    and [item["phase"] for item in resumes] == ["resume"] * 8
                    and all(
                        item["phase_index"] == activity_phase_index
                        and item.get("depends_on_phase_completion")
                        == training_count - 1
                        for item in (*pollution, *resumes)
                    )
                    and [item["phase_ordinal"] for item in pollution] == list(range(8))
                    and [item["relative_release_ms"] for item in pollution]
                    == pollution_offsets
                    and [item["phase_ordinal"] for item in resumes] == list(range(8))
                    and all(
                        item["relative_release_ms"] == item.get("hint_lead_ms", 250)
                        and item.get("hint_lead_ms", 250) in {50, 250, 750}
                        for item in resumes
                    ),
                    "activation training schedule contract",
                )
                if post_load_count or recovery_count:
                    _require(
                        post_load_count == 8
                        and recovery_count == 4
                        and [item["phase"] for item in post_load] == ["load"] * 8
                        and [item["phase_index"] for item in post_load]
                        == [activity_phase_index + 1] * 8
                        and [item["phase_ordinal"] for item in post_load]
                        == list(range(8))
                        and all(
                            item.get("depends_on_phase_completion")
                            == activity_phase_index
                            for item in post_load
                        )
                        and [item["relative_release_ms"] for item in post_load]
                        == activation_training.get(
                            "post_activity_load_release_offsets_ms"
                        )
                        and [item["phase"] for item in recovery] == ["recovery"] * 4
                        and [item["phase_index"] for item in recovery]
                        == [activity_phase_index + 2] * 4
                        and [item["phase_ordinal"] for item in recovery]
                        == list(range(4))
                        and all(
                            item.get("depends_on_phase_completion")
                            == activity_phase_index + 1
                            for item in recovery
                        )
                        and [item["relative_release_ms"] for item in recovery]
                        == activation_training.get(
                            "post_drain_recovery_release_offsets_ms"
                        ),
                        "activation post-drain load/recovery schedule contract",
                    )
                hint_publication_order = activation_training.get(
                    "hint_publication_order"
                )
                if hint_publication_order is not None:
                    hint_only_ids = [
                        item.get("hint_id")
                        for item in causal_extension.get("hint_only_events", [])
                    ]
                    expected_publication_ids = [
                        item["request_id"] for item in resumes
                    ] + hint_only_ids
                    _require(
                        activation_training.get("serialize_hint_publication") is True
                        and isinstance(hint_publication_order, list)
                        and all(
                            isinstance(item, str) and item
                            for item in hint_publication_order
                        )
                        and len(hint_publication_order)
                        == len(set(hint_publication_order))
                        and set(hint_publication_order)
                        == set(expected_publication_ids),
                        "activation hint publication order contract",
                    )
            else:
                _require(
                    len(causal) == 32
                    and [item["request_index"] for item in causal] == list(range(32)),
                    "causal tool-resume episode must have 32 contiguous requests",
                )
                _require(
                    [item["phase"] for item in causal[16:24]] == ["pollution"] * 8
                    and [item["phase"] for item in causal[24:]] == ["resume"] * 8,
                    "causal tool-resume pollution/resume ordering",
                )
                for pair_index in range(8):
                    warm1, warm2 = causal[2 * pair_index : 2 * pair_index + 2]
                    _require(
                        warm1["phase"] == "warm1"
                        and warm2["phase"] == "warm2"
                        and warm1["session_id"] == warm2["session_id"],
                        "causal tool-resume warm pairs must be adjacent",
                    )
                pollution = causal[16:24]
                resumes = causal[24:]
                _require(
                    all(
                        item["phase_index"] == 16
                        and item.get("depends_on_phase_completion") == 15
                        for item in (*pollution, *resumes)
                    )
                    and [item["phase_ordinal"] for item in pollution] == list(range(8))
                    and [item["relative_release_ms"] for item in pollution]
                    == [125 * index for index in range(8)]
                    and [item["phase_ordinal"] for item in resumes] == list(range(8))
                    and all(
                        item["relative_release_ms"] == item.get("hint_lead_ms", 250)
                        and item.get("hint_lead_ms", 250) in {50, 250, 750}
                        for item in resumes
                    ),
                    "causal tool-resume activity schedule differs from frozen graph",
                )
                _require(
                    isinstance(causal_extension, Mapping)
                    and causal_extension.get("hint_boundary")
                    == "final_warm2_completion"
                    and causal_extension.get("hint_eligibility_offset_ms") == 0
                    and causal_extension.get("pollution_release_offsets_ms")
                    == [125 * index for index in range(8)]
                    and causal_extension.get("resume_release_offset_ms")
                    in {250, "per_request"}
                    and causal_extension.get("activity_phase_index") == 16
                    and causal_extension.get("activity_completion_dependency") == 15
                    and causal_extension.get("activity_completion_rule")
                    == "all_16_responses"
                    and causal_extension.get("hint_acknowledgement_gates_release")
                    is False,
                    "causal tool-resume schedule contract",
                )
        reusable = sorted(
            episode_requests[(split, context_budget, "reusable_prefix")],
            key=lambda item: item["request_index"],
        )
        if episode_states["reusable_prefix"] == "ready":
            _require(
                len(reusable) == 32, "reusable-prefix episode must have 32 requests"
            )
            _require(
                [item["request_index"] for item in reusable] == list(range(32)),
                "reusable-prefix request indexes must be contiguous",
            )
            for pair_index in range(8):
                warm1, warm2 = reusable[2 * pair_index : 2 * pair_index + 2]
                _require(
                    warm1["phase"] == "warm1"
                    and warm2["phase"] == "warm2"
                    and warm1["session_id"] == warm2["session_id"]
                    and warm2.get("depends_on_phase_completion")
                    == warm1["phase_index"],
                    "each reusable target must warm in an adjacent dependent pair",
                )
            _require(
                [item["phase"] for item in reusable[16:24]] == ["pollution"] * 8
                and [item["phase"] for item in reusable[24:]] == ["revisit"] * 8,
                "reusable-prefix pollution/revisit ordering",
            )
        topic = sorted(
            episode_requests[(split, context_budget, "distinct_topic_shift")],
            key=lambda item: item["request_index"],
        )
        if episode_states["distinct_topic_shift"] == "ready":
            _require(len(topic) == 28, "topic-shift episode must have 28 requests")
            _require(
                [item["request_index"] for item in topic] == list(range(28)),
                "topic-shift request indexes must be contiguous",
            )
            for pair_index in range(8):
                warm1, warm2 = topic[2 * pair_index : 2 * pair_index + 2]
                expected_topic = "a" if pair_index < 4 else "b"
                _require(
                    warm1["phase"] == f"{expected_topic}_warm1"
                    and warm2["phase"] == f"{expected_topic}_warm2"
                    and warm1["session_id"] == warm2["session_id"]
                    and warm2.get("depends_on_phase_completion")
                    == warm1["phase_index"],
                    "topic targets must warm in adjacent dependent pairs",
                )
            _require(
                [item["phase"] for item in topic[16:24]] == ["pollution"] * 8
                and [item["phase"] for item in topic[24:]] == ["b_revisit"] * 4,
                "topic-shift pollution/revisit ordering",
            )
    order_digests = document.get("episode_order_digests", {})
    _require(isinstance(order_digests, Mapping), "episode_order_digests")
    reusable_digests = {
        value
        for name, value in order_digests.items()
        if str(name).endswith(":reusable_prefix")
    }
    topic_digests = {
        value
        for name, value in order_digests.items()
        if str(name).endswith(":distinct_topic_shift")
    }
    _require(
        not (
            episode_states["reusable_prefix"] == "ready"
            and episode_states["distinct_topic_shift"] == "ready"
            and reusable_digests
            and reusable_digests == topic_digests
        ),
        "reusable and topic-shift episodes cannot use the same ordering",
    )
    return {"request_count": len(requests), "counts": dict(counts)}


def _apply_delta(config: Mapping[str, object], delta: Sequence[object]) -> dict:
    result = copy.deepcopy(dict(config))
    for item in delta:
        _require(isinstance(item, Mapping), "config_delta entries must be objects")
        path = item.get("path")
        _require(isinstance(path, str) and path, "config_delta path")
        parts = path.split(".")
        cursor: dict = result
        for part in parts[:-1]:
            _require(isinstance(cursor.get(part), dict), f"unknown delta path {path}")
            cursor = cursor[part]
        leaf = parts[-1]
        _require(
            cursor.get(leaf) == item.get("before"), f"delta before mismatch: {path}"
        )
        cursor[leaf] = item.get("after")
    return result


_FEATURE_CONFIG_PATHS = {
    "E01": {"prefix_aware": "extra_config.prefix_aware"},
    "E02": {"ema_rank_weight_floor": "extra_config.ema_rank_weight_floor"},
    "E03": {
        "capture_stride": "extra_config.capture_stride",
        "scoring_mode": "extra_config.scoring_mode",
    },
    "E04": {
        "native_store_threshold": "extra_config.store_threshold",
        "prefix_cap": "request_controls.max_offload_tokens",
        "admission_mode": "extra_config.admission_mode",
    },
    "E05": {
        "prefetch_trigger": "extra_config.prefetch_trigger",
        "tool_gap_ms": "request_controls.tool_gap_ms",
        "soft_retention": "extra_config.soft_retention",
    },
    "E06": {
        "staging_mode": "extra_config.staging_mode",
        "fraction": "extra_config.prefetch_budget_fraction",
    },
    "E07": {"prefetch_selector": "extra_config.prefetch_selector"},
    "E08": {"scheduler": "scheduler_backend"},
    "E09": {"alpha": "extra_config.alpha"},
}


def _config_value(config: Mapping[str, object], path: str) -> object:
    value: object = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _setting_value_matches(name: str, expected: object, actual: object) -> bool:
    if name == "prefix_cap" and expected == "none":
        return actual is None
    if name == "soft_retention" and expected == "disabled":
        return actual in (None, False, "disabled")
    if name == "soft_retention" and expected == "off":
        return actual in (None, False, "disabled", "off")
    if name == "admission_mode" and expected == "off":
        return actual in (None, "disabled", "off")
    return actual == expected


def _arm_realizes_feature_delta(
    arm: Mapping[str, object],
    feature: str,
    arms: Mapping[str, Mapping[str, object]],
) -> bool:
    comparator_id = arm.get("comparator_arm_id")
    if comparator_id not in arms:
        return False
    comparator = arms[comparator_id]
    paths = _FEATURE_CONFIG_PATHS[feature]
    return any(
        _config_value(arm["config"], path) != _config_value(comparator["config"], path)
        for path in paths.values()
    )


def _major_feature_enabled(config: Mapping[str, object], feature: str) -> bool:
    values = {
        name: _config_value(config, path)
        for name, path in _FEATURE_CONFIG_PATHS[feature].items()
    }
    if feature == "E01":
        return values["prefix_aware"] is True
    if feature == "E02":
        return values["ema_rank_weight_floor"] in {0.01, 0.03, 0.05, 0.1}
    if feature == "E03":
        return values["scoring_mode"] == "asynchronous"
    if feature == "E04":
        return values["admission_mode"] in {
            "historical_frequency",
            "historical_frequency_cost_per_byte",
        }
    if feature == "E05":
        return (
            values["prefetch_trigger"] == "known_prefix_wait_hint"
            and values["soft_retention"] == "bounded_hint"
        )
    if feature == "E06":
        return (
            values["staging_mode"] == "owned_reclaimable"
            and isinstance(values["fraction"], (int, float))
            and float(values["fraction"]) > 0.0
        )
    if feature == "E07":
        return values["prefetch_selector"] == "useful_on_time"
    if feature == "E08":
        return values["scheduler"] == "cache_affinity"
    if feature == "E09":
        return values["alpha"] == 0.6
    return False


def _set_config_value(config: dict, path: str, value: object) -> None:
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _neutralized_feature_config(
    full_config: Mapping[str, object],
    feature: str,
    neutral_settings: Mapping[str, Mapping[str, object]],
) -> dict:
    config = copy.deepcopy(dict(full_config))
    neutral = neutral_settings[feature]
    for name, value in neutral.items():
        normalized = (
            None
            if name == "prefix_cap" and value == "none"
            else "disabled"
            if name == "admission_mode" and value == "off"
            else value
        )
        _set_config_value(config, _FEATURE_CONFIG_PATHS[feature][name], normalized)
    return config


def _with_config_value(
    config: Mapping[str, object], path: str, value: object
) -> dict[str, object]:
    changed = copy.deepcopy(dict(config))
    _set_config_value(changed, path, value)
    return changed


def _behavior_config(config: Mapping[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(dict(config))
    extra = normalized.get("extra_config")
    _require(isinstance(extra, dict), "control extra_config must be an object")
    for field in _PROFILE_BOUND_EXTRA_CONFIG:
        extra.pop(field, None)
    return normalized


def _original_control_configs(
    original_config: Mapping[str, object], scheduler_backend: str
) -> dict[str, dict[str, object]]:
    dynamic = _with_config_value(
        original_config, "extra_config.staging_mode", "dynamic_scratch"
    )
    _set_config_value(dynamic, "extra_config.prefetch_budget_fraction", 0.05)
    return {
        "original_semantic_no_prefetch": copy.deepcopy(dict(original_config)),
        "original_semantic_dynamic_prefetch": dynamic,
        "ARC_same_admission_prefetch_staging": _with_config_value(
            original_config, "policy", "arc"
        ),
        "LRU_same_admission_prefetch_staging": _with_config_value(
            original_config, "policy", "lru"
        ),
        "best_supported_native_scheduler": _with_config_value(
            original_config, "scheduler_backend", scheduler_backend
        ),
    }


def _control_config_matches_anchor(
    name: str,
    config: Mapping[str, object],
    *,
    anchor: Mapping[str, object],
) -> bool:
    if not str(anchor.get("policy", "")).lower().startswith("semantic"):
        return False
    if name == "ARC_same_admission_prefetch_staging":
        expected = _with_config_value(anchor, "policy", "arc")
        if float(
            _config_value(anchor, "extra_config.prefetch_budget_fraction") or 0
        ) > 0 and _config_value(anchor, "extra_config.prefetch_selector") in {
            "semantic",
            "useful_on_time",
        }:
            _set_config_value(expected, "extra_config.shared_prefetch_scorer", True)
        return config == expected
    if name == "LRU_same_admission_prefetch_staging":
        expected = _with_config_value(anchor, "policy", "lru")
        if float(
            _config_value(anchor, "extra_config.prefetch_budget_fraction") or 0
        ) > 0 and _config_value(anchor, "extra_config.prefetch_selector") in {
            "semantic",
            "useful_on_time",
        }:
            _set_config_value(expected, "extra_config.shared_prefetch_scorer", True)
        return config == expected
    if name == "best_supported_native_scheduler":
        scheduler = config.get("scheduler_backend")
        return scheduler in {"native_sync_fcfs", "native_async_if_supported"} and (
            config == _with_config_value(anchor, "scheduler_backend", scheduler)
        )
    return False


def _named_control_is_exact(
    name: str,
    arm: Mapping[str, object],
    *,
    arms: Mapping[str, Mapping[str, object]],
    reference: Mapping[str, object],
    original_controls: Mapping[str, Mapping[str, object]],
) -> bool:
    config = arm["config"]
    if (
        arm.get("control_id") != name
        or config != reference["config"]
        or arm["arm_config_hash"] != reference["config_hash"]
    ):
        return False
    if name in {
        "original_semantic_no_prefetch",
        "original_semantic_dynamic_prefetch",
    }:
        return config == original_controls[name]
    comparator = arms.get(str(arm.get("comparator_arm_id")))
    return (
        comparator is not None
        and comparator.get("execution_state") == "ready"
        and comparator.get("split") == arm.get("split")
        and _control_config_matches_anchor(name, config, anchor=comparator["config"])
    )


def _profile_payload(
    manifest: Mapping[str, object], profile: Mapping[str, object]
) -> dict[str, object]:
    return {
        "source": manifest["source"],
        "corpus_selection": manifest["corpus_selection"],
        "request_manifest": manifest["request_manifest"],
        "protocol_bindings": manifest["protocol_bindings"],
        "feature_neutral_settings": manifest["feature_neutral_settings"],
        "control_references": manifest["control_references"],
        "effective_environment": manifest["effective_environment"],
        "comparator_config": profile["comparator_config"],
        "workload": profile["workload"],
        "model_runtime": profile["model_runtime"],
        "scheduler_backend": profile["scheduler_backend"],
        "resources": profile["resources"],
        "replication_plan": profile["replication_plan"],
    }


def _nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _validate_async_scoring_evidence(
    events: object, planned_cell: Mapping[str, object]
) -> None:
    _require(isinstance(events, list) and events, "async scoring evidence is missing")
    expected_age_ns = planned_cell.get("async_scoring_max_snapshot_age_ns")
    statuses = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("event") == "status"
    ]
    _require(statuses, "async scoring terminal status evidence is missing")
    final_status = statuses[-1]
    _require(
        final_status.get("pending_items") == 0
        and final_status.get("pending_snapshot_bytes") == 0
        and final_status.get("max_snapshot_age_ns") == expected_age_ns,
        "async scoring terminal status is not drained or age-bound",
    )
    published_events = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("event") == "published"
    ]
    published = {event.get("observation_sequence"): event for event in published_events}
    consumed = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("event") == "consumed"
    ]
    rejected = [
        event
        for event in events
        if isinstance(event, Mapping) and event.get("event") == "consumption_rejected"
    ]
    _require(
        len(published) == len(published_events)
        and all(
            isinstance(sequence, int) and not isinstance(sequence, bool)
            for sequence in published
        ),
        "async scoring published observations are duplicated or unsequenced",
    )
    if not published:
        next_sequence = final_status.get("next_observation_sequence")
        producer_terminal_sequences = {
            event.get("observation_sequence")
            for event in events
            if isinstance(event, Mapping)
            and event.get("event")
            in {"submission_dropped", "scoring_failed", "publication_rejected"}
        }
        _require(
            isinstance(next_sequence, int)
            and next_sequence >= 0
            and (
                next_sequence == 0
                or producer_terminal_sequences == set(range(next_sequence))
            ),
            "async scoring zero-publication run lacks complete zero-action evidence",
        )
        for event in events:
            if not (
                isinstance(event, Mapping)
                and event.get("event") == "publication_rejected"
                and event.get("reason") == "snapshot_expired"
            ):
                continue
            capture_ns = event.get("snapshot_captured_ns")
            input_ready_ns = event.get("observation_available_ns")
            publication_ns = event.get("publication_ns")
            expires_at_ns = event.get("expires_at_ns")
            _require(
                isinstance(capture_ns, int)
                and isinstance(input_ready_ns, int)
                and isinstance(publication_ns, int)
                and expires_at_ns == capture_ns + expected_age_ns
                and capture_ns <= input_ready_ns <= publication_ns
                and expires_at_ns < publication_ns,
                "async scoring producer expiry does not derive from capture",
            )
        return
    terminal_sequences = [
        event.get("observation_sequence") for event in (*consumed, *rejected)
    ]
    _require(
        len(terminal_sequences) == len(set(terminal_sequences))
        and set(terminal_sequences) == set(published),
        "async scoring publication/consumption join is incomplete",
    )
    for publication in published.values():
        capture_ns = publication.get("snapshot_captured_ns")
        input_ready_ns = publication.get("observation_available_ns")
        publication_ns = publication.get("publication_ns")
        expires_at_ns = publication.get("expires_at_ns")
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (capture_ns, input_ready_ns, publication_ns)
            )
            and isinstance(expected_age_ns, int)
            and expected_age_ns > 0
            and expires_at_ns == capture_ns + expected_age_ns
            and capture_ns <= input_ready_ns <= publication_ns <= expires_at_ns
            and publication.get("max_snapshot_age_ns") == expected_age_ns,
            "async scoring publication deadline does not derive from capture",
        )
    for event in consumed:
        publication = published[event["observation_sequence"]]
        expires_at_ns = publication.get("expires_at_ns")
        clocks = [
            publication.get("snapshot_captured_ns"),
            publication.get("observation_available_ns"),
            publication.get("publication_ns"),
            event.get("consumer_decision_ns"),
            event.get("consumer_use_ns"),
        ]
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in clocks
            )
            and clocks == sorted(clocks)
            and publication.get("request_ids") == event.get("request_ids")
            and isinstance(expires_at_ns, int)
            and event.get("snapshot_captured_ns") == clocks[0]
            and event.get("observation_available_ns") == clocks[1]
            and event.get("expires_at_ns") == expires_at_ns
            and clocks[-1] <= expires_at_ns,
            "async scoring consumed evidence violates causal age bounds",
        )
        request_bindings = event.get("request_bindings")
        _require(
            isinstance(request_bindings, list)
            and request_bindings
            and all(isinstance(binding, Mapping) for binding in request_bindings),
            "async scoring consumption request bindings are missing",
        )
        for binding in request_bindings:
            _require(
                binding.get("run_id") == planned_cell["policy_run_id"]
                and binding.get("cell_id") == planned_cell["policy_cell_id"],
                "async scoring consumption request binding mismatch",
            )
    for event in rejected:
        publication = published[event["observation_sequence"]]
        decision_ns = event.get("consumer_decision_ns")
        if event.get("reason") == "snapshot_expired_at_consumer":
            _require(
                isinstance(decision_ns, int)
                and decision_ns > publication.get("expires_at_ns", decision_ns),
                "async scoring expiry rejection lacks elapsed deadline",
            )


def _wait_hint_action_counts(
    publications: Sequence[Mapping[str, object]],
    runtime_events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    acknowledgement_statuses = Counter(
        response.get("status")
        for publication in publications
        if isinstance((response := publication.get("response")), Mapping)
        and isinstance(response.get("status"), str)
    )
    runtime_event_counts = Counter(
        event.get("event")
        for event in runtime_events
        if isinstance(event.get("event"), str)
    )
    terminal_event_ids = {
        event.get("event_id")
        for event in runtime_events
        if event.get("event_id")
        and (
            event.get("event") in {"staging_consumed", "staging_retired"}
            or (
                event.get("event") == "hint_result"
                and event.get("status")
                in {"expired", "no_action", "suppressed", "rejected"}
            )
        )
    }
    return {
        "publication_count": len(publications),
        "acknowledgement_statuses": dict(sorted(acknowledgement_statuses.items())),
        "runtime_event_counts": dict(sorted(runtime_event_counts.items())),
        "terminal_event_id_count": len(terminal_event_ids),
    }


def _validate_wait_hint_publications(
    result: Mapping[str, object],
    planned_cell: Mapping[str, object],
    bindings: Sequence[Mapping[str, object]],
    raw_requests: Sequence[Mapping[str, object]],
) -> None:
    publications = result.get("wait_hint_publications")
    _require(isinstance(publications, list), "wait hint publication evidence")
    causal_bindings = [
        binding
        for binding in bindings
        if binding.get("episode") == "causal_tool_resume"
    ]
    if not causal_bindings:
        _require(not publications, "non-wait episode published wait hints")
        return
    resume_bindings = [
        binding for binding in causal_bindings if binding.get("phase") == "resume"
    ]
    trigger = planned_cell.get("prefetch_trigger")
    tool_gap_ms = planned_cell.get("tool_gap_ms")
    retention_ms = planned_cell.get("wait_hint_retention_ms")
    _require(
        isinstance(tool_gap_ms, int)
        and not isinstance(tool_gap_ms, bool)
        and tool_gap_ms in {0, 50, 250},
        "planned wait-hint tool gap is invalid",
    )
    _require(
        isinstance(retention_ms, int)
        and not isinstance(retention_ms, bool)
        and retention_ms > 0,
        "planned wait-hint retention is invalid",
    )
    if trigger != "known_prefix_wait_hint":
        _require(
            not publications
            and not any(
                isinstance(event, Mapping) and event.get("event_id")
                for event in result.get("wait_hint_events", [])
            ),
            "reactive control published early wait hints",
        )
        return
    demand_publications = [
        publication
        for publication in publications
        if isinstance(publication, Mapping) and publication.get("hint_only") is not True
    ]
    hint_only_publications = [
        publication
        for publication in publications
        if isinstance(publication, Mapping) and publication.get("hint_only") is True
    ]
    _require(
        len(demand_publications) == len(resume_bindings) == 8,
        "known-prefix wait treatment requires eight hint publications",
    )
    by_manifest_id = {
        publication.get("manifest_request_id"): publication
        for publication in demand_publications
        if isinstance(publication, Mapping)
    }
    requests_by_id = {
        request["manifest_request_id"]: request for request in raw_requests
    }
    from semantic_offload.wait_hint import KnownPrefixWaitHint

    for binding in resume_bindings:
        expected_lead_ms = int(binding.get("hint_lead_ms", tool_gap_ms))
        manifest_request_id = binding["request_id"]
        publication = by_manifest_id.get(manifest_request_id)
        _require(
            isinstance(publication, Mapping),
            f"missing wait hint publication for {manifest_request_id}",
        )
        payload = publication.get("payload")
        _require(isinstance(payload, Mapping), "wait hint payload evidence missing")
        hint = KnownPrefixWaitHint.from_mapping(payload)
        expected_event_id = _wait_hint_event_id(binding, planned_cell)
        expected_session = _opaque_token(
            str(planned_cell["opaque_identity_namespace"]),
            "s",
            [binding["session_id"], binding["generation"]],
        )
        response = publication.get("response")
        response_status = (
            response.get("status") if isinstance(response, Mapping) else None
        )
        response_reason = (
            response.get("reason") if isinstance(response, Mapping) else None
        )
        _require(
            publication.get("payload_digest") == digest(payload)
            and publication.get("future_suffix_published") is False
            and hint.event_id == expected_event_id
            and hint.owner_id == binding["external_request_id"]
            and hint.run_id == planned_cell["policy_run_id"]
            and hint.cell_id == planned_cell["policy_cell_id"]
            and hint.session_id == expected_session
            and hint.generation == binding["generation"]
            and hint.prefix_boundary_tokens == binding["known_prefix_boundary_tokens"]
            and hint.declared_wait_ms == expected_lead_ms
            and publication.get("http_status") == 200
            and publication.get("error") is None
            and isinstance(response, Mapping)
            and response.get("event_id") == expected_event_id
            and response.get("run_id") == hint.run_id
            and response.get("cell_id") == hint.cell_id
            and response.get("session_id") == hint.session_id
            and response.get("generation") == hint.generation
            and response.get("nomination_deadline_ns")
            == hint.input_publication_ns + expected_lead_ms * 1_000_000
            and response.get("expires_at_ns")
            == hint.input_publication_ns + (expected_lead_ms + retention_ms) * 1_000_000
            and response_status in {"accepted", "expired", "no_action"}
            and (
                response_status == "accepted"
                or response_reason
                in {
                    "nomination_deadline_elapsed",
                    "observed_prefix_unavailable",
                    "resident_prefix_unavailable",
                    "prefix_already_gpu_resident",
                }
            ),
            "wait hint publication identity or acknowledgement mismatch",
        )
        raw_request = requests_by_id[manifest_request_id]
        phase_origin_ns = publication.get("phase_origin_ns")
        publication_ns = publication.get("publication_ns")
        completed_ns = publication.get("completed_ns")
        planned_resume_ns = publication.get("planned_resume_ns")
        send_ns = raw_request.get("send_ns")
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (
                    phase_origin_ns,
                    publication_ns,
                    completed_ns,
                    planned_resume_ns,
                    send_ns,
                )
            )
            and isinstance(publication.get("hint_boundary_ns"), int)
            and phase_origin_ns
            <= publication["hint_boundary_ns"]
            <= publication_ns
            <= completed_ns
            and hint.input_publication_ns == publication_ns
            and planned_resume_ns
            == publication["hint_boundary_ns"] + expected_lead_ms * 1_000_000
            and raw_request.get("phase_origin_ns") == phase_origin_ns
            and raw_request.get("planned_release_ns") == planned_resume_ns
            and send_ns >= planned_resume_ns
            and publication.get("acknowledgement_late_by_ns")
            == max(0, completed_ns - planned_resume_ns),
            "wait hint publication/resume clocks are noncausal",
        )
    runtime_events = result.get("wait_hint_events")
    _require(isinstance(runtime_events, list), "wait hint runtime events are missing")
    _require(
        result.get("wait_hint_action_counts")
        == _wait_hint_action_counts(publications, runtime_events),
        "wait hint action accounting mismatch",
    )
    expected_event_ids = {
        _wait_hint_event_id(binding, planned_cell) for binding in resume_bindings
    }
    expected_event_ids.update(
        publication.get("event_id")
        for publication in hint_only_publications
        if isinstance(publication.get("event_id"), str)
    )
    runtime_event_ids = {
        event.get("event_id")
        for event in runtime_events
        if isinstance(event, Mapping) and event.get("event_id")
    }
    _require(
        runtime_event_ids.issubset(expected_event_ids),
        "wait hint runtime evidence contains foreign events",
    )
    expected_identity_by_event = {
        _wait_hint_event_id(binding, planned_cell): {
            "run_id": planned_cell["policy_run_id"],
            "cell_id": planned_cell["policy_cell_id"],
            "session_id": _opaque_token(
                str(planned_cell["opaque_identity_namespace"]),
                "s",
                [binding["session_id"], binding["generation"]],
            ),
            "generation": binding["generation"],
        }
        for binding in resume_bindings
    }
    for publication in hint_only_publications:
        payload = publication.get("payload")
        _require(
            isinstance(payload, Mapping)
            and publication.get("future_suffix_published") is False
            and publication.get("http_status") == 200
            and publication.get("error") is None
            and payload.get("event_id") == publication.get("event_id")
            and payload.get("owner_id")
            == _expected_external_request_id(
                planned_cell, str(publication.get("manifest_request_id"))
            ),
            "hint-only publication identity mismatch",
        )
        expected_identity_by_event[str(publication["event_id"])] = {
            "run_id": payload["run_id"],
            "cell_id": payload["cell_id"],
            "session_id": payload["session_id"],
            "generation": payload["generation"],
        }
    for event in runtime_events:
        if not isinstance(event, Mapping) or not event.get("event_id"):
            continue
        expected_identity = expected_identity_by_event.get(event["event_id"])
        _require(
            expected_identity is not None
            and all(
                event.get(field) == value for field, value in expected_identity.items()
            ),
            "wait hint runtime run/cell/generation join mismatch",
        )
    terminal_event_ids = {
        event.get("event_id")
        for event in runtime_events
        if isinstance(event, Mapping)
        and (
            event.get("event") == "staging_consumed"
            or event.get("event") == "staging_retired"
            or (
                event.get("event") == "hint_result"
                and event.get("status") in {"expired", "no_action", "suppressed"}
            )
        )
    }
    _require(
        terminal_event_ids == expected_event_ids,
        "wait hint runtime evidence lacks terminal accounting",
    )


def _validate_raw_cell_evidence(
    result: Mapping[str, object],
    plan: Mapping[str, object],
    planned_cell: Mapping[str, object],
    *,
    source: Mapping[str, object],
) -> tuple[
    list[Mapping[str, object]], Mapping[str, object], Mapping[str, object], float
]:
    proof = planned_cell.get("request_proof")
    _require(isinstance(proof, Mapping), "planned cell request proof is missing")
    bindings = proof.get("bindings")
    ordered_ids = proof.get("ordered_manifest_request_ids")
    _require(
        isinstance(bindings, list)
        and bindings
        and proof.get("count") == len(bindings)
        and proof.get("bindings_digest") == digest(bindings)
        and isinstance(ordered_ids, list)
        and ordered_ids == [binding.get("request_id") for binding in bindings],
        "planned cell request proof is inconsistent",
    )
    _require(
        result.get("request_manifest_sha256") == proof.get("request_manifest_sha256"),
        "measured result request manifest mismatch",
    )
    raw_requests = result.get("requests")
    _require(
        isinstance(raw_requests, list)
        and len(raw_requests) == len(bindings)
        and [request.get("manifest_request_id") for request in raw_requests]
        == ordered_ids,
        "measured result request set/order differs from plan",
    )
    for request, expected in zip(raw_requests, bindings):
        _require(
            request.get("manifest_request_id") == expected["request_id"]
            and request.get("request_index") == expected["request_index"]
            and request.get("phase_index") == expected["phase_index"]
            and request.get("external_request_id") == expected["external_request_id"],
            "measured request identity differs from plan",
        )
        clocks = [
            request.get("eligibility_ns"),
            request.get("send_ns"),
            request.get("request_start_ns"),
            request.get("request_end_ns"),
            request.get("completion_ns"),
        ]
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in clocks
            )
            and clocks == sorted(clocks),
            "measured request clocks are missing or noncausal",
        )
        if expected.get("episode") == "causal_tool_resume":
            phase_origin_ns = request.get("phase_origin_ns")
            planned_release_ns = request.get("planned_release_ns")
            release_lateness_ns = request.get("release_lateness_ns")
            _require(
                isinstance(phase_origin_ns, int)
                and not isinstance(phase_origin_ns, bool)
                and isinstance(planned_release_ns, int)
                and not isinstance(planned_release_ns, bool)
                and isinstance(release_lateness_ns, int)
                and not isinstance(release_lateness_ns, bool)
                and planned_release_ns
                == phase_origin_ns + int(expected["relative_release_ms"]) * 1_000_000
                and request.get("send_ns") >= planned_release_ns
                and release_lateness_ns == request.get("send_ns") - planned_release_ns,
                "causal request release differs from frozen schedule",
            )
        output_token_ids = request.get("output_token_ids")
        token_receipts = request.get("token_receipt_ns")
        chunk_receipts = request.get("chunk_receipt_ns")
        _require(
            isinstance(output_token_ids, list)
            and all(
                isinstance(token, int) and not isinstance(token, bool) and token >= 0
                for token in output_token_ids
            )
            and request.get("output_token_digest") == digest(output_token_ids)
            and isinstance(token_receipts, list)
            and len(token_receipts) == len(output_token_ids)
            and all(
                isinstance(clock, int) and clocks[2] <= clock <= clocks[3]
                for clock in token_receipts
            )
            and token_receipts == sorted(token_receipts)
            and isinstance(chunk_receipts, list)
            and chunk_receipts
            and all(
                isinstance(clock, int) and clocks[2] <= clock <= clocks[3]
                for clock in chunk_receipts
            )
            and chunk_receipts == sorted(chunk_receipts),
            "measured request exact-token evidence is invalid",
        )
        first_token_ns = request.get("first_token_ns")
        if output_token_ids:
            _require(
                first_token_ns == token_receipts[0]
                and _nonnegative_number(request.get("ttft_ms"))
                and math.isclose(
                    float(request["ttft_ms"]),
                    (first_token_ns - clocks[2]) / 1_000_000.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "measured request TTFT does not recompute from clocks",
            )
        else:
            _require(
                first_token_ns is None and request.get("ttft_ms") is None,
                "zero-token request has fabricated first-token evidence",
            )
        _require(
            request.get("http_status") == 200
            and request.get("error") is None
            and request.get("done_seen") is True
            and request.get("terminal_complete") is True
            and request.get("success") is True
            and isinstance(request.get("finish_reasons"), list)
            and len(request["finish_reasons"]) == 1
            and request["finish_reasons"][0] in {"stop", "length"}
            and _nonnegative_number(request.get("total_ms"))
            and math.isclose(
                float(request["total_ms"]),
                (clocks[3] - clocks[2]) / 1_000_000.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "measured request terminal proof is incomplete",
        )

    if bindings and all(
        binding.get("episode") == "causal_tool_resume" for binding in bindings
    ):
        activity_phase_indexes = {
            int(binding["phase_index"])
            for binding in bindings
            if binding.get("phase") == "resume"
        }
        _require(
            len(activity_phase_indexes) == 1,
            "causal resume requests do not share one activity phase",
        )
        activity_phase_index = next(iter(activity_phase_indexes))
        activity_requests = [
            request
            for request in raw_requests
            if request.get("phase_index") == activity_phase_index
        ]
        final_warm = [
            request
            for request in raw_requests
            if request.get("phase_index") == activity_phase_index - 1
        ]
        _require(
            len(activity_requests) == 16
            and len(final_warm) == 1
            and all(
                request.get("phase_origin_ns") == final_warm[0].get("completion_ns")
                for request in activity_requests
            ),
            "causal activity origin differs from final warm completion",
        )

    _validate_wait_hint_publications(result, planned_cell, bindings, raw_requests)

    drain = result.get("drain")
    _require(isinstance(drain, Mapping), "measured result drain proof is missing")
    drain_started_ns = drain.get("started_ns")
    drain_completed_ns = drain.get("completed_ns")
    _require(
        drain.get("status") == "complete"
        and isinstance(drain_started_ns, int)
        and isinstance(drain_completed_ns, int)
        and drain_started_ns
        >= max(request["completion_ns"] for request in raw_requests)
        and drain_completed_ns >= drain_started_ns
        and drain.get("duration_ms")
        == (drain_completed_ns - drain_started_ns) / 1_000_000.0
        and drain.get("preservation_event_count") == len(bindings),
        "measured result drain did not complete from terminal requests",
    )
    gauges = drain.get("gauges")
    _require(
        isinstance(gauges, Mapping)
        and gauges.get("vllm:num_requests_running") == 0.0
        and gauges.get("vllm:num_requests_waiting") == 0.0,
        "measured result drain lacks idle scheduler gauges",
    )
    if planned_cell.get("scoring_mode") == "asynchronous":
        async_status = drain.get("async_scoring_status")
        _require(
            gauges.get("vllm:semantic_async_scoring_pending") == 0.0
            or (
                isinstance(async_status, Mapping)
                and async_status.get("pending_items") == 0
                and async_status.get("pending_snapshot_bytes") == 0
            ),
            "measured result drain lacks completed async scoring proof",
        )
        async_status = drain.get("async_scoring_status")
        _require(
            isinstance(async_status, Mapping)
            and async_status.get("max_snapshot_age_ns")
            == planned_cell.get("async_scoring_max_snapshot_age_ns"),
            "measured result drain async age bound differs from plan",
        )
        _validate_async_scoring_evidence(
            result.get("async_scoring_events"), planned_cell
        )
    episode_start_ns = min(request["eligibility_ns"] for request in raw_requests)
    episode_makespan_ms = (drain_completed_ns - episode_start_ns) / 1_000_000.0
    _require(
        result.get("episode_start_ns") == episode_start_ns
        and result.get("episode_end_ns") == drain_completed_ns
        and result.get("episode_makespan_ms") == episode_makespan_ms,
        "measured result episode makespan does not recompute from clocks",
    )

    raw_events = result.get("preservation_events")
    _require(
        isinstance(raw_events, list)
        and len(raw_events) == len(bindings)
        and result.get("preservation_event_count") == len(raw_events),
        "measured result raw preservation events are missing",
    )
    validation_profiles = plan.get("validation_profiles")
    profile = (
        validation_profiles.get(planned_cell["profile_id"])
        if isinstance(validation_profiles, Mapping)
        else None
    )
    _require(isinstance(profile, Mapping), "plan validation profile is missing")
    binding_errors = validate_campaign_event_bindings(
        raw_events,
        bindings,
        cell=planned_cell,
        profile=profile,
        implementation_revision=str(source["implementation_revision"]),
        require_passive_key_evidence=(
            source.get("passive_key_evidence_schema")
            == "semantic-passive-key-evidence-v1"
        ),
    )
    _require(
        binding_errors == [] and result.get("campaign_binding_errors") == [],
        "measured result preservation event bindings are invalid",
    )
    from benchmarks.preservation import (
        map_preservation_events,
        summarize_gpu_model_runner_work,
        summarize_prefill_cuda_work,
    )

    recomputed_preservation = map_preservation_events(
        raw_events,
        expected_request_ids=[binding["external_request_id"] for binding in bindings],
        run_id=str(planned_cell["policy_run_id"]),
        cell_id=str(planned_cell["policy_cell_id"]),
    )
    _require(
        result.get("preservation_mapping") == recomputed_preservation
        and recomputed_preservation.get("preservation_reason")
        in {"authoritative_request_events", "mixed_request_outcomes"},
        "measured result preservation mapping does not recompute from raw events",
    )
    if "gpu_prefill_work_accounting" in result:
        _require(
            result.get("gpu_prefill_work_accounting")
            == summarize_prefill_cuda_work(raw_events),
            "measured result GPU prefill work does not recompute from raw events",
        )
    if "gpu_model_runner_work_accounting" in result:
        _require(
            result.get("gpu_model_runner_work_accounting")
            == summarize_gpu_model_runner_work(raw_events),
            "measured result GPU model-runner work does not recompute",
        )

    metrics_before = result.get("metrics_before")
    metrics_after = result.get("metrics_after")
    from harness import metrics as metrics_mod

    endpoint_evidence = result.get("metrics_endpoint_evidence")
    _require(
        isinstance(endpoint_evidence, Mapping)
        and set(endpoint_evidence) == {"before", "after"},
        "measured result metric endpoint evidence is missing",
    )
    for phase, expected_values in (
        ("before", metrics_before),
        ("after", metrics_after),
    ):
        phase_evidence = endpoint_evidence[phase]
        _require(
            isinstance(phase_evidence, Mapping)
            and isinstance(phase_evidence.get("selected_exposition_lines"), list)
            and all(
                isinstance(line, str)
                for line in phase_evidence["selected_exposition_lines"]
            ),
            f"measured result {phase} metric exposition evidence is malformed",
        )
        reparsed = metrics_mod.parse_metrics_evidence(
            "\n".join(phase_evidence["selected_exposition_lines"])
        )
        _require(
            reparsed == phase_evidence
            and phase_evidence.get("values") == expected_values,
            f"measured result {phase} metrics do not reparse from exposition",
        )
    _require(
        isinstance(metrics_before, Mapping)
        and isinstance(metrics_after, Mapping)
        and set(metrics_before) == set(metrics_mod.TRACKED_METRICS)
        and set(metrics_after) == set(metrics_mod.TRACKED_METRICS)
        and all(
            value is None or _nonnegative_number(value)
            for value in (*metrics_before.values(), *metrics_after.values())
        ),
        "measured result Prometheus snapshots are missing or invalid",
    )
    recomputed_deltas, recomputed_delta_provenance = metrics_mod.diff_with_provenance(
        dict(metrics_before), dict(metrics_after)
    )
    _require(
        result.get("metrics_delta") == recomputed_deltas,
        "measured result metric deltas do not recompute from snapshots",
    )
    _require(
        result.get("metrics_delta_provenance") == recomputed_delta_provenance,
        "measured result metric delta provenance does not recompute",
    )
    for metric in (
        "vllm:kv_offload_load_bytes_total",
        "vllm:kv_offload_store_bytes_total",
        "vllm:num_preemptions_total",
    ):
        value = recomputed_deltas.get(metric)
        _require(
            _nonnegative_number(value) and float(value).is_integer(),
            f"measured result {metric} is not a nonnegative integral delta",
        )
    for field in ("preservation_recomputed_tokens", "preservation_reused_tokens"):
        value = recomputed_preservation.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"measured result {field} is not a nonnegative token count",
        )
    recomputed_physical_transfer = _physical_transfer_accounting(
        raw_events,
        recomputed_deltas,
        wait_hint_events=result.get("wait_hint_events", []),
        wait_hint_publications=result.get("wait_hint_publications", []),
    )
    has_explicit_speculative_accounting = any(
        "speculative_transfer_accounting" in event for event in raw_events
    )
    _require(
        (
            not has_explicit_speculative_accounting
            and result.get("physical_transfer_accounting") is None
        )
        or result.get("physical_transfer_accounting") == recomputed_physical_transfer,
        "measured result physical transfer accounting does not recompute",
    )
    return (
        raw_requests,
        recomputed_preservation,
        recomputed_deltas,
        episode_makespan_ms,
        (recomputed_physical_transfer if has_explicit_speculative_accounting else None),
    )


def _validate_measured_cell(
    result: Mapping[str, object],
    plan: Mapping[str, object],
    *,
    source: Mapping[str, object],
    metric_contracts: Mapping[str, Mapping[str, object]],
    execution_manifest_sha256: str,
) -> Mapping[str, object]:
    matching_cells = [
        cell
        for cell in plan.get("cells", [])
        if cell.get("cell_id") == result.get("cell_id")
    ]
    _require(len(matching_cells) == 1, "measured result cell is not in plan")
    planned_cell = matching_cells[0]
    _require(
        planned_cell.get("execution_state") == "ready"
        and result.get("schema") == "ep-campaign-cell-result-v1"
        and result.get("valid") is True
        and result.get("campaign_manifest_sha256") == execution_manifest_sha256
        and result.get("plan_digest") == plan.get("plan_digest"),
        "measured result execution identity mismatch",
    )
    for field in (
        "cell_digest",
        "profile_id",
        "baseline_profile_hash",
        "replicate_id",
        "seed",
        "order_schedule_digest",
        "arm_id",
        "arm_config_hash",
    ):
        _require(
            result.get(field) == planned_cell.get(field),
            f"measured result {field} differs from plan",
        )
    _require(
        result.get("metric_contract_digest") == digest(metric_contracts),
        "measured result metric contract mismatch",
    )
    deployed_extension = result.get("deployed_extension", {})
    deployed_runtime = result.get("deployed_runtime", {})
    _require(
        deployed_extension.get("revision") == source["implementation_revision"]
        and deployed_extension.get("tree") == source["implementation_tree"]
        and deployed_extension.get("clean") is True
        and deployed_runtime.get("revision") == source["runtime_revision"]
        and (
            source.get("runtime_tree") is None
            or deployed_runtime.get("tree") == source["runtime_tree"]
        )
        and deployed_runtime.get("clean") is True,
        "measured result deployed source/runtime mismatch",
    )
    _require(
        result.get("effective_environment") == planned_cell.get("effective_environment")
        and result.get("effective_environment", {}).get(
            "SEMANTIC_OFFLOAD_DISABLE_PREFETCH"
        )
        == "0",
        "measured result effective environment mismatch",
    )
    expected_scheduler = planned_cell.get("scheduler_resolution")
    expected_runtime = planned_cell.get("runtime_resolution")
    actual_scheduler = result.get("resolved_scheduler_runtime")
    _require(
        isinstance(expected_scheduler, Mapping)
        and result.get("resolved_scheduler") == expected_scheduler
        and isinstance(expected_runtime, Mapping)
        and actual_scheduler == expected_runtime,
        "measured result scheduler runtime differs from plan",
    )
    normalized_metrics = result.get("normalized_metrics")
    _require(
        isinstance(normalized_metrics, Mapping)
        and set(normalized_metrics) == REQUIRED_METRICS,
        "measured result normalized metric vector mismatch",
    )
    for metric, contract in metric_contracts.items():
        entry = normalized_metrics[metric]
        value = entry.get("value") if isinstance(entry, Mapping) else None
        _require(
            isinstance(entry, Mapping)
            and entry.get("unit") == contract["unit"]
            and entry.get("available") is True
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"measured result metric {metric} is unavailable or unbound",
        )
    (
        raw_requests,
        raw_preservation,
        raw_metric_deltas,
        episode_makespan_ms,
        raw_physical_transfer_accounting,
    ) = _validate_raw_cell_evidence(
        result,
        plan,
        planned_cell,
        source=source,
    )
    recomputed_metrics = _normalized_metric_vector(
        metric_contracts=metric_contracts,
        requests=raw_requests,
        preservation=raw_preservation,
        metric_deltas=raw_metric_deltas,
        episode_makespan_ms=episode_makespan_ms,
        physical_transfer_accounting=raw_physical_transfer_accounting,
    )
    _require(
        normalized_metrics == recomputed_metrics,
        "measured result normalized metrics do not recompute from raw evidence",
    )
    expected_valid = (
        all(request.get("success") is True for request in raw_requests)
        and result.get("drain", {}).get("status") == "complete"
        and raw_preservation.get("preservation_reason")
        in {"authoritative_request_events", "mixed_request_outcomes"}
        and result.get("campaign_binding_errors") == []
        and all(entry["available"] for entry in recomputed_metrics.values())
    )
    _require(
        expected_valid and result.get("valid") is True,
        "measured result validity does not recompute from drained evidence",
    )
    return planned_cell


def validate_campaign_manifest(
    manifest: Mapping[str, object], manifest_path: Path
) -> tuple[dict[str, Mapping[str, object]], dict[str, Mapping[str, object]], Path]:
    _require(manifest.get("schema") == SCHEMA, "campaign manifest schema")
    _require(
        isinstance(manifest.get("port_base"), int)
        and 1024 <= manifest["port_base"] <= 60000,
        "port_base must be an unprivileged port",
    )
    identity_namespace = manifest.get("opaque_identity_namespace")
    _require(
        isinstance(identity_namespace, str)
        and re.fullmatch(r"[0-9a-f]{64}", identity_namespace) is not None,
        "opaque_identity_namespace must be a frozen 256-bit hex value",
    )
    source = manifest.get("source")
    _require(isinstance(source, Mapping), "source is required")
    _require(source.get("extension_revision") == SOURCE_BASELINE, "source baseline")
    runtime_variant_id = source.get("runtime_variant")
    if runtime_variant_id is None:
        _require(
            source.get("runtime_revision") == RUNTIME_REVISION,
            "legacy runtime revision must be original149",
        )
        runtime_variant = NATIVE_RUNTIME_VARIANTS["original149"]
    else:
        _require(
            runtime_variant_id in NATIVE_RUNTIME_VARIANTS,
            "runtime variant is not in the frozen registry",
        )
        runtime_variant = NATIVE_RUNTIME_VARIANTS[str(runtime_variant_id)]
        _require(
            source.get("runtime_variants_binding_sha256")
            == NATIVE_RUNTIME_VARIANTS_BINDING_SHA256
            and source.get("runtime_revision") == runtime_variant["revision"]
            and source.get("runtime_tree") == runtime_variant["tree"]
            and source.get("runtime_source_path") == runtime_variant["path"]
            and source.get("runtime_wait_hint_utility")
            is runtime_variant["wait_hint_utility"],
            "runtime variant binding differs from frozen registry",
        )
        for optional_binding in (
            "patch_sha256",
            "native_binary_inventory_sha256",
        ):
            expected = runtime_variant.get(optional_binding)
            _require(
                source.get(f"runtime_{optional_binding}") == expected,
                f"runtime variant {optional_binding} binding mismatch",
            )
    for field in ("implementation_revision", "implementation_tree"):
        _require(
            isinstance(source.get(field), str)
            and re.fullmatch(r"[0-9a-f]{40}", source[field]) is not None,
            f"source {field}",
        )
    _require(
        source.get("passive_key_evidence_schema")
        in {None, "semantic-passive-key-evidence-v1"},
        "source passive key evidence schema",
    )
    _require(source.get("runtime_source_path"), "runtime_source_path is required")
    base = manifest_path.resolve().parent
    _load_bound_file(manifest.get("corpus_selection", {}), base, "corpus_selection")
    request_path = _load_bound_file(
        manifest.get("request_manifest", {}), base, "request_manifest"
    )
    with request_path.open(encoding="utf-8") as source_file:
        request_document = json.load(source_file)
    validate_request_manifest(request_document)
    protocol_bindings = manifest.get("protocol_bindings")
    _require(isinstance(protocol_bindings, Mapping), "protocol_bindings are required")
    comparisons_path = _load_bound_file(
        protocol_bindings.get("required_comparisons", {}),
        base,
        "protocol_bindings.required_comparisons",
    )
    with comparisons_path.open(encoding="utf-8") as source_file:
        comparisons = json.load(source_file)
    _require(
        comparisons.get("schema") == "ep-required-comparisons-v3",
        "required comparisons must use the final v3 schema",
    )
    episode_path = _load_bound_file(
        protocol_bindings.get("episode_expansion", {}),
        base,
        "protocol_bindings.episode_expansion",
    )
    _load_bound_file(
        protocol_bindings.get("combination_semantics", {}),
        base,
        "protocol_bindings.combination_semantics",
    )
    original_control_path = _load_bound_file(
        protocol_bindings.get("original_control_semantics", {}),
        base,
        "protocol_bindings.original_control_semantics",
    )
    with original_control_path.open(encoding="utf-8") as source_file:
        original_control_semantics = json.load(source_file)
    _require(
        original_control_semantics.get("schema") == "ep-original-control-semantics-v1"
        and original_control_semantics.get("source_revision") == SOURCE_BASELINE
        and original_control_semantics.get("original_semantic_no_prefetch")
        == FROZEN_ORIGINAL_SEMANTIC_CONFIG
        and original_control_semantics.get(
            "original_semantic_dynamic_prefetch", {}
        ).get("only_behavior_delta")
        == {"extra_config.prefetch_budget_fraction": 0.05},
        "original control semantics differ from the accepted source contract",
    )
    _require(
        {
            feature: comparisons.get("items", {}).get(feature, {}).get("settings")
            for feature in FEATURES
        }
        == REQUIRED_SETTINGS,
        "required comparisons settings differ from harness settings",
    )
    comparison_episode = comparisons.get("episode_manifest", {}).get(
        "expansion_spec", {}
    )
    _require(
        comparison_episode.get("sha256") == file_digest(episode_path),
        "required comparisons episode binding mismatch",
    )

    noise = manifest.get("noise_protocol")
    _require(isinstance(noise, Mapping), "noise_protocol is required")
    _require(tuple(noise.get("seeds", ())) == SEEDS, "five frozen seeds required")
    _require(noise.get("aa_pairs") == 5, "five independent A/A pairs required")
    _require(noise.get("treatment_repetitions") == 5, "five treatment runs required")
    _require(noise.get("bootstrap_draws") == BOOTSTRAP_DRAWS, "bootstrap draws")
    _require(noise.get("bootstrap_seed") == BOOTSTRAP_SEED, "bootstrap seed")
    _require(
        noise.get("outer_resampling_unit") == "independent_run_pair"
        and noise.get("nested_resampling_unit") == "session",
        "bootstrap must resample run pairs outermost and sessions within pair",
    )
    metric_contracts = noise.get("metric_contracts")
    _require(isinstance(metric_contracts, Mapping), "metric_contracts are required")
    _require(
        set(metric_contracts) == REQUIRED_METRICS,
        "metric_contracts must exactly enumerate required serving metrics",
    )
    for metric, contract in metric_contracts.items():
        _require(isinstance(contract, Mapping), f"metric contract {metric}")
        _require(contract.get("unit"), f"metric contract {metric} unit")
        _require(
            contract.get("effect_formula") in {"treatment_minus_control", "ratio"},
            f"metric contract {metric} effect formula",
        )
        _require(
            contract.get("beneficial_direction") in {"higher", "lower"},
            f"metric contract {metric} beneficial direction",
        )

    _require(
        manifest.get("effective_environment") == REQUIRED_EFFECTIVE_ENVIRONMENT,
        "effective environment must disable no campaign behavior",
    )

    arms_raw = manifest.get("arms")
    _require(isinstance(arms_raw, list) and arms_raw, "arms must be non-empty")
    arms: dict[str, Mapping[str, object]] = {}
    for arm in arms_raw:
        _require(isinstance(arm, Mapping), "arm must be an object")
        arm_id = arm.get("id")
        _require(
            isinstance(arm_id, str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", arm_id) is not None
            and arm_id not in arms,
            "unique filesystem-safe arm id",
        )
        _require(arm.get("split") in {"screen", "heldout"}, f"{arm_id}: split")
        features = arm.get("features")
        _require(
            isinstance(features, list) and set(features).issubset(FEATURES),
            f"{arm_id}: invalid features",
        )
        config = arm.get("config")
        _require(isinstance(config, Mapping), f"{arm_id}: config")
        arm_extra = config.get("extra_config", {})
        _require(isinstance(arm_extra, Mapping), f"{arm_id}: extra_config")
        policy_name = str(config.get("policy", "")).lower()
        shared_scorer_required = (
            policy_name in {"arc", "lru"}
            and float(arm_extra.get("prefetch_budget_fraction", 0.0)) > 0.0
            and arm_extra.get("prefetch_selector") in {"semantic", "useful_on_time"}
        )
        _require(
            not shared_scorer_required
            or arm_extra.get("shared_prefetch_scorer") is True,
            f"{arm_id}: ARC/LRU score-consuming prefetch requires shared scorer",
        )
        if arm_extra.get("scoring_mode", "synchronous") == "asynchronous":
            max_age_ms = arm_extra.get("async_scoring_max_snapshot_age_ms", 1000)
            _require(
                isinstance(max_age_ms, int)
                and not isinstance(max_age_ms, bool)
                and max_age_ms > 0,
                f"{arm_id}: async scoring requires positive snapshot age",
            )
        serialized = json.dumps(config, sort_keys=True).lower()
        _require(
            "future_reuse" not in serialized and "oracle" not in serialized,
            f"{arm_id}: future/oracle serving inputs are forbidden",
        )
        _require(arm.get("arm_config_hash") == digest(config), f"{arm_id}: config hash")
        _require(
            arm.get("execution_state") in {"ready", "pending"},
            f"{arm_id}: execution_state",
        )
        if (
            arm.get("execution_state") == "ready"
            and arm_extra.get("prefetch_trigger") == "known_prefix_wait_hint"
        ):
            _require(
                runtime_variant_id == "wait_hint_utility_v1"
                and runtime_variant.get("wait_hint_utility") is True,
                f"{arm_id}: ready wait-hint arm requires pinned utility child",
            )
        if arm.get("execution_state") == "ready":
            argv = arm.get("runner_argv")
            _require(
                isinstance(argv, list)
                and argv
                and all(isinstance(arg, str) for arg in argv),
                f"{arm_id}: ready arm needs runner_argv",
            )
            _validate_runner_argv_template(argv)
        arms[arm_id] = arm

    for arm_id, arm in arms.items():
        comparator_id = arm.get("comparator_arm_id")
        delta = arm.get("config_delta")
        _require(isinstance(delta, list), f"{arm_id}: config_delta")
        if comparator_id is None:
            _require(not delta, f"{arm_id}: baseline delta must be empty")
            continue
        _require(comparator_id in arms, f"{arm_id}: unknown comparator")
        _require(
            _apply_delta(arms[comparator_id]["config"], delta) == arm["config"],
            f"{arm_id}: undeclared config difference from comparator",
        )
        if set(arm["features"]) & {"E01", "E02", "E09"}:
            _require(
                arm["split"] == "screen" or manifest.get("heldout_selection_lock"),
                f"{arm_id}: E01/E02/E09 cannot be selected on heldout",
            )

    neutral_settings = manifest.get("feature_neutral_settings")
    _require(
        neutral_settings == REQUIRED_NEUTRAL_SETTINGS,
        "feature_neutral_settings must match the frozen neutral contract",
    )
    control_references = manifest.get("control_references")
    required_control_names = {
        obligation.removeprefix("control:")
        for obligation in COMBINATION_OBLIGATIONS
        if obligation.startswith("control:")
    }
    _require(
        isinstance(control_references, Mapping)
        and set(control_references) == required_control_names,
        "control_references must enumerate every named control",
    )
    for control_name, reference in control_references.items():
        _require(
            isinstance(reference, Mapping)
            and isinstance(reference.get("config"), Mapping)
            and reference.get("config_hash") == digest(reference["config"]),
            f"control reference {control_name} hash",
        )
    original_comparator = control_references["original_semantic_no_prefetch"]["config"]
    _require(
        _behavior_config(original_comparator) == FROZEN_ORIGINAL_SEMANTIC_CONFIG,
        "named original comparator differs from accepted original behavior",
    )
    original_controls = _original_control_configs(
        original_comparator,
        str(original_comparator.get("scheduler_backend")),
    )
    for control_name in (
        "original_semantic_no_prefetch",
        "original_semantic_dynamic_prefetch",
    ):
        _require(
            control_references[control_name]["config"]
            == original_controls[control_name],
            f"control reference {control_name} differs from frozen original",
        )
    semantic_anchors = [
        original_comparator,
        *[
            arm["config"]
            for arm in arms.values()
            if str(arm["config"].get("policy", "")).lower().startswith("semantic")
        ],
    ]
    for control_name in (
        "ARC_same_admission_prefetch_staging",
        "LRU_same_admission_prefetch_staging",
        "best_supported_native_scheduler",
    ):
        _require(
            any(
                _control_config_matches_anchor(
                    control_name,
                    control_references[control_name]["config"],
                    anchor=anchor,
                )
                for anchor in semantic_anchors
            ),
            f"control reference {control_name} is not an exact machinery control",
        )

    measured_result_records: list[tuple[Path, Mapping[str, object]]] = []
    measured_result_paths: set[Path] = set()
    coverage = manifest.get("coverage")
    _require(isinstance(coverage, Mapping), "coverage is required")
    _require(set(coverage) == set(FEATURES), "coverage must enumerate E01--E09")
    for feature in FEATURES:
        entry = coverage[feature]
        _require(isinstance(entry, Mapping), f"coverage {feature}")
        _require(
            entry.get("status") in {"ready", "pending", "measured"},
            f"coverage {feature} status",
        )
        _require(
            entry.get("settings") == REQUIRED_SETTINGS[feature],
            f"coverage {feature} settings",
        )
        bound_arms = entry.get("arm_ids")
        _require(isinstance(bound_arms, list), f"coverage {feature} arm_ids")
        _require(all(arm in arms for arm in bound_arms), f"coverage {feature} arm")
        if entry["status"] != "pending":
            _require(bound_arms, f"coverage {feature} cannot be empty")
            _require(
                all(arms[arm]["execution_state"] == "ready" for arm in bound_arms),
                f"coverage {feature} binds a pending arm",
            )
            for setting_name, values in REQUIRED_SETTINGS[feature].items():
                path = _FEATURE_CONFIG_PATHS[feature][setting_name]
                for expected in values:
                    _require(
                        any(
                            _setting_value_matches(
                                setting_name,
                                expected,
                                _config_value(arms[arm_id]["config"], path),
                            )
                            for arm_id in bound_arms
                        ),
                        f"coverage {feature} does not realize {setting_name}={expected}",
                    )
            if entry["status"] == "measured":
                result_bindings = entry.get("result_bindings")
                _require(
                    isinstance(result_bindings, list)
                    and len(result_bindings) >= 5 * len(bound_arms),
                    f"measured coverage {feature} requires five results per arm",
                )
                observed_replicates: dict[str, set[object]] = defaultdict(set)
                for index, binding in enumerate(result_bindings):
                    result_path = _load_bound_file(
                        binding, base, f"coverage {feature} result {index}"
                    )
                    with result_path.open(encoding="utf-8") as source_file:
                        measured_result = json.load(source_file)
                    _require(
                        result_path not in measured_result_paths,
                        "measured result path is reused",
                    )
                    measured_result_paths.add(result_path)
                    measured_result_records.append((result_path, measured_result))
                    arm_id = measured_result.get("arm_id")
                    _require(
                        measured_result.get("schema") == "ep-campaign-cell-result-v1"
                        and measured_result.get("valid") is True
                        and arm_id in bound_arms
                        and measured_result.get("arm_config_hash")
                        == arms[arm_id]["arm_config_hash"],
                        f"coverage {feature} result binding mismatch",
                    )
                    observed_replicates[arm_id].add(measured_result.get("replicate_id"))
                _require(
                    set(observed_replicates) == set(bound_arms)
                    and all(
                        len(replicates) == 5
                        for replicates in observed_replicates.values()
                    ),
                    f"coverage {feature} requires five distinct replicates per arm",
                )
    obligations = manifest.get("combination_obligations")
    _require(isinstance(obligations, list) and obligations, "combination obligations")
    _require(
        all(
            isinstance(item, Mapping)
            and item.get("status") in {"ready", "pending", "measured"}
            for item in obligations
        ),
        "every combination obligation needs explicit status",
    )
    _require(
        {item.get("id") for item in obligations} == COMBINATION_OBLIGATIONS,
        "combination obligations must enumerate interactions, controls, and E01--E09 leave-outs",
    )
    for obligation in obligations:
        status = obligation["status"]
        if status == "pending":
            continue
        arm_ids = obligation.get("arm_ids")
        _require(
            isinstance(arm_ids, list) and arm_ids,
            f"combination {obligation['id']} requires concrete arms",
        )
        _require(
            all(
                arm_id in arms and arms[arm_id]["execution_state"] == "ready"
                for arm_id in arm_ids
            ),
            f"combination {obligation['id']} binds unknown or pending arms",
        )
        obligation_id = str(obligation["id"])
        required_features = set(re.findall(r"E\d{2}", obligation_id))
        if obligation_id.startswith("interaction:"):
            if "full_compatible_bundle" in obligation_id:
                required_features = set(FEATURES)
            _require(
                any(
                    required_features.issubset(set(arms[arm_id]["features"]))
                    and all(
                        _arm_realizes_feature_delta(arms[arm_id], feature, arms)
                        and _major_feature_enabled(arms[arm_id]["config"], feature)
                        for feature in required_features
                    )
                    for arm_id in arm_ids
                ),
                f"combination {obligation_id} does not realize exact behavior deltas",
            )
        elif obligation_id.startswith("control:"):
            control_name = obligation_id.removeprefix("control:")
            reference = control_references[control_name]
            _require(
                any(
                    _named_control_is_exact(
                        control_name,
                        arms[arm_id],
                        arms=arms,
                        reference=reference,
                        original_controls=original_controls,
                    )
                    for arm_id in arm_ids
                ),
                f"combination {obligation_id} lacks its named control",
            )
        elif obligation_id.startswith("leave_one_out:"):
            removed = obligation_id.removeprefix("leave_one_out:")
            full_bundle_arm_id = obligation.get("full_bundle_arm_id")
            _require(
                full_bundle_arm_id in arms
                and set(arms[full_bundle_arm_id]["features"]) == set(FEATURES)
                and all(
                    _major_feature_enabled(arms[full_bundle_arm_id]["config"], feature)
                    for feature in FEATURES
                )
                and any(
                    arms[arm_id].get("leave_one_out") == removed
                    and set(arms[arm_id]["features"]) == set(FEATURES) - {removed}
                    and arms[arm_id]["config"]
                    == _neutralized_feature_config(
                        arms[full_bundle_arm_id]["config"],
                        removed,
                        neutral_settings,
                    )
                    for arm_id in arm_ids
                ),
                f"combination {obligation_id} lacks its leave-out arm",
            )
        if status == "measured":
            result_bindings = obligation.get("result_bindings")
            _require(
                isinstance(result_bindings, list) and result_bindings,
                f"measured combination {obligation['id']} requires results",
            )
            result_replicates: dict[str, set[object]] = defaultdict(set)
            for index, binding in enumerate(result_bindings):
                result_path = _load_bound_file(
                    binding, base, f"combination {obligation['id']} result {index}"
                )
                with result_path.open(encoding="utf-8") as source_file:
                    measured_result = json.load(source_file)
                _require(
                    result_path not in measured_result_paths,
                    "measured result path is reused",
                )
                measured_result_paths.add(result_path)
                measured_result_records.append((result_path, measured_result))
                result_arm = measured_result.get("arm_id")
                _require(
                    measured_result.get("schema") == "ep-campaign-cell-result-v1"
                    and measured_result.get("valid") is True
                    and result_arm in arm_ids
                    and measured_result.get("arm_config_hash")
                    == arms[result_arm]["arm_config_hash"],
                    f"combination {obligation['id']} result binding mismatch",
                )
                result_replicates[result_arm].add(measured_result.get("replicate_id"))
            _require(
                set(result_replicates) == set(arm_ids)
                and all(
                    len(replicates) == 5 for replicates in result_replicates.values()
                ),
                f"combination {obligation['id']} requires five results per arm",
            )

    heldout_arms = [arm for arm in arms.values() if arm["split"] == "heldout"]
    selection_lock = manifest.get("heldout_selection_lock")
    selected_config_hashes: set[str] = set()
    if heldout_arms:
        _require(
            isinstance(selection_lock, Mapping), "structured heldout selection lock"
        )
        _require(
            selection_lock.get("schema") == "ep-heldout-selection-lock-v1"
            and selection_lock.get("selection_split") == "screen"
            and selection_lock.get("frozen_before_heldout") is True,
            "heldout selection lock must be frozen from screen",
        )
        screen_plan_path = _load_bound_file(
            selection_lock.get("screen_plan", {}), base, "screen plan"
        )
        screen_result_path = _load_bound_file(
            selection_lock.get("screen_result", {}), base, "screen result"
        )
        screen_manifest_path = _load_bound_file(
            selection_lock.get("screen_manifest", {}), base, "screen manifest"
        )
        with screen_manifest_path.open(encoding="utf-8") as source_file:
            screen_manifest = json.load(source_file)
        screen_arms, screen_profiles, _ = validate_campaign_manifest(
            screen_manifest, screen_manifest_path
        )
        _require(
            all(arm["split"] == "screen" for arm in screen_arms.values())
            and all(
                profile["workload"]["split"] == "screen"
                for profile in screen_profiles.values()
            )
            and screen_manifest.get("source") == manifest.get("source")
            and screen_manifest.get("request_manifest")
            == manifest.get("request_manifest")
            and screen_manifest.get("protocol_bindings")
            == manifest.get("protocol_bindings")
            and screen_manifest.get("noise_protocol") == manifest.get("noise_protocol"),
            "heldout screen manifest is not a compatible screen-only campaign",
        )
        with screen_plan_path.open(encoding="utf-8") as source_file:
            screen_plan = json.load(source_file)
        declared_plan_digest = screen_plan.get("plan_digest")
        plan_without_digest = dict(screen_plan)
        plan_without_digest.pop("plan_digest", None)
        _require(
            screen_plan.get("schema") == PLAN_SCHEMA
            and declared_plan_digest == digest(plan_without_digest)
            and screen_plan == build_plan(screen_manifest, screen_manifest_path)
            and bool(screen_plan.get("cells")),
            "heldout screen plan schema/digest",
        )
        with screen_result_path.open(encoding="utf-8") as source_file:
            screen_result = json.load(source_file)
        selected = selection_lock.get("selected_configs")
        _require(
            isinstance(selected, list) and 0 < len(selected) <= 3,
            "heldout selection must contain at most three configurations",
        )
        _require(selection_lock.get("selection_rule"), "heldout selection rule")
        _require(
            isinstance(selection_lock.get("frozen_at_ns"), int)
            and selection_lock["frozen_at_ns"] >= 0,
            "heldout selection freeze time",
        )
        _require(
            screen_result.get("schema") == "ep-screen-selection-result-v1"
            and screen_result.get("valid") is True
            and screen_result.get("selection_split") == "screen"
            and screen_result.get("screen_plan_sha256") == file_digest(screen_plan_path)
            and screen_result.get("selection_rule")
            == selection_lock.get("selection_rule")
            and screen_result.get("selected_configs") == selected
            and isinstance(screen_result.get("completed_at_ns"), int)
            and screen_result["completed_at_ns"] <= selection_lock["frozen_at_ns"],
            "heldout screen selection result binding",
        )
        screen_cell_results = screen_result.get("cell_results")
        _require(
            isinstance(screen_cell_results, list) and screen_cell_results,
            "heldout selection requires screen cell results",
        )
        evidence_replicates: dict[str, set[object]] = defaultdict(set)
        for index, binding in enumerate(screen_cell_results):
            cell_result_path = _load_bound_file(
                binding, screen_result_path.parent, f"screen cell result {index}"
            )
            with cell_result_path.open(encoding="utf-8") as source_file:
                cell_result = json.load(source_file)
            _validate_measured_cell(
                cell_result,
                screen_plan,
                source=screen_manifest["source"],
                metric_contracts=screen_manifest["noise_protocol"]["metric_contracts"],
                execution_manifest_sha256=file_digest(screen_manifest_path),
            )
            evidence_replicates[cell_result["arm_id"]].add(
                cell_result.get("replicate_id")
            )
        for selected_config in selected:
            _require(
                isinstance(selected_config, Mapping)
                and isinstance(selected_config.get("arm_config_hash"), str)
                and selected_config.get("screen_arm_id") in arms
                and arms[selected_config["screen_arm_id"]]["split"] == "screen"
                and arms[selected_config["screen_arm_id"]]["arm_config_hash"]
                == selected_config["arm_config_hash"],
                "selected config binding",
            )
            selected_config_hashes.add(selected_config["arm_config_hash"])
            _require(
                len(evidence_replicates.get(selected_config["screen_arm_id"], set()))
                == 5,
                "screen-selected config lacks five valid replicate results",
            )
        for arm in heldout_arms:
            source_arm_id = arm.get("selected_from_screen_arm_id")
            _require(
                source_arm_id in arms
                and arms[source_arm_id]["split"] == "screen"
                and arms[source_arm_id]["arm_config_hash"] == arm["arm_config_hash"]
                and arm["arm_config_hash"] in selected_config_hashes,
                f"heldout arm {arm['id']} was not screen-selected",
            )

    profiles_raw = manifest.get("baseline_profiles")
    _require(isinstance(profiles_raw, list) and profiles_raw, "baseline profiles")
    profiles: dict[str, Mapping[str, object]] = {}
    scheduled_ready_arms: set[str] = set()
    scheduled_ready_counts: Counter[str] = Counter()
    planned_cell_ids: set[str] = set()
    first_heldout_cell_id: str | None = None
    for profile in profiles_raw:
        _require(isinstance(profile, Mapping), "baseline profile object")
        profile_id = profile.get("id")
        _require(
            isinstance(profile_id, str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", profile_id) is not None
            and profile_id not in profiles,
            "unique filesystem-safe profile id",
        )
        replication = profile.get("replication_plan")
        _require(isinstance(replication, list) and len(replication) == 5, "replicates")
        replicate_ids = [item.get("replicate_id") for item in replication]
        _require(
            len(replicate_ids) == len(set(replicate_ids)),
            f"{profile_id}: replicate ids must be unique",
        )
        treatment_arm_ids = profile.get("treatment_arm_ids")
        _require(
            isinstance(treatment_arm_ids, list)
            and treatment_arm_ids
            and len(treatment_arm_ids) == len(set(treatment_arm_ids)),
            f"{profile_id}: treatment_arm_ids",
        )
        _require(
            all(arm_id in arms for arm_id in treatment_arm_ids),
            f"{profile_id}: unknown treatment arm",
        )
        _require(
            tuple(item.get("seed") for item in replication) == SEEDS,
            f"{profile_id}: replicate seeds",
        )
        for item in replication:
            _require(item.get("replicate_id"), "replicate binding")
            run_schedule = item.get("run_schedule")
            _require(
                isinstance(run_schedule, list) and len(run_schedule) >= 3,
                "replicate run_schedule must contain an A/A pair and treatment",
            )
            slot_ids = [entry.get("slot_id") for entry in run_schedule]
            _require(
                all(
                    isinstance(slot_id, str)
                    and re.fullmatch(r"[A-Za-z0-9_.-]+", slot_id) is not None
                    for slot_id in slot_ids
                )
                and len(slot_ids) == len(set(slot_ids)),
                "replicate slot ids must be unique and filesystem-safe",
            )
            aa_entries = [
                entry for entry in run_schedule if entry.get("role") == "aa_baseline"
            ]
            _require(
                len(aa_entries) == 2, "each replicate requires exactly two A/A slots"
            )
            _require(
                all(
                    entry.get("role") in {"aa_baseline", "treatment"}
                    for entry in run_schedule
                ),
                "run_schedule contains an unknown role",
            )
            _require(
                all(entry.get("arm_id") in arms for entry in run_schedule),
                "run_schedule references unknown arm",
            )
            _require(
                all(
                    arms[entry["arm_id"]]["config"] == profile["comparator_config"]
                    for entry in aa_entries
                ),
                "A/A slots must use the exact comparator config",
            )
            scheduled_treatments = [
                entry["arm_id"]
                for entry in run_schedule
                if entry.get("role") == "treatment"
            ]
            _require(
                sorted(scheduled_treatments) == sorted(treatment_arm_ids),
                "each replicate must run every treatment arm exactly once",
            )
            if any(
                arms[arm_id]["execution_state"] == "ready"
                for arm_id in treatment_arm_ids
            ):
                _require(
                    all(
                        arms[entry["arm_id"]]["execution_state"] == "ready"
                        for entry in run_schedule
                    ),
                    "ready treatment requires ready A/A comparator slots",
                )
            request_order = item.get("request_order")
            _require(
                request_order == "manifest_order",
                "replicate request_order must preserve the frozen manifest graph",
            )
            expected_schedule_digest = digest(
                {
                    "seed": item["seed"],
                    "run_schedule": run_schedule,
                    "request_order": request_order,
                }
            )
            _require(
                item.get("order_schedule_digest") == expected_schedule_digest,
                "order schedule digest mismatch",
            )
        workload = profile.get("workload")
        resources = profile.get("resources")
        model_runtime = profile.get("model_runtime")
        _require(isinstance(workload, Mapping), f"{profile_id}: workload")
        _require(isinstance(resources, Mapping), f"{profile_id}: resources")
        _require(isinstance(model_runtime, Mapping), f"{profile_id}: model_runtime")
        for field in (
            "model",
            "model_revision",
            "tokenizer_revision",
            "adapter_namespace",
            "layout_namespace",
            "dtype",
            "tensor_parallel_size",
        ):
            _require(
                model_runtime.get(field) not in (None, ""), f"{profile_id}: {field}"
            )
        split = workload.get("split")
        context_tokens = workload.get("context_tokens")
        if any(
            arms[arm_id]["execution_state"] == "ready" for arm_id in treatment_arm_ids
        ):
            _scheduler_server_args(profile.get("scheduler_backend"))
            for arm_id in treatment_arm_ids:
                arm_extra = arms[arm_id]["config"].get("extra_config", {})
                _require(
                    arm_extra.get("implementation_revision")
                    == source["implementation_revision"],
                    f"{profile_id}: {arm_id} implementation revision mismatch",
                )
        if split == "heldout" or context_tokens == 4096:
            _require(
                selected_config_hashes
                and all(
                    arms[arm_id]["arm_config_hash"] in selected_config_hashes
                    for arm_id in treatment_arm_ids
                ),
                f"{profile_id}: heldout/4096 config is not screen-locked",
            )
        _require(
            workload.get("episode")
            in {
                "reusable_prefix",
                "distinct_topic_shift",
                "nonreuse_null",
                "causal_tool_resume",
            },
            f"{profile_id}: executable episode",
        )
        profile_requests = [
            request
            for request in request_document["requests"]
            if request["split"] == split and request["context_budget"] == context_tokens
        ]
        _require(profile_requests, f"{profile_id}: no bound requests")
        max_required_tokens = max(
            len(request["prompt_token_ids"]) + request["suffix_budget_tokens"]
            for request in profile_requests
        )
        max_model_len = resources.get("max_model_len")
        block_size_tokens = resources.get("kv_block_size_tokens")
        total_gpu_blocks = resources.get("total_gpu_blocks")
        _require(
            isinstance(max_model_len, int) and max_model_len >= max_required_tokens,
            f"{profile_id}: full prompt plus suffix exceeds max_model_len",
        )
        _require(
            isinstance(block_size_tokens, int) and block_size_tokens > 0,
            f"{profile_id}: kv_block_size_tokens",
        )
        required_blocks = (
            max_required_tokens + block_size_tokens - 1
        ) // block_size_tokens
        _require(
            isinstance(total_gpu_blocks, int) and total_gpu_blocks >= required_blocks,
            f"{profile_id}: full prompt plus suffix exceeds GPU KV block budget",
        )
        _require(
            max_model_len <= total_gpu_blocks * block_size_tokens,
            f"{profile_id}: max_model_len exceeds physical GPU KV token capacity",
        )
        _require(
            isinstance(resources.get("gpu_memory_utilization"), (int, float))
            and not isinstance(resources["gpu_memory_utilization"], bool)
            and 0.0 < float(resources["gpu_memory_utilization"]) <= 1.0,
            f"{profile_id}: gpu_memory_utilization",
        )
        actual_engine_profile = {
            "dtype": model_runtime["dtype"],
            "tensor_parallel_size": model_runtime["tensor_parallel_size"],
            "enforce_eager": resources["enforce_eager"],
            "max_num_seqs": resources["max_num_seqs"],
            "max_num_batched_tokens": resources["max_num_batched_tokens"],
            "kv_block_size_tokens": block_size_tokens,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": resources["gpu_memory_utilization"],
        }
        scheduled_profile_arm_ids = {
            scheduled_run["arm_id"]
            for replicate in replication
            for scheduled_run in replicate["run_schedule"]
        }
        bounded_prefill_cuda_timing = bool(
            scheduled_profile_arm_ids
            and all(
                arms[arm_id]["config"]
                .get("extra_config", {})
                .get("bounded_prefill_cuda_timing")
                is True
                for arm_id in scheduled_profile_arm_ids
            )
        )
        _require(
            _engine_profile_matches(
                actual_engine_profile,
                total_gpu_blocks,
                bounded_prefill_cuda_timing=bounded_prefill_cuda_timing,
            ),
            f"{profile_id}: engine profile differs from frozen readiness profile",
        )
        for replicate in replication:
            for scheduled_run in replicate["run_schedule"]:
                arm_id = scheduled_run["arm_id"]
                cell_id = (
                    f"{profile_id}-{replicate['replicate_id']}-"
                    f"{scheduled_run['slot_id']}"
                )
                _require(cell_id not in planned_cell_ids, "planned cell id collision")
                planned_cell_ids.add(cell_id)
                if split == "heldout" and first_heldout_cell_id is None:
                    first_heldout_cell_id = cell_id
                if arms[arm_id]["execution_state"] == "ready":
                    scheduled_ready_arms.add(arm_id)
                    scheduled_ready_counts[arm_id] += 1
                _require(
                    arms[arm_id]["split"] == split,
                    f"{profile_id}: arm split mismatch for {arm_id}",
                )
                arm_scheduler = arms[arm_id]["config"].get(
                    "scheduler_backend", profile["scheduler_backend"]
                )
                _require(
                    arm_scheduler == profile["scheduler_backend"],
                    f"{profile_id}: arm scheduler mismatch for {arm_id}",
                )
        expected_hash = digest(_profile_payload(manifest, profile))
        _require(
            profile.get("baseline_profile_hash") == expected_hash,
            f"{profile_id}: baseline_profile_hash mismatch",
        )
        profiles[profile_id] = profile
    ready_arm_ids = {
        arm_id for arm_id, arm in arms.items() if arm["execution_state"] == "ready"
    }
    _require(
        ready_arm_ids == scheduled_ready_arms,
        "every globally ready arm must belong to a compatible frozen profile",
    )
    for arm_id in ready_arm_ids:
        if arms[arm_id].get("comparator_arm_id") is not None:
            _require(
                scheduled_ready_counts[arm_id] == 5,
                f"ready treatment {arm_id} must have exactly five planned runs",
            )
    if heldout_arms:
        first_launch = selection_lock.get("first_heldout_launch")
        _require(
            isinstance(first_launch, Mapping)
            and first_launch.get("cell_id") == first_heldout_cell_id
            and isinstance(first_launch.get("not_before_ns"), int)
            and first_launch["not_before_ns"] >= selection_lock["frozen_at_ns"],
            "first heldout launch identity/freeze ordering",
        )
    canonical_plan = (
        _build_plan_from_validated(
            manifest,
            manifest_path,
            arms=arms,
            profiles=profiles,
            request_path=request_path,
        )
        if measured_result_records
        else None
    )
    for result_path, measured_result in measured_result_records:
        plan_binding = measured_result.get("plan_artifact")
        _require(isinstance(plan_binding, Mapping), "measured result plan binding")
        plan_path = _load_bound_file(
            plan_binding, result_path.parent, "measured result plan"
        )
        with plan_path.open(encoding="utf-8") as source_file:
            measured_plan = json.load(source_file)
        plan_digest = measured_plan.get("plan_digest")
        plan_without_digest = dict(measured_plan)
        plan_without_digest.pop("plan_digest", None)
        _require(
            measured_plan.get("schema") == PLAN_SCHEMA
            and plan_digest == digest(plan_without_digest)
            and measured_plan.get("source") == manifest["source"]
            and measured_plan == canonical_plan,
            "measured result plan identity mismatch",
        )
        _validate_measured_cell(
            measured_result,
            measured_plan,
            source=source,
            metric_contracts=noise["metric_contracts"],
            execution_manifest_sha256=measured_plan["manifest_sha256"],
        )
    return arms, profiles, request_path


def validate_causal_online_input(record: Mapping[str, object]) -> None:
    _require(record.get("schema") == CAUSAL_SCHEMA, "causal schema")
    for field in (
        "schema_version",
        "engine_run_id",
        "model_revision",
        "tokenizer_revision",
        "adapter_namespace",
        "layout_namespace",
        "request_id",
        "session_id",
        "generation",
        "scheduler_step",
        "observation_sequence",
        "key_chain_digest",
        "feature_provenance",
        "cost_sample_provenance",
        "candidate_config_hash",
    ):
        _require(record.get(field) not in (None, ""), f"causal input missing {field}")
    times = [record.get(field) for field in ONLINE_TIME_FIELDS]
    _require(
        all(isinstance(value, int) and value >= 0 for value in times),
        "causal timestamps must be non-negative integers",
    )
    _require(times == sorted(times), "causal publication/consumption order violated")
    forbidden = {"future_reuse", "eventual_demand", "consumed", "discarded", "output"}
    _require(
        not forbidden.intersection(record), "diagnostic future fields in online input"
    )


def build_plan(
    manifest: Mapping[str, object], manifest_path: Path
) -> dict[str, object]:
    arms, profiles, request_path = validate_campaign_manifest(manifest, manifest_path)
    return _build_plan_from_validated(
        manifest,
        manifest_path,
        arms=arms,
        profiles=profiles,
        request_path=request_path,
    )


def _profile_requests(
    request_document: Mapping[str, object], workload: Mapping[str, object]
) -> list[Mapping[str, object]]:
    selected = [
        request
        for request in request_document["requests"]
        if request["split"] == workload["split"]
        and request["context_budget"] == workload["context_tokens"]
        and request["episode"] == workload["episode"]
    ]
    return sorted(selected, key=lambda request: int(request["request_index"]))


def _expected_external_request_id(
    cell: Mapping[str, object], manifest_request_id: str
) -> str:
    sample_index = int(
        _opaque_token(
            str(cell["opaque_identity_namespace"]),
            "",
            manifest_request_id,
            12,
        ),
        16,
    )
    return (
        f"sc|{cell['policy_run_id']}|{cell['policy_cell_id']}|"
        f"{cell['policy_workload_id']}|sample-{sample_index}"
    )


def _request_validation_binding(
    request: Mapping[str, object], cell: Mapping[str, object]
) -> dict[str, object]:
    fields = (
        "request_id",
        "request_index",
        "phase_index",
        "relative_release_ms",
        "content_origin",
        "arrival_origin",
        "wait_origin",
        "pressure_origin",
        "source_id",
        "session_id",
        "family",
        "split",
        "generation",
        "prompt_token_digest",
        "key_chain_digest",
        "episode",
        "phase",
        "known_prefix_boundary_tokens",
        "known_prefix_token_digest",
        "hint_lead_ms",
        "hint_only",
    )
    binding = {field: request.get(field) for field in fields}
    binding["context_token_span"] = request.get("context_token_span")
    binding["prompt_token_count"] = len(request["prompt_token_ids"])
    binding["external_request_id"] = _expected_external_request_id(
        cell, str(request["request_id"])
    )
    return binding


def _build_plan_from_validated(
    manifest: Mapping[str, object],
    manifest_path: Path,
    *,
    arms: Mapping[str, Mapping[str, object]],
    profiles: Mapping[str, Mapping[str, object]],
    request_path: Path,
) -> dict[str, object]:
    with request_path.open(encoding="utf-8") as source_file:
        request_document = json.load(source_file)
    cells: list[dict[str, object]] = []
    output_root = Path(str(manifest.get("output_root", "campaign-results"))).resolve()
    for profile_id, profile in profiles.items():
        selected_requests = _profile_requests(request_document, profile["workload"])
        _require(selected_requests, f"{profile_id}: profile selected no requests")
        for replicate in profile["replication_plan"]:
            for scheduled_run in replicate["run_schedule"]:
                arm_id = scheduled_run["arm_id"]
                _require(arm_id in arms, f"{profile_id}: unknown arm {arm_id}")
                arm = arms[arm_id]
                slot_id = scheduled_run["slot_id"]
                cell_id = f"{profile_id}-{replicate['replicate_id']}-{slot_id}"
                port = int(manifest["port_base"]) + len(cells)
                opaque_namespace = str(manifest["opaque_identity_namespace"])
                policy_run_id = _opaque_token(
                    opaque_namespace,
                    "r",
                    [profile_id, replicate["replicate_id"]],
                )
                policy_cell_id = _opaque_token(
                    opaque_namespace,
                    "c",
                    [profile_id, replicate["replicate_id"]],
                )
                policy_workload_id = _opaque_token(
                    opaque_namespace,
                    "w",
                    profile["workload"],
                )
                argv = arm.get("runner_argv")
                expanded = None
                if argv is not None:
                    replacements = {
                        "arm_id": arm_id,
                        "seed": str(replicate["seed"]),
                        "replicate_id": str(replicate["replicate_id"]),
                        "slot_id": str(slot_id),
                        "cell_id": cell_id,
                        "request_manifest": str(request_path),
                        "manifest": str(manifest_path.resolve()),
                        "profile_id": profile_id,
                        "port": str(port),
                        "output_dir": str(output_root / cell_id),
                    }
                    expanded = [argument.format(**replacements) for argument in argv]
                cell = {
                    "cell_id": cell_id,
                    "profile_id": profile_id,
                    "baseline_profile_hash": profile["baseline_profile_hash"],
                    "replicate_id": replicate["replicate_id"],
                    "slot_id": slot_id,
                    "schedule_role": scheduled_run["role"],
                    "seed": replicate["seed"],
                    "request_order": replicate["request_order"],
                    "port": port,
                    "policy_run_id": policy_run_id,
                    "policy_cell_id": policy_cell_id,
                    "policy_workload_id": policy_workload_id,
                    "opaque_identity_namespace": opaque_namespace,
                    "order_schedule_digest": replicate["order_schedule_digest"],
                    "arm_id": arm_id,
                    "arm_config_hash": arm["arm_config_hash"],
                    "config_delta": arm["config_delta"],
                    "execution_state": arm["execution_state"],
                    "scoring_mode": arm["config"]
                    .get("extra_config", {})
                    .get("scoring_mode", "synchronous"),
                    "prefetch_trigger": arm["config"]
                    .get("extra_config", {})
                    .get("prefetch_trigger", "preemption"),
                    "tool_gap_ms": arm["config"]
                    .get("request_controls", {})
                    .get("tool_gap_ms", 0),
                    "wait_hint_retention_ms": arm["config"]
                    .get("extra_config", {})
                    .get("wait_hint_ttl_ms", 2000),
                    "async_scoring_max_snapshot_age_ns": int(
                        arm["config"]
                        .get("extra_config", {})
                        .get("async_scoring_max_snapshot_age_ms", 1000)
                    )
                    * 1_000_000,
                    "scheduler_resolution": _scheduler_resolution(
                        profile["scheduler_backend"]
                    ),
                    "runtime_resolution": _expected_runtime_resolution(
                        profile, profile["scheduler_backend"]
                    ),
                    "effective_environment": {
                        **manifest["effective_environment"],
                        "VLLM_PLUGINS": (
                            "semantic_wait_hint"
                            if arm["config"]
                            .get("extra_config", {})
                            .get("prefetch_trigger")
                            == "known_prefix_wait_hint"
                            else ""
                        ),
                    },
                    "argv": expanded,
                }
                request_bindings = [
                    _request_validation_binding(request, cell)
                    for request in selected_requests
                ]
                cell["request_proof"] = {
                    "request_manifest_sha256": file_digest(request_path),
                    "count": len(selected_requests),
                    "ordered_manifest_request_ids": [
                        request["request_id"] for request in selected_requests
                    ],
                    "selected_request_digest": digest(selected_requests),
                    "bindings": request_bindings,
                    "bindings_digest": digest(request_bindings),
                }
                cell["cell_digest"] = digest(cell)
                cells.append(cell)
    plan = {
        "schema": PLAN_SCHEMA,
        "manifest_sha256": file_digest(manifest_path),
        "source": manifest["source"],
        "coverage": manifest["coverage"],
        "combination_obligations": manifest["combination_obligations"],
        "noise_protocol": manifest["noise_protocol"],
        "effective_environment": manifest["effective_environment"],
        "validation_profiles": {
            profile_id: {
                "workload": profile["workload"],
                "model_runtime": profile["model_runtime"],
                "resources": profile["resources"],
            }
            for profile_id, profile in profiles.items()
        },
        "cells": cells,
    }
    plan["plan_digest"] = digest(plan)
    return plan


def _stable_request_seed(seed: int, request_id: str) -> int:
    value = hashlib.blake2s(f"{seed}\0{request_id}".encode(), digest_size=4).digest()
    return int.from_bytes(value, "big") & 0x7FFFFFFF


def _campaign_namespace(profile: Mapping[str, object]) -> dict[str, str]:
    model_runtime = profile["model_runtime"]
    return {
        "model_revision": str(model_runtime["model_revision"]),
        "tokenizer_revision": str(model_runtime["tokenizer_revision"]),
        "adapter_namespace": str(model_runtime["adapter_namespace"]),
        "layout_namespace": str(model_runtime["layout_namespace"]),
    }


def _campaign_observed_prefix_id(
    request: Mapping[str, object], profile: Mapping[str, object]
) -> str:
    boundary = request.get("known_prefix_boundary_tokens")
    _require(
        isinstance(boundary, int) and not isinstance(boundary, bool) and boundary > 0,
        "causal tool-resume request lacks known prefix boundary",
    )
    tokens = request["prompt_token_ids"][:boundary]
    _require(
        request.get("known_prefix_token_digest") == digest(tokens),
        "causal tool-resume known prefix digest mismatch",
    )
    return digest({"namespace": _campaign_namespace(profile), "token_ids": tokens})


def _wait_hint_event_id(
    request: Mapping[str, object], cell: Mapping[str, object]
) -> str:
    return digest(
        {
            "schema": "ep-campaign-wait-hint-event-v1",
            "cell_id": cell["cell_id"],
            "manifest_request_id": request["request_id"],
            "generation": request["generation"],
        }
    )


def build_serving_payload(
    *,
    request: Mapping[str, object],
    arm: Mapping[str, object],
    profile: Mapping[str, object],
    cell: Mapping[str, object],
) -> dict[str, object]:
    """Build the only fields published to serving; diagnostic phases stay out."""
    config = arm["config"]
    model_runtime = profile["model_runtime"]
    run_id = str(cell["policy_run_id"])
    policy_cell_id = str(cell["policy_cell_id"])
    policy_namespace = str(cell["opaque_identity_namespace"])
    external_id = _expected_external_request_id(cell, str(request["request_id"]))
    policy_session_id = _opaque_token(
        policy_namespace, "s", [request["session_id"], request["generation"]]
    )
    params = {
        "campaign_schema": "ep-campaign-serving-v1",
        "external_request_id": external_id,
        "semantic_request_id": external_id,
        "semantic_sample_id": external_id,
        "seed": cell["seed"],
        "run_id": run_id,
        "cell_id": policy_cell_id,
        "session_id": policy_session_id,
        "generation": request["generation"],
        "key_chain_digest": request["key_chain_digest"],
        "candidate_config_hash": cell["arm_config_hash"],
        "campaign_provenance": {
            "manifest_request_id": request["request_id"],
            "manifest_request_index": request["request_index"],
            "human_cell_id": cell["cell_id"],
            "human_profile_id": cell["profile_id"],
            "schedule_role": cell["schedule_role"],
            "content_origin": request["content_origin"],
            "arrival_origin": request["arrival_origin"],
            "wait_origin": request["wait_origin"],
            "pressure_origin": request["pressure_origin"],
            "source_id": request["source_id"],
            "session_id": request["session_id"],
            "context_token_span": request.get("context_token_span"),
            "family": request["family"],
            "split": request["split"],
        },
    }
    if request.get("episode") == "causal_tool_resume":
        params["observed_prefix_id"] = _campaign_observed_prefix_id(request, profile)
        params["observed_prefix_boundary_tokens"] = request[
            "known_prefix_boundary_tokens"
        ]
        if (
            request.get("phase") == "resume"
            and config.get("extra_config", {}).get("prefetch_trigger")
            == "known_prefix_wait_hint"
        ):
            params["wait_hint_event_id"] = _wait_hint_event_id(request, cell)
    request_controls = config.get("request_controls", {})
    prefix_cap = request_controls.get("max_offload_tokens")
    if prefix_cap == "known_shared_prefix_tokens":
        context_span = request.get("context_token_span")
        _require(
            isinstance(context_span, list)
            and len(context_span) == 2
            and all(isinstance(item, int) for item in context_span),
            "known prefix cap requires context_token_span",
        )
        params["max_offload_tokens"] = context_span[1]
    elif prefix_cap is not None:
        _require(
            isinstance(prefix_cap, int) and prefix_cap >= 0,
            "max_offload_tokens must be null, known prefix, or non-negative int",
        )
        params["max_offload_tokens"] = prefix_cap
    admission_mode = config.get("extra_config", {}).get("admission_mode", "disabled")
    if admission_mode == "historical_frequency_cost_per_byte":
        serving_inputs = request["serving_inputs"]
        _require(
            "admission_cost_per_byte" in serving_inputs
            and serving_inputs.get("admission_cost_provenance"),
            "cost-aware ready arm requires frozen historical request cost",
        )
        params["admission_cost_per_byte"] = serving_inputs["admission_cost_per_byte"]
        params["admission_cost_provenance"] = serving_inputs[
            "admission_cost_provenance"
        ]
        # The frozen sample existed before this request, but monotonic clocks
        # are process-local. Record when it became available to this process.
        available_ns = time.monotonic_ns()
        params["admission_cost_observation_available_ns"] = available_ns
        params["admission_cost_input_publication_ns"] = time.monotonic_ns()
    return {
        "model": model_runtime["model"],
        "request_id": external_id,
        "prompt": request["prompt_token_ids"],
        "max_tokens": request["suffix_budget_tokens"],
        "temperature": request.get("temperature", 0.0),
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "seed": _stable_request_seed(int(cell["seed"]), external_id),
        "stream": True,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
        "kv_transfer_params": params,
    }


async def execute_request_schedule(
    requests: Sequence[Mapping[str, object]],
    send,
    on_phase_boundary=None,
    before_phase=None,
    *,
    serialize_within_phase: bool = False,
) -> list[dict[str, object]]:
    """Execute phase barriers and within-phase releases from the frozen graph."""
    _require(
        isinstance(serialize_within_phase, bool),
        "serialize_within_phase must be a boolean",
    )
    by_phase: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for request in requests:
        by_phase[int(request["phase_index"])].append(request)
    results: list[dict[str, object]] = []
    previous_phase = -1
    for phase_index in sorted(by_phase):
        _require(phase_index == previous_phase + 1, "phase indexes must be contiguous")
        previous_phase = phase_index
        phase_origin_ns = (
            max(int(result["completion_ns"]) for result in results)
            if results
            else time.monotonic_ns()
        )
        phase_release_ns: Mapping[str, int] = {}
        if before_phase is not None:
            provided_release_ns = await before_phase(
                phase_index, tuple(by_phase[phase_index]), phase_origin_ns
            )
            if provided_release_ns is not None:
                _require(
                    isinstance(provided_release_ns, Mapping),
                    "before_phase release constraints must be a mapping",
                )
                phase_release_ns = provided_release_ns
        if on_phase_boundary is not None:
            on_phase_boundary("start", phase_index)

        async def dispatch(
            request: Mapping[str, object],
            phase_index: int = phase_index,
            phase_origin_ns: int = phase_origin_ns,
            phase_release_ns: Mapping[str, int] = phase_release_ns,
        ) -> dict[str, object]:
            eligibility_ns = time.monotonic_ns()
            release_base_ns = max(
                phase_origin_ns,
                int(phase_release_ns.get(request["request_id"], 0)),
            )
            planned_release_ns = (
                release_base_ns + int(request["relative_release_ms"]) * 1_000_000
            )
            remaining = (planned_release_ns - time.monotonic_ns()) / 1_000_000_000
            if remaining > 0:
                await asyncio.sleep(remaining)
            send_ns = time.monotonic_ns()
            response = await send(request)
            completion_ns = time.monotonic_ns()
            return {
                "manifest_request_id": request["request_id"],
                "request_index": request["request_index"],
                "phase_index": phase_index,
                "phase_origin_ns": phase_origin_ns,
                "eligibility_ns": eligibility_ns,
                "planned_release_ns": planned_release_ns,
                "release_lateness_ns": max(0, send_ns - planned_release_ns),
                "send_ns": send_ns,
                "completion_ns": completion_ns,
                **response,
            }

        if serialize_within_phase:
            phase_results = []
            for request in sorted(
                by_phase[phase_index],
                key=lambda item: (
                    int(item["relative_release_ms"]),
                    int(item["request_index"]),
                ),
            ):
                phase_results.append(await dispatch(request))
        else:
            phase_results = await asyncio.gather(
                *(dispatch(request) for request in by_phase[phase_index])
            )
        results.extend(phase_results)
        if on_phase_boundary is not None:
            on_phase_boundary("complete", phase_index)
    return sorted(results, key=lambda result: int(result["request_index"]))


def _parse_preservation_events(log_path: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    marker = "SEMANTIC_PRESERVATION_EVENT "
    decoder = json.JSONDecoder()
    with open(log_path, encoding="utf-8", errors="replace") as source:
        for line in source:
            offset = line.find(marker)
            if offset < 0:
                continue
            try:
                event, _ = decoder.raw_decode(line[offset + len(marker) :])
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _parse_async_scoring_events(log_path: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    marker = "SEMANTIC_ASYNC_SCORING_EVENT "
    decoder = json.JSONDecoder()
    with open(log_path, encoding="utf-8", errors="replace") as source:
        for line in source:
            offset = line.find(marker)
            if offset < 0:
                continue
            try:
                event, _ = decoder.raw_decode(line[offset + len(marker) :])
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _parse_wait_hint_events(log_path: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    marker = "SEMANTIC_WAIT_HINT_EVENT "
    decoder = json.JSONDecoder()
    with open(log_path, encoding="utf-8", errors="replace") as source:
        for line in source:
            offset = line.find(marker)
            if offset < 0:
                continue
            try:
                event, _ = decoder.raw_decode(line[offset + len(marker) :])
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _parse_runtime_resolutions(log_path: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    marker = "SEMANTIC_RUNTIME_RESOLUTION "
    decoder = json.JSONDecoder()
    with open(log_path, encoding="utf-8", errors="replace") as source:
        for line in source:
            offset = line.find(marker)
            if offset < 0:
                continue
            try:
                event, _ = decoder.raw_decode(line[offset + len(marker) :])
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


async def _wait_for_campaign_drain(
    handle,
    *,
    expected_events: int,
    metrics_mod,
    background_mode: str,
    timeout_s: float = 30.0,
) -> dict[str, object]:
    started_ns = time.monotonic_ns()
    deadline = time.monotonic() + timeout_s
    last_gauges: dict[str, float | None] = {}
    last_event_count = 0
    while time.monotonic() < deadline:
        events = _parse_preservation_events(handle.log_path)
        last_event_count = len(events)
        try:
            last_gauges = metrics_mod.snapshot_gauges(handle.metrics_url())
        except (OSError, ValueError):
            last_gauges = {}
        running = last_gauges.get("vllm:num_requests_running")
        waiting = last_gauges.get("vllm:num_requests_waiting")
        background_pending = last_gauges.get("vllm:semantic_async_scoring_pending")
        async_events = (
            _parse_async_scoring_events(handle.log_path)
            if background_mode == "asynchronous"
            else []
        )
        latest_async_status = next(
            (
                event
                for event in reversed(async_events)
                if event.get("event") == "status"
            ),
            None,
        )
        background_drained = (
            background_mode == "synchronous"
            or background_pending == 0.0
            or (
                latest_async_status is not None
                and latest_async_status.get("pending_items") == 0
                and latest_async_status.get("pending_snapshot_bytes") == 0
            )
        )
        if (
            last_event_count == expected_events
            and running == 0.0
            and waiting == 0.0
            and background_drained
        ):
            completed_ns = time.monotonic_ns()
            return {
                "status": "complete",
                "started_ns": started_ns,
                "completed_ns": completed_ns,
                "duration_ms": (completed_ns - started_ns) / 1_000_000.0,
                "preservation_event_count": last_event_count,
                "gauges": last_gauges,
                "owned_background_work": (
                    "not_applicable"
                    if background_mode == "synchronous"
                    else "gauge_or_worker_status_zero"
                ),
                "async_scoring_status": latest_async_status,
            }
        await asyncio.sleep(0.1)
    completed_ns = time.monotonic_ns()
    return {
        "status": "timeout",
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "duration_ms": (completed_ns - started_ns) / 1_000_000.0,
        "preservation_event_count": last_event_count,
        "gauges": last_gauges,
        "unavailable": [
            "complete_preservation_event_set",
            "zero_running_waiting_requests",
            "owned_background_work_completion",
        ],
    }


def _occupancy_sample(
    *,
    phase: str,
    phase_index: int | None,
    gauges: Mapping[str, float | None],
    resources: Mapping[str, object],
) -> dict[str, object]:
    gpu_usage = gauges.get("vllm:kv_cache_usage_perc")
    cpu_usage = gauges.get("vllm:kv_offload_cpu_cache_usage_perc")
    total_gpu_blocks = int(resources["total_gpu_blocks"])
    cpu_bytes = int(resources["cpu_bytes"])
    return {
        "timestamp_ns": time.monotonic_ns(),
        "phase": phase,
        "phase_index": phase_index,
        "gauges": dict(gauges),
        "configured_gpu_blocks": total_gpu_blocks,
        "occupied_gpu_blocks": (
            gpu_usage * total_gpu_blocks
            if isinstance(gpu_usage, (int, float))
            else None
        ),
        "configured_cpu_bytes": cpu_bytes,
        "occupied_cpu_bytes": (
            cpu_usage * cpu_bytes if isinstance(cpu_usage, (int, float)) else None
        ),
        "speculative_reserved_blocks": None,
        "speculative_inflight_blocks": None,
        "speculative_ready_blocks": None,
        "unavailable_fields": [
            field
            for field, value in (
                ("occupied_gpu_blocks", gpu_usage),
                ("occupied_cpu_bytes", cpu_usage),
                ("speculative_reserved_blocks", None),
                ("speculative_inflight_blocks", None),
                ("speculative_ready_blocks", None),
            )
            if value is None
        ],
    }


def validate_campaign_event_bindings(
    events: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    *,
    cell: Mapping[str, object],
    profile: Mapping[str, object],
    implementation_revision: str | None = None,
    require_passive_key_evidence: bool = False,
) -> list[str]:
    """Check campaign provenance that the generic preservation mapper cannot know."""
    run_id = str(cell["policy_run_id"])
    policy_cell_id = str(cell["policy_cell_id"])
    workload_id = str(cell["policy_workload_id"])
    policy_namespace = str(cell["opaque_identity_namespace"])
    expected: dict[str, Mapping[str, object]] = {}
    for request in requests:
        policy_sample_index = int(
            _opaque_token(policy_namespace, "", request["request_id"], 12), 16
        )
        external_id = (
            f"sc|{run_id}|{policy_cell_id}|{workload_id}|sample-{policy_sample_index}"
        )
        expected[external_id] = request
    errors: list[str] = []
    if len(events) != len(expected):
        errors.append("event_count_mismatch")
    model_runtime = profile["model_runtime"]
    model_namespace = {
        "model_revision": model_runtime["model_revision"],
        "tokenizer_revision": model_runtime["tokenizer_revision"],
        "adapter_namespace": model_runtime["adapter_namespace"],
        "layout_namespace": model_runtime["layout_namespace"],
    }
    for event in events:
        external_id = event.get("external_request_id")
        request = expected.get(external_id)
        if request is None:
            errors.append(f"foreign_event:{external_id}")
            continue
        for event_field, request_field in (
            ("manifest_request_id", "request_id"),
            ("manifest_request_index", "request_index"),
            ("content_origin", "content_origin"),
            ("arrival_origin", "arrival_origin"),
            ("wait_origin", "wait_origin"),
            ("pressure_origin", "pressure_origin"),
            ("source_id", "source_id"),
            ("family", "family"),
            ("split", "split"),
            ("generation", "generation"),
            ("prompt_token_digest", "prompt_token_digest"),
            ("context_token_span", "context_token_span"),
        ):
            if event.get(event_field) != request.get(request_field):
                errors.append(f"{external_id}:{event_field}_mismatch")
        for event_field, expected_value in (
            ("human_cell_id", cell.get("cell_id")),
            ("human_profile_id", cell.get("profile_id")),
            ("schedule_role", cell.get("schedule_role")),
        ):
            if event.get(event_field) != expected_value:
                errors.append(f"{external_id}:{event_field}_mismatch")
        expected_policy_session = _opaque_token(
            policy_namespace, "s", [request["session_id"], request["generation"]]
        )
        observed_policy_session = event.get(
            "policy_session_id", event.get("session_id")
        )
        if observed_policy_session != expected_policy_session:
            errors.append(f"{external_id}:policy_session_id_mismatch")
        if "policy_session_id" in event and event.get("session_id") != request.get(
            "session_id"
        ):
            errors.append(f"{external_id}:session_id_mismatch")
        for decision in event.get("admission_decisions", []):
            if decision.get("candidate_config_hash") != cell["arm_config_hash"]:
                errors.append(f"{external_id}:candidate_config_hash_mismatch")
            if decision.get("engine_run_id") != run_id:
                errors.append(f"{external_id}:decision_run_id_mismatch")
            if decision.get("session_id") != expected_policy_session:
                errors.append(f"{external_id}:decision_session_id_mismatch")
            if decision.get("generation") != request["generation"]:
                errors.append(f"{external_id}:decision_generation_mismatch")
            if decision.get("key_chain_digest") != request["key_chain_digest"]:
                errors.append(f"{external_id}:decision_key_chain_digest_mismatch")
            for field, value in model_namespace.items():
                if decision.get(field) != value:
                    errors.append(f"{external_id}:decision_{field}_mismatch")
        prefetch_attribution = event.get("prefetch_attribution")
        provenance = (
            prefetch_attribution.get("provenance", {})
            if isinstance(prefetch_attribution, Mapping)
            else {}
        )
        provenance_config = (
            provenance.get("config", {}) if isinstance(provenance, Mapping) else {}
        )
        if (
            implementation_revision is not None
            and provenance.get("extension_version") != implementation_revision
        ):
            errors.append(f"{external_id}:implementation_revision_mismatch")
        if (
            provenance_config.get("cache_block_size")
            != profile["resources"]["kv_block_size_tokens"]
        ):
            errors.append(f"{external_id}:runtime_block_size_mismatch")
        if require_passive_key_evidence:
            errors.extend(
                _validate_passive_key_evidence(
                    event,
                    request,
                    external_id=external_id,
                    tokens_per_block=int(profile["resources"]["kv_block_size_tokens"]),
                    blocks_per_chunk=int(provenance_config.get("blocks_per_chunk", 1)),
                    kv_group_count=int(provenance_config.get("kv_group_count", 0)),
                )
            )
    return errors


def _validate_passive_key_evidence(
    event: Mapping[str, object],
    request: Mapping[str, object],
    *,
    external_id: str,
    tokens_per_block: int,
    blocks_per_chunk: int,
    kv_group_count: int,
) -> list[str]:
    """Validate source-backed key ordering/lifecycle without inferring cache facts."""
    errors: list[str] = []
    positions = event.get("ordered_key_positions")
    if event.get("ordered_key_positions_complete") is not True or not isinstance(
        positions, list
    ):
        return [f"{external_id}:ordered_key_positions_unavailable"]
    prompt_token_count = request.get("prompt_token_count")
    if not isinstance(prompt_token_count, int):
        prompt_tokens = request.get("prompt_token_ids")
        prompt_token_count = (
            len(prompt_tokens) if isinstance(prompt_tokens, list) else -1
        )
    tokens_per_chunk = tokens_per_block * blocks_per_chunk
    expected_chunks_per_group = prompt_token_count // tokens_per_chunk
    expected_by_group: dict[int, int] = defaultdict(int)
    seen_positions: set[tuple[int, int]] = set()
    known_keys: set[str] = set()
    for position in positions:
        if not isinstance(position, Mapping):
            errors.append(f"{external_id}:ordered_key_position_invalid")
            continue
        group_idx = position.get("group_idx")
        chunk_index = position.get("chunk_index")
        key = position.get("key")
        identity = (group_idx, chunk_index)
        valid = (
            isinstance(group_idx, int)
            and not isinstance(group_idx, bool)
            and group_idx >= 0
            and isinstance(chunk_index, int)
            and not isinstance(chunk_index, bool)
            and chunk_index == expected_by_group[group_idx]
            and isinstance(key, str)
            and bool(key)
            and position.get("token_start") == chunk_index * tokens_per_chunk
            and position.get("token_end")
            == min(
                prompt_token_count,
                (chunk_index + 1) * tokens_per_chunk,
            )
            and position.get("provenance") == "scheduler_request_group_state"
            and position.get("state") == "planned"
            and identity not in seen_positions
        )
        if not valid:
            errors.append(f"{external_id}:ordered_key_position_invalid")
        if isinstance(group_idx, int) and isinstance(chunk_index, int):
            expected_by_group[group_idx] = chunk_index + 1
            seen_positions.add((group_idx, chunk_index))
        if isinstance(key, str):
            known_keys.add(key)
    if (
        kv_group_count <= 0
        or set(expected_by_group) != set(range(kv_group_count))
        or any(
            expected_by_group[group_idx] != expected_chunks_per_group
            for group_idx in range(kv_group_count)
        )
    ):
        errors.append(f"{external_id}:ordered_key_full_prompt_coverage_invalid")
    context_span = request.get("context_token_span")
    if (
        isinstance(context_span, list)
        and len(context_span) == 2
        and all(isinstance(value, int) for value in context_span)
    ):
        context_positions = {
            (position.get("group_idx"), position.get("chunk_index"))
            for position in positions
            if isinstance(position, Mapping)
            and position.get("complete_chunk") is True
            and isinstance(position.get("token_start"), int)
            and isinstance(position.get("token_end"), int)
            and position["token_start"] >= context_span[0]
            and position["token_end"] <= context_span[1]
        }
        expected_context_positions = {
            (group_idx, chunk_index)
            for group_idx in range(kv_group_count)
            for chunk_index in range(expected_chunks_per_group)
            if chunk_index * tokens_per_chunk >= context_span[0]
            and (chunk_index + 1) * tokens_per_chunk <= context_span[1]
        }
        if context_positions != expected_context_positions:
            errors.append(f"{external_id}:ordered_key_context_coverage_invalid")

    lifecycle = event.get("key_lifecycle")
    if not isinstance(lifecycle, Mapping):
        errors.append(f"{external_id}:key_lifecycle_unavailable")
        lifecycle = {}
    for kind, expected_state, clock_field in (
        ("cpu_commits", "committed_cpu_ready", "completion_observed_at_ns"),
        ("cpu_evictions", "committed_cpu_eviction", "observed_at_ns"),
    ):
        records = lifecycle.get(kind, [])
        if not isinstance(records, list):
            errors.append(f"{external_id}:{kind}_invalid")
            continue
        prior_sequence = -1
        for record in records:
            sequence = record.get("sequence") if isinstance(record, Mapping) else None
            valid = (
                isinstance(record, Mapping)
                and record.get("key") in known_keys
                and record.get("state") == expected_state
                and isinstance(sequence, int)
                and not isinstance(sequence, bool)
                and sequence > prior_sequence
                and isinstance(record.get(clock_field), int)
                and isinstance(record.get("generation"), int)
                and isinstance(record.get("run_id"), str)
                and isinstance(record.get("cell_id"), str)
            )
            if valid and kind == "cpu_commits":
                valid = (
                    isinstance(record.get("committed_at_ns"), int)
                    and record["committed_at_ns"] <= record["completion_observed_at_ns"]
                )
            if not valid:
                errors.append(f"{external_id}:{kind}_invalid")
            elif kind == "cpu_evictions" and (
                not isinstance(record.get("prior_commit_sequence"), int)
                or record["prior_commit_sequence"] >= sequence
                or not isinstance(record.get("prior_commit_ns"), int)
                or record["prior_commit_ns"] > record["observed_at_ns"]
            ):
                errors.append(f"{external_id}:cpu_eviction_commit_join_invalid")
            if isinstance(sequence, int):
                prior_sequence = sequence

    first = event.get("first_admission")
    if not isinstance(first, Mapping):
        errors.append(f"{external_id}:first_admission_unavailable")
    else:
        if (
            first.get("admission_index") != 0
            or first.get("episode_id") != 0
            or not isinstance(first.get("admitted_at_ns"), int)
            or first.get("provenance") != "scheduler_update_state_after_alloc"
        ):
            errors.append(f"{external_id}:first_admission_invalid")
        for name in (
            "local_keys",
            "planned_load_keys",
            "completed_load_keys",
            "recomputed_keys",
        ):
            records = first.get(name)
            if not isinstance(records, list) or any(
                not isinstance(record, Mapping) or record.get("key") not in known_keys
                for record in records
            ):
                errors.append(f"{external_id}:first_admission_{name}_invalid")
        local_ids = {
            (record.get("group_idx"), record.get("chunk_index"))
            for record in first.get("local_keys", [])
        }
        planned_load_ids = {
            (record.get("group_idx"), record.get("chunk_index"))
            for record in first.get("planned_load_keys", [])
        }
        completed_load_ids = {
            (record.get("group_idx"), record.get("chunk_index"))
            for record in first.get("completed_load_keys", [])
        }
        recomputed_ids = {
            (record.get("group_idx"), record.get("chunk_index"))
            for record in first.get("recomputed_keys", [])
        }
        if (
            not completed_load_ids.issubset(planned_load_ids)
            or local_ids & planned_load_ids
            or local_ids & recomputed_ids
            or completed_load_ids & recomputed_ids
        ):
            errors.append(f"{external_id}:first_admission_partition_invalid")
    return errors


def _physical_transfer_accounting(
    events: Sequence[Mapping[str, object]],
    metric_deltas: Mapping[str, float | None],
    *,
    wait_hint_events: Sequence[Mapping[str, object]] = (),
    wait_hint_publications: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    physical_load_bytes = metric_deltas.get("vllm:kv_offload_load_bytes_total")
    _require(
        _nonnegative_number(physical_load_bytes),
        "physical load bytes are unavailable",
    )
    totals = {
        "job_count": 0,
        "successful_job_count": 0,
        "failed_job_count": 0,
        "submitted_bytes": 0,
        "completed_bytes": 0,
        "consumed_bytes": 0,
        "discarded_bytes": 0,
        "released_reservation_bytes": 0,
        "uncompleted_submitted_bytes": 0,
        "unconsumed_completed_bytes": 0,
        "actual_contributors": 0,
        "expected_contributors": 0,
    }
    device_time_s = 0.0
    job_ids: set[tuple[object, object, object]] = set()
    demand_job_ids: set[tuple[object, object, object]] = set()
    demand_completed_from_jobs = 0
    completed_from_jobs = 0
    for event in events:
        for status in event.get("transfer_job_statuses", []):
            _require(
                isinstance(status, Mapping),
                "demand physical transfer job record",
            )
            if status.get("direction") != "load":
                continue
            identity = (
                event.get("run_id"),
                event.get("cell_id"),
                status.get("job_id"),
            )
            _require(
                identity not in demand_job_ids,
                "demand physical transfer job is duplicated",
            )
            demand_job_ids.add(identity)
            if status.get("success") is True:
                transfer_bytes = status.get("bytes")
                _require(
                    isinstance(transfer_bytes, int)
                    and not isinstance(transfer_bytes, bool)
                    and transfer_bytes > 0,
                    "completed demand transfer bytes",
                )
                demand_completed_from_jobs += transfer_bytes
    for event in events:
        accounting = event.get("speculative_transfer_accounting")
        statuses = event.get("speculative_transfer_job_statuses")
        if accounting is None and statuses is None:
            continue
        _require(
            isinstance(accounting, Mapping) and isinstance(statuses, list),
            "speculative physical transfer evidence is incomplete",
        )
        for name in totals:
            value = accounting.get(name)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"speculative physical transfer {name}",
            )
            totals[name] += value
        current_time = accounting.get("device_time_s")
        _require(
            isinstance(current_time, (int, float))
            and not isinstance(current_time, bool)
            and math.isfinite(float(current_time))
            and float(current_time) >= 0.0,
            "speculative physical transfer device time",
        )
        device_time_s += float(current_time)
        for status in statuses:
            _require(
                isinstance(status, Mapping),
                "speculative physical transfer job record",
            )
            identity = (
                event.get("run_id"),
                event.get("cell_id"),
                status.get("job_id"),
            )
            _require(
                identity not in job_ids and identity not in demand_job_ids,
                "speculative physical transfer job is duplicated",
            )
            job_ids.add(identity)
            if status.get("success") is True:
                transfer_bytes = status.get("bytes")
                _require(
                    isinstance(transfer_bytes, int)
                    and not isinstance(transfer_bytes, bool)
                    and transfer_bytes > 0,
                    "completed speculative transfer bytes",
                )
                completed_from_jobs += transfer_bytes
    hint_only_ids = {
        publication.get("event_id")
        for publication in wait_hint_publications
        if isinstance(publication, Mapping)
        and publication.get("hint_only") is True
        and isinstance(publication.get("event_id"), str)
    }
    hint_only_retired_ids = {
        event.get("event_id")
        for event in wait_hint_events
        if isinstance(event, Mapping)
        and event.get("event") == "staging_retired"
        and event.get("event_id") in hint_only_ids
    }
    for event in wait_hint_events:
        if (
            not isinstance(event, Mapping)
            or event.get("event") != "staging_ready"
            or event.get("event_id") not in hint_only_ids
        ):
            continue
        transfer_bytes = event.get("transfer_bytes")
        transfer_time_s = event.get("transfer_time_s")
        _require(
            isinstance(transfer_bytes, int)
            and not isinstance(transfer_bytes, bool)
            and transfer_bytes > 0
            and isinstance(transfer_time_s, (int, float))
            and not isinstance(transfer_time_s, bool)
            and math.isfinite(float(transfer_time_s))
            and float(transfer_time_s) >= 0.0,
            "hint-only physical transfer evidence",
        )
        identity = (
            event.get("run_id"),
            event.get("cell_id"),
            event.get("job_id"),
        )
        _require(
            identity not in job_ids and identity not in demand_job_ids,
            "hint-only physical transfer job is duplicated",
        )
        job_ids.add(identity)
        completed_from_jobs += transfer_bytes
        totals["job_count"] += 1
        totals["successful_job_count"] += 1
        totals["submitted_bytes"] += transfer_bytes
        totals["completed_bytes"] += transfer_bytes
        totals["actual_contributors"] += 1
        totals["expected_contributors"] += 1
        device_time_s += float(transfer_time_s)
        if event.get("event_id") in hint_only_retired_ids:
            totals["discarded_bytes"] += transfer_bytes
            totals["released_reservation_bytes"] += transfer_bytes
        else:
            totals["unconsumed_completed_bytes"] += transfer_bytes
    _require(
        totals["job_count"] == len(job_ids)
        and totals["completed_bytes"] == completed_from_jobs,
        "speculative physical transfer totals do not reconcile",
    )
    _require(
        float(physical_load_bytes)
        == demand_completed_from_jobs + totals["completed_bytes"],
        "physical load counter does not reconcile completion jobs",
    )
    return {
        "prometheus_physical_load_bytes": float(physical_load_bytes),
        "demand_load_job_count": len(demand_job_ids),
        "demand_load_bytes": demand_completed_from_jobs,
        "speculative_job_count": totals["job_count"],
        "speculative_successful_job_count": totals["successful_job_count"],
        "speculative_failed_job_count": totals["failed_job_count"],
        "speculative_submitted_h2d_bytes": totals["submitted_bytes"],
        "speculative_completed_h2d_bytes": totals["completed_bytes"],
        "speculative_consumed_h2d_bytes": totals["consumed_bytes"],
        "speculative_discarded_h2d_bytes": totals["discarded_bytes"],
        "speculative_released_reservation_bytes": totals["released_reservation_bytes"],
        "speculative_uncompleted_submitted_bytes": totals[
            "uncompleted_submitted_bytes"
        ],
        "speculative_unconsumed_completed_h2d_bytes": totals[
            "unconsumed_completed_bytes"
        ],
        "speculative_actual_contributors": totals["actual_contributors"],
        "speculative_expected_contributors": totals["expected_contributors"],
        "speculative_device_time_s": device_time_s,
        "total_physical_h2d_bytes": float(physical_load_bytes),
        "unmatched_physical_load_bytes": 0.0,
        "source": "prometheus_physical_load_reconciled_with_speculative_completion",
    }


def _normalized_metric_vector(
    *,
    metric_contracts: Mapping[str, Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    preservation: Mapping[str, object],
    metric_deltas: Mapping[str, float | None],
    episode_makespan_ms: float | None,
    physical_transfer_accounting: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    ttfts = [
        float(request["ttft_ms"])
        for request in requests
        if isinstance(request.get("ttft_ms"), (int, float))
    ]
    output_tokens = sum(
        len(request.get("output_token_ids", [])) for request in requests
    )
    values = {
        "ttft_ms": statistics.median(ttfts) if ttfts else None,
        "throughput_tokens_per_s": (
            output_tokens / (episode_makespan_ms / 1000.0)
            if episode_makespan_ms is not None and episode_makespan_ms > 0
            else None
        ),
        "episode_makespan_ms": episode_makespan_ms,
        "recomputed_prompt_tokens": preservation.get("preservation_recomputed_tokens"),
        "preservation_reused_tokens": preservation.get("preservation_reused_tokens"),
        "load_bytes": (
            physical_transfer_accounting.get("total_physical_h2d_bytes")
            if physical_transfer_accounting is not None
            else metric_deltas.get("vllm:kv_offload_load_bytes_total")
        ),
        "store_bytes": metric_deltas.get("vllm:kv_offload_store_bytes_total"),
        "preemptions": metric_deltas.get("vllm:num_preemptions_total"),
    }
    sources = {
        "ttft_ms": "median_request_stream_first_exact_token",
        "throughput_tokens_per_s": "exact_output_tokens_over_drained_episode_makespan",
        "episode_makespan_ms": "first_eligibility_to_drain_complete",
        "recomputed_prompt_tokens": "authoritative_preservation_event_sum",
        "preservation_reused_tokens": "authoritative_preservation_event_sum",
        "load_bytes": (
            "prometheus_physical_load_reconciled_with_speculative_completion"
            if physical_transfer_accounting is not None
            else "prometheus_drained_interval_delta"
        ),
        "store_bytes": "prometheus_drained_interval_delta",
        "preemptions": "prometheus_drained_interval_delta",
    }
    return {
        metric: {
            "value": values[metric],
            "unit": metric_contracts[metric]["unit"],
            "source": sources[metric],
            "available": values[metric] is not None,
            "unavailable_reason": (
                None if values[metric] is not None else "source_measurement_unavailable"
            ),
        }
        for metric in sorted(REQUIRED_METRICS)
    }


from harness.streaming import consume_campaign_sse


async def execute_serving_cell(
    manifest: Mapping[str, object],
    manifest_path: Path,
    *,
    cell_id: str,
    profile_id: str,
    request_manifest_path: Path,
    output: Path,
    port: int,
) -> dict[str, object]:
    """Launch one fresh server and execute one explicit profile/arm/replicate cell."""
    arms, profiles, request_path = validate_campaign_manifest(manifest, manifest_path)
    plan = build_plan(manifest, manifest_path)
    plan_artifact_path = output.with_suffix(output.suffix + ".plan.json")
    _write_json(plan_artifact_path, plan)
    matches = [cell for cell in plan["cells"] if cell["cell_id"] == cell_id]
    _require(len(matches) == 1, f"unknown or duplicate cell_id {cell_id}")
    cell = matches[0]
    _require(cell["profile_id"] == profile_id, "execute-cell profile differs from plan")
    _require(
        request_manifest_path.resolve() == request_path.resolve(),
        "execute-cell request manifest differs from campaign binding",
    )
    _require(port == cell["port"], "execute-cell port differs from frozen plan")
    _require(cell["execution_state"] == "ready", f"{cell_id} is pending")
    arm = arms[cell["arm_id"]]
    profile = profiles[cell["profile_id"]]
    config = arm["config"]
    extra_config = config.get("extra_config")
    _require(isinstance(extra_config, Mapping), "arm extra_config")
    _require(extra_config.get("preservation_events") is True, "events are required")
    model_runtime = profile["model_runtime"]
    extension_root = Path(__file__).resolve().parents[1]
    deployed_extension = _git_identity(extension_root)
    runtime_root = Path(str(manifest["source"]["runtime_source_path"]))
    _require(runtime_root.is_dir(), "declared runtime_source_path does not exist")
    deployed_runtime = _git_identity(runtime_root)
    _require(
        deployed_extension["revision"] == manifest["source"]["implementation_revision"]
        and deployed_extension["tree"] == manifest["source"]["implementation_tree"]
        and deployed_extension["clean"] is True,
        "deployed extension identity/tree/cleanliness mismatch",
    )
    _require(
        deployed_runtime["revision"] == manifest["source"]["runtime_revision"]
        and (
            manifest["source"].get("runtime_tree") is None
            or deployed_runtime["tree"] == manifest["source"]["runtime_tree"]
        )
        and deployed_runtime["clean"] is True,
        "deployed vLLM revision/tree/cleanliness mismatch",
    )
    expected_namespace = {
        "model_revision": model_runtime["model_revision"],
        "tokenizer_revision": model_runtime["tokenizer_revision"],
        "adapter_namespace": model_runtime["adapter_namespace"],
        "layout_namespace": model_runtime["layout_namespace"],
    }
    _require(
        extra_config.get("admission_namespace") == expected_namespace,
        "admission namespace must match the baseline profile",
    )
    with request_path.open(encoding="utf-8") as source:
        request_document = json.load(source)
    workload = profile["workload"]
    requests = _profile_requests(request_document, workload)
    _require(requests, "profile selected no requests")
    payloads = {
        request["request_id"]: build_serving_payload(
            request=request, arm=arm, profile=profile, cell=cell
        )
        for request in requests
    }
    external_ids = [payload["request_id"] for payload in payloads.values()]
    _require(
        len(external_ids) == len(set(external_ids)),
        "opaque policy request identity collision",
    )
    import aiohttp

    from benchmarks.preservation import (
        map_preservation_events,
        summarize_gpu_model_runner_work,
        summarize_prefill_cuda_work,
    )
    from harness import metrics as metrics_mod
    from harness import policies as policies_mod
    from harness.server import launch_server

    resources = profile["resources"]
    kv_config = policies_mod.kv_transfer_config(
        config["policy"], resources["cpu_bytes"], extra_config=dict(extra_config)
    )
    server_args = _engine_server_args(profile, seed=int(cell["seed"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = launch_server(
        model_runtime["model"],
        port,
        str(output.parent),
        gpu_memory_utilization=float(resources["gpu_memory_utilization"]),
        max_model_len=int(resources["max_model_len"]),
        num_gpu_blocks_override=int(resources["total_gpu_blocks"]),
        kv_transfer_config=kv_config,
        extra_args=server_args,
        extra_env={
            **cell["effective_environment"],
            "PYTHONHASHSEED": str(cell["seed"]),
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    [
                        str(extension_root),
                        str(runtime_root),
                        os.environ.get("PYTHONPATH"),
                    ],
                )
            ),
        },
        log_label=cell_id,
    )
    before_metrics: dict[str, float | None] = {}
    after_metrics: dict[str, float | None] = {}
    before_metric_evidence: dict[str, object] = {}
    after_metric_evidence: dict[str, object] = {}
    results: list[dict[str, object]] = []
    phase_occupancy_samples: list[dict[str, object]] = []
    wait_hint_publications: list[dict[str, object]] = []
    wait_hint_tasks: list[asyncio.Task[dict[str, object]]] = []
    drain = {"status": "not_started"}
    resolved_scheduler_runtime: dict[str, object] = {}
    try:
        runtime_resolutions = _parse_runtime_resolutions(handle.log_path)
        unique_runtime_resolutions = {
            digest(resolution): resolution for resolution in runtime_resolutions
        }
        _require(
            len(unique_runtime_resolutions) == 1,
            "server did not emit one consistent scheduler runtime resolution",
        )
        resolved_scheduler_runtime = next(iter(unique_runtime_resolutions.values()))
        expected_scheduler = cell["runtime_resolution"]
        _require(
            resolved_scheduler_runtime == expected_scheduler,
            "server scheduler runtime differs from frozen plan",
        )
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            before_metric_evidence = metrics_mod.snapshot_evidence(handle.metrics_url())
            before_metrics = dict(before_metric_evidence["values"])

            async def send(request: Mapping[str, object]) -> dict[str, object]:
                payload = payloads[request["request_id"]]
                start_ns = time.monotonic_ns()
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
                error = None
                status = None
                try:
                    async with session.post(
                        f"{handle.base_url()}/v1/completions", json=payload
                    ) as response:
                        status = response.status
                        if status >= 400:
                            error = (await response.text())[-2000:]
                        else:
                            stream_result = await consume_campaign_sse(response.content)
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    error = repr(exc)
                end_ns = time.monotonic_ns()
                first_token_ns = stream_result["first_token_ns"]
                terminal_complete = bool(stream_result["terminal_complete"])
                return {
                    "external_request_id": payload["request_id"],
                    "http_status": status,
                    "success": error is None and status == 200 and terminal_complete,
                    "request_start_ns": start_ns,
                    "request_end_ns": end_ns,
                    "ttft_ms": (
                        (first_token_ns - start_ns) / 1_000_000.0
                        if isinstance(first_token_ns, int)
                        else None
                    ),
                    "total_ms": (end_ns - start_ns) / 1_000_000.0,
                    **stream_result,
                    "error": error,
                }

            async def before_phase(
                phase_index: int,
                phase_requests: Sequence[Mapping[str, object]],
                phase_origin_ns: int,
            ) -> dict[str, int] | None:
                if workload["episode"] != "causal_tool_resume":
                    return None
                resume_requests = [
                    request
                    for request in phase_requests
                    if request.get("phase") == "resume"
                ]
                if not resume_requests:
                    return None
                activation_training = request_document.get(
                    "causal_tool_resume_extension", {}
                ).get("activation_training")
                expected_activity_phase = 16
                expected_phase_request_count = 16
                if (
                    isinstance(activation_training, Mapping)
                    and activation_training.get("split") == workload.get("split")
                    and activation_training.get("context_budget")
                    == workload.get("context_tokens")
                ):
                    expected_activity_phase = int(
                        activation_training["activity_phase_index"]
                    )
                _require(
                    phase_index == expected_activity_phase
                    and len(phase_requests) == expected_phase_request_count
                    and len(resume_requests) == 8,
                    "causal activity phase differs from frozen graph",
                )
                request_controls = config.get("request_controls", {})
                tool_gap_ms = request_controls.get("tool_gap_ms", 0)
                _require(
                    isinstance(tool_gap_ms, int)
                    and not isinstance(tool_gap_ms, bool)
                    and tool_gap_ms in {0, 50, 250},
                    "causal tool-resume gap differs from frozen settings",
                )
                _require(
                    all(
                        request.get("relative_release_ms")
                        == request.get("hint_lead_ms", tool_gap_ms)
                        and request.get("hint_lead_ms", tool_gap_ms) in {50, 250, 750}
                        for request in resume_requests
                    ),
                    "performance activity has an invalid declared hint lead",
                )
                pre_hint_idle_ms = (
                    activation_training.get("pre_hint_idle_ms", 0)
                    if isinstance(activation_training, Mapping)
                    else 0
                )
                _require(
                    isinstance(pre_hint_idle_ms, int)
                    and not isinstance(pre_hint_idle_ms, bool)
                    and 0 <= pre_hint_idle_ms <= 15_000,
                    "causal pre-hint idle interval",
                )
                if pre_hint_idle_ms:
                    await asyncio.sleep(pre_hint_idle_ms / 1_000.0)
                hint_boundary_ns = time.monotonic_ns()
                release_ns = {
                    request["request_id"]: hint_boundary_ns
                    for request in phase_requests
                }
                if extra_config.get("prefetch_trigger") == "known_prefix_wait_hint":

                    async def publish_one(
                        request: Mapping[str, object],
                        *,
                        manifest_request_id: str | None = None,
                        declared_wait_ms: int | None = None,
                        hint_only: bool = False,
                    ) -> dict[str, object]:
                        hint_request_id = manifest_request_id or str(
                            request["request_id"]
                        )
                        lead_ms = (
                            int(declared_wait_ms)
                            if declared_wait_ms is not None
                            else int(request.get("hint_lead_ms", tool_gap_ms))
                        )
                        available_ns = time.monotonic_ns()
                        publication_ns = time.monotonic_ns()
                        request_payload = payloads[request["request_id"]]
                        payload = {
                            "schema": "semantic-known-prefix-wait-hint-v1",
                            "event_id": _wait_hint_event_id(
                                {**request, "request_id": hint_request_id}, cell
                            ),
                            "owner_id": (
                                _expected_external_request_id(cell, hint_request_id)
                                if hint_only
                                else request_payload["request_id"]
                            ),
                            "run_id": request_payload["kv_transfer_params"]["run_id"],
                            "cell_id": request_payload["kv_transfer_params"]["cell_id"],
                            "session_id": request_payload["kv_transfer_params"][
                                "session_id"
                            ],
                            "generation": request["generation"],
                            "observed_prefix_id": _campaign_observed_prefix_id(
                                request, profile
                            ),
                            "prefix_boundary_tokens": request[
                                "known_prefix_boundary_tokens"
                            ],
                            "observation_available_ns": available_ns,
                            "input_publication_ns": publication_ns,
                            "declared_wait_ms": lead_ms,
                            "wait_provenance": ("driver_declared_tool_wait_budget"),
                            "namespace": _campaign_namespace(profile),
                        }
                        status = None
                        response_payload: object = None
                        error = None
                        try:
                            async with session.post(
                                f"{handle.base_url()}"
                                "/plugins/semantic-offload/wait-hints",
                                json=payload,
                            ) as response:
                                status = response.status
                                response_payload = await response.json()
                        except (
                            aiohttp.ClientError,
                            asyncio.TimeoutError,
                            ValueError,
                        ) as exc:
                            error = repr(exc)
                        completed_ns = time.monotonic_ns()
                        planned_resume_ns = hint_boundary_ns + lead_ms * 1_000_000
                        return {
                            "event_id": payload["event_id"],
                            "manifest_request_id": hint_request_id,
                            "hint_only": hint_only,
                            "hint_boundary_ns": hint_boundary_ns,
                            "phase_origin_ns": phase_origin_ns,
                            "planned_resume_ns": planned_resume_ns,
                            "publication_ns": publication_ns,
                            "completed_ns": completed_ns,
                            "acknowledgement_late_by_ns": max(
                                0, completed_ns - planned_resume_ns
                            ),
                            "http_status": status,
                            "response": response_payload,
                            "error": error,
                            "payload": payload,
                            "payload_digest": digest(payload),
                            "future_suffix_published": False,
                        }

                    hint_only_specs = request_document.get(
                        "causal_tool_resume_extension", {}
                    ).get("hint_only_events", [])
                    requests_by_id = {
                        str(request["request_id"]): request for request in requests
                    }
                    publications = [
                        {
                            "request": request,
                            "manifest_request_id": str(request["request_id"]),
                            "declared_wait_ms": None,
                            "hint_only": False,
                        }
                        for request in resume_requests
                    ]
                    for spec in hint_only_specs:
                        _require(
                            isinstance(spec, Mapping)
                            and isinstance(spec.get("hint_id"), str)
                            and spec["hint_id"]
                            and spec.get("source_request_id") in requests_by_id
                            and spec.get("lead_ms") in {50, 250, 750},
                            "invalid hint-only event",
                        )
                        publications.append(
                            {
                                "request": requests_by_id[
                                    str(spec["source_request_id"])
                                ],
                                "manifest_request_id": str(spec["hint_id"]),
                                "declared_wait_ms": int(spec["lead_ms"]),
                                "hint_only": True,
                            }
                        )
                    publication_order = (
                        activation_training.get("hint_publication_order")
                        if isinstance(activation_training, Mapping)
                        else None
                    )
                    if publication_order is not None:
                        publications_by_id = {
                            str(item["manifest_request_id"]): item
                            for item in publications
                        }
                        _require(
                            len(publications_by_id) == len(publications)
                            and set(publication_order) == set(publications_by_id),
                            "runtime hint publication order differs from frozen graph",
                        )
                        publications = [
                            publications_by_id[str(item)] for item in publication_order
                        ]
                    serialize_publication = bool(
                        isinstance(activation_training, Mapping)
                        and activation_training.get("serialize_hint_publication")
                    )
                    if serialize_publication:
                        for item in publications:
                            wait_hint_publications.append(
                                await publish_one(
                                    item["request"],
                                    manifest_request_id=item["manifest_request_id"],
                                    declared_wait_ms=item["declared_wait_ms"],
                                    hint_only=bool(item["hint_only"]),
                                )
                            )
                    else:
                        wait_hint_tasks.extend(
                            asyncio.create_task(
                                publish_one(
                                    item["request"],
                                    manifest_request_id=item["manifest_request_id"],
                                    declared_wait_ms=item["declared_wait_ms"],
                                    hint_only=bool(item["hint_only"]),
                                )
                            )
                            for item in publications
                        )
                return release_ns

            def sample_phase(boundary: str, phase_index: int) -> None:
                try:
                    gauges = metrics_mod.snapshot_gauges(handle.metrics_url())
                except (OSError, ValueError):
                    gauges = {}
                phase_occupancy_samples.append(
                    _occupancy_sample(
                        phase=boundary,
                        phase_index=phase_index,
                        gauges=gauges,
                        resources=resources,
                    )
                )

            results = await execute_request_schedule(
                requests,
                send,
                on_phase_boundary=sample_phase,
                before_phase=before_phase,
                serialize_within_phase=bool(
                    request_document.get("causal_tool_resume_extension", {})
                    .get("activation_training", {})
                    .get("serialize_within_phase", False)
                ),
            )
            if wait_hint_tasks:
                wait_hint_publications.extend(await asyncio.gather(*wait_hint_tasks))
            drain = await _wait_for_campaign_drain(
                handle,
                expected_events=len(results),
                metrics_mod=metrics_mod,
                background_mode=str(extra_config.get("scoring_mode", "synchronous")),
            )
            after_metric_evidence = metrics_mod.snapshot_evidence(handle.metrics_url())
            after_metrics = dict(after_metric_evidence["values"])
    finally:
        handle.shutdown()
    events = _parse_preservation_events(handle.log_path)
    async_scoring_events = _parse_async_scoring_events(handle.log_path)
    wait_hint_events = _parse_wait_hint_events(handle.log_path)
    expected_ids = [result["external_request_id"] for result in results]
    mapped = map_preservation_events(
        events,
        expected_request_ids=expected_ids,
        run_id=cell["policy_run_id"],
        cell_id=cell["policy_cell_id"],
    )
    gpu_prefill_work_accounting = summarize_prefill_cuda_work(events)
    gpu_model_runner_work_accounting = summarize_gpu_model_runner_work(events)
    campaign_binding_errors = validate_campaign_event_bindings(
        events,
        requests,
        cell=cell,
        profile=profile,
        implementation_revision=manifest["source"]["implementation_revision"],
        require_passive_key_evidence=(
            manifest["source"].get("passive_key_evidence_schema")
            == "semantic-passive-key-evidence-v1"
        ),
    )
    event_occupancy = [
        {
            "external_request_id": event.get("external_request_id"),
            "block_pool_snapshots": event.get("prefetch_attribution", {}).get(
                "block_pool_snapshots", []
            ),
            "cpu_cache_snapshots": event.get("prefetch_attribution", {}).get(
                "cpu_cache_snapshots", []
            ),
            "allocator_peak_stats": event.get("prefetch_attribution", {}).get(
                "allocator_peak_stats", {}
            ),
        }
        for event in events
        if isinstance(event.get("prefetch_attribution"), Mapping)
    ]
    episode_start_ns = min(request["eligibility_ns"] for request in results)
    episode_end_ns = int(drain.get("completed_ns", 0))
    episode_makespan_ms = (
        (episode_end_ns - episode_start_ns) / 1_000_000.0
        if episode_end_ns >= episode_start_ns
        else None
    )
    metric_deltas, metric_delta_provenance = metrics_mod.diff_with_provenance(
        before_metrics, after_metrics
    )
    physical_transfer_accounting = _physical_transfer_accounting(
        events,
        metric_deltas,
        wait_hint_events=wait_hint_events,
        wait_hint_publications=wait_hint_publications,
    )
    metric_contracts = manifest["noise_protocol"]["metric_contracts"]
    normalized_metrics = _normalized_metric_vector(
        metric_contracts=metric_contracts,
        requests=results,
        preservation=mapped,
        metric_deltas=metric_deltas,
        episode_makespan_ms=episode_makespan_ms,
        physical_transfer_accounting=physical_transfer_accounting,
    )
    occupancy_peaks = {
        field: (
            max(values)
            if (
                values := [
                    sample[field]
                    for sample in phase_occupancy_samples
                    if isinstance(sample.get(field), (int, float))
                ]
            )
            else None
        )
        for field in ("occupied_gpu_blocks", "occupied_cpu_bytes")
    }
    result = {
        "schema": "ep-campaign-cell-result-v1",
        **{
            key: cell[key]
            for key in (
                "cell_id",
                "profile_id",
                "baseline_profile_hash",
                "replicate_id",
                "slot_id",
                "schedule_role",
                "seed",
                "request_order",
                "order_schedule_digest",
                "arm_id",
                "arm_config_hash",
                "config_delta",
            )
        },
        "request_manifest_sha256": file_digest(request_path),
        "campaign_manifest_sha256": file_digest(manifest_path),
        "plan_digest": plan["plan_digest"],
        "plan_artifact": {
            "path": str(plan_artifact_path.resolve()),
            "sha256": file_digest(plan_artifact_path),
        },
        "cell_digest": cell["cell_digest"],
        "deployed_extension": deployed_extension,
        "deployed_runtime": deployed_runtime,
        "resolved_scheduler_backend": profile["scheduler_backend"],
        "resolved_scheduler": _scheduler_resolution(profile["scheduler_backend"]),
        "resolved_scheduler_runtime": resolved_scheduler_runtime,
        "resolved_kv_block_size_tokens": resources["kv_block_size_tokens"],
        "resolved_launch_argv": handle.command,
        "effective_environment": dict(cell["effective_environment"]),
        "server_log": handle.log_path,
        "requests": results,
        "wait_hint_publications": wait_hint_publications,
        "wait_hint_events": wait_hint_events,
        "wait_hint_action_counts": _wait_hint_action_counts(
            wait_hint_publications, wait_hint_events
        ),
        "episode_start_ns": episode_start_ns,
        "episode_end_ns": episode_end_ns,
        "episode_makespan_ms": episode_makespan_ms,
        "drain": drain,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "metrics_endpoint_evidence": {
            "before": before_metric_evidence,
            "after": after_metric_evidence,
        },
        "metrics_delta": metric_deltas,
        "metrics_delta_provenance": metric_delta_provenance,
        "physical_transfer_accounting": physical_transfer_accounting,
        "gpu_prefill_work_accounting": gpu_prefill_work_accounting,
        "gpu_model_runner_work_accounting": gpu_model_runner_work_accounting,
        "normalized_metrics": normalized_metrics,
        "preservation_event_count": len(events),
        "preservation_events": events,
        "async_scoring_events": async_scoring_events,
        "preservation_mapping": mapped,
        "campaign_binding_errors": campaign_binding_errors,
        "physical_occupancy_evidence": {
            "phase_samples": phase_occupancy_samples,
            "event_samples": event_occupancy,
            "phase_peaks": occupancy_peaks,
            "configured_gpu_blocks": resources["total_gpu_blocks"],
            "configured_cpu_bytes": resources["cpu_bytes"],
        },
        "metric_contract_digest": digest(metric_contracts),
        "valid": (
            all(request["success"] for request in results)
            and drain.get("status") == "complete"
            and mapped.get("preservation_reason")
            in {"authoritative_request_events", "mixed_request_outcomes"}
            and not campaign_binding_errors
            and all(metric["available"] for metric in normalized_metrics.values())
        ),
    }
    _write_json(output, result)
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--manifest", required=True, type=Path)
    plan_parser.add_argument("--output", required=True, type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    run_parser.add_argument("--cell-id", action="append")
    causal_parser = subparsers.add_parser("validate-causal-events")
    causal_parser.add_argument("--events", required=True, type=Path)
    causal_parser.add_argument("--output", required=True, type=Path)
    request_parser = subparsers.add_parser("build-requests")
    request_parser.add_argument("--contexts", required=True, type=Path)
    request_parser.add_argument("--prepared-manifest", required=True, type=Path)
    request_parser.add_argument("--output", required=True, type=Path)
    wait_request_parser = subparsers.add_parser("extend-wait-requests")
    wait_request_parser.add_argument("--base", required=True, type=Path)
    wait_request_parser.add_argument("--output", required=True, type=Path)
    cell_parser = subparsers.add_parser("execute-cell")
    cell_parser.add_argument("--manifest", required=True, type=Path)
    cell_parser.add_argument("--request-manifest", required=True, type=Path)
    cell_parser.add_argument("--cell-id", required=True)
    cell_parser.add_argument("--profile-id", required=True)
    cell_parser.add_argument("--output", required=True, type=Path)
    cell_parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args(argv)

    if args.command == "build-requests":
        request_manifest = build_request_manifest(args.contexts, args.prepared_manifest)
        _write_json(args.output, request_manifest)
        return 0

    if args.command == "extend-wait-requests":
        with args.base.open(encoding="utf-8") as source:
            base_manifest = json.load(source)
        extended_manifest = extend_causal_tool_resume_manifest(
            base_manifest, base_sha256=file_digest(args.base)
        )
        _write_json(args.output, extended_manifest)
        return 0

    if args.command == "execute-cell":
        with args.manifest.open(encoding="utf-8") as source:
            manifest = json.load(source)
        asyncio.run(
            execute_serving_cell(
                manifest,
                args.manifest,
                cell_id=args.cell_id,
                profile_id=args.profile_id,
                request_manifest_path=args.request_manifest,
                output=args.output,
                port=args.port,
            )
        )
        return 0

    if args.command == "validate-causal-events":
        records = []
        with args.events.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    record = json.loads(line)
                    try:
                        validate_causal_online_input(record)
                    except ManifestError as exc:
                        raise ManifestError(f"line {line_number}: {exc}") from exc
                    records.append(record)
        _write_json(
            args.output,
            {
                "schema": "ep-causal-validation-v1",
                "valid": True,
                "record_count": len(records),
            },
        )
        return 0

    with args.manifest.open(encoding="utf-8") as source:
        manifest = json.load(source)
    plan = build_plan(manifest, args.manifest)
    _write_json(args.output, plan)
    if args.command == "plan":
        return 0
    selected = set(args.cell_id or [])
    cells = [
        cell for cell in plan["cells"] if not selected or cell["cell_id"] in selected
    ]
    _require(cells, "no cells selected")
    for cell in cells:
        _require(cell["execution_state"] == "ready", f"{cell['cell_id']} is pending")
        _require(cell["argv"], f"{cell['cell_id']} has no runner argv")
        subprocess.run(cell["argv"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
