# SPDX-License-Identifier: Apache-2.0
"""Aggregate staged environment failures before any model process starts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from benchmarks.eviction_prefetch_campaign import digest, file_digest
from benchmarks.native_child_runtime import verify_child_inventory
from benchmarks.wait_hint_serving_diagnostic import verify_binary_inventory

SCHEMA = "ep-staged-environment-preflight-v1"
REPORT_SCHEMA = "ep-staged-environment-preflight-report-v1"


class PreflightError(ValueError):
    pass


class CheckFailure(PreflightError):
    def __init__(self, message: str, evidence: object):
        super().__init__(message)
        self.evidence = evidence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    _require(path.is_file(), f"{label} is missing")
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _git_identity(path: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "path": str(path.resolve()),
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "tracked_clean": not git("status", "--porcelain", "--untracked-files=no"),
    }


def _runtime_environment(context: Mapping[str, object]) -> dict[str, str]:
    environment = os.environ.copy()
    pythonpath = context.get("pythonpath")
    _require(isinstance(pythonpath, list) and pythonpath, "context PYTHONPATH")
    _require(
        all(
            isinstance(path, str)
            and path.startswith("/home/")
            and "/Users/" not in path
            and "/System/Volumes/" not in path
            for path in pythonpath
        ),
        "context contains a non-WSL execution path",
    )
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    overrides = context.get("environment", {})
    _require(isinstance(overrides, Mapping), "context environment must be an object")
    environment.update({str(key): str(value) for key, value in overrides.items()})
    return environment


def _run_probe(
    context: Mapping[str, object], code: str, *, timeout: int = 60
) -> dict[str, object]:
    argv = [str(context["interpreter"]), "-c", code]
    started_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(context["cwd"]),
            env=_runtime_environment(context),
            capture_output=True,
            text=False,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        stdout_bytes = completed.stdout
        stderr_bytes = completed.stderr
        error = None
    except (OSError, subprocess.TimeoutExpired) as exc:
        returncode = None
        stdout_bytes = getattr(exc, "stdout", None)
        stderr_bytes = getattr(exc, "stderr", None)
        error = repr(exc)
    completed_ns = time.monotonic_ns()
    return {
        "argv": argv,
        "cwd": str(context["cwd"]),
        "pythonpath": list(context["pythonpath"]),
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "duration_ns": completed_ns - started_ns,
        "returncode": returncode,
        **_stream_evidence(stdout_bytes, stderr_bytes),
        "error": error,
    }


def _stream_evidence(stdout: object, stderr: object) -> dict[str, object]:
    """Keep exact captured bytes separately from the human-readable projection."""
    evidence: dict[str, object] = {}
    decode_errors = []
    for name, value in (("stdout", stdout), ("stderr", stderr)):
        raw = (
            value
            if isinstance(value, bytes)
            else (value.encode("utf-8") if isinstance(value, str) else b"")
        )
        try:
            display = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            display = raw.decode("utf-8", errors="replace")
            decode_errors.append({"stream": name, "error": str(exc)})
        evidence[name] = display
        evidence[f"{name}_raw"] = {
            "encoding": "base64",
            "data": base64.b64encode(raw).decode("ascii"),
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    evidence["decode_errors"] = decode_errors
    return evidence


def _probe_payload(probe: Mapping[str, object], label: str) -> Mapping[str, object]:
    """Retain command evidence even when launch, parsing or validation fails."""
    try:
        _require(probe["returncode"] == 0, f"{label} failed")
        _require(not probe.get("decode_errors"), f"{label} output is not UTF-8")
        result = json.loads(str(probe["stdout"]).splitlines()[-1])
        _require(isinstance(result, Mapping), f"{label} result must be an object")
        return result
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        raise CheckFailure(str(exc), {"probe": dict(probe)}) from exc


def _counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        "planned": len(records),
        **{
            state: sum(item.get("status") == state for item in records)
            for state in ("pass", "fail", "blocked", "not_run")
        },
    }


def _check_input_closure(manifest: Mapping[str, object]) -> dict[str, object]:
    inputs = manifest.get("inputs")
    _require(isinstance(inputs, list) and inputs, "preflight input closure")
    verified = []
    errors = []
    seen = set()
    for index, binding in enumerate(inputs):
        if not isinstance(binding, Mapping):
            errors.append(f"input[{index}] is not an object")
            continue
        path = Path(str(binding.get("runtime_path")))
        if (
            not path.is_absolute()
            or "/Users/" in str(path)
            or "/System/Volumes/" in str(path)
        ):
            errors.append(f"input[{index}] is not a WSL execution path: {path}")
            continue
        if path in seen:
            errors.append(f"duplicate runtime input path: {path}")
            continue
        seen.add(path)
        if not path.is_file():
            errors.append(f"missing runtime input: {path}")
            continue
        actual = file_digest(path)
        if actual != binding.get("sha256"):
            errors.append(f"runtime input hash mismatch: {path}")
            continue
        verified.append({"path": str(path), "sha256": actual})
    if errors:
        raise CheckFailure(
            "; ".join(errors),
            {"verified": verified, "errors": errors},
        )
    return {"verified": verified, "verified_count": len(verified)}


def _check_context_identity(context: Mapping[str, object]) -> dict[str, object]:
    source = context.get("source")
    runtime = context.get("runtime")
    _require(isinstance(source, Mapping), "context source identity")
    _require(isinstance(runtime, Mapping), "context runtime identity")
    source_actual = _git_identity(Path(str(source.get("path"))))
    runtime_actual = _git_identity(Path(str(runtime.get("path"))))
    for expected, actual, label in (
        (source, source_actual, "source"),
        (runtime, runtime_actual, "runtime"),
    ):
        _require(
            actual["revision"] == expected.get("revision")
            and actual["tree"] == expected.get("tree")
            and actual["tracked_clean"] is True,
            f"context {label} revision/tree/clean mismatch",
        )
    interpreter = Path(str(context.get("interpreter")))
    _require(interpreter.is_file() and os.access(interpreter, os.X_OK), "interpreter")
    return {
        "source": source_actual,
        "runtime": runtime_actual,
        "interpreter": str(interpreter.resolve()),
    }


def _check_inventory(context: Mapping[str, object]) -> dict[str, object]:
    base = context.get("base_binary_inventory")
    _require(isinstance(base, Mapping), "context base binary inventory")
    base_result = verify_binary_inventory(
        Path(str(base.get("path"))), str(base.get("sha256"))
    )
    child = context.get("child_binary_inventory")
    child_result = None
    if child is not None:
        _require(isinstance(child, Mapping), "context child binary inventory")
        child_result = verify_child_inventory(
            Path(str(child.get("path"))),
            str(child.get("sha256")),
            run_import_probe=False,
        )
    return {"base": base_result, "child": child_result}


def _check_native_imports(context: Mapping[str, object]) -> dict[str, object]:
    modules = context.get("native_modules")
    _require(isinstance(modules, list) and modules, "context native module list")
    _require(
        all(isinstance(name, str) and name for name in modules), "native module names"
    )
    _require(len(set(modules)) == len(modules), "duplicate native module names")
    code = (
        "import importlib,json,sys\nout={}\nerrors={}\n"
        f"mods={modules!r}\n"
        "for name in mods:\n"
        "    try:\n"
        "        out[name]=getattr(importlib.import_module(name),'__file__',None)\n"
        "    except Exception as exc:\n"
        "        errors[name]=repr(exc)\n"
        "print(json.dumps({'sys_executable':sys.executable,'modules':out,'errors':errors},sort_keys=True))\n"
    )
    probe = _run_probe(context, code)
    payload = _probe_payload(probe, "native module import probe")
    origins = payload.get("modules")
    errors = payload.get("errors")
    if (
        not isinstance(origins, Mapping)
        or not isinstance(errors, Mapping)
        or errors
        or set(origins) != set(modules)
        or not all(
            isinstance(origins.get(name), str) and origins[name] for name in modules
        )
    ):
        raise CheckFailure(
            "native module import/origin unavailable",
            {"probe": probe, "origins": payload},
        )
    return {"probe": probe, "origins": payload}


def _check_plugin(context: Mapping[str, object]) -> dict[str, object]:
    plugin = context.get("plugin")
    if plugin is None:
        return {"required": False}
    _require(isinstance(plugin, Mapping), "context plugin")
    code = (
        "import json; from pathlib import Path; "
        "from benchmarks.wait_hint_serving_diagnostic import verify_plugin_discovery; "
        f"expected=json.loads({json.dumps(json.dumps(dict(plugin)))}); "
        "r=verify_plugin_discovery(Path(expected['plugin_root']),expected); "
        "print(json.dumps(r,sort_keys=True))"
    )
    probe = _run_probe(context, code)
    result = _probe_payload(probe, "plugin disabled/enabled discovery")
    if not (
        result.get("verified") is True
        and result.get("native_loader_disabled_count") == 0
        and result.get("native_loader_enabled_names")
        == [plugin.get("entry_point_name")]
    ):
        raise CheckFailure(
            "plugin discovery result", {"probe": probe, "result": result}
        )
    return {"probe": probe, "result": result}


def _check_native_constructor(context: Mapping[str, object]) -> dict[str, object]:
    code = """
import json,torch
from vllm.config import CacheConfig,DeviceConfig,VllmConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheGroupSpec,KVCacheTensor
from vllm.v1.kv_offload.config import OffloadingCacheConfig,OffloadingConfig,OffloadingGroupConfig,OffloadingModelConfig,OffloadingParallelConfig
from semantic_offload.connector import SemanticOffloadingConnectorScheduler
from semantic_offload.spec import SemanticOffloadingSpec
config=OffloadingConfig(groups=(OffloadingGroupConfig(16,('layer',)),),worker_kv_bytes_per_block=458752,enable_kv_cache_events=False,extra_config={'cpu_bytes_to_use':268435456,'eviction_policy':'arc','instrumentation_only':True,'preservation_events':True,'prefetch_budget_fraction':0.0},engine_id='preflight',model=OffloadingModelConfig('Qwen/Qwen2.5-1.5B-Instruct','bfloat16'),cache=OffloadingCacheConfig(16,1),parallel=OffloadingParallelConfig(0,1,1,1,1,1,0,True),replicated_layout=False)
spec=SemanticOffloadingSpec(config)
vllm_config=VllmConfig(device_config=DeviceConfig('cpu'),cache_config=CacheConfig(block_size=16))
attention=FullAttentionSpec(16,num_kv_heads=2,head_size=128,dtype=torch.bfloat16)
kv=KVCacheConfig(640,[KVCacheTensor(size=458752*640,shared_by=['layer'])],[KVCacheGroupSpec(['layer'],attention)])
scheduler=SemanticOffloadingConnectorScheduler(spec,vllm_config,kv)
print(json.dumps({'spec_type':type(spec).__name__,'scheduler_config_type':type(scheduler.config).__name__,'scheduler_config_fields':list(scheduler.config._fields),'transfer_bytes_per_key':scheduler._prefetch_bytes_per_key,'allocation_bytes_per_key':scheduler._prefetch_allocation_bytes_per_key,'prefetch_trigger':scheduler._prefetch_trigger},sort_keys=True))
"""
    probe = _run_probe(context, code)
    result = _probe_payload(probe, "real native constructor probe")
    if not (
        result.get("spec_type") == "SemanticOffloadingSpec"
        and result.get("scheduler_config_type") == "SchedulerOffloadConfig"
        and result.get("transfer_bytes_per_key") == 458752
        and result.get("allocation_bytes_per_key") == 458752
        and result.get("prefetch_trigger") == "preemption"
    ):
        raise CheckFailure(
            "real native constructor result", {"probe": probe, "result": result}
        )
    return {"probe": probe, "result": result}


def _check_validators(
    manifest: Mapping[str, object], context_status: Mapping[str, str] | None = None
) -> dict[str, object]:
    validators = manifest.get("validators")
    _require(isinstance(validators, list) and validators, "preflight validators")
    results = []
    errors = []
    contexts = {
        context.get("id"): context
        for context in manifest["contexts"]
        if isinstance(context, Mapping) and isinstance(context.get("id"), str)
    }
    seen_ids: set[str] = set()
    for index, validator in enumerate(validators):
        record = {
            "index": index,
            "id": None,
            "status": "fail",
            "returncode": None,
            **_stream_evidence(None, None),
            "error": None,
        }
        results.append(record)
        if not isinstance(validator, Mapping):
            record["error"] = f"validator[{index}] is not an object"
            errors.append(record["error"])
            continue
        record["id"] = validator.get("id")
        validator_id = validator.get("id")
        if (
            not isinstance(validator_id, str)
            or not validator_id
            or validator_id in seen_ids
        ):
            record["error"] = f"missing or duplicate validator id at index {index}"
            errors.append(record["error"])
            continue
        seen_ids.add(validator_id)
        context_id = validator.get("context_id")
        if not isinstance(context_id, str):
            record["error"] = f"invalid validator context: {context_id!r}"
            errors.append(record["error"])
            continue
        context = contexts.get(validator.get("context_id"))
        if context is None:
            record["error"] = f"unknown validator context: {context_id}"
            errors.append(record["error"])
            continue
        if context_status is not None and context_status.get(context_id) != "pass":
            record.update(status="blocked", blocked_by=[f"{context_id}:identity"])
            errors.append(f"validator blocked: {validator_id}")
            continue
        argv = validator.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(arg, str) and arg for arg in argv)
            or any(
                forbidden in " ".join(map(str, argv))
                for forbidden in ("execute-cell", "execute-case", "execute-batch")
            )
        ):
            record["error"] = f"unsafe validator argv: {validator.get('id')}"
            errors.append(record["error"])
            continue
        started_ns = time.monotonic_ns()
        try:
            completed = subprocess.run(
                list(argv),
                cwd=str(context["cwd"]),
                env=_runtime_environment(context),
                capture_output=True,
                text=False,
                timeout=120,
                check=False,
            )
            record.update(
                returncode=completed.returncode,
                **_stream_evidence(completed.stdout, completed.stderr),
                status="pass" if completed.returncode == 0 else "fail",
            )
            if record["decode_errors"]:
                record.update(status="fail", error="validator output is not UTF-8")
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            record.update(
                error=repr(exc),
                **_stream_evidence(
                    getattr(exc, "stdout", None), getattr(exc, "stderr", None)
                ),
            )
        completed_ns = time.monotonic_ns()
        record.update(
            {
                "id": validator.get("id"),
                "context_id": validator.get("context_id"),
                "argv": argv,
                "started_ns": started_ns,
                "completed_ns": completed_ns,
                "duration_ns": completed_ns - started_ns,
            }
        )
        if record["status"] != "pass":
            errors.append(f"validator failed: {validator.get('id')}")
    if errors:
        raise CheckFailure(
            "; ".join(errors),
            {"results": results, "errors": errors, "counts": _counts(results)},
        )
    return {"results": results, "counts": _counts(results)}


def _check_execution_contract(manifest: Mapping[str, object]) -> dict[str, object]:
    contract = manifest.get("execution_contract")
    _require(isinstance(contract, Mapping), "preflight execution contract")
    outputs = contract.get("output_paths")
    _require(isinstance(outputs, list) and outputs, "preflight output paths")
    _require(
        len(set(outputs)) == len(outputs)
        and all(
            isinstance(path, str)
            and path.startswith("/home/")
            and not Path(path).exists()
            for path in outputs
        ),
        "output paths are duplicated, non-WSL, or not fresh",
    )
    geometry = contract.get("geometry")
    _require(
        isinstance(geometry, Mapping)
        and (
            geometry.get("gpu_blocks"),
            geometry.get("max_model_len"),
        )
        in {(320, 5120), (640, 8192)}
        and geometry.get("cpu_bytes") == 256 * 1024 * 1024
        and geometry.get("block_size_tokens") == 16
        and geometry.get("max_num_seqs") == 4
        and geometry.get("max_num_batched_tokens") == 8192,
        "execution geometry",
    )
    _require(
        int(contract.get("maximum_prompt_tokens", 0))
        + int(contract.get("max_new_tokens", 0))
        <= int(geometry["max_model_len"]),
        "workload exceeds model token limit",
    )
    supervisor = contract.get("supervisor")
    _require(isinstance(supervisor, Mapping), "supervisor binding")
    supervisor_path = Path(str(supervisor.get("path")))
    _require(
        supervisor_path.is_file()
        and supervisor.get("sha256") == file_digest(supervisor_path)
        and supervisor.get("route") == "sc-gpu"
        and supervisor.get("first_failure_stop") is True
        and supervisor.get("automatic_retries") == 0,
        "supervisor manifest/route/fail-fast binding",
    )
    return {"geometry": dict(geometry), "supervisor": dict(supervisor)}


def run_preflight(
    manifest: Mapping[str, object], manifest_path: Path
) -> dict[str, object]:
    body = dict(manifest)
    manifest_digest = body.pop("manifest_digest", None)
    _require(
        manifest.get("schema") == SCHEMA and manifest_digest == digest(body),
        "preflight manifest schema/digest",
    )
    contexts = manifest.get("contexts")
    _require(isinstance(contexts, list) and contexts, "preflight contexts")
    checks: list[dict[str, object]] = []
    status: dict[str, str] = {}

    def check(
        check_id: str,
        function: Callable[[], object],
        dependencies: Sequence[str] = (),
    ) -> None:
        blocked_by = [
            dependency
            for dependency in dependencies
            if status.get(dependency) != "pass"
        ]
        if blocked_by:
            checks.append(
                {"id": check_id, "status": "blocked", "blocked_by": blocked_by}
            )
            status[check_id] = "blocked"
            return
        started_ns = time.monotonic_ns()
        try:
            evidence = function()
            outcome = "pass"
            error = None
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as exc:
            evidence = getattr(exc, "evidence", None)
            outcome = "fail"
            error = repr(exc)
        completed_ns = time.monotonic_ns()
        checks.append(
            {
                "id": check_id,
                "status": outcome,
                "error": error,
                "started_ns": started_ns,
                "completed_ns": completed_ns,
                "duration_ns": completed_ns - started_ns,
                "evidence": evidence,
            }
        )
        status[check_id] = outcome

    check("input_closure", lambda: _check_input_closure(manifest))
    context_ids = [
        context.get("id") if isinstance(context, Mapping) else None
        for context in contexts
    ]
    reserved_ids = {value for value in context_ids if isinstance(value, str)}
    validator_context_status: dict[str, str] = {}
    for index, context in enumerate(contexts):
        supplied_id = context_ids[index]
        well_formed = (
            isinstance(supplied_id, str)
            and bool(supplied_id)
            and context_ids.count(supplied_id) == 1
        )
        context_id = supplied_id if well_formed else f"context[{index}]"
        while not well_formed and context_id in reserved_ids:
            context_id += "_"
        reserved_ids.add(context_id)
        identity_id = f"{context_id}:identity"

        def identity(context=context, well_formed=well_formed):
            _require(
                well_formed,
                "context must be an object with a unique nonempty string id",
            )
            return _check_context_identity(context)

        check(identity_id, identity)
        if isinstance(supplied_id, str):
            validator_context_status[supplied_id] = status[identity_id]
        check(
            f"{context_id}:binary_inventory",
            lambda context=context: _check_inventory(context),
            (identity_id,),
        )
        check(
            f"{context_id}:native_imports",
            lambda context=context: _check_native_imports(context),
            (identity_id,),
        )
        check(
            f"{context_id}:plugin_discovery",
            lambda context=context: _check_plugin(context),
            (identity_id,),
        )
        check(
            f"{context_id}:native_constructor",
            lambda context=context: _check_native_constructor(context),
            (identity_id,),
        )
    check(
        "validators",
        lambda: _check_validators(manifest, validator_context_status),
        ("input_closure",),
    )
    check("execution_contract", lambda: _check_execution_contract(manifest))
    failed = [item["id"] for item in checks if item["status"] == "fail"]
    blocked = [item["id"] for item in checks if item["status"] == "blocked"]
    return {
        "schema": REPORT_SCHEMA,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_digest(manifest_path),
        "manifest_digest": manifest["manifest_digest"],
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "sys_executable": sys.executable,
            "cwd": str(Path.cwd()),
            "execution_route": manifest.get("execution_route"),
        },
        "checks": checks,
        "counts": _counts(checks),
        "failed_checks": failed,
        "blocked_checks": blocked,
        "valid": not failed and not blocked,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = _load_json(args.manifest, "preflight manifest")
        report = run_preflight(manifest, args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {
            "schema": REPORT_SCHEMA,
            "manifest_path": str(args.manifest.resolve()),
            "valid": False,
            "failed_checks": ["manifest"],
            "blocked_checks": [],
            "checks": [{"id": "manifest", "status": "fail", "error": repr(exc)}],
            "counts": {"planned": 1, "pass": 0, "fail": 1, "blocked": 0, "not_run": 0},
        }
    _write_json(args.output, report)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
