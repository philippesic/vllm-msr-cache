from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tools.benchmarking.server import server_command
from tools.config import benchmark_environment
from tools.process import process_identity, stop_session
from tools.runtime_verify import snapshot, write_record


def request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    call = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local-benchmark",
        },
    )
    with urllib.request.urlopen(call, timeout=30) as response:
        result: dict[str, Any] = json.load(response)
    return result


def digest_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def probe(
    root: Path,
    python: Path,
    model: Path,
    output: Path,
    seconds: int,
    system: str = "gpu-only",
    backend: str = "TRITON_ATTN",
    coding: tuple[Path, Path, str] | None = None,
) -> int:
    if not python.is_file() or not (model / "config.json").is_file():
        raise ValueError("runtime Python and a complete local model are required")
    if output.is_relative_to(root) or not 0 < seconds <= 1800:
        raise ValueError("new external output and timeout in 1..1800 required")
    if system not in ("gpu-only", "native-arc", "wait-hint", "lmcache"):
        raise ValueError("unsupported system")
    output.mkdir(parents=True, exist_ok=False)
    identities = snapshot(root, output / "source")
    model_identity = {
        str(path): digest_file(path)
        for path in sorted(model.rglob("*"))
        if path.is_file()
    }
    runtime = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata,json; print(json.dumps(dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))))",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command, namespace = server_command(python, model, system, port, backend)
    started = time.monotonic()
    write_record(
        output / "admission.json",
        {
            "schema": "benchmark-model-probe-v3",
            "command": command,
            "source_sha256": identities,
            "budget_seconds": seconds,
            "scope": "model-loading-generation-and-optional-restoration",
            "system": system,
            "model_sha256": model_identity,
            "runtime_packages": json.loads(runtime.stdout),
            "coding": None
            if coding is None
            else {
                "python": str(coding[0]),
                "dataset": str(coding[1]),
                "dataset_sha256": digest_file(coding[1]),
                "instance": coding[2],
            },
        },
    )
    result: dict[str, Any] = {
        "status": "failed",
        "schema": "benchmark-model-terminal-v3",
        "scope": "model-loading-generation-and-optional-restoration",
    }
    with (output / "execution.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=output / "source",
            env=benchmark_environment(
                output / "source", model, system in ("native-arc", "wait-hint")
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        owner = process_identity(child.pid)
        timer = None
        if owner is not None:
            timer = threading.Timer(seconds, stop_session, args=(child.pid, owner))
            timer.daemon = True
            timer.start()
        try:
            while time.monotonic() - started < seconds:
                if child.poll() is not None:
                    result["reason"] = "server_exited"
                    result["exit_code"] = child.returncode
                    break
                try:
                    request(f"http://127.0.0.1:{port}/v1/models")
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    time.sleep(1)
                    continue
                try:
                    result["response"] = request(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        {
                            "model": "qualification",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "What is 2 + 2? Answer with the number only.",
                                }
                            ],
                            "temperature": 0,
                            "max_tokens": 32,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    if (
                        result["response"]["choices"][0]["message"]["content"].strip()
                        != "4"
                    ):
                        raise ValueError("generation sanity answer differs")
                    if system in ("native-arc", "wait-hint"):
                        result["restore"] = restoration(
                            f"http://127.0.0.1:{port}", namespace, system
                        )
                        result["status"] = "restoration_observed"
                    else:
                        result["status"] = "generation_observed"
                    if coding is not None:
                        from tools.benchmarking.harness import run_swebench
                        from tools.benchmarking.proxy import local_proxy

                        harness_python, dataset, instance = coding
                        if seconds - (time.monotonic() - started) < 420:
                            raise ValueError(
                                "insufficient remaining pilot budget for acquisition and grading"
                            )
                        with local_proxy(
                            f"http://127.0.0.1:{port}",
                            system,
                            namespace,
                            output / "requests",
                        ) as endpoint:
                            result["coding"] = run_swebench(
                                output / "source",
                                harness_python,
                                dataset,
                                instance,
                                endpoint,
                                output / "coding",
                                240,
                            )
                        if result["coding"]["quality"] == "ungraded":
                            result["status"] = "failed"
                            result["reason"] = (
                                "coding acquisition or grading incomplete"
                            )
                    break
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result["status"] = "failed"
                    result["reason"] = str(error)
                    break
            else:
                result["reason"] = "deadline"
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
            if owner is not None:
                stop_session(child.pid, owner)
            child.wait()
            result["elapsed_seconds"] = time.monotonic() - started
            log.flush()
            result["artifacts_sha256"] = {
                str(path.relative_to(output)): digest_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
                and not path.is_relative_to(output / "source")
                and path != output / "admission.json"
            }
            result["admission_sha256"] = digest_file(output / "admission.json")
            write_record(output / "terminal.json", result)
    return (
        0 if result["status"] in ("generation_observed", "restoration_observed") else 1
    )


def verify_probe(output: Path) -> dict[str, Any]:
    output = output.resolve()
    admission = json.loads((output / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((output / "terminal.json").read_text())
    if (
        admission.get("schema") != "benchmark-model-probe-v3"
        or terminal.get("schema") != "benchmark-model-terminal-v3"
    ):
        raise ValueError("unsupported probe receipt schema")
    if set(admission) != {
        "schema",
        "command",
        "source_sha256",
        "budget_seconds",
        "scope",
        "system",
        "model_sha256",
        "runtime_packages",
        "coding",
    } or admission["system"] not in ("gpu-only", "native-arc", "wait-hint", "lmcache"):
        raise ValueError("invalid probe admission")
    if (
        admission["scope"] != "model-loading-generation-and-optional-restoration"
        or terminal["scope"] != admission["scope"]
        or not admission["source_sha256"]
        or not admission["model_sha256"]
        or type(admission["budget_seconds"]) is not int
        or not 0 < admission["budget_seconds"] <= 1800
    ):
        raise ValueError("invalid probe scope, inputs or budget")
    for relative, digest in admission["source_sha256"].items():
        path = (output / "source" / relative).resolve()
        if not path.is_relative_to(output / "source") or digest_file(path) != digest:
            raise ValueError("source snapshot drift")
    for filename, digest in admission["model_sha256"].items():
        if digest_file(Path(filename)) != digest:
            raise ValueError("model artifact drift")
    if terminal["admission_sha256"] != digest_file(output / "admission.json"):
        raise ValueError("admission drift")
    actual = {
        str(path.relative_to(output)): digest_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and not path.is_relative_to(output / "source")
        and path != output / "admission.json"
        and path != output / "terminal.json"
    }
    if terminal["artifacts_sha256"] != actual:
        raise ValueError("execution artifact drift")
    expected = (
        "generation_observed"
        if admission["system"] in ("gpu-only", "lmcache")
        else "restoration_observed"
    )
    if terminal["status"] != expected:
        raise ValueError(
            f"probe did not pass: {terminal.get('reason', terminal['status'])}"
        )
    if not 0 < terminal["elapsed_seconds"] <= admission["budget_seconds"] + 60:
        raise ValueError("probe exceeded execution budget")
    if terminal["response"]["choices"][0]["message"]["content"].strip() != "4":
        raise ValueError("invalid sanity response")
    if admission["system"] in ("native-arc", "wait-hint"):
        observation = terminal["restore"]
        if (
            not observation["initial"]["kv_transfer_params"]["wait_hint"]["registered"]
            or observation["acknowledgement"]["ready_cpu_prefix_keys"] <= 0
            or observation["acknowledgement"]["decision"]
            != ("refreshed" if admission["system"] == "wait-hint" else "disabled")
            or not observation["restored"]["choices"][0].get("token_ids")
            or observation["restored"]["choices"][0]["token_ids"]
            != observation["fresh"]["choices"][0]["token_ids"]
            or observation["restored"]["usage"]["prompt_tokens_details"][
                "cached_tokens"
            ]
            <= 0
            or observation["fresh"]["usage"]["prompt_tokens_details"]["cached_tokens"]
            != 0
            or observation["restored"]["choices"][0]["text"]
            != observation["fresh"]["choices"][0]["text"]
        ):
            raise ValueError("invalid restoration observations")
    if admission["coding"] is not None:
        coding = admission["coding"]
        if digest_file(Path(coding["dataset"])) != coding["dataset_sha256"] or terminal[
            "coding"
        ]["quality"] not in ("resolved", "unresolved"):
            raise ValueError("coding input drift or missing grading")
        acquisition = json.loads((output / "coding/acquisition.json").read_text())
        grading = json.loads((output / "coding/grading/terminal.json").read_text())
        reports = list(
            (output / "coding/grading/logs/run_evaluation").rglob("report.json")
        )
        if (
            acquisition != terminal["coding"]
            or grading != acquisition["grading"]
            or grading["exit_code"] != 0
            or len(reports) != 1
            or json.loads(reports[0].read_text()) != grading["report"]
            or type(grading["report"][coding["instance"]]["resolved"]) is not bool
            or acquisition["quality"]
            != (
                "resolved"
                if grading["report"][coding["instance"]]["resolved"]
                else "unresolved"
            )
        ):
            raise ValueError("coding verdict disagrees with original grader")
    return terminal


def restoration(base: str, namespace: dict[str, str], system: str) -> dict[str, Any]:
    prompt = list(range(1000, 4168))
    parameters = {
        "model": "qualification",
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 8,
        "seed": 0,
        "return_token_ids": True,
    }
    registration = {
        "schema": "wait-registration-v2",
        "scope": "probe",
        "session": "resume",
        "incarnation": "instance",
        "generation": 1,
    }
    initial = request(
        base + "/v1/completions",
        {**parameters, "kv_transfer_params": {"wait_hint": registration}},
    )
    observed = initial["kv_transfer_params"]["wait_hint"]
    if not observed["registered"]:
        raise ValueError(f"registration refused: {observed}")
    now = time.monotonic_ns()
    hint = {
        "schema": "wait-hint-v2",
        **{key: value for key, value in registration.items() if key != "schema"},
        "event": "tool-wait",
        "namespace": namespace,
        **{
            key: observed[key]
            for key in ("prefix", "boundary", "observed_ns", "clock_domain")
        },
        "published_ns": now,
        "deadline_ns": now + 60_000_000_000,
    }
    acknowledgement = request(base + "/v1/wait-hints", hint)
    if not acknowledgement["ready_cpu_prefix_keys"] or acknowledgement["decision"] != (
        "refreshed" if system == "wait-hint" else "disabled"
    ):
        raise ValueError(f"ready prefix not exercised: {acknowledgement}")
    if not request(base + "/reset_prefix_cache", {})["success"]:
        raise ValueError("GPU displacement refused")
    restored = request(base + "/v1/completions", parameters)
    if not request(base + "/reset_prefix_cache?reset_external=true", {})["success"]:
        raise ValueError("fresh reset refused")
    fresh = request(base + "/v1/completions", parameters)
    restored_count = restored["usage"]["prompt_tokens_details"]["cached_tokens"]
    fresh_count = fresh["usage"]["prompt_tokens_details"]["cached_tokens"]
    if (
        restored_count <= 0
        or fresh_count != 0
        or not restored["choices"][0].get("token_ids")
        or restored["choices"][0]["token_ids"] != fresh["choices"][0]["token_ids"]
        or restored["choices"][0]["text"] != fresh["choices"][0]["text"]
    ):
        raise ValueError("restored continuation differs or reuse not exercised")
    return {
        "initial": initial,
        "acknowledgement": acknowledgement,
        "restored": restored,
        "fresh": fresh,
        "scope": "forced-displacement matched continuation; not a natural workload",
    }
