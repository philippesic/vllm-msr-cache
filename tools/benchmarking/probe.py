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


def request(
    url: str, payload: dict[str, Any] | None = None, timeout: int = 30
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    call = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer local-benchmark",
        },
    )
    with urllib.request.urlopen(call, timeout=timeout) as response:
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
    coding_kind: str = "repair",
    panel: tuple[Path, list[dict[str, Any]]] | None = None,
) -> int:
    if not python.is_file() or not (model / "config.json").is_file():
        raise ValueError("runtime Python and a complete local model are required")
    if output.is_relative_to(root) or not 0 < seconds <= (108000 if panel else 1800):
        raise ValueError("new external output and supported bounded timeout required")
    if panel is not None and coding is not None:
        raise ValueError("use either a panel or a single coding probe")
    if system not in ("gpu-only", "native-arc", "wait-hint", "lmcache"):
        raise ValueError("unsupported system")
    if coding_kind not in (
        "repair",
        "regression",
        "navigation",
        "terminal",
        "grounded",
        "branching",
    ):
        raise ValueError("unsupported coding task kind")
    if coding is not None:
        harness_python, dataset, instance = coding
        if not harness_python.is_file() or not dataset.is_file():
            raise ValueError("coding requires existing harness and dataset")
        rows = json.loads(dataset.read_text())
        if sum(row.get("instance_id") == instance for row in rows) != 1:
            raise ValueError("coding requires exactly one matching dataset instance")
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
    runtime_sources = subprocess.run(
        [
            str(python),
            "-c",
            "import sysconfig,pathlib,hashlib,json,sys; base=pathlib.Path(sysconfig.get_path('purelib')); targets=json.loads(sys.argv[1]); print(json.dumps({str(base/t):hashlib.sha256((base/t).read_bytes()).hexdigest() for t in targets if (base/t).is_file()}))",
            json.dumps(
                [
                    record["target"]
                    for record in json.loads(
                        (root / "benchmarks/runtime-patches/manifest.json").read_text()
                    )
                ]
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command, namespace = server_command(python, model, system, port, backend)
    started = time.monotonic()
    write_record(
        output / "admission.json",
        {
            "schema": "benchmark-model-probe-v6"
            if panel
            else "benchmark-model-probe-v5",
            "command": command,
            "source_sha256": identities,
            "budget_seconds": seconds,
            "scope": "model-loading-generation-and-optional-restoration",
            "system": system,
            "model_sha256": model_identity,
            "runtime_packages": json.loads(runtime.stdout),
            "runtime_sources": json.loads(runtime_sources.stdout),
            **({"panel": {"python": str(panel[0]), "jobs": panel[1]}} if panel else {}),
            "coding": None
            if coding is None
            else {
                "python": str(coding[0]),
                "dataset": str(coding[1]),
                "dataset_sha256": digest_file(coding[1]),
                "instance": coding[2],
                "kind": coding_kind,
            },
        },
    )
    result: dict[str, Any] = {
        "status": "failed",
        "schema": "benchmark-model-terminal-v6"
        if panel
        else "benchmark-model-terminal-v5",
        "scope": "model-loading-generation-and-optional-restoration",
    }
    with (output / "execution.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=output / "source",
            env=benchmark_environment(
                output / "source",
                model,
                system in ("native-arc", "wait-hint"),
                system == "lmcache",
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
                    if system == "lmcache":
                        result["external_restore"] = external_restoration(
                            f"http://127.0.0.1:{port}"
                        )
                        validate_external_restoration(result["external_restore"])
                        result["status"] = "restoration_observed"
                    if panel is not None:
                        from tools.benchmarking.panel import acquire_panel

                        result["panel"] = acquire_panel(
                            root,
                            panel[0],
                            panel[1],
                            f"http://127.0.0.1:{port}",
                            namespace,
                            system,
                            output / "panel",
                            started + seconds - 20,
                            2 * 1024**3
                            if json.loads((model / "config.json").read_text()).get(
                                "model_type"
                            )
                            == "gemma4_unified"
                            else 256 * 1024**2,
                        )
                        if not result["panel"]["complete"]:
                            result["status"] = "failed"
                            result["reason"] = "panel has failed or unexecuted cells"
                    if coding is not None:
                        from tools.benchmarking.harness import run_swebench
                        from tools.benchmarking.proxy import local_proxy

                        harness_python, dataset, instance = coding
                        if seconds - (time.monotonic() - started) < (
                            600 if coding_kind == "branching" else 420
                        ):
                            raise ValueError(
                                "insufficient remaining pilot budget for acquisition and grading"
                            )
                        with local_proxy(
                            f"http://127.0.0.1:{port}",
                            system,
                            namespace,
                            output / "requests",
                            "terminus-json"
                            if coding_kind == "terminal"
                            else "mini-swe",
                        ) as endpoint:
                            if coding_kind == "branching":
                                from tools.benchmarking.branching import run_branching

                                result["coding"] = run_branching(
                                    root,
                                    harness_python,
                                    dataset,
                                    instance,
                                    endpoint,
                                    output / "coding",
                                    240,
                                )
                            elif coding_kind == "terminal":
                                from tools.benchmarking.terminal import run_terminal

                                result["coding"] = run_terminal(
                                    root,
                                    harness_python,
                                    dataset,
                                    instance,
                                    endpoint,
                                    output / "coding",
                                    240,
                                )
                            else:
                                result["coding"] = run_swebench(
                                    output / "source",
                                    harness_python,
                                    dataset,
                                    instance,
                                    endpoint,
                                    output / "coding",
                                    240,
                                    coding_kind,
                                )
                        if result["coding"]["quality"] == "ungraded":
                            result["status"] = "failed"
                            result["reason"] = (
                                "coding acquisition or grading incomplete"
                            )
                        elif result["coding"]["execution_exit"] != 0:
                            result["status"] = "failed"
                            result["reason"] = "coding acquisition did not complete"
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


def verify_probe(output: Path, *, allow_failed_panel: bool = False) -> dict[str, Any]:
    output = output.resolve()
    admission = json.loads((output / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((output / "terminal.json").read_text())
    if admission.get("schema") not in (
        "benchmark-model-probe-v3",
        "benchmark-model-probe-v4",
        "benchmark-model-probe-v5",
        "benchmark-model-probe-v6",
    ) or terminal.get("schema") != admission["schema"].replace("probe", "terminal"):
        raise ValueError("unsupported probe receipt schema")
    expected_keys = {
        "schema",
        "command",
        "source_sha256",
        "budget_seconds",
        "scope",
        "system",
        "model_sha256",
        "runtime_packages",
        "coding",
    }
    if admission["schema"] == "benchmark-model-probe-v6":
        expected_keys.add("panel")
    if admission["schema"] in (
        "benchmark-model-probe-v4",
        "benchmark-model-probe-v5",
        "benchmark-model-probe-v6",
    ):
        expected_keys.add("runtime_sources")
        if not admission["runtime_sources"]:
            raise ValueError("missing runtime source identity")
    if set(admission) != expected_keys or admission["system"] not in (
        "gpu-only",
        "native-arc",
        "wait-hint",
        "lmcache",
    ):
        raise ValueError("invalid probe admission")
    if (
        admission["scope"] != "model-loading-generation-and-optional-restoration"
        or terminal["scope"] != admission["scope"]
        or not admission["source_sha256"]
        or not admission["model_sha256"]
        or type(admission["budget_seconds"]) is not int
        or not 0
        < admission["budget_seconds"]
        <= (108000 if "panel" in admission else 1800)
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
        if admission["system"] == "gpu-only"
        or (
            admission["system"] == "lmcache"
            and admission["schema"]
            in ("benchmark-model-probe-v3", "benchmark-model-probe-v4")
        )
        else "restoration_observed"
    )
    if terminal["status"] != expected and not (
        allow_failed_panel
        and "panel" in admission
        and "panel" in terminal
        and terminal["status"] == "failed"
    ):
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
    if (
        admission["system"] == "lmcache"
        and admission["schema"]
        in ("benchmark-model-probe-v5", "benchmark-model-probe-v6")
        and "external_restore" not in terminal
    ):
        raise ValueError("missing external restoration")
    if "external_restore" in terminal:
        if admission["system"] != "lmcache":
            raise ValueError("unexpected external restoration")
        validate_external_restoration(terminal["external_restore"])
    if "panel" in admission:
        from tools.benchmarking.panel import verify_panel

        observed_panel = verify_panel(output / "panel")
        if observed_panel != terminal["panel"] or (
            not observed_panel["complete"] and not allow_failed_panel
        ):
            raise ValueError("panel incomplete or differs from observations")
    if admission["coding"] is not None:
        coding = admission["coding"]
        if digest_file(Path(coding["dataset"])) != coding["dataset_sha256"] or terminal[
            "coding"
        ]["quality"] not in ("resolved", "unresolved"):
            raise ValueError("coding input drift or missing grading")
        acquisition = json.loads((output / "coding/acquisition.json").read_text())
        if coding.get("kind", "repair") == "branching":
            from tools.benchmarking.branching import verify_branching

            if (
                verify_branching(output / "coding") != acquisition
                or acquisition != terminal["coding"]
            ):
                raise ValueError(
                    "branch acquisition differs from original observations"
                )
            return terminal
        if coding.get("kind", "repair") == "terminal":
            from tools.benchmarking.terminal import verify_terminal

            if (
                verify_terminal(output / "coding") != acquisition
                or acquisition != terminal["coding"]
            ):
                raise ValueError(
                    "terminal acquisition differs from original observations"
                )
            return terminal
        grading = json.loads((output / "coding/grading/terminal.json").read_text())
        if "image" in acquisition:
            grading_admission = json.loads(
                (output / "coding/grading/admission.json").read_text()
            )
            pinned = json.loads((output / "coding/dataset/test.json").read_text())
            if (
                acquisition["image"] != grading_admission["image"]
                or json.loads((output / "coding/image.json").read_text())
                != acquisition["image"]
                or len(pinned) != 1
                or pinned[0]["instance_id"] != coding["instance"]
                or pinned[0]["image_name"] != acquisition["image"]["id"]
            ):
                raise ValueError("coding and grader image identities differ")
        if acquisition != terminal["coding"] or grading != acquisition["grading"]:
            raise ValueError("coding acquisition differs from grading receipt")
        if coding.get("kind", "repair") in ("regression", "navigation", "grounded"):
            from tools.benchmarking.grounded import verify_grounded
            from tools.benchmarking.navigation import verify_navigation
            from tools.benchmarking.regression import verify_regression

            reader = {
                "regression": verify_regression,
                "navigation": verify_navigation,
                "grounded": verify_grounded,
            }[coding["kind"]]
            verified = reader(output / "coding/grading")
            if verified["quality"] != acquisition["quality"]:
                raise ValueError("regression task quality differs from observations")
        else:
            from tools.benchmarking.harness import verify_coding_grade

            if (
                verify_coding_grade(output / "coding/grading") != grading
                or acquisition["quality"] != grading["quality"]
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


def external_restoration(base: str) -> dict[str, Any]:
    parameters = {
        "model": "qualification",
        "prompt": list(range(1000, 4168)),
        "temperature": 0,
        "max_tokens": 8,
        "seed": 0,
        "return_token_ids": True,
    }
    initial = request(base + "/v1/completions", parameters)
    if not request(base + "/reset_prefix_cache", {})["success"]:
        raise ValueError("external probe GPU displacement refused")
    restored = request(base + "/v1/completions", parameters)
    if not request(base + "/reset_prefix_cache?reset_external=true", {})["success"]:
        raise ValueError("external probe fresh reset refused")
    fresh = request(base + "/v1/completions", parameters)
    observation = {"initial": initial, "restored": restored, "fresh": fresh}
    return observation


def validate_external_restoration(observation: dict[str, Any]) -> None:
    restored = observation["restored"]
    fresh = observation["fresh"]
    if (
        restored["usage"]["prompt_tokens_details"]["cached_tokens"] <= 0
        or fresh["usage"]["prompt_tokens_details"]["cached_tokens"] != 0
        or not restored["choices"][0].get("token_ids")
        or restored["choices"][0]["token_ids"] != fresh["choices"][0]["token_ids"]
        or restored["choices"][0]["text"] != fresh["choices"][0]["text"]
    ):
        raise ValueError("external restoration differs or reuse not exercised")
