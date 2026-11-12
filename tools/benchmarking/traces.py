from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from tools.benchmarking.probe import digest_file, request
from tools.runtime_verify import write_record


def metrics(base: str) -> str:
    call = urllib.request.Request(
        base + "/metrics", headers={"Authorization": "Bearer local-benchmark"}
    )
    with urllib.request.urlopen(call, timeout=10) as response:
        content: bytes = response.read(4 * 1024 * 1024)
        return content.decode()


def metric_total(text: str, name: str) -> float:
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.split("{", 1)[0].split(" ", 1)[0] == name
    )


def trace_activity(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    records = [
        json.loads(path.read_text()) for path in sorted(output.glob("request-*.json"))
    ]
    by_label = {record["label"]: record for record in records}
    events = json.loads((output / "hints.json").read_text())["events"]
    cached = {
        label: record["response"]["usage"]["prompt_tokens_details"]["cached_tokens"]
        for label, record in by_label.items()
    }
    native = admission["system"] in ("native-arc", "wait-hint")
    required = {"cold", "resident", "returning", "cpu-fit-return"}
    pressure = admission["case"] != "T1"
    if pressure:
        required.update(("abandoned", "after-competition-return", "unhinted-return"))
    return {
        "phases_present": required <= by_label.keys(),
        "gpu_reuse_observed": cached.get("resident", 0) > 0,
        "cpu_restore_observed": cached.get("cpu-fit-return", 0) > 0,
        "ready_advice_observed": any(
            event["acknowledgement"].get("ready_cpu_prefix_keys", 0) > 0
            for event in events
        ),
        "native_advice_required": native and admission["case"] not in ("O1", "O2"),
        "return_cached_tokens": cached.get("after-competition-return"),
        "unhinted_cached_tokens": cached.get("unhinted-return"),
        "competing_prefixes": sum(
            label.startswith("competitor-") for label in by_label
        ),
        "cached_tokens_are_cpu_only_after_displacement": True,
        "natural_wait_prevalence": "not measured by this controlled fixture",
    }


def acquire_trace(
    base: str,
    namespace: dict[str, str],
    system: str,
    case: str,
    output: Path,
    seconds: int,
    cpu_bytes: int,
    pressure_prefixes: int = 24,
) -> dict[str, Any]:
    if output.exists() or not 0 < seconds <= 7200:
        raise ValueError("trace requires a new output and a bounded duration")
    output.mkdir(parents=True)
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    hints: list[dict[str, Any]] = []
    incarnation = digest_file(Path(__file__))[:16] + output.parent.name[-16:]
    specification = {
        "schema": "benchmark-trace-v1",
        "case": case,
        "system": system,
        "seconds": seconds,
        "cpu_bytes": cpu_bytes,
        "prompt_tokens": 3072,
        "pressure_policy": "frozen per-model count; observed store-byte ratio establishes or refuses CPU pressure coverage",
        "pressure_prefixes": pressure_prefixes,
        "hint_policy": "causal completion then tool-wait; refresh returning session after each competing prefix; abandoned session never returns",
        "displacement": "explicit GPU reset retains external cache; full reset between independent phases",
        "timing": "fixed-work completion; offered/started/completed times retained, no capacity claim",
    }
    write_record(output / "admission.json", specification)

    def reset(external: bool) -> None:
        suffix = "?reset_external=true" if external else ""
        if not request(base + "/reset_prefix_cache" + suffix, {})["success"]:
            raise ValueError("trace cache reset refused")

    def generate(
        label: str, seed: int, generated: int = 8, register: bool = False
    ) -> dict[str, Any]:
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 2:
            raise TimeoutError("trace acquisition deadline")
        payload: dict[str, Any] = {
            "model": "qualification",
            "prompt": list(range(1000 + seed * 1000, 4072 + seed * 1000)),
            "temperature": 0,
            "seed": 0,
            "max_tokens": generated,
            "return_token_ids": True,
        }
        if register and system in ("native-arc", "wait-hint"):
            payload["kv_transfer_params"] = {
                "wait_hint": {
                    "schema": "wait-registration-v2",
                    "scope": "trace",
                    "session": label,
                    "incarnation": incarnation,
                    "generation": 1,
                }
            }
        before = time.monotonic_ns()
        injected = 0
        if case == "S1":
            time.sleep(0.2)
            injected = time.monotonic_ns() - before
        response = request(
            base + "/v1/completions", payload, timeout=min(90, int(remaining))
        )
        return {
            "label": label,
            "seed": seed,
            "request": payload,
            "response": response,
            "started_ns": before,
            "completed_ns": time.monotonic_ns(),
            "injected_delay_ns": injected,
        }

    def publish(record: dict[str, Any], event: int) -> None:
        observed = (record["response"].get("kv_transfer_params") or {}).get(
            "wait_hint", {}
        )
        if not observed.get("registered"):
            return
        now = time.monotonic_ns()
        registration = record["request"]["kv_transfer_params"]["wait_hint"]
        hint = {
            **{key: value for key, value in registration.items() if key != "schema"},
            "schema": "wait-hint-v2",
            "event": "tool-wait-" + str(event),
            "namespace": namespace,
            **{
                key: observed[key]
                for key in ("prefix", "boundary", "observed_ns", "clock_domain")
            },
            "published_ns": now,
            "deadline_ns": now + 60_000_000_000,
        }
        hints.append(
            {
                "after_request": record["label"],
                "event_index": event,
                "hint": hint,
                "acknowledgement": request(base + "/v1/wait-hints", hint),
            }
        )

    trace_result: dict[str, Any] = {"complete": False, "error": None}
    try:
        reset(True)
        (output / "metrics-before.txt").write_text(metrics(base))
        records.append(generate("cold", 0))
        records.append(generate("resident", 0))
        reset(True)
        baseline = metrics(base)
        returning = generate("returning", 1, register=case != "O1")
        records.append(returning)
        stored = max(
            0,
            metric_total(metrics(base), "vllm:kv_offload_store_bytes")
            - metric_total(baseline, "vllm:kv_offload_store_bytes"),
        )
        trace_result["observed_prefix_store_bytes"] = stored
        if case not in ("O1", "O2"):
            publish(returning, 0)
        reset(False)
        records.append(generate("cpu-fit-return", 1))
        if case != "T1":
            abandoned = generate("abandoned", 2, register=case != "O1")
            records.append(abandoned)
            if case not in ("O1", "O2"):
                publish(abandoned, 0)
            count = pressure_prefixes
            trace_result["competing_prefixes"] = count
            for index in range(count):
                records.append(generate("competitor-" + str(index), index + 3))
                if case not in ("O1", "O2"):
                    publish(returning, index + 1)
                reset(False)
            records.append(generate("after-competition-return", 1))
            reset(False)
            records.append(generate("unhinted-return", 3))
        if case in ("T3", "R1", "A1", "S1"):
            for phase, offered in (("low", 1), ("near", 2), ("overload", 4)):
                launch = time.monotonic_ns()
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=offered
                ) as workers:
                    futures = [
                        workers.submit(
                            generate, phase + "-" + str(index), index + 51, 64
                        )
                        for index in range(offered)
                    ]
                    for future in futures:
                        record = future.result()
                        record["offered_ns"] = launch
                        record["load_phase"] = phase
                        records.append(record)
            reset(False)
            records.append(generate("recovery", 1))
        if case == "sustained":
            index = 0
            while seconds - (time.monotonic() - started) > 5:
                entry = generate("soak-" + str(index), index % 40, register=True)
                records.append(entry)
                publish(entry, 0)
                reset(False)
                if seconds - (time.monotonic() - started) > 5:
                    records.append(generate("soak-return-" + str(index), index % 40))
                index += 1
            trace_result["operating_seconds"] = time.monotonic() - started
        (output / "metrics-after.txt").write_text(metrics(base))
        trace_result["complete"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        trace_result["error"] = str(error)
    finally:
        for index, record in enumerate(records):
            write_record(output / f"request-{index:03d}.json", record)
        write_record(output / "hints.json", {"events": hints})
        trace_result["activity"] = trace_activity(output)
        trace_result["elapsed_seconds"] = time.monotonic() - started
        trace_result["admission_sha256"] = digest_file(output / "admission.json")
        trace_result["artifacts_sha256"] = {
            path.name: digest_file(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "admission.json"
        }
        write_record(output / "terminal.json", trace_result)
    return trace_result


def verify_trace(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((output / "terminal.json").read_text())
    if (
        admission["schema"] != "benchmark-trace-v1"
        or digest_file(output / "admission.json") != terminal["admission_sha256"]
        or {
            path.name: digest_file(path)
            for path in output.iterdir()
            if path.is_file() and path.name not in ("admission.json", "terminal.json")
        }
        != terminal["artifacts_sha256"]
    ):
        raise ValueError("trace evidence drift")
    records = [
        json.loads(path.read_text()) for path in sorted(output.glob("request-*.json"))
    ]
    labels = {record["label"]: record for record in records}
    for record in records:
        usage = record["response"]["usage"]
        if (
            record["completed_ns"] < record["started_ns"]
            or usage["prompt_tokens"] != len(record["request"]["prompt"])
            or not 0
            <= usage["prompt_tokens_details"]["cached_tokens"]
            <= usage["prompt_tokens"]
            or not record["response"]["choices"][0]["token_ids"]
        ):
            raise ValueError("trace timing, token accounting or output missing")
    if terminal["complete"]:
        if not {"cold", "resident", "returning", "cpu-fit-return"} <= labels.keys():
            raise ValueError("trace missing mandatory phases")
        if (
            labels["returning"]["response"]["choices"][0]["token_ids"]
            != labels["cpu-fit-return"]["response"]["choices"][0]["token_ids"]
        ):
            raise ValueError("trace restored continuation differs")
        if terminal["elapsed_seconds"] > admission["seconds"] + 15:
            raise ValueError("trace exceeded its bound")
    if terminal["activity"] != trace_activity(output):
        raise ValueError("trace activity differs from observations")
    return terminal
