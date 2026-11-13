from __future__ import annotations

import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from tools.benchmarking.probe import digest_file, request
from tools.benchmarking.resources import sample_resources, verify_resources
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
    resets = (
        json.loads((output / "resets.json").read_text())["events"]
        if admission["schema"] == "benchmark-trace-v2"
        else []
    )
    cached = {
        label: record["response"]["usage"]["prompt_tokens_details"]["cached_tokens"]
        for label, record in by_label.items()
    }
    native = admission["system"] in ("native-arc", "wait-hint")
    required = {"cold", "resident", "returning", "cpu-fit-return"}
    pressure = admission["case"] not in ("T1", "calibration", "I1")
    if pressure:
        required.update(("abandoned", "after-competition-return", "unhinted-return"))
    result = {
        "phases_present": required <= by_label.keys(),
        "gpu_reuse_observed": cached.get("resident", 0) > 0,
        "cpu_restore_observed": cached.get("cpu-fit-return", 0) > 0,
        "ready_advice_observed": any(
            (event["acknowledgement"].get("ready_cpu_prefix_keys") or 0) > 0
            for event in events
        ),
        "native_advice_required": native and admission["case"] not in ("O1", "O2"),
        "return_cached_tokens": cached.get("after-competition-return"),
        "unhinted_cached_tokens": cached.get("unhinted-return"),
        "competing_prefixes": sum(
            label.startswith("competitor-") for label in by_label
        ),
        "natural_wait_prevalence": "not measured by this controlled fixture",
    }
    if admission["schema"] == "benchmark-trace-v1":
        result["cached_tokens_are_cpu_only_after_displacement"] = True
    else:
        result["cpu_return_follows_gpu_displacement"] = any(
            not reset["external"]
            and reset["response"].get("success")
            and by_label.get("returning", {}).get("completed_ns", float("inf"))
            <= reset["completed_ns"]
            <= by_label.get("cpu-fit-return", {}).get("started_ns", 0)
            for reset in resets
        )
    return result


def acquire_trace(
    base: str,
    namespace: dict[str, str],
    system: str,
    case: str,
    output: Path,
    seconds: int,
    cpu_bytes: int,
    pressure_prefixes: int = 24,
    session: int | None = None,
    load_intervals: dict[str, float] | None = None,
) -> dict[str, Any]:
    if output.exists() or not 0 < seconds <= 7200:
        raise ValueError("trace requires a new output and a bounded duration")
    output.mkdir(parents=True)
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    hints: list[dict[str, Any]] = []
    resets: list[dict[str, Any]] = []
    offers: list[dict[str, Any]] = []
    sampled = started
    sample_count = 0
    incarnation = digest_file(Path(__file__))[:16] + output.parent.name[-16:]
    specification: dict[str, Any] = {
        "schema": "benchmark-trace-v2",
        "case": case,
        "system": system,
        "seconds": seconds,
        "cpu_bytes": cpu_bytes,
        "prompt_tokens": 3072,
        "pressure_prompt_tokens": 1024,
        "pressure_policy": "frozen per-model count; observed store-byte ratio establishes or refuses CPU pressure coverage",
        "pressure_prefixes": pressure_prefixes,
        "hint_policy": "causal completion then tool-wait; refresh returning session after each competing prefix; abandoned session never returns",
        "displacement": "explicit GPU reset retains external cache; full reset between independent phases",
        "timing": "fixed-work completion; offered/started/completed times retained, no capacity claim",
        "load_intervals": load_intervals,
        "load_phase_requests": {"low": 2, "near": 3, "overload": 4},
    }
    write_record(output / "admission.json", specification)

    def reset(external: bool) -> None:
        suffix = "?reset_external=true" if external else ""
        response = request(base + "/reset_prefix_cache" + suffix, {})
        resets.append(
            {
                "external": external,
                "completed_ns": time.monotonic_ns(),
                "response": response,
            }
        )
        if not response["success"]:
            raise ValueError("trace cache reset refused")

    def generate(
        label: str,
        seed: int,
        generated: int = 8,
        register: bool = False,
        prompt_tokens: int = 3072,
        logprobs: bool = False,
    ) -> dict[str, Any]:
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 2:
            raise TimeoutError("trace acquisition deadline")
        payload: dict[str, Any] = {
            "model": "qualification",
            "prompt": list(
                range(1000 + seed * 1000, 1000 + seed * 1000 + prompt_tokens)
            ),
            "temperature": 0,
            "seed": 0,
            "max_tokens": generated,
            "return_token_ids": True,
        }
        if logprobs:
            payload["logprobs"] = 5
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
        if session is not None:
            write_record(output / "resource-start.json", sample_resources(session))
        reset(True)
        (output / "metrics-before.txt").write_text(metrics(base))
        records.append(generate("cold", 0))
        records.append(generate("resident", 0))
        reset(True)
        baseline = metrics(base)
        pressure_tokens = 3072 if case in ("T1", "calibration", "D1") else 1024
        returning = generate(
            "returning", 1, register=case != "O1", prompt_tokens=pressure_tokens
        )
        records.append(returning)
        stored = max(
            0,
            metric_total(metrics(base), "vllm:kv_offload_store_bytes_total")
            - metric_total(baseline, "vllm:kv_offload_store_bytes_total"),
        )
        trace_result["observed_prefix_store_bytes"] = stored
        observed = (returning["response"].get("kv_transfer_params") or {}).get(
            "wait_hint", {}
        )
        trace_result["cache_layout"] = observed.get("cache_layout")
        if case not in ("O1", "O2"):
            publish(returning, 0)
        reset(False)
        records.append(generate("cpu-fit-return", 1, prompt_tokens=pressure_tokens))
        if case == "calibration":
            samples = []
            for index in range(4):
                reset(True)
                record = generate("calibration-" + str(index), index + 51, 32)
                records.append(record)
                samples.append((record["completed_ns"] - record["started_ns"]) / 1e9)
            service = statistics.median(samples)
            trace_result["load_intervals"] = {
                phase: service / load for phase, load in (("low", 0.3), ("near", 0.8))
            }
            trace_result["load_intervals"]["overload"] = 0.0
        if case == "I1":
            for label, external in (
                ("invariance-fresh-a", True),
                ("invariance-fresh-b", True),
                ("invariance-resident", None),
                ("invariance-restored", False),
                ("invariance-fresh-c", True),
            ):
                if external is not None:
                    reset(external)
                records.append(generate(label, 4, 32, logprobs=True))
            trace_result["invariance"] = invariance_observation(records, system)
        if case not in ("T1", "calibration", "I1"):
            abandoned = generate(
                "abandoned", 2, register=case != "O1", prompt_tokens=pressure_tokens
            )
            records.append(abandoned)
            if case not in ("O1", "O2"):
                publish(abandoned, 0)
            count = pressure_prefixes
            trace_result["competing_prefixes"] = count
            for index in range(count):
                if case not in ("O1", "O2"):
                    publish(returning, index + 1)
                records.append(
                    generate(
                        "competitor-" + str(index),
                        index + 3,
                        prompt_tokens=pressure_tokens,
                    )
                )
                reset(False)
                if session is not None and time.monotonic() - sampled >= 30:
                    write_record(
                        output / f"resource-{sample_count:04d}.json",
                        sample_resources(session),
                    )
                    sample_count += 1
                    sampled = time.monotonic()
            records.append(
                generate("after-competition-return", 1, prompt_tokens=pressure_tokens)
            )
            reset(False)
            records.append(
                generate("unhinted-return", 3, prompt_tokens=pressure_tokens)
            )
        if case in ("T3", "R1", "A1", "S1"):
            if load_intervals is None or set(load_intervals) != {
                "low",
                "near",
                "overload",
            }:
                raise ValueError(
                    "load phases require frozen native-control calibration"
                )
            for phase, interval in load_intervals.items():
                if not 0 <= interval < 120 or (interval == 0 and phase != "overload"):
                    raise ValueError("invalid offered-arrival interval")
                reset(True)
                launch = time.monotonic()
                launch_ns = time.monotonic_ns()
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
                    futures = []
                    offset = len(offers)
                    for index in range(specification["load_phase_requests"][phase]):
                        due = launch + index * interval
                        offers.append(
                            {
                                "label": phase + "-" + str(index),
                                "offered_ns": launch_ns + int(index * interval * 1e9),
                                "status": "offered",
                            }
                        )
                        time.sleep(max(0, due - time.monotonic()))
                        futures.append(
                            (
                                index,
                                workers.submit(
                                    generate, phase + "-" + str(index), index + 51, 32
                                ),
                            )
                        )
                    for index, future in futures:
                        try:
                            record = future.result()
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            offers[offset + index].update(
                                status="failed", error=str(error)
                            )
                            continue
                        record["offered_ns"] = launch_ns + int(index * interval * 1e9)
                        record["load_phase"] = phase
                        records.append(record)
                        offers[offset + index]["status"] = "complete"
            if any(offer["status"] != "complete" for offer in offers):
                raise ValueError("offered load includes failed requests")
            reset(False)
            records.append(generate("recovery", 1))
        if case == "sustained":
            index = 0
            while seconds - (time.monotonic() - started) > 60:
                prompt_tokens = (1024, 3072, 6144)[index % 3]
                generated = (8, 32, 64)[index % 3]
                seed = index % 40 + 40 * ((index // 128) % 2)
                entry = generate(
                    "soak-" + str(index), seed, generated, True, prompt_tokens
                )
                records.append(entry)
                publish(entry, 0)
                reset(False)
                if index % 5 != 0 and seconds - (time.monotonic() - started) > 30:
                    returned = generate(
                        "soak-return-" + str(index),
                        seed,
                        generated,
                        prompt_tokens=prompt_tokens,
                    )
                    records.append(returned)
                    if (
                        entry["response"]["choices"][0]["token_ids"]
                        != returned["response"]["choices"][0]["token_ids"]
                    ):
                        raise ValueError("sustained restored continuation differs")
                index += 1
                if session is not None and time.monotonic() - sampled >= 30:
                    write_record(
                        output / f"resource-{sample_count:04d}.json",
                        sample_resources(session),
                    )
                    sample_count += 1
                    sampled = time.monotonic()
            trace_result["active_seconds"] = time.monotonic() - started
            quiet_started = time.monotonic()
            time.sleep(max(0, seconds - (quiet_started - started) - 2))
            trace_result["quiet_seconds"] = time.monotonic() - quiet_started
            if hints:
                previous = hints[-1]
                response = request(base + "/v1/wait-hints", previous["hint"])
                if response.get("metadata") != previous["acknowledgement"].get(
                    "metadata"
                ):
                    raise ValueError("quiet replay changed retained metadata")
                hints.append({**previous, "acknowledgement": response})
            trace_result["operating_seconds"] = time.monotonic() - started
        (output / "metrics-after.txt").write_text(metrics(base))
        trace_result["observed_store_bytes_during_trace"] = max(
            0,
            metric_total(
                (output / "metrics-after.txt").read_text(),
                "vllm:kv_offload_store_bytes_total",
            )
            - metric_total(
                (output / "metrics-before.txt").read_text(),
                "vllm:kv_offload_store_bytes_total",
            ),
        )
        trace_result["complete"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        trace_result["error"] = str(error)
    finally:
        if session is not None:
            write_record(output / "resource-end.json", sample_resources(session))
        for index, record in enumerate(records):
            write_record(output / f"request-{index:03d}.json", record)
        write_record(output / "hints.json", {"events": hints})
        write_record(output / "resets.json", {"events": resets})
        write_record(output / "load-offers.json", {"offers": offers})
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


def invariance_observation(
    records: list[dict[str, Any]], system: str
) -> dict[str, Any]:
    selected = {
        record["label"]: record
        for record in records
        if record["label"].startswith("invariance-")
    }
    names = ("fresh-a", "fresh-b", "resident", "restored", "fresh-c")
    if set(selected) != {"invariance-" + name for name in names}:
        raise ValueError("invariance diagnostic missing a condition")
    reference = selected["invariance-fresh-b"]
    result = {}
    for name in names:
        record = selected["invariance-" + name]
        choice = record["response"]["choices"][0]
        logprobs = choice.get("logprobs") or {}
        if (
            record["request"] != reference["request"]
            or not logprobs.get("top_logprobs")
            or len(logprobs["top_logprobs"]) != len(choice["token_ids"])
        ):
            raise ValueError("invariance parameters or logprobs missing")
        cached = record["response"]["usage"]["prompt_tokens_details"]["cached_tokens"]
        expected_cached = name == "resident" or (
            name == "restored" and system != "gpu-only"
        )
        if (cached > 0) != expected_cached:
            raise ValueError("invariance cache condition not exercised")
        result[name] = {
            "cached_tokens": cached,
            "matches_fresh_b": choice["token_ids"]
            == reference["response"]["choices"][0]["token_ids"],
        }
    return result


def verify_trace(output: Path) -> dict[str, Any]:
    admission = json.loads((output / "admission.json").read_text())
    terminal: dict[str, Any] = json.loads((output / "terminal.json").read_text())
    if (
        admission["schema"] not in ("benchmark-trace-v1", "benchmark-trace-v2")
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
        if admission["case"] == "I1" and terminal.get(
            "invariance"
        ) != invariance_observation(records, admission["system"]):
            raise ValueError("invariance summary differs from observations")
        if not {"cold", "resident", "returning", "cpu-fit-return"} <= labels.keys():
            raise ValueError("trace missing mandatory phases")
        if (
            labels["returning"]["response"]["choices"][0]["token_ids"]
            != labels["cpu-fit-return"]["response"]["choices"][0]["token_ids"]
        ):
            raise ValueError("trace restored continuation differs")
        if terminal["elapsed_seconds"] > admission["seconds"] + 15:
            raise ValueError("trace exceeded its bound")
        if admission["case"] == "sustained":
            returns = [
                record
                for label, record in labels.items()
                if label.startswith("soak-return-")
            ]
            if not returns:
                raise ValueError("sustained trace missing restored continuations")
            for returned in returns:
                original = labels[returned["label"].replace("soak-return-", "soak-")]
                parameters = {
                    key: value
                    for key, value in original["request"].items()
                    if key != "kv_transfer_params"
                }
                if (
                    parameters != returned["request"]
                    or original["response"]["choices"][0]["token_ids"]
                    != returned["response"]["choices"][0]["token_ids"]
                ):
                    raise ValueError("sustained restored continuation differs")
        if admission["case"] == "calibration":
            service = statistics.median(
                (
                    labels["calibration-" + str(index)]["completed_ns"]
                    - labels["calibration-" + str(index)]["started_ns"]
                )
                / 1e9
                for index in range(4)
            )
            if terminal["load_intervals"] != {
                **{
                    phase: service / load
                    for phase, load in (("low", 0.3), ("near", 0.8))
                },
                "overload": 0.0,
            }:
                raise ValueError("load calibration differs from native observations")
        if admission["schema"] == "benchmark-trace-v2" and admission["case"] in (
            "T3",
            "R1",
            "A1",
            "S1",
        ):
            offers = json.loads((output / "load-offers.json").read_text())["offers"]
            if len(offers) != 9 or len({offer["label"] for offer in offers}) != 9:
                raise ValueError("offered-load denominator differs")
            for offer in offers:
                record = labels[offer["label"]]
                if (
                    offer["status"] != "complete"
                    or record["offered_ns"] != offer["offered_ns"]
                    or record["started_ns"] < record["offered_ns"]
                ):
                    raise ValueError(
                        "offered-load request missing or started before arrival"
                    )
    if terminal["activity"] != trace_activity(output):
        raise ValueError("trace activity differs from observations")
    for path in output.glob("resource-*.json"):
        verify_resources(path)
    return terminal
