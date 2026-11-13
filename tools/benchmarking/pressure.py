from __future__ import annotations

import hashlib
import json
import random
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
from collections import Counter
from pathlib import Path
from typing import Any

from tools.benchmarking.probe import digest_file, request
from tools.benchmarking.server import server_command
from tools.benchmarking.traces import metric_total, metrics
from tools.benchmarking.workload import Session, cohort, schedule
from tools.config import benchmark_environment
from tools.process import process_identity, stop_session
from tools.runtime_verify import snapshot, write_record

DEFINITION = Path("benchmarks/pressure.json")
DEFINITION_SCHEMA = "pressure-definition-v1"
CELL_SCHEMA = "pressure-cell-v1"
REPORT_SCHEMA = "pressure-report-v1"
TOKEN_RANGE = (1000, 30000)
PROBE_TOKENS = 2112
ENDPOINT_ARMS = ("native-arc", "wait-hint")
MAX_MODEL_LEN = 6144
SIMULATOR_FIELDS = {
    "groups",
    "batch_tokens",
    "prefill_tokens_per_second",
    "decode_seconds_per_token",
    "gpu_tokens",
    "gpu_prefix_reuse",
}


def load_definition(root: Path) -> dict[str, Any]:
    raw: dict[str, Any] = json.loads((root / DEFINITION).read_text())
    fields = {"schema", "workload", "seeds", "models", "arms", "profiles", "controls"}
    if set(raw) != fields or raw["schema"] != DEFINITION_SCHEMA:
        raise ValueError("unsupported pressure definition")
    for profile in raw["profiles"].values():
        if set(profile["arms"]) - set(raw["arms"]) or profile["cell_seconds"] > 1800:
            raise ValueError("profile arms or cell budget outside the definition")
    return raw


def workload_for(definition: dict[str, Any], model: str) -> dict[str, Any]:
    override = definition["models"][model].get("workload", {})
    if set(override) - set(definition["workload"]):
        raise ValueError("model workload override names unknown fields")
    return {**definition["workload"], **override}


def model_identity(definition: dict[str, Any], model: str) -> str:
    material = {
        "workload": workload_for(definition, model),
        "model": {
            key: value
            for key, value in definition["models"][model].items()
            if key not in SIMULATOR_FIELDS
        },
        "arms": definition["arms"],
        "controls": definition["controls"],
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def sessions_for(definition: dict[str, Any], seed: int, model: str) -> list[Session]:
    return schedule({**workload_for(definition, model), "seed": seed})


def covers_cohorts(definition: dict[str, Any], seed: int, model: str) -> bool:
    returning = [s for s in sessions_for(definition, seed, model) if len(s.turns) > 1]
    return (
        sum(s.hinted for s in returning) >= 2
        and any(not s.hinted for s in returning)
        and any(s.abandons for s in returning)
    )


def profile_seeds(
    definition: dict[str, Any], profile: str, model: str, chosen: list[int]
) -> list[int]:
    if chosen:
        return chosen
    seeds = definition["seeds"][profile]
    return list(seeds[model] if isinstance(seeds, dict) else seeds)


def pool_bytes(model: dict[str, Any], pool: str) -> int:
    if pool == "none":
        return 0
    return int(model["cpu_bytes" if pool == "pressure" else "oracle_cpu_bytes"])


def pool_keys(model: dict[str, Any], pool: str) -> int:
    return pool_bytes(model, pool) // int(model["cpu_chunk_bytes"])


def replay_scenario(
    definition: dict[str, Any], model_id: str, seeds: list[int]
) -> dict[str, Any]:
    model = definition["models"][model_id]
    return {
        "name": f"pressure-{model_id}",
        "model": model_id,
        "geometry": {
            "block_tokens": model["block_tokens"],
            "groups": model["groups"],
            "gpu_blocks": model["gpu_tokens"] // model["block_tokens"],
            "cpu_keys": pool_keys(model, "pressure"),
            "max_running": 2,
            "batch_tokens": model["batch_tokens"],
            "prefill_tokens_per_second": model["prefill_tokens_per_second"],
            "decode_seconds_per_token": model["decode_seconds_per_token"],
            "store_seconds": 0.02,
            "hint_delay_seconds": 0.02,
            "hint_deadline_seconds": 60.0,
            "hold_share": 0.5,
            "gpu_prefix_reuse": model["gpu_prefix_reuse"],
        },
        "schedule": {**workload_for(definition, model_id), "seed": seeds[0]},
        "seeds": len(seeds),
    }


def gpu_idle() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    used, utilization = (int(value) for value in query.stdout.split(","))
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.split()
    return {
        "memory_used_mib": used,
        "utilization_percent": utilization,
        "compute_processes": len(processes),
        "idle": used < 1024 and utilization < 10 and not processes,
    }


def offload_metrics(base: str) -> dict[str, float]:
    text = metrics(base)
    names = {
        line.split("{", 1)[0].split(" ", 1)[0]
        for line in text.splitlines()
        if line.startswith(("vllm:kv_offload", "vllm:prefix_cache", "vllm:external"))
    }
    return {name: metric_total(text, name) for name in sorted(names)}


def counter(observed: dict[str, float], fragment: str) -> float:
    return sum(
        value
        for name, value in observed.items()
        if fragment in name and not name.endswith(("_created", "_bucket"))
    )


def token_ids(rng: random.Random, count: int) -> list[int]:
    return [rng.randrange(*TOKEN_RANGE) for _ in range(count)]


def generate(
    base: str, prompt: list[int], generated: int, extra: dict[str, Any], timeout: float
) -> dict[str, Any]:
    payload = {
        "model": "qualification",
        "prompt": prompt,
        "max_tokens": generated,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "return_token_ids": True,
        **extra,
    }
    started = time.monotonic_ns()
    response = request(base + "/v1/completions", payload, timeout=max(1, int(timeout)))
    return {
        "started_ns": started,
        "completed_ns": time.monotonic_ns(),
        "response": response,
    }


def registration(
    arm: dict[str, Any], session: str, incarnation: str, turn: int
) -> dict[str, Any]:
    if arm["system"] not in ENDPOINT_ARMS:
        return {}
    return {
        "kv_transfer_params": {
            "wait_hint": {
                "schema": "wait-registration-v2",
                "scope": "pressure",
                "session": session,
                "incarnation": incarnation,
                "generation": turn + 1,
            }
        }
    }


def publish(base: str, registered: dict[str, Any], event: str) -> dict[str, Any]:
    observed = (registered["response"].get("kv_transfer_params") or {}).get(
        "wait_hint", {}
    )
    if not observed.get("registered"):
        return {"published": False, "registration": observed}
    request_registration = registered["registration"]
    now = time.monotonic_ns()
    hint = {
        **{k: v for k, v in request_registration.items() if k != "schema"},
        "schema": "wait-hint-v2",
        "event": event,
        "prefix": observed["prefix"],
        "boundary": observed["boundary"],
        "namespace": registered["namespace"],
        "observed_ns": observed["observed_ns"],
        "published_ns": now,
        "deadline_ns": now + 60_000_000_000,
        "clock_domain": observed["clock_domain"],
    }
    return {
        "published": True,
        "hint": hint,
        "acknowledgement": request(base + "/v1/wait-hints", hint),
    }


def replay_sessions(
    base: str,
    arm: dict[str, Any],
    namespace: dict[str, str],
    sessions: list[Session],
    seed: int,
    deadline: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    lock = threading.Lock()
    origin = time.monotonic()
    incarnation = f"seed-{seed}-{time.monotonic_ns()}"

    def run(session: Session) -> None:
        rng = random.Random(f"{seed}:{session.name}")
        sequence: list[int] = []
        try:
            time.sleep(max(0.0, origin + session.start - time.monotonic()))
            for index, turn in enumerate(session.turns):
                if index:
                    time.sleep(turn.gap)
                prompt = sequence[: turn.shared] + token_ids(
                    rng, turn.prompt - turn.shared
                )
                extra = registration(arm, session.name, incarnation, index)
                remaining = deadline - time.monotonic()
                if remaining < 5:
                    raise TimeoutError("cell deadline")
                result = generate(
                    base, prompt, turn.generated, extra, min(180, remaining)
                )
                choice = result["response"]["choices"][0]
                sequence = prompt + list(choice["token_ids"])
                last = index + 1 == len(session.turns)
                wants_hint = session.hinted and (not last or session.abandons)
                row = {
                    "session": session.name,
                    "turn": index,
                    "cohort": cohort(session),
                    "abandons": session.abandons,
                    "prompt_tokens": turn.prompt,
                    "shared_tokens": turn.shared,
                    "generated_tokens": turn.generated,
                    "gap_seconds": turn.gap,
                    "started_ns": result["started_ns"],
                    "completed_ns": result["completed_ns"],
                    "usage": result["response"]["usage"],
                    "metrics": result["response"].get("metrics"),
                    "registration": (
                        result["response"].get("kv_transfer_params") or {}
                    ).get("wait_hint"),
                    "output_tokens": len(choice["token_ids"]),
                    "wants_hint": wants_hint,
                    "hint": None,
                }
                if wants_hint and extra:
                    row["hint"] = publish(
                        base,
                        {
                            **result,
                            "registration": extra["kv_transfer_params"]["wait_hint"],
                            "namespace": namespace,
                        },
                        f"wait-{index}",
                    )
                with lock:
                    rows.append(row)
        except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
            with lock:
                errors.append(f"{session.name}: {error}")

    threads = [threading.Thread(target=run, args=(s,), daemon=True) for s in sessions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(1.0, deadline - time.monotonic() + 5))
    if any(thread.is_alive() for thread in threads):
        errors.append("session threads exceeded the cell deadline")
    return sorted(rows, key=lambda row: row["started_ns"]), errors


def reset(base: str, external: bool) -> dict[str, Any]:
    suffix = "?reset_external=true" if external else ""
    response = request(base + "/reset_prefix_cache" + suffix, {})
    if not response.get("success", True):
        raise ValueError("cache reset refused")
    return response


def restoration_check(
    base: str, arm: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    prompt = token_ids(random.Random("restoration-probe"), PROBE_TOKENS)
    before = offload_metrics(base)
    first = generate(base, prompt, 8, {}, 120)
    time.sleep(2)
    stored = offload_metrics(base)
    reset(base, False)
    returned = generate(base, prompt, 8, {}, 120)
    cached = returned["response"]["usage"]["prompt_tokens_details"]["cached_tokens"]
    store_bytes = counter(stored, "store_bytes") - counter(before, "store_bytes")
    keys = store_bytes / int(model["cpu_chunk_bytes"])
    reset(base, True)
    return {
        "prompt_tokens": PROBE_TOKENS,
        "first_cached_tokens": first["response"]["usage"]["prompt_tokens_details"][
            "cached_tokens"
        ],
        "restored_cached_tokens": cached,
        "restore_observed": cached >= int(model["hit_alignment"]),
        "stored_keys": keys,
        "keys_per_token": keys / PROBE_TOKENS if keys else None,
        "same_output": first["response"]["choices"][0]["token_ids"]
        == returned["response"]["choices"][0]["token_ids"],
    }


def wait_ready(base: str, child: subprocess.Popen[bytes], seconds: float) -> float:
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        if child.poll() is not None:
            raise ValueError(f"server exited with {child.returncode}")
        try:
            request(base + "/v1/models")
            return time.monotonic() - started
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2)
    raise TimeoutError("server did not become ready")


def serve_arm(
    source: Path,
    output: Path,
    python: Path,
    model_path: Path,
    model_id: str,
    arm_name: str,
    arm: dict[str, Any],
    definition: dict[str, Any],
    seeds: list[int],
    cell_seconds: int,
) -> list[dict[str, Any]]:
    model = definition["models"][model_id]
    idle = gpu_idle()
    for _ in range(30):
        if idle["idle"]:
            break
        time.sleep(2)
        idle = gpu_idle()
    if not idle["idle"]:
        raise ValueError(f"GPU is not idle: {idle}")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command, namespace = server_command(
        python,
        model_path,
        arm["system"],
        port,
        cpu_bytes=pool_bytes(model, arm["pool"]) or None,
        mechanism=arm["mechanism"],
        gpu_blocks=model["gpu_blocks"],
        max_model_len=MAX_MODEL_LEN,
    )
    base = f"http://127.0.0.1:{port}"
    output.mkdir(parents=True)
    cells = []
    started = time.monotonic()
    with (output / "execution.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=source,
            env=benchmark_environment(
                source,
                model_path,
                arm["system"] in ENDPOINT_ARMS,
                arm["system"] == "lmcache",
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        owner = process_identity(child.pid)
        try:
            startup = wait_ready(base, child, 420)
            for seed in seeds:
                cell: dict[str, Any] = {
                    "schema": CELL_SCHEMA,
                    "arm": arm_name,
                    "arm_definition": arm,
                    "seed": seed,
                    "command": command,
                    "pool_bytes": pool_bytes(model, arm["pool"]),
                    "gpu_before": idle,
                    "startup_seconds": startup,
                    "status": "failed",
                }
                began = time.monotonic()
                try:
                    reset(base, True)
                    cell["restoration"] = (
                        None
                        if arm["system"] == "gpu-only"
                        else restoration_check(base, arm, model)
                    )
                    before = offload_metrics(base)
                    sessions = sessions_for(definition, seed, model_id)
                    rows, errors = replay_sessions(
                        base,
                        arm,
                        namespace,
                        sessions,
                        seed,
                        time.monotonic() + cell_seconds,
                    )
                    time.sleep(2)
                    cell["metrics_before"] = before
                    cell["metrics_after"] = offload_metrics(base)
                    cell["requests"] = rows
                    cell["errors"] = errors
                    expected = sum(len(session.turns) for session in sessions)
                    cell["status"] = (
                        "complete" if not errors and len(rows) == expected else "failed"
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    cell["errors"] = [str(error)]
                cell["workload_seconds"] = time.monotonic() - began
                write_record(output / f"cell-{seed}.json", cell)
                cells.append(cell)
        finally:
            if owner is not None:
                stop_session(child.pid, owner)
            child.wait()
    write_record(
        output / "server.json",
        {"elapsed_seconds": time.monotonic() - started, "exit_code": child.returncode},
    )
    return cells


def precheck(
    root: Path,
    runtime_python: Path,
    definition: dict[str, Any],
    output: Path,
    profile: str,
    chosen: list[int],
) -> dict[str, Any]:
    from tools.benchmarking.replay import simulate

    pressure = [
        replay_scenario(
            definition, model, profile_seeds(definition, profile, model, chosen)
        )
        for model in definition["models"]
    ]
    oracle = []
    for scenario in pressure:
        model = definition["models"][scenario["model"]]
        oracle.append(
            {
                **scenario,
                "name": scenario["name"] + "-oracle-pool",
                "geometry": {
                    **scenario["geometry"],
                    "cpu_keys": pool_keys(model, "oracle"),
                },
            }
        )
    scenarios = output / "precheck-scenarios.json"
    output.mkdir(parents=True, exist_ok=True)
    named = {entry["name"]: entry for entry in pressure + oracle}
    write_record(
        scenarios,
        {
            "schema": "replay-scenarios-v1",
            "geometries": {name: entry["geometry"] for name, entry in named.items()},
            "schedules": {name: entry["schedule"] for name, entry in named.items()},
            "scenarios": [
                {
                    "name": name,
                    "model": entry["model"],
                    "geometry": name,
                    "schedule": name,
                    "seeds": entry["seeds"],
                }
                for name, entry in named.items()
            ],
        },
    )
    code = simulate(
        root,
        runtime_python,
        scenarios,
        root / "benchmarks/replay/release-01-shapes.json",
        [],
        output / "precheck",
    )
    if code:
        raise ValueError("pre-GPU simulation failed")
    simulated = json.loads((output / "precheck/simulation.json").read_text())
    by_name = {entry["name"]: entry for entry in simulated["scenarios"]}
    verdicts = {}
    for scenario in pressure:
        control = by_name[scenario["name"]]["arms"]
        ceiling = by_name[scenario["name"] + "-oracle-pool"]["arms"]["native"]
        separation = share(control["gpu-only"]) - share(ceiling)
        verdicts[scenario["model"]] = {
            "control_minus_oracle_share": round(separation, 4),
            "status": "discriminating"
            if separation >= definition["controls"]["separation"]
            else "non-discriminating",
            "native_restores": control["native"]["activity"].get("cpu_restores", 0),
        }
    return verdicts


def share(arm: dict[str, Any], name: str = "hinted") -> float:
    totals = arm["cohorts"][name]
    reusable = totals.get("reusable_tokens", 0)
    return float(totals.get("recomputed_tokens", 0)) / reusable if reusable else 0.0


def run_pressure(
    root: Path,
    profile: str,
    models: dict[str, Path],
    runtime_python: Path,
    lmcache_python: Path | None,
    output: Path,
    arms: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    definition = load_definition(root)
    if profile not in definition["profiles"] or output.exists():
        raise ValueError("known profile and a new output directory required")
    if set(models) - set(definition["models"]):
        raise ValueError("unknown pressure model")
    selected = arms or definition["profiles"][profile]["arms"]
    if set(selected) - set(definition["profiles"][profile]["arms"]):
        raise ValueError("arm outside the selected profile")
    if "lmcache" in selected and lmcache_python is None:
        raise ValueError("lmcache arm requires --lmcache-python")
    uncovered = [
        (model, seed)
        for model in models
        for seed in profile_seeds(definition, profile, model, seeds)
        if not covers_cohorts(definition, seed, model)
    ]
    if uncovered:
        raise ValueError(
            f"seeds without hinted, unhinted and abandoned cohorts: {uncovered}"
        )
    output.mkdir(parents=True)
    identities = snapshot(root, output / "source")
    write_record(
        output / "admission.json",
        {
            "schema": "pressure-admission-v1",
            "profile": profile,
            "arms": selected,
            "seeds": {
                model: profile_seeds(definition, profile, model, seeds)
                for model in models
            },
            "definition": definition,
            "definition_sha256": digest_file(root / DEFINITION),
            "models": {
                name: {
                    "path": str(path),
                    "config_sha256": digest_file(path / "config.json"),
                }
                for name, path in models.items()
            },
            "source_sha256": identities,
            "runtime_python": str(runtime_python),
            "lmcache_python": str(lmcache_python) if lmcache_python else None,
        },
    )
    verdicts = precheck(root, runtime_python, definition, output, profile, seeds)
    write_record(output / "precheck.json", verdicts)
    refused = [name for name, v in verdicts.items() if v["status"] != "discriminating"]
    if refused:
        raise ValueError(f"pre-GPU check predicts non-discriminating cells: {refused}")
    seconds = definition["profiles"][profile]["cell_seconds"]
    for model_id, path in models.items():
        for arm_name in selected:
            arm = definition["arms"][arm_name]
            python = lmcache_python if arm["system"] == "lmcache" else runtime_python
            assert python is not None
            serve_arm(
                output / "source",
                output / "cells" / model_id / arm_name,
                python,
                path,
                model_id,
                arm_name,
                arm,
                definition,
                profile_seeds(definition, profile, model_id, seeds),
                seconds,
            )
    return verify_pressure(output)


def cohort_summary(rows: list[dict[str, Any]], alignment: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("hinted", "unhinted"):
        returns = [row for row in rows if row["turn"] and row["cohort"] == name]
        reusable = recomputed = 0
        ttft = []
        for row in returns:
            usable = min(row["shared_tokens"], row["prompt_tokens"] - 1)
            usable = usable // alignment * alignment
            cached = row["usage"]["prompt_tokens_details"]["cached_tokens"] or 0
            reusable += usable
            recomputed += usable - min(cached, usable)
            if row.get("metrics"):
                ttft.append(row["metrics"]["time_to_first_token_ms"])
        result[name] = {
            "returns": len(returns),
            "reusable_tokens": reusable,
            "recomputed_tokens": recomputed,
            "recomputed_share": round(recomputed / reusable, 4) if reusable else None,
            "mean_ttft_ms": round(statistics.fmean(ttft), 2) if ttft else None,
            "median_ttft_ms": round(statistics.median(ttft), 2) if ttft else None,
        }
    return result


def advice_activity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: Counter[str] = Counter()
    mechanisms: Counter[str] = Counter()
    held = refreshed = 0
    for row in rows:
        hint = row.get("hint") or {}
        acknowledgement = hint.get("acknowledgement")
        if not acknowledgement:
            continue
        decisions[acknowledgement["decision"]] += 1
        mechanisms[str(acknowledgement["mechanisms"].get("advice"))] += 1
        held += acknowledgement.get("held_keys", 0)
        refreshed += acknowledgement.get("refreshed_keys", 0)
    return {
        "hints_wanted": sum(row["wants_hint"] for row in rows),
        "hints_published": sum(
            bool((row.get("hint") or {}).get("published")) for row in rows
        ),
        "abandoned_hints": sum(
            row["wants_hint"] and row["abandons"] and row.get("final", False)
            for row in rows
        ),
        "decisions": dict(decisions),
        "mechanisms": dict(mechanisms),
        "held_keys": held,
        "refreshed_keys": refreshed,
    }


def cell_summary(
    cell: dict[str, Any], model_id: str, definition: dict[str, Any]
) -> dict[str, Any]:
    model = definition["models"][model_id]
    rows = cell.get("requests", [])
    sessions = sessions_for(definition, cell["seed"], model_id)
    finals = {(s.name, len(s.turns) - 1) for s in sessions}
    rows = [{**row, "final": (row["session"], row["turn"]) in finals} for row in rows]
    before, after = cell.get("metrics_before", {}), cell.get("metrics_after", {})
    stored_keys = (
        counter(after, "store_bytes") - counter(before, "store_bytes")
    ) / int(model["cpu_chunk_bytes"])
    loaded = counter(after, "load_bytes") - counter(before, "load_bytes")
    failures = counter(after, "failure") - counter(before, "failure")
    largest = max(
        (
            max(t.prompt + t.generated for t in s.turns)
            for s in sessions
            if len(s.turns) > 1
        ),
        default=0,
    )
    return {
        "arm": cell["arm"],
        "seed": cell["seed"],
        "status": cell["status"],
        "errors": cell.get("errors", []),
        "cohorts": cohort_summary(rows, int(model["hit_alignment"])),
        "advice": advice_activity(rows),
        "stored_keys": round(stored_keys, 1),
        "loaded_bytes": loaded,
        "store_failures": failures,
        "pool_keys": cell["pool_bytes"] // int(model["cpu_chunk_bytes"]),
        "largest_conversation_tokens": largest,
        "distinct_tokens": sum(
            t.prompt - t.shared + t.generated for s in sessions for t in s.turns
        ),
        "restoration": cell.get("restoration"),
        "workload_seconds": round(cell.get("workload_seconds", 0.0), 1),
        "startup_seconds": round(cell.get("startup_seconds", 0.0), 1),
    }


def contracts(
    summary: dict[str, Any], arm: dict[str, Any], keys_per_token: float | None
) -> dict[str, bool | None]:
    cohorts = summary["cohorts"]
    advice = summary["advice"]
    checks: dict[str, bool | None] = {
        "complete": summary["status"] == "complete",
        "hinted_returns": cohorts["hinted"]["returns"] > 0,
        "unhinted_returns": cohorts["unhinted"]["returns"] > 0,
        "abandoned_hints": advice["abandoned_hints"] > 0,
    }
    if arm["system"] == "gpu-only":
        checks["gpu_displacement"] = any(
            (cohorts[name]["recomputed_tokens"] or 0) > 0 for name in cohorts
        )
        return checks
    restoration = summary["restoration"] or {}
    checks["separate_prefix_restore"] = bool(restoration.get("restore_observed"))
    if arm["system"] == "lmcache":
        checks["cpu_restores"] = None
        checks["cpu_evictions"] = (
            None
            if keys_per_token is None
            else summary["distinct_tokens"] * keys_per_token > summary["pool_keys"]
        )
    elif arm["pool"] == "oracle":
        checks["cpu_restores"] = summary["loaded_bytes"] > 0
        checks["oracle_pool_never_full"] = (
            summary["stored_keys"] <= summary["pool_keys"]
        )
    else:
        checks["cpu_restores"] = summary["loaded_bytes"] > 0
        checks["cpu_evictions"] = summary["stored_keys"] > summary["pool_keys"]
    checks["pool_holds_one_conversation"] = (
        None
        if keys_per_token is None
        else summary["largest_conversation_tokens"] * keys_per_token
        <= summary["pool_keys"]
    )
    if arm["system"] in ENDPOINT_ARMS:
        expected = arm["mechanism"] if arm["system"] == "wait-hint" else "disabled"
        mechanisms = advice["mechanisms"]
        checks["hints_acknowledged"] = (
            advice["hints_published"] == advice["hints_wanted"] > 0
        )
        checks["advice_mechanism"] = bool(mechanisms) and set(mechanisms) == {expected}
        if arm["system"] == "wait-hint":
            checks["advice_applied"] = advice["decisions"].get("refreshed", 0) > 0
        if arm["mechanism"] == "promote-hold" and arm["system"] == "wait-hint":
            checks["holds_applied"] = advice["held_keys"] > 0
    return checks


def pooled(summaries: list[dict[str, Any]], name: str) -> float | None:
    reusable = sum(s["cohorts"][name]["reusable_tokens"] for s in summaries)
    recomputed = sum(s["cohorts"][name]["recomputed_tokens"] for s in summaries)
    return round(recomputed / reusable, 4) if reusable else None


def model_controls(
    arms: dict[str, list[dict[str, Any]]], separation: float
) -> dict[str, Any]:
    needed = ("gpu-only", "oracle", "native-arc", "native-arc-repeat")
    if any(not [s for s in arms.get(name, []) if s["valid"]] for name in needed):
        return {"status": "controls-missing"}
    worse = pooled(arms["gpu-only"], "hinted") or 0.0
    better = pooled(arms["oracle"], "hinted") or 0.0
    first = pooled(arms["native-arc"], "hinted") or 0.0
    second = pooled(arms["native-arc-repeat"], "hinted") or 0.0
    gap = worse - better
    aa = abs(first - second)
    status = (
        "discriminating" if gap >= separation and aa < gap else "non-discriminating"
    )
    return {
        "status": status,
        "known_worse_share": worse,
        "known_better_share": better,
        "separation": round(gap, 4),
        "required_separation": separation,
        "aa_difference": round(aa, 4),
    }


def advice_rule(
    arms: dict[str, list[dict[str, Any]]], tolerance: float
) -> dict[str, Any]:
    names = ("native-arc", "wait-hint", "candidate")
    valid = {name: [s for s in arms.get(name, []) if s["valid"]] for name in names}
    if any(len(valid[name]) < 3 for name in names):
        return {
            "status": "insufficient-repetitions",
            "valid": {k: len(v) for k, v in valid.items()},
        }
    reference = {s["seed"]: s for s in valid["wait-hint"]}
    paired = [
        (
            s["cohorts"]["hinted"]["recomputed_tokens"],
            reference[s["seed"]]["cohorts"]["hinted"]["recomputed_tokens"],
        )
        for s in valid["candidate"]
        if s["seed"] in reference
    ]
    harm = (pooled(valid["candidate"], "unhinted") or 0.0) - (
        pooled(valid["native-arc"], "unhinted") or 0.0
    )
    failures = sum(s["store_failures"] for s in valid["candidate"])
    control_failures = sum(s["store_failures"] for s in valid["native-arc"])
    rule = {
        "paired_hinted_recomputed": paired,
        "every_repetition_no_worse": len(paired) >= 3
        and all(a <= b for a, b in paired),
        "total_lower": sum(a for a, _ in paired) < sum(b for _, b in paired),
        "unhinted_share_change": round(harm, 4),
        "unhinted_within_tolerance": harm <= tolerance,
        "candidate_store_failures": failures,
        "control_store_failures": control_failures,
        "no_added_store_failures": failures <= control_failures,
    }
    rule["met"] = all(
        rule[key]
        for key in (
            "every_repetition_no_worse",
            "total_lower",
            "unhinted_within_tolerance",
            "no_added_store_failures",
        )
    )
    rule["status"] = "evaluated"
    return rule


def verify_pressure(*directories: Path) -> dict[str, Any]:
    cells: dict[str, dict[str, list[dict[str, Any]]]] = {}
    identities: dict[str, set[str]] = {}
    definition: dict[str, Any] = {}
    for directory in directories:
        admission = json.loads((directory / "admission.json").read_text())
        definition = admission["definition"]
        for path in sorted(directory.glob("cells/*/*/cell-*.json")):
            model_id, arm_name = path.parent.parent.name, path.parent.name
            identities.setdefault(model_id, set()).add(
                model_identity(definition, model_id)
            )
            cell = json.loads(path.read_text())
            summary = cell_summary(cell, model_id, definition)
            cells.setdefault(model_id, {}).setdefault(arm_name, []).append(summary)
    if any(len(found) != 1 for found in identities.values()):
        raise ValueError("pressure directories use different model definitions")
    report: dict[str, Any] = {"schema": REPORT_SCHEMA, "models": {}}
    for model_id, arms in cells.items():
        rates = [
            s["restoration"]["keys_per_token"]
            for summaries in arms.values()
            for s in summaries
            if (s["restoration"] or {}).get("keys_per_token")
        ]
        keys_per_token = statistics.median(rates) if rates else None
        for arm_name, summaries in arms.items():
            for summary in summaries:
                checks = contracts(
                    summary, definition["arms"][arm_name], keys_per_token
                )
                summary["contracts"] = checks
                summary["valid"] = all(value is not False for value in checks.values())
                summary["state"] = (
                    "failed"
                    if not checks["complete"]
                    else "valid"
                    if summary["valid"]
                    else "unexercised"
                )
        controls = model_controls(arms, definition["controls"]["separation"])
        comparable = controls["status"] == "discriminating"
        report["models"][model_id] = {
            "keys_per_token": keys_per_token,
            "controls": controls,
            "arms": {
                name: {
                    "cells": summaries,
                    "pooled_hinted_share": pooled(summaries, "hinted")
                    if comparable or controls["status"] == "controls-missing"
                    else None,
                    "pooled_unhinted_share": pooled(summaries, "unhinted")
                    if comparable or controls["status"] == "controls-missing"
                    else None,
                }
                for name, summaries in sorted(arms.items())
            },
            "advice_rule": advice_rule(
                arms, definition["controls"]["unhinted_tolerance"]
            )
            if controls["status"] != "non-discriminating"
            else {"status": "withheld: non-discriminating controls"},
        }
    return report


def projection(root: Path, profile: str, calibration: list[Path]) -> dict[str, Any]:
    definition = load_definition(root)
    arms = definition["profiles"][profile]["arms"]
    measured: dict[tuple[str, str], tuple[float, float]] = {}
    for directory in calibration:
        for path in directory.glob("cells/*/*/cell-*.json"):
            cell = json.loads(path.read_text())
            measured[(path.parent.parent.name, path.parent.name)] = (
                float(cell.get("startup_seconds", 0.0)),
                float(cell.get("workload_seconds", 0.0)),
            )
    rows = []
    total = 0.0
    for model in definition["models"]:
        for arm in arms:
            startup, workload = measured.get(
                (model, arm),
                (420.0, float(definition["profiles"][profile]["cell_seconds"])),
            )
            seeds = profile_seeds(definition, profile, model, [])
            seconds = startup + 30 + len(seeds) * workload
            total += seconds
            rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "repetitions": len(seeds),
                    "seconds": round(seconds),
                    "basis": "measured" if (model, arm) in measured else "budget",
                }
            )
    return {
        "schema": "pressure-plan-v1",
        "profile": profile,
        "cells": rows,
        "projected_gpu_hours": round(total / 3600, 2),
        "note": "servers start once per model/arm; repetitions reuse the server after a full cache reset",
    }
