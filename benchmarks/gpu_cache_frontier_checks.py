"""Freeze and verify targeted operational/KV checks for the smaller live profile."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from harness.gpu_lock import gpu_reserved

from .gpu_cache_frontier import (
    CAMPAIGN,
    ROOT,
    idle_gpu,
    native_delta,
    read,
    sha,
    verify_pins,
    write,
)


def prepare(args):
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    base = read(args.base_plan)
    candidate = next(x for x in base["runs"] if x["loader_role"] == "bounded1280" and x["seed"] == 16001)
    runs = []
    for arm in ("native8", "candidate"):
        command = list(candidate["args"])
        name = f"{args.prefix}-byte-{arm}"
        command[command.index("--name") + 1] = name
        if arm == "native8":
            command[command.index("--arm") + 1] = "arc"
            offset = command.index("--source-label")
            del command[offset:offset + 2]
            command.remove("--gpu-gated-cpu-retention")
            command.remove("--no-audit")
            command.append("--native-arc")
        command.append("--audit")
        runs.append(dict(candidate, name=name, arm=arm, args=command, gate_enabled=arm == "candidate",
                         check="byte", runner=base["runner"], planned_EOS=72))
    for phase, waves, interval in (("persistent", 6, 0), ("overlap", 3, 20)):
        command = list(candidate["args"])
        name = args.prefix + "-" + phase
        command[command.index("--name") + 1] = name
        command[command.index("--seed") + 1] = "14009"
        command[command.index("--timeout") + 1] = "7200"
        command += ["--waves", str(waves), "--wave-seeds", "14009,15013,16001", "--wave-settle-s", "2",
                    "--wave-release-interval-s", str(interval), "--registry-limit", "4096", "--cancel-probe"]
        runs.append(dict(candidate, name=name, phase=phase, arm="candidate", args=command, seed=14009,
                         check="operational", runner=str(CAMPAIGN / "run_live_operational_v2_nov3.py"),
                         planned_EOS=waves * 72, normal_natural_eos_planned=waves * 72,
                         health_natural_eos_planned=4, deliberate_abort_planned=1))
    for item in runs:
        if (CAMPAIGN / "runs" / item["name"]).exists():
            raise FileExistsError(item["name"])
    files = {Path(p) for p in base["file_pins"]}
    files.update(CAMPAIGN.glob("*.py"))
    files.update((CAMPAIGN / "hooks-operational-v2").glob("*.py"))
    files.add(Path(__file__).resolve())
    plan = dict(base, schema="small-GPU-cache-targeted-check-plan-v1", created_ns=time.time_ns(),
                output=str(destination), runs=runs, file_pins={str(p): sha(p) for p in sorted(files)},
                base_plan=str(Path(args.base_plan).resolve()), base_plan_sha256=sha(args.base_plan),
                scope="1280 GPU blocks,2048 CPU blocks,8active,priority8000ms,bounded AWQ512. All transfers byte-audited on pressure trace16001 separately from performance; six persistent waves then three genuinely overlapping waves, each with exact native abort and health recovery. Natural EOS, original history, no reset within endpoint. No indefinite soak or interactive SLA claim.")
    write(destination / "plan.json", plan)
    print(json.dumps(dict(cases=len(runs), normal_EOS=sum(x["planned_EOS"] for x in runs), health_EOS=8)))


def compress_audits(directory):
    for path in directory.glob("*audit*.jsonl"):
        expected = sha(path)
        target = path.with_suffix(path.suffix + ".gz")
        if target.exists():
            raise FileExistsError(target)
        with path.open("rb") as source, gzip.open(target, "wb", compresslevel=3) as output:
            shutil.copyfileobj(source, output)
        with gzip.open(target, "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Audit compression integrity failed")
        path.unlink()


@gpu_reserved
def run(args):
    plan = read(args.plan)
    output = Path(plan["output"])
    receipt = output / "execution.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    state = dict(status="running", plan_sha256=sha(args.plan), started_ns=time.time_ns(), runs=[])
    try:
        for item in plan["runs"]:
            verify_pins(plan)
            idle_gpu()
            state["current"] = item["name"]
            write(receipt, state)
            print("START " + item["name"], flush=True)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", GPU_CACHE_RUNTIME_STAGE=plan["runtime_stage"],
                       VLLM_AWQ_REPACK_CHUNK_ROWS="512", VLLM_CLI_MODULE="benchmarks.awq_loading_runtime_cli")
            env.pop("VLLM_CLI", None)
            env.pop("PYTHONPATH", None)
            with (output / (item["name"] + ".log")).open("w") as stream:
                completed = subprocess.run([plan["python"], item["runner"], *item["args"]],
                                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            directory = CAMPAIGN / "runs" / item["name"]
            if directory.exists():
                shutil.copyfile(CAMPAIGN / "return_priority_nov3.py", directory / "return_priority_helper_astra.py")
                compress_audits(directory)
            raw_path = directory / "raw.json.gz"
            raw = read(raw_path) if raw_path.exists() else {}
            row = dict(name=item["name"], exit_code=completed.returncode, raw=str(raw_path),
                       raw_sha256=sha(raw_path) if raw_path.exists() else None,
                       status=raw.get("status"), error=raw.get("error"),
                       natural_EOS=sum(q.get("natural_eos", False) for q in raw.get("requests", [])),
                       descendants=raw.get("owned_descendants_remaining"))
            state["runs"].append(row)
            state.pop("current", None)
            write(receipt, state)
            print("END " + json.dumps(row), flush=True)
            verify_pins(plan)
            idle_gpu()
            if completed.returncode or row["status"] != "complete" or row["natural_EOS"] != item["planned_EOS"] or row["descendants"]:
                raise RuntimeError("Targeted check failed; all attempted evidence retained")
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="stopped", error=repr(error))
        raise
    finally:
        state["completed_ns"] = time.time_ns()
        write(receipt, state)


def byte_receipt(raw, directory):
    counts, byte_counts = collections.Counter(), collections.Counter()
    load_counts, load_bytes = collections.Counter(), collections.Counter()
    components = 0
    identities = set()
    audits = []
    for path in directory.glob("byte-audit-*.jsonl.gz"):
        audits.append(dict(path=str(path), sha256=sha(path)))
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                identity = (row["at_ns"], row["job_id"], row["direction"])
                assert identity not in identities
                identities.add(identity)
                pages = len(row["source_physical_ids"])
                assert pages == len(row["destination_physical_ids"]) and pages > 0
                assert row["intrusive_synchronization"] is True
                assert len(row["components"]) == 28 and {c["tensor_idx"] for c in row["components"]} == set(range(28))
                for component in row["components"]:
                    assert component["equal"] and component["invalid_byte_rejected"]
                    assert component["page_size_bytes"] == 32768 and component["bytes"] == pages * 32768
                    assert re.fullmatch("[0-9a-f]{64}", component["sha256"])
                    components += 1
                assert row["total_bytes"] == sum(c["bytes"] for c in row["components"]) == pages * 917504
                direction = row["direction"]
                assert direction in ("D2H", "H2D")
                counts[direction] += 1
                byte_counts[direction] += row["total_bytes"]
                if row["at_ns"] >= raw["load_start_ns"]:
                    load_counts[direction] += 1
                    load_bytes[direction] += row["total_bytes"]
    for direction, label in (("D2H", "GPU_to_CPU"), ("H2D", "CPU_to_GPU")):
        assert counts[direction] > 0
        assert native_delta(raw, "vllm:kv_offload_total_bytes_total", f'transfer_type="{label}"') == load_bytes[direction]
        assert native_delta(raw, "vllm:kv_offload_size_count", f'transfer_type="{label}"') == load_counts[direction]
        for metric, total in (("vllm:kv_offload_total_bytes_total", byte_counts[direction]), ("vllm:kv_offload_size_count", counts[direction])):
            final = [float(line.rsplit(" ", 1)[1]) for line in raw["metrics_final"].splitlines()
                     if line.startswith(metric + "{") and f'transfer_type="{label}"' in line]
            assert final and sum(final) == total
    return dict(qualified=True, audit_files=audits, jobs=dict(counts), bytes=dict(byte_counts),
                component_comparisons_and_executed_one_byte_controls=components,
                aggregate_counts_and_bytes_reconciled=True,
                per_job_native_byte_field_observed=False,
                scope="Mapped physical pages times independently pinned 28-tensor geometry; all native aggregate counts/bytes reconciled. Intrusive byte audits are separate from performance.")


def report(args):
    plan = read(args.plan)
    verify_pins(plan)
    from .research_checks.http import verify_http
    from .research_checks.operational import analyze
    from .research_checks.priority import verify_priority
    rows = []
    for item in plan["runs"]:
        directory = CAMPAIGN / "runs" / item["name"]
        raw = read(directory / "raw.json.gz")
        assert raw["geometry"]["gpu_blocks"] == 1280 and raw["geometry"]["cpu_blocks"] == 2048
        assert raw["status"] == "complete" and raw["owned_descendants_remaining"] == []
        if item["check"] == "operational":
            result = analyze(item, CAMPAIGN)
            result.pop("first_prompts_by_wave", None)
        else:
            assert raw["args"]["audit"] and not raw["performance_eligible"]
            assert len(raw["requests"]) == 72 and all(q["success"] and q["natural_eos"] for q in raw["requests"])
            assert verify_http(raw)["qualified"] and verify_priority(raw, item, CAMPAIGN)["qualified"]
            result = dict(name=item["name"], natural_eos=72, preemptions=native_delta(raw, "vllm:num_preemptions_total"),
                          byte_receipt=byte_receipt(raw, directory))
        result["raw_sha256"] = sha(directory / "raw.json.gz")
        rows.append(result)
    write(Path(plan["output"]) / "report.json", dict(schema="small-cache-targeted-check-report-v1", qualified=True,
          plan_sha256=sha(args.plan), rows=rows, scope=plan["scope"]))
    print(json.dumps(dict(qualified=True, cases=len(rows), rows=rows), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--base-plan", default=str(ROOT / "output/gpu-cache-frontier-20261103/bounded-loading/plan.json"))
    prepare_parser.add_argument("--prefix", default="live-nov3-vram-check")
    for name in ("run", "report"):
        sub.add_parser(name).add_argument("--plan", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
