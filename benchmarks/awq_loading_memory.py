"""Bit-integrity and real-serving experiments for bounded AWQ initialization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sysconfig
import time
from pathlib import Path

from harness.gpu_lock import GPULock, gpu_reserved

from .gpu_cache_frontier import (
    CAMPAIGN,
    ROOT,
    idle_gpu,
    read,
    sha,
    verify_pins,
    write,
)

OVERRIDES = ("envs.py", "model_executor/layers/quantization/auto_awq.py")


def gpu_proof(output):
    """Compare every output word on real Qwen linear geometries and a short tail."""
    path = Path(output).resolve()
    if path.exists():
        raise FileExistsError(path)
    rows = []
    idle_gpu()
    with GPULock():
        import torch
        from vllm.model_executor.layers.quantization.auto_awq import (
            _repack_awq_weight_rows,
        )

        generator = torch.Generator().manual_seed(8213)
        for K, N in ((3584, 4608), (3584, 3584), (3584, 37888), (18944, 3584), (520, 56)):
            cpu = torch.randint(-(2**31), 2**31, (K, N // 8), dtype=torch.int32, generator=generator)
            weights = cpu.to("cuda")
            values = []
            expected = None
            for chunk in (0, 512):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter_ns()
                packed = _repack_awq_weight_rows(weights, 4, chunk)
                torch.cuda.synchronize()
                measurement = dict(chunk_rows=chunk, elapsed_ns=time.perf_counter_ns() - started,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
                actual = packed.cpu()
                if expected is None:
                    expected = actual
                else:
                    if not torch.equal(actual, expected):
                        raise AssertionError("Packed AWQ weights differ")
                    corrupt = actual.clone()
                    corrupt.view(torch.uint8).reshape(-1)[0] ^= 1
                    if torch.equal(corrupt, expected):
                        raise AssertionError("One-byte corruption oracle failed")
                    del corrupt
                if not torch.equal(cpu, weights.cpu()):
                    raise AssertionError("Source AWQ weights mutated")
                measurement["output_words"] = actual.numel()
                values.append(measurement)
                del packed, actual
            rows.append(dict(K=K, N=N, exact_int32_words=True, one_byte_oracle=True, measurements=values))
            del expected, weights, cpu
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    write(path, dict(schema="bounded-AWQ-GPU-word-proof-v1", passed=True,
                     source_sha256=sha(ROOT / "vendor/vllm/vllm" / OVERRIDES[1]),
                     geometries=rows, claim="Exact format conversion and temporary allocator peaks; standalone proof is not serving throughput."))
    print(json.dumps(dict(passed=True, geometries=len(rows), output_words=sum(r["measurements"][0]["output_words"] for r in rows))))


def prepare(args):
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    installed = Path(sysconfig.get_paths()["purelib"]) / "vllm"
    stage = destination / "runtime-stage"
    stage.mkdir()
    # Preserve every installed package asset, with two owned regular-file
    # overrides. Symlinks avoid copying shared binaries or modifying the wheel.
    shutil.copytree(installed, stage / "vllm", copy_function=os.symlink,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), symlinks=True)
    for relative in OVERRIDES:
        target = stage / "vllm" / relative
        target.unlink()
        shutil.copyfile(ROOT / "vendor/vllm/vllm" / relative, target)
    identity = dict(schema="installed-wheel-bounded-AWQ-stage-v1", installed=str(installed),
                    overrides={name: sha(stage / "vllm" / name) for name in OVERRIDES})
    write(stage / "identity.json", identity)
    base = read(ROOT / "output/gpu-cache-frontier-20261103/confirm/plan.json")
    original = read(ROOT / "evidence/live-optimization-2080ti-20261103/priority-confirmation-frozen.json")
    orders = [("native1536", "candidate1280", "bounded1280"),
              ("candidate1280", "bounded1280", "native1536"),
              ("bounded1280", "native1536", "candidate1280")]
    runs = []
    for seed, order in zip((14009, 15013, 16001), orders):
        for role in order:
            arm = "native8" if role == "native1536" else "candidate"
            blocks = 1536 if arm == "native8" else 1280
            template = next(x for x in base["runs"] if x["seed"] == seed and x["arm"] == arm and x["gpu_blocks"] == blocks)
            name = f"{args.prefix}-s{seed}-{role}"
            command = list(template["args"])
            command[command.index("--name") + 1] = name
            if (CAMPAIGN / "runs" / name).exists():
                raise FileExistsError(name)
            runs.append(dict(template, name=name, args=command, repeat=0,
                             loader_role=role, loader_chunk_rows=512 if role == "bounded1280" else 0))
    files = {Path(p) for p in base["file_pins"]}
    files.update([Path(__file__).resolve(), ROOT / "benchmarks/awq_loading_runtime_cli.py",
                  stage / "identity.json", *[stage / "vllm" / name for name in OVERRIDES],
                  *[ROOT / "vendor/vllm/vllm" / name for name in OVERRIDES]])
    plan = dict(base, schema="bounded-AWQ-live-memory-plan-v1", created_ns=time.time_ns(),
                python=original["python"], output=str(destination), runs=runs, mode="screen",
                runtime_stage=str(stage), file_pins={str(p): sha(p) for p in sorted(files)},
                run_prefix=args.prefix,
                stop_conditions=["changed frozen source or stage", "non-idle GPU before invocation", "any incomplete serving case or owned cleanup failure"],
                interpretation="Nine fresh live executions on three reused traces. Native1536 and candidate1280 use identical staged code with legacy chunk0; bounded1280 differs only in loading chunk512. The 36-run preceding confirmation measures steady serving without this loading change. Startup improvement is generic to the model loader, not unique cache-policy attribution.",
                analysis_unit="Three whole-workflow triples; descriptive serving follow-through plus separate exact packed-word proof. No new production/quality qualification.")
    plan.pop("groups", None)
    write(destination / "plan.json", plan)
    print(json.dumps(dict(plan=str(destination / "plan.json"), invocations=len(runs), stage=str(stage))))


@gpu_reserved
def run(args):
    plan = read(args.plan)
    output = Path(plan["output"])
    receipt = output / "execution.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    state = dict(status="running", started_ns=time.time_ns(), runs=[])
    try:
        for item in plan["runs"]:
            verify_pins(plan)
            idle_gpu()
            state["current"] = item["name"]
            write(receipt, state)
            print("START " + item["name"], flush=True)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       GPU_CACHE_RUNTIME_STAGE=plan["runtime_stage"],
                       VLLM_AWQ_REPACK_CHUNK_ROWS=str(item["loader_chunk_rows"]),
                       VLLM_CLI_MODULE="benchmarks.awq_loading_runtime_cli")
            env.pop("VLLM_CLI", None)
            env.pop("PYTHONPATH", None)
            with (output / (item["name"] + ".log")).open("w") as log:
                completed = subprocess.run([plan["python"], plan["runner"], *item["args"]],
                                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            directory = CAMPAIGN / "runs" / item["name"]
            shutil.copyfile(CAMPAIGN / "return_priority_nov3.py", directory / "return_priority_helper_astra.py")
            raw_path = directory / "raw.json.gz"
            raw = read(raw_path)
            row = dict(name=item["name"], exit_code=completed.returncode,
                       status=raw.get("status"), raw=str(raw_path), raw_sha256=sha(raw_path),
                       natural_eos=sum(q.get("natural_eos", False) for q in raw.get("requests", [])),
                       owned_descendants_remaining=raw.get("owned_descendants_remaining"))
            state["runs"].append(row)
            state.pop("current", None)
            write(receipt, state)
            print("END " + json.dumps(row), flush=True)
            verify_pins(plan)
            idle_gpu()
            if completed.returncode or raw.get("status") != "complete" or row["natural_eos"] != 72 or row["owned_descendants_remaining"]:
                raise RuntimeError("Bounded AWQ serving or cleanup failed; preserve and inspect")
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="stopped", error=repr(error))
        raise
    finally:
        state["completed_ns"] = time.time_ns()
        write(receipt, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("gpu-proof").add_argument("--output", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--prefix", default="live-nov3-vram-bounded")
    sub.add_parser("run").add_argument("--plan", required=True)
    args = parser.parse_args()
    if args.command == "gpu-proof":
        gpu_proof(args.output)
    else:
        {"prepare": prepare, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
