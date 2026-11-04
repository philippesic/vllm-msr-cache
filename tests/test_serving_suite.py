"""Contract, transport, failure, statistics, and owned-process controls."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from benchmarks.serving_suite.contracts import (
    PROFILES,
    SCHEMA,
    fixture_corpus,
    validate_corpus,
    validate_plan,
    write_json,
)
from benchmarks.serving_suite.report import (
    evaluate_case,
    paired_comparison,
    percentile,
    summarize_requests,
)
from benchmarks.serving_suite.resources import GPULock, OwnedServer, free_port
from benchmarks.serving_suite.runner import metric, metric_delta, select_sessions
from benchmarks.serving_suite.workload import completion


def plan():
    return dict(
        PROFILES["smoke"],
        schema=SCHEMA,
        tier="smoke",
        gpu_blocks=128,
        cpu_bytes=2**20,
        max_model_len=2048,
        seed=42,
        startup_timeout_s=10,
        request_timeout_s=2,
        case_timeout_s=10,
        sample_interval_s=0.25,
        gpu_memory_utilization=0.5,
        required_activity=["store", "restore"],
    )


@pytest.mark.parametrize(
    "key",
    [
        "repeats",
        "sessions",
        "turns",
        "concurrency",
        "waves",
        "max_tokens",
        "gpu_blocks",
        "cpu_bytes",
        "seed",
    ],
)
@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_integer_plan(key, bad):
    p = plan()
    p[key] = bad
    with pytest.raises(ValueError):
        validate_plan(p)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1])
def test_deadline_validation(bad):
    p = plan()
    p["request_timeout_s"] = bad
    with pytest.raises(ValueError):
        validate_plan(p)


def test_tier_minimums():
    p = plan()
    p["tier"] = "confirm"
    with pytest.raises(ValueError):
        validate_plan(p)
    p["tier"] = "operational"
    with pytest.raises(ValueError):
        validate_plan(p)


def test_corpus_and_disjoint_confirmation():
    c = fixture_corpus()
    p = plan()
    p.update(tier="confirm", sessions=6, repeats=3)
    groups = [{s["id"] for s in select_sessions(c, p, i)} for i in range(3)]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
    assert select_sessions(c, p, 0) == select_sessions(c, p, 0)
    with pytest.raises(ValueError):
        select_sessions(c, p, 99)
    c["sessions"].append(c["sessions"][0])
    with pytest.raises(ValueError):
        validate_corpus(c)


@pytest.mark.parametrize(
    "change",
    [
        {"turns": []},
        {"context": 123},
        {"wait_s": [-1]},
        {"wait_s": [float("nan")]},
        {"id": ""},
    ],
)
def test_corpus_rejects_malformed_session(change):
    c = fixture_corpus(1)
    c["sessions"][0].update(change)
    with pytest.raises(ValueError):
        validate_corpus(c)


def test_atomic_output_rejects_nonfinite(tmp_path):
    p = tmp_path / "result.json"
    write_json(p, {"x": 1})
    with pytest.raises(ValueError):
        write_json(p, {"x": float("nan")})
    assert json.loads(p.read_text()) == {"x": 1}


def test_metric_semantics():
    raw = 'vllm:request_success_total{finished_reason="abort",model_name="x"} 2\nvllm:request_success_total{finished_reason="stop",model_name="x"} 3\n'
    assert metric(raw, "vllm:request_success_total") == 5
    assert metric(raw, "vllm:request_success_total", {"finished_reason": "abort"}) == 2
    assert metric(raw, "missing") is None
    before = "vllm:kv_offload_store_bytes_total 100\n"
    assert (
        metric_delta(before, "vllm:kv_offload_store_bytes_total 200\n")[
            "vllm:kv_offload_store_bytes_total"
        ]
        == 100
    )
    assert (
        metric_delta(before, "vllm:kv_offload_store_bytes_total 50\n")[
            "vllm:kv_offload_store_bytes_total"
        ]
        is None
    )
    assert metric_delta(before, "")["vllm:kv_offload_store_bytes_total"] is None


def valid_case(arm="arc", repeat=0):
    return dict(
        name=arm,
        arm=arm,
        repeat=repeat,
        expected_requests=1,
        cleanup={"verified": True},
        telemetry={"samples": 2, "gpu_peak_bytes": 100},
        activity={"store": 1, "restore": 1},
        summary={
            "completed": 1,
            "failed": 0,
            "output_tokens_per_s": 10,
            "output_tokens": 10,
            "ttft_mean_s": 0.1,
            "e2e_p95_s": 1,
        },
    )


def test_missing_and_zero_activity_are_inconclusive():
    c = valid_case()
    c["activity"]["restore"] = 0
    assert evaluate_case(c, ["restore"])["status"] == "inconclusive"
    c["activity"]["restore"] = None
    assert evaluate_case(c, ["restore"])["unavailable_activity"] == ["restore"]
    c["error"] = "failed transfer"
    assert evaluate_case(c, ["restore"])["status"] == "failed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("cleanup", {"verified": False}),
        ("telemetry", {"samples": 0}),
        ("expected_requests", 2),
        ("operational", {"passed": False}),
    ],
)
def test_incomplete_data_cannot_pass(field, value):
    c = valid_case()
    c[field] = value
    assert evaluate_case(c, [])["status"] == "failed"


def test_stats_preserve_failure_and_termination():
    good = {
        "success": True,
        "output_token_ids": [1, 2],
        "ttft_s": 0.2,
        "latency_s": 1,
        "finish_reason": "length",
    }
    stats = summarize_requests([good, {"success": False}], 2)
    assert stats["output_tokens_per_s"] == 1 and stats["failed"] == 1
    assert stats["finish_reasons"] == {"length": 1}
    assert percentile([], 0.95) is None
    assert percentile([1, 2, 3], 0.5) == 2


def test_pairing_no_selective_failure_exclusion():
    cases = []
    for rep in range(3):
        for arm in ("arc", "candidate"):
            c = valid_case(arm, rep)
            c["evaluation"] = evaluate_case(c, [])
            cases.append(c)
    result = paired_comparison(cases)
    assert result["throughput_ratio_95ci"] == [1, 1]
    cases[0]["evaluation"]["status"] = "inconclusive"
    result = paired_comparison(cases)
    assert result["decision"] == "inconclusive" and result["eligible_pairs"] == 2


def test_cross_process_lock(tmp_path):
    path = tmp_path / "gpu.lock"
    with GPULock(path):
        code = (
            "from benchmarks.serving_suite.resources import GPULock; GPULock("
            + repr(str(path))
            + ").__enter__()"
        )
        child = subprocess.run([sys.executable, "-c", code], capture_output=True)
        assert child.returncode != 0 and b"lock held" in child.stderr
    with GPULock(path):
        pass


def test_start_failure_and_cleanup(tmp_path):
    server = OwnedServer(
        [sys.executable, "-c", "raise SystemExit(7)"],
        tmp_path,
        dict(os.environ),
        free_port(),
    )
    with pytest.raises(RuntimeError, match="exited"):
        server.start(2)
    assert server.cleanup["verified"] and server.proc.returncode == 7


def test_readiness_timeout_kills_owned_process(tmp_path):
    server = OwnedServer(
        [sys.executable, "-c", "import time;time.sleep(30)"],
        tmp_path,
        dict(os.environ),
        free_port(),
    )
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        server.start(0.2)
    assert server.cleanup["verified"] and time.monotonic() - start < 5


def test_parent_exit_does_not_leave_child(tmp_path):
    pidfile = tmp_path / "child.pid"
    code = f'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);open({str(pidfile)!r},"w").write(str(p.pid));time.sleep(.2)'
    server = OwnedServer(
        [sys.executable, "-c", code], tmp_path, dict(os.environ), free_port()
    )
    with pytest.raises(RuntimeError):
        server.start(3)
    import psutil

    pid = int(pidfile.read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )
    assert server.cleanup["verified"]


async def transport_case(mode):
    attempts = []

    async def handler(request):
        attempts.append(await request.json())
        if mode == "http_error":
            return web.Response(status=503, text="deliberate error")
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if mode == "timeout":
            await asyncio.sleep(0.2)
            return response
        chunks = [
            {"choices": [{"token_ids": [10, 11], "text": "x", "finish_reason": None}]},
            {"choices": [{"token_ids": [12], "text": "y", "finish_reason": "stop"}]},
        ]
        if mode == "missing_tokens":
            chunks[0]["choices"][0].pop("token_ids")
        if mode == "malformed":
            await response.write(b"data: {bad}\n\n")
            return response
        for chunk in chunks:
            await response.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        if mode != "truncated":
            await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    records = []
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=0.05 if mode == "timeout" else 2)
        ) as client:
            result = await completion(
                client,
                f"http://127.0.0.1:{port}",
                {"request_id": "test"},
                {},
                records.append,
            )
        assert len(records) == 1 and len(attempts) == 1
        return result
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    "mode", ["http_error", "timeout", "missing_tokens", "malformed", "truncated"]
)
def test_transport_failures_retained_without_retry(mode):
    row = asyncio.run(transport_case(mode))
    assert not row["success"] and row["error"]


def test_real_http_stream_parser():
    row = asyncio.run(transport_case("success"))
    assert row["success"] and row["output_token_ids"] == [10, 11, 12]
    assert row["finish_reason"] == "stop" and row["ttft_s"] > 0


def test_cli_plan_is_gpu_free_and_refuses_overwrite(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    out = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.serving_suite.runner",
        "smoke",
        "--plan",
        "--model",
        str(model),
        "--output",
        str(out),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    first = subprocess.run(cmd, capture_output=True, env=env)
    assert first.returncode == 0, first.stderr.decode()
    assert json.loads((out / "plan.json").read_text())["synthetic"] is True
    second = subprocess.run(cmd, capture_output=True, env=env)
    assert second.returncode != 0


def test_public_corpus_conversion_preserves_content():
    from benchmarks.serving_suite.import_corpus import convert

    d = {
        "schema": "quantized-realworld-sessions-v1",
        "sessions": [
            {"id": "a", "cohort": "agent", "turns": [{"user_text": "original input"}]}
        ],
    }
    c = convert(d, "hash", "agent")
    assert c["sessions"][0]["turns"] == ["original input"]
    assert not c["synthetic"] and "recorded tools are not executed" in c["provenance"]


def test_detached_worker_is_recovered_after_parent_exit(tmp_path):
    pidfile = tmp_path / "detached.pid"
    code = f'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],start_new_session=True);open({str(pidfile)!r},"w").write(str(p.pid));time.sleep(.2)'
    server = OwnedServer(
        [sys.executable, "-c", code], tmp_path, dict(os.environ), free_port()
    )
    with pytest.raises(RuntimeError):
        server.start(3)
    import psutil

    pid = int(pidfile.read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )
    assert server.cleanup["verified"]


def test_abort_parser_exact_id_and_mixed_logs(tmp_path):
    from benchmarks.serving_suite.runner import abort_receipts

    log = tmp_path / "server.log"
    event = {
        "schema": "semantic-backend-abort-v1",
        "probe_id": "exact",
        "status": "FINISHED_ABORTED",
    }
    log.write_text(
        "prefix SEMANTIC_BENCHMARK_ABORT " + json.dumps(event) + " trailing log\n"
    )
    assert abort_receipts(log, "exact") == [event]
    assert abort_receipts(log, "ex") == []
    log.write_text("SEMANTIC_BENCHMARK_ABORT {bad}")
    with pytest.raises(ValueError):
        abort_receipts(log, "exact")


def test_offline_integrity_detects_mutation_and_missing_files(tmp_path):
    from benchmarks.serving_suite.verify import seal, verify

    data = {"schema": SCHEMA, "status": "failed", "cases": []}
    write_json(tmp_path / "result.json", data)
    seal(tmp_path)
    assert verify(tmp_path)["cases"] == 0
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed artifact"):
        verify(tmp_path)
    (tmp_path / "result.json").unlink()
    with pytest.raises(ValueError, match="Missing artifact"):
        verify(tmp_path)


def test_raw_sse_retained_when_stream_is_incomplete(tmp_path):
    async def exercise():
        async def handler(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(
                b'data: {"choices":[{"token_ids":[1],"text":"x","finish_reason":null}]}\n\n'
            )
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        path = tmp_path / "stream.sse"
        rows = []
        try:
            async with aiohttp.ClientSession() as client:
                row = await completion(
                    client,
                    f"http://127.0.0.1:{port}",
                    {"request_id": "raw"},
                    {},
                    rows.append,
                    stream_path=path,
                )
            assert not row["success"] and path.read_bytes().startswith(b"data: ")
            assert rows[0]["raw_stream"] == str(path)
        finally:
            await runner.cleanup()

    asyncio.run(exercise())


def test_quality_assertion_failure_is_not_hidden():
    c = valid_case()
    c["summary"]["content_checks_failed"] = 1
    assert evaluate_case(c, [])["status"] == "failed"


def test_runtime_stage_preserves_source_and_supplies_distribution_assets(tmp_path):
    from tools.vllm_runtime import stage_package

    source, wheel, staged = (tmp_path / n for n in ("source", "wheel", "staged"))
    source.mkdir()
    wheel.mkdir()
    (source / "patched.py").write_text("repository")
    (wheel / "patched.py").write_text("wheel")
    (wheel / "support.py").write_text("support")
    (wheel / "untracked.py").write_text("must not stage")
    (wheel / "binary.so").write_bytes(b"binary")
    (wheel / "_version.py").write_text("pinned")
    stage_package(
        source,
        wheel,
        staged,
        [Path(n) for n in ["patched.py", "support.py", "binary.so", "_version.py"]],
    )
    assert (staged / "patched.py").read_text() == "repository"
    assert (staged / "support.py").read_text() == "support"
    assert (staged / "binary.so").is_symlink()
    assert (staged / "_version.py").read_text() == "pinned"
    assert not (staged / "_version.py").is_symlink()
    assert not (staged / "untracked.py").exists()


@pytest.mark.parametrize(
    "cfg", [[], {}, {"kv_connector": "wrong", "kv_connector_extra_config": {}}]
)
def test_bad_candidate_rejected_before_server(cfg):
    p = plan()
    p["candidate_config"] = cfg
    with pytest.raises(ValueError, match="Candidate"):
        validate_plan(p)


def test_bad_wait_shape_rejected():
    c = fixture_corpus()
    c["sessions"][0]["wait_s"] = "invalid"
    with pytest.raises(ValueError, match="wait_s"):
        validate_corpus(c)


def test_launcher_sigterm_cleans_staging_and_child(tmp_path):
    import psutil

    ready = tmp_path / "ready.json"
    staging = tmp_path / "staging"
    script = (
        "import os,json,time;from pathlib import Path;import vllm;"
        f"Path({str(ready)!r}).write_text(json.dumps([os.getpid(),vllm.__file__]));"
        "time.sleep(60)"
    )
    env = dict(os.environ, SEMANTIC_VLLM_STAGE_PARENT=str(staging))
    launcher = subprocess.Popen(
        [sys.executable, "tools/vllm_runtime.py", sys.executable, "-c", script],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            assert launcher.poll() is None
            time.sleep(0.05)
        child_pid, path = json.loads(ready.read_text())
        assert str(staging) in path
        launcher.terminate()
        launcher.wait(timeout=8)
        assert not psutil.pid_exists(child_pid)
        assert not list(staging.iterdir())
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait()
        if child_pid and psutil.pid_exists(child_pid):
            psutil.Process(child_pid).kill()


def test_offline_verifier_recomputes_aggregates_and_rejects_duplicate_ids(tmp_path):
    from benchmarks.serving_suite.verify import seal, verify

    row = {
        "wave": 0,
        "source_id": "s",
        "turn": 0,
        "success": True,
        "output_token_ids": [1, 2],
        "payload": {"prompt": [3]},
        "ttft_s": 0.1,
        "latency_s": 0.2,
        "finish_reason": "stop",
    }
    case = {
        "name": "case",
        "load_duration_s": 1,
        "summary": summarize_requests([row], 1),
        "evaluation": {"status": "failed"},
    }
    result = {"schema": SCHEMA, "status": "failed", "cases": [case]}
    directory = tmp_path / "case"
    directory.mkdir()
    records = directory / "requests.jsonl"
    records.write_text(json.dumps(row) + "\n")
    write_json(tmp_path / "result.json", result)
    seal(tmp_path)
    assert verify(tmp_path)["request_records"] == 1
    case["summary"]["output_tokens"] = 99
    write_json(tmp_path / "result.json", result)
    seal(tmp_path)
    with pytest.raises(ValueError, match="Wrong request aggregate"):
        verify(tmp_path)
    records.write_text((json.dumps(row) + "\n") * 2)
    case["summary"] = summarize_requests([row, row], 1)
    write_json(tmp_path / "result.json", result)
    seal(tmp_path)
    with pytest.raises(ValueError, match="Duplicate request identity"):
        verify(tmp_path)


def test_resource_row_failure_reaches_evaluation(tmp_path):
    import threading
    from types import SimpleNamespace

    from benchmarks.serving_suite.resources import Sampler

    sampler = Sampler(1, tmp_path / "unused")
    sampler.stop_event = threading.Event()
    sampler.thread = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: False)
    sampler.rows = [
        {
            "process_cpu_s": {},
            "gpu_used_bytes": 100,
            "rss_bytes": 100,
            "error": "lost process",
        }
    ]
    c = valid_case()
    c["telemetry"] = sampler.close()
    assert "resource_sampler_error" in evaluate_case(c, [])["errors"]


@pytest.mark.parametrize("valid_ack", [True, False])
def test_hint_http_workflow_binds_full_session_identity(tmp_path, valid_ack):
    from benchmarks.serving_suite.workload import run_wave

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1] * 32

        def decode(self, *args, **kwargs):
            return "answer"

    async def exercise():
        payloads = []
        published = []

        async def generate(request):
            payloads.append(await request.json())
            chunk = {
                "choices": [
                    {"token_ids": [2], "text": "answer", "finish_reason": "stop"}
                ]
            }
            return web.Response(
                text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
                content_type="text/event-stream",
            )

        async def hint(request):
            published.append(await request.json())
            return web.json_response(
                {
                    "schema": "semantic-known-prefix-wait-hint-ack-v1"
                    if valid_ack
                    else "unsupported",
                    "event_id": published[-1]["event_id"],
                    "status": "accepted",
                    "canonical_event_id": "accepted-event",
                }
            )

        app = web.Application()
        app.router.add_post("/v1/completions", generate)
        app.router.add_post("/plugins/semantic-offload/wait-hints", hint)
        server = web.AppRunner(app)
        await server.setup()
        site = web.TCPSite(server, "127.0.0.1", 0)
        await site.start()
        p = plan()
        p.update(tier="screen", publish_hints=True, concurrency=1)
        rows = []
        hints = []
        try:
            await run_wave(
                f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
                p,
                [{"id": "s", "turns": ["one", "two"]}],
                Tokenizer(),
                0,
                rows.append,
                "candidate",
                {"model_revision": "test"},
                hints,
                tmp_path / "streams",
            )
            assert len(rows) == 2 and all(r["success"] for r in rows)
            assert len(published) == 1
            for payload in payloads:
                params = payload["kv_transfer_params"]
                assert all(
                    params[k] == published[0][k]
                    for k in ["run_id", "cell_id", "session_id", "generation"]
                )
            if valid_ack:
                assert (
                    payloads[1]["kv_transfer_params"]["wait_hint_event_id"]
                    == "accepted-event"
                )
            else:
                assert "wait_hint_event_id" not in payloads[1]["kv_transfer_params"]
                assert "schema/identity mismatch" in hints[0]["error"]
                c = valid_case()
                c["hints"] = hints
                assert evaluate_case(c, [])["status"] == "failed"
            assert "two" not in json.dumps(published[0])
        finally:
            await server.cleanup()

    asyncio.run(exercise())


def test_cpu_recency_activity_does_not_mislabel_prefetch_admission():
    from benchmarks.serving_suite.report import activity

    hints = [
        {
            "response": {
                "status": "no_action",
                "reason": "cpu_prefix_recency_refreshed",
                "ready_cpu_prefix_keys": 5,
            }
        },
        {
            "response": {
                "status": "no_action",
                "reason": "cpu_prefix_recency_refreshed",
                "ready_cpu_prefix_keys": 0,
            }
        },
    ]
    a = activity({}, hints)
    assert a["cpu_recency"] == 5 and a["hint_accepted"] == 0
    assert activity({}, hints[1:])["cpu_recency"] == 0
