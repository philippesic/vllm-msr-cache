from pathlib import Path

import pytest

from tools.benchmarking.plan import cells, describe
from tools.benchmarking.probe import probe


def test_release_schedule_preserves_budget_and_comparison_denominators() -> None:
    schedule = cells("release")
    assert len({cell.identity for cell in schedule}) == len(schedule)
    assert describe("release")["counts"] == {
        "functional": 0,
        "application": 204,
        "trace": 52,
        "soak": 4,
    }
    assert sum(cell.seconds for cell in schedule if cell.kind == "soak") == 21600
    for model in ("qwen35-4b-int4", "gemma4-12b-int4"):
        for block in (0, 1):
            tasks = {
                arm: {
                    cell.case
                    for cell in schedule
                    if cell.model == model
                    and cell.block == block
                    and cell.system == arm
                    and cell.kind == "application"
                }
                for arm in (
                    "gpu-only",
                    "native-arc",
                    "wait-hint",
                    "lmcache",
                    "continuum",
                )
            }
            assert tasks["gpu-only"] == tasks["native-arc"] == tasks["wait-hint"]
            assert len(tasks["wait-hint"]) == 13
            assert tasks["lmcache"] == tasks["continuum"]
            assert len(tasks["lmcache"]) == 6
            assert tasks["lmcache"] <= tasks["wait-hint"]
    assert describe("release")["cell_timeout_sum_seconds"] <= 108000


def test_compare_is_one_block_and_unknown_profiles_refuse() -> None:
    scheduled = cells("compare")
    assert len(scheduled) == 15
    assert {cell.block for cell in scheduled} == {0}
    assert {cell.model for cell in scheduled} == {"qwen35-4b-int4"}
    assert describe("compare")["status"] == "unqualified_schedule"
    with pytest.raises(ValueError, match="unknown profile"):
        cells("typo")


def test_model_probe_refuses_missing_assets_before_creating_attempt(
    tmp_path: Path,
) -> None:
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="complete local model"):
        probe(tmp_path / "source", tmp_path / "missing", tmp_path / "model", out, 60)
    assert not out.exists()


def test_coding_proxy_registers_each_turn_and_only_hints_real_tool_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import json
    from email.message import Message
    from typing import Any

    from tools.benchmarking.proxy import Handler, Proxy

    calls: list[tuple[str, dict[str, Any] | None]] = []

    def upstream(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        calls.append((url, payload))
        if url.endswith("/tokenize"):
            return {"count": 64}
        if url.endswith("/v1/wait-hints"):
            return {"decision": "disabled", "refreshed_keys": 0}
        return {
            "choices": [
                {"message": {"content": "```mswea_bash_command\ncat app.py\n```"}}
            ],
            "kv_transfer_params": {
                "wait_hint": {
                    "registered": True,
                    "prefix": "digest",
                    "boundary": 16,
                    "observed_ns": 1,
                    "clock_domain": "engine",
                }
            },
        }

    monkeypatch.setattr("tools.benchmarking.proxy.request", upstream)
    server = Proxy("http://local", "native-arc", {}, tmp_path)
    try:
        for generation in (1, 2):
            handler = Handler.__new__(Handler)
            handler.server = server
            handler.path = "/v1/chat/completions"
            vars(handler).update(
                request_version="HTTP/1.1",
                requestline="POST /v1/chat/completions HTTP/1.1",
                client_address=("127.0.0.1", 12345),
            )
            handler.command = "POST"
            handler.headers = Message()
            body = json.dumps(
                {"model": "qualification", "messages": [], "stream": False}
            ).encode()
            handler.headers["Content-Length"] = str(len(body))
            handler.rfile = io.BytesIO(body)
            handler.wfile = io.BytesIO()
            Handler.do_POST(handler)
            assert b"200 OK" in handler.wfile.getvalue()
            observed = json.loads(
                (tmp_path / f"request-{generation:04d}.json").read_text()
            )
            assert observed["hint"]["generation"] == generation
            assert observed["acknowledgement"]["decision"] == "disabled"
            assert (
                observed["start_ns"]
                <= observed["received_ns"]
                <= observed["forwarded_ns"]
            )
        assert len(calls) == 6
    finally:
        server.server_close()
