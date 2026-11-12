from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import urllib.error
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, cast

from tools.benchmarking.probe import request
from tools.runtime_verify import write_record


def expects_tool_return(text: str, protocol: str) -> bool:
    if protocol == "mini-swe":
        actions = re.findall(r"```mswea_bash_command\s*\n(.*?)\n```", text, re.DOTALL)
        return (
            len(actions) == 1
            and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in actions[0]
        )
    if protocol != "terminus-json":
        raise ValueError("unsupported tool protocol")
    try:
        first, last = text.index("{"), text.rindex("}") + 1
        parsed = json.loads(text[first:last])
        commands = parsed["commands"]
        return (
            isinstance(parsed["analysis"], str)
            and isinstance(parsed["plan"], str)
            and parsed.get("task_complete", False) is False
            and isinstance(commands, list)
            and bool(commands)
            and all(
                isinstance(command["keystrokes"], str)
                and bool(command["keystrokes"])
                and type(command["duration"]) in (int, float)
                and math.isfinite(command["duration"])
                and 0 <= command["duration"] <= 60
                for command in commands
            )
        )
    except (ValueError, KeyError, TypeError):
        return False


class Proxy(ThreadingHTTPServer):
    def __init__(
        self,
        upstream: str,
        system: str,
        namespace: dict[str, str],
        output: Path,
        tool_protocol: str = "mini-swe",
    ) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.upstream = upstream
        self.system = system
        self.namespace = namespace
        self.output = output
        self.sequence = 0
        self.serial = threading.Lock()
        self.tool_protocol = tool_protocol
        self.closing = False

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.serial:
            if self.closing:
                raise ValueError("benchmark proxy is closing")
            session = payload.get("benchmark_session", "task")
            if not isinstance(session, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]{1,64}", session
            ):
                raise ValueError("invalid benchmark session")
            self.sequence += 1
            generation = self.sequence
            start = time.monotonic_ns()
            registration = {
                "schema": "wait-registration-v2",
                "scope": "coding-pilot",
                "session": session,
                "incarnation": hashlib.sha256(
                    str(self.output.resolve()).encode()
                ).hexdigest()[:32],
                "generation": generation,
            }
            forwarded = {**payload, "stream": False}
            forwarded.pop("benchmark_session", None)
            write_record(
                self.output / f"request-{generation:04d}.admission.json",
                {"request": payload, "start_ns": start, "generation": generation},
            )
            removed = 0
            while True:
                tokenized = request(
                    self.upstream + "/tokenize",
                    {
                        "model": forwarded["model"],
                        "messages": forwarded["messages"],
                        "add_generation_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                if tokenized["count"] + forwarded.get("max_tokens", 512) <= 8192:
                    break
                messages = forwarded["messages"]
                if len(messages) <= 4:
                    raise ValueError(
                        "task plus latest tool turn exceeds context budget"
                    )
                forwarded["messages"] = messages[:2] + messages[4:]
                removed += 2
            if self.system in ("native-arc", "wait-hint"):
                forwarded["kv_transfer_params"] = {"wait_hint": registration}
            result = request(
                self.upstream + "/v1/chat/completions", forwarded, timeout=90
            )
            received = time.monotonic_ns()
            record: dict[str, Any] = {
                "schema": "benchmark-request-v2",
                "tool_protocol": self.tool_protocol,
                "session": session,
                "request": forwarded,
                "response": result,
                "start_ns": start,
                "received_ns": received,
                "system": self.system,
                "hint": None,
                "acknowledgement": None,
                "compaction": {
                    "policy": "retain-system-task-and-newest-complete-turns-v1",
                    "removed_messages": removed,
                    "forwarded_tokens": tokenized["count"],
                },
            }
            text = result["choices"][0]["message"].get("content") or ""
            if self.system in ("native-arc", "wait-hint") and expects_tool_return(
                text, self.tool_protocol
            ):
                observed = (result.get("kv_transfer_params") or {}).get("wait_hint", {})
                if observed.get("registered"):
                    now = time.monotonic_ns()
                    hint = {
                        "schema": "wait-hint-v2",
                        **{
                            key: value
                            for key, value in registration.items()
                            if key != "schema"
                        },
                        "event": "tool-wait",
                        "namespace": self.namespace,
                        **{
                            key: observed[key]
                            for key in (
                                "prefix",
                                "boundary",
                                "observed_ns",
                                "clock_domain",
                            )
                        },
                        "published_ns": now,
                        "deadline_ns": now + 60_000_000_000,
                    }
                    record["hint"] = hint
                    record["acknowledgement"] = request(
                        self.upstream + "/v1/wait-hints", hint
                    )
            record["forwarded_ns"] = time.monotonic_ns()
            write_record(self.output / f"request-{generation:04d}.json", record)
            return result


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4 * 1024 * 1024:
                raise ValueError("request size")
            payload = json.loads(self.rfile.read(length))
            result = cast(Proxy, self.server).generate(payload)
            content = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            return
        except urllib.error.HTTPError as error:
            content = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.send_error(502, str(error))


@contextmanager
def local_proxy(
    upstream: str,
    system: str,
    namespace: dict[str, str],
    output: Path,
    tool_protocol: str = "mini-swe",
) -> Iterator[str]:
    output.mkdir(parents=True, exist_ok=False)
    server = Proxy(upstream, system, namespace, output, tool_protocol)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.closing = True
        server.shutdown()
        drained = server.serial.acquire(timeout=95)
        try:
            if not drained:
                raise TimeoutError("benchmark proxy drain exceeded its deadline")
        finally:
            server.server_close()
            if drained:
                server.serial.release()
            thread.join(timeout=5)
