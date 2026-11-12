from pathlib import Path

import pytest

from tools.benchmarking.plan import cells, describe
from tools.benchmarking.probe import probe


@pytest.mark.parametrize(
    ("before", "after", "quality"),
    (
        ("<failure/>", "", "resolved"),
        ("", "", "unresolved"),
        ("<failure/>", "<failure/>", "unresolved"),
        ("<error/>", "", "unresolved"),
        ("<failure/>", "<skipped/>", "unresolved"),
    ),
)
def test_regression_checker_requires_failure_then_same_test_passes(
    tmp_path: Path, before: str, after: str, quality: str
) -> None:
    from tools.benchmarking.regression import regression_outcome

    for name, status in (("before", before), ("after", after)):
        (tmp_path / f"{name}.xml").write_text(
            '<testsuite><testcase classname="regression" name="test_issue">'
            + status
            + "</testcase></testsuite>"
        )
    assert (
        regression_outcome(tmp_path / "before.xml", tmp_path / "after.xml")["quality"]
        == quality
    )
    (tmp_path / "after.xml").write_text(
        '<testsuite><testcase classname="regression" name="different_test"/></testsuite>'
    )
    with pytest.raises(ValueError, match="identities changed"):
        regression_outcome(tmp_path / "before.xml", tmp_path / "after.xml")


def test_release_schedule_preserves_budget_and_comparison_denominators() -> None:
    schedule = cells("release")
    assert len({cell.identity for cell in schedule}) == len(schedule)
    assert describe("release")["counts"] == {
        "functional": 0,
        "application": 180,
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
                )
            }
            assert tasks["gpu-only"] == tasks["native-arc"] == tasks["wait-hint"]
            assert len(tasks["wait-hint"]) == 13
            assert len(tasks) == 4
            assert len(tasks["lmcache"]) == 6
            assert tasks["lmcache"] <= tasks["wait-hint"]
    assert describe("release")["cell_timeout_sum_seconds"] <= 108000


def test_compare_is_one_block_and_unknown_profiles_refuse() -> None:
    scheduled = cells("compare")
    assert len(scheduled) == 12
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


def test_coding_image_drift_refuses_before_tagging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    import subprocess

    from tools.benchmarking.harness import pin_swebench_image

    commands: list[list[str]] = []

    def inspect(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output and text and check and timeout > 0
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, json.dumps([{"Id": "sha256:" + "a" * 64}])
        )

    monkeypatch.setattr("tools.benchmarking.harness.subprocess.run", inspect)
    with pytest.raises(ValueError, match="image changed"):
        pin_swebench_image("pallets__flask-5014", "sha256:" + "b" * 64)
    assert len(commands) == 1
    assert "image inspect" in commands[0][-1]


def test_coding_deadline_retains_patch_before_owned_container_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import subprocess
    from typing import Any

    from tools.benchmarking.harness import run_swebench

    dataset = tmp_path / "test.json"
    dataset.write_text(json.dumps([{"instance_id": "project__issue"}]))
    image = {"id": "sha256:" + "a" * 64, "tag": "benchmark-a"}
    operations: list[str] = []

    def docker(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        operations.append(command[-1])
        value = "owned-container" if " ps " in command[-1] else ""
        if " exec " in command[-1]:
            value = "retained patch"
        return subprocess.CompletedProcess(command, 0, value)

    def grader(*args: Any) -> dict[str, str]:
        prediction = json.loads(args[3].read_text())
        assert prediction["project__issue"]["model_patch"] == "retained patch"
        assert args[-1] == image["id"]
        return {"quality": "unresolved"}

    monkeypatch.setattr("tools.benchmarking.harness.subprocess.run", docker)
    monkeypatch.setattr("tools.benchmarking.harness.run_owned", lambda *_args: 124)
    monkeypatch.setattr(
        "tools.benchmarking.harness.pin_swebench_image", lambda _: image
    )
    monkeypatch.setattr("tools.benchmarking.harness.grade_swebench", grader)
    result = run_swebench(
        tmp_path / "source",
        tmp_path / "python",
        dataset,
        "project__issue",
        "http://local",
        tmp_path / "attempt",
        240,
    )
    assert result["execution_exit"] == 124
    assert result["prediction_origin"] == "deadline-snapshot"
    assert result["quality"] == "unresolved"
    assert " exec " in operations[-2]
    assert " rm " in operations[-1]


def test_coding_proxy_registers_each_turn_and_only_hints_real_tool_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import json
    from email.message import Message
    from typing import Any

    from tools.benchmarking.proxy import Handler, Proxy

    calls: list[tuple[str, dict[str, Any] | None]] = []

    def upstream(
        url: str, payload: dict[str, Any] | None = None, timeout: int = 30
    ) -> dict[str, Any]:
        calls.append((url, payload))
        if url.endswith("/tokenize"):
            assert payload is not None
            return {"count": 9000 if len(payload["messages"]) > 6 else 64}
        if url.endswith("/v1/wait-hints"):
            return {"decision": "disabled", "refreshed_keys": 0}
        assert timeout == 90
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
                {
                    "model": "qualification",
                    "messages": [
                        {"role": "system", "content": "instructions"},
                        {"role": "user", "content": "task"},
                    ]
                    + [
                        {"role": role, "content": str(turn)}
                        for turn in range(3)
                        for role in ("assistant", "user")
                    ],
                    "stream": False,
                }
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
            assert observed["compaction"]["removed_messages"] == 2
            assert len(observed["request"]["messages"]) == 6
            assert (
                observed["start_ns"]
                <= observed["received_ns"]
                <= observed["forwarded_ns"]
            )
        assert len(calls) == 8
    finally:
        server.server_close()


def test_probe_reader_rejects_drift_failed_attempt_and_unobserved_restoration(
    tmp_path: Path,
) -> None:
    import json
    from typing import Any

    from tools.benchmarking.probe import digest_file, verify_probe

    (tmp_path / "source").mkdir()
    source = tmp_path / "source/input.py"
    source.write_text("value = 1\n")
    log = tmp_path / "execution.log"
    log.write_text("generation completed\n")
    admission = {
        "schema": "benchmark-model-probe-v3",
        "scope": "model-loading-generation-and-optional-restoration",
        "command": ["runtime", "-m", "server"],
        "runtime_packages": {"runtime": "1"},
        "system": "gpu-only",
        "budget_seconds": 30,
        "source_sha256": {"input.py": digest_file(source)},
        "model_sha256": {str(source): digest_file(source)},
        "coding": None,
    }
    (tmp_path / "admission.json").write_text(json.dumps(admission))
    terminal = {
        "schema": "benchmark-model-terminal-v3",
        "scope": "model-loading-generation-and-optional-restoration",
        "admission_sha256": digest_file(tmp_path / "admission.json"),
        "status": "generation_observed",
        "elapsed_seconds": 2,
        "artifacts_sha256": {"execution.log": digest_file(log)},
        "response": {"choices": [{"message": {"content": "4"}}]},
    }
    receipt = tmp_path / "terminal.json"
    receipt.write_text(json.dumps(terminal))
    assert verify_probe(tmp_path)["status"] == "generation_observed"
    log.write_text("different execution\n")
    with pytest.raises(ValueError, match="artifact drift"):
        verify_probe(tmp_path)
    log.write_text("generation completed\n")
    receipt.write_text(json.dumps({**terminal, "status": "failed"}))
    with pytest.raises(ValueError, match="did not pass"):
        verify_probe(tmp_path)
    admission["system"] = "wait-hint"
    (tmp_path / "admission.json").write_text(json.dumps(admission))
    receipt.write_text(
        json.dumps(
            {**terminal, "admission_sha256": digest_file(tmp_path / "admission.json")}
        )
    )
    with pytest.raises(ValueError, match="did not pass"):
        verify_probe(tmp_path)
    restored = {
        "choices": [{"text": "same decoded text", "token_ids": [42]}],
        "usage": {"prompt_tokens_details": {"cached_tokens": 16}},
    }
    fresh: dict[str, Any] = {
        "choices": [{"text": "same decoded text", "token_ids": [42]}],
        "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
    }
    observation = {
        "initial": {"kv_transfer_params": {"wait_hint": {"registered": True}}},
        "acknowledgement": {"decision": "refreshed", "ready_cpu_prefix_keys": 1},
        "restored": restored,
        "fresh": fresh,
    }
    nested = tmp_path / "coding/grading/admission.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}\n")
    terminal.update(
        status="restoration_observed",
        restore=observation,
        admission_sha256=digest_file(tmp_path / "admission.json"),
        artifacts_sha256={
            "execution.log": digest_file(log),
            "coding/grading/admission.json": digest_file(nested),
        },
    )
    receipt.write_text(json.dumps(terminal))
    assert verify_probe(tmp_path)["status"] == "restoration_observed"
    fresh["choices"][0]["token_ids"] = [43]
    receipt.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="restoration observations"):
        verify_probe(tmp_path)
    fresh["choices"][0]["token_ids"] = [42]
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]\n")
    admission["coding"] = {
        "dataset": str(dataset),
        "dataset_sha256": digest_file(dataset),
        "instance": "case",
    }
    (tmp_path / "admission.json").write_text(json.dumps(admission))
    grading = {
        "exit_code": 0,
        "quality": "resolved",
        "report": {"case": {"resolved": False}},
    }
    acquisition = {"quality": "resolved", "grading": grading}
    terminal["coding"] = acquisition
    (tmp_path / "coding/acquisition.json").write_text(json.dumps(acquisition))
    (tmp_path / "coding/grading/terminal.json").write_text(json.dumps(grading))
    report = tmp_path / "coding/grading/logs/run_evaluation/run/model/case/report.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(grading["report"]))
    terminal["admission_sha256"] = digest_file(tmp_path / "admission.json")
    terminal["artifacts_sha256"] = {
        str(path.relative_to(tmp_path)): digest_file(path)
        for path in tmp_path.rglob("*")
        if path.is_file()
        and not path.is_relative_to(tmp_path / "source")
        and path not in (tmp_path / "admission.json", receipt)
    }
    receipt.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="original grader"):
        verify_probe(tmp_path)


def test_external_restoration_requires_cpu_reuse_and_fresh_equal_tokens() -> None:
    import copy
    from typing import Any

    from tools.benchmarking.probe import validate_external_restoration

    response: dict[str, Any] = {
        "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
        "choices": [{"token_ids": [42], "text": "answer"}],
    }
    observation = {
        name: copy.deepcopy(response) for name in ("initial", "restored", "fresh")
    }
    observation["restored"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 528
    validate_external_restoration(observation)
    for name, count in (("restored", 0), ("fresh", 528)):
        invalid = copy.deepcopy(observation)
        invalid[name]["usage"]["prompt_tokens_details"]["cached_tokens"] = count
        with pytest.raises(ValueError, match="reuse not exercised"):
            validate_external_restoration(invalid)
    invalid = copy.deepcopy(observation)
    invalid["fresh"]["choices"][0]["token_ids"] = [43]
    with pytest.raises(ValueError, match="restoration differs"):
        validate_external_restoration(invalid)


def test_runtime_patch_requires_exact_source_and_preserves_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json
    import subprocess

    from tools.benchmarking.patches import apply_runtime_patches

    root = tmp_path / "project"
    directory = root / "benchmarks/runtime-patches"
    directory.mkdir(parents=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    target = runtime / "module.py"
    before, after = b"setting = 1\n", b"setting = 2\n"
    target.write_bytes(before)
    (directory / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "name": "test",
                    "target": "module.py",
                    "original_sha256": hashlib.sha256(before).hexdigest(),
                    "patched_sha256": hashlib.sha256(after).hexdigest(),
                    "edits": [{"old": before.decode(), "new": after.decode()}],
                }
            ]
        )
    )
    monkeypatch.setattr(
        "tools.benchmarking.patches.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, str(runtime)),
    )
    apply_runtime_patches(root, Path("python"), ["test"])
    assert target.read_bytes() == after
    assert target.with_suffix(".py.benchmark-original").read_bytes() == before
    apply_runtime_patches(root, Path("python"), ["test"])
    target.write_text("unrelated active edits\n")
    with pytest.raises(ValueError, match="source differs"):
        apply_runtime_patches(root, Path("python"), ["test"])
    assert target.read_text() == "unrelated active edits\n"


def test_campaign_refuses_unimplemented_coverage_before_acquisition(
    tmp_path: Path,
) -> None:
    import json

    from tools.benchmarking.campaign import run_campaign

    root = tmp_path / "project"
    (root / "benchmarks").mkdir(parents=True)
    (root / "benchmarks/workloads.json").write_text(json.dumps({"cases": []}))
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="not implemented/admitted"):
        run_campaign(
            root,
            "release",
            {},
            Path("python"),
            Path("lmcache"),
            Path("harness"),
            Path("dataset"),
            output,
            [],
        )
    assert not output.exists()


def test_v5_lmcache_receipt_cannot_pass_without_external_restore(
    tmp_path: Path,
) -> None:
    import json
    from typing import Any

    from tools.benchmarking.probe import digest_file, verify_probe

    output = tmp_path / "attempt"
    (output / "source").mkdir(parents=True)
    source = output / "source/file.py"
    source.write_text("value = 1\n")
    model = tmp_path / "weights"
    model.write_bytes(b"weights")
    admission = {
        "schema": "benchmark-model-probe-v5",
        "command": [],
        "source_sha256": {"file.py": digest_file(source)},
        "budget_seconds": 60,
        "scope": "model-loading-generation-and-optional-restoration",
        "system": "lmcache",
        "model_sha256": {str(model): digest_file(model)},
        "runtime_packages": {},
        "runtime_sources": {"module.py": "a" * 64},
        "coding": None,
    }
    (output / "admission.json").write_text(json.dumps(admission))
    terminal = {
        "schema": "benchmark-model-terminal-v5",
        "scope": admission["scope"],
        "status": "restoration_observed",
        "elapsed_seconds": 5,
        "admission_sha256": digest_file(output / "admission.json"),
        "artifacts_sha256": {},
        "response": {"choices": [{"message": {"content": "4"}}]},
    }
    (output / "terminal.json").write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="missing external restoration"):
        verify_probe(output)
    response: dict[str, Any] = {
        "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
        "choices": [{"text": "answer", "token_ids": [42]}],
    }
    import copy

    restored = copy.deepcopy(response)
    restored["usage"]["prompt_tokens_details"]["cached_tokens"] = 528
    terminal["external_restore"] = {
        "fresh": response,
        "restored": restored,
        "initial": response,
    }
    (output / "terminal.json").write_text(json.dumps(terminal))
    assert verify_probe(output)["status"] == "restoration_observed"


def test_campaign_keeps_failed_cells_in_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from tools.benchmarking.campaign import run_campaign, verify_campaign
    from tools.benchmarking.plan import Cell
    from tools.benchmarking.probe import digest_file

    root = tmp_path / "project"
    (root / "benchmarks").mkdir(parents=True)
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([{"instance_id": "issue"}]))
    definitions = root / "benchmarks/workloads.json"
    definitions.write_text(
        json.dumps(
            {
                "dataset_sha256": digest_file(dataset),
                "cases": [{"id": "C1-1", "instance": "issue"}],
            }
        )
    )
    binary = tmp_path / "python"
    binary.touch()
    model = tmp_path / "model"
    model.mkdir()
    roster = tuple(
        Cell("model", system, "C1-1", 0, "application", 240)
        for system in ("native-arc", "lmcache")
    )
    monkeypatch.setattr("tools.benchmarking.campaign.cells", lambda profile: roster)
    monkeypatch.setattr(
        "tools.repo_policy.git_paths", lambda path: [Path("benchmarks/workloads.json")]
    )
    monkeypatch.setattr(
        "tools.benchmarking.campaign.git_paths",
        lambda path: [Path("benchmarks/workloads.json")],
    )

    def fake_probe(
        _root: Path,
        _python: Path,
        _model: Path,
        output: Path,
        _seconds: int,
        system: str,
        coding: tuple[Path, Path, str],
    ) -> int:
        output.mkdir()
        (output / "admission.json").write_text(
            json.dumps(
                {
                    "source_sha256": {
                        "benchmarks/workloads.json": digest_file(definitions)
                    },
                    "system": system,
                    "command": ["--model", str(model)],
                    "coding": {
                        "instance": coding[2],
                        "dataset_sha256": digest_file(dataset),
                    },
                }
            )
        )
        return int(system == "lmcache")

    monkeypatch.setattr("tools.benchmarking.campaign.probe", fake_probe)
    monkeypatch.setattr(
        "tools.benchmarking.campaign.pin_swebench_image",
        lambda instance, expected_identity=None: {"id": "sha256:" + "a" * 64},
    )
    monkeypatch.setattr(
        "tools.benchmarking.campaign.verify_probe",
        lambda path: {"coding": {"image": {"id": "sha256:" + "a" * 64}}},
    )
    output = tmp_path / "campaign"
    assert (
        run_campaign(
            root,
            "release",
            {"model": model},
            binary,
            binary,
            binary,
            dataset,
            output,
            ["C1-1"],
        )
        == 1
    )
    terminal = json.loads((output / "terminal.json").read_text())
    assert [result["status"] for result in terminal["results"]] == [
        "complete",
        "failed",
    ]
    with pytest.raises(ValueError, match="cell incomplete"):
        verify_campaign(output)
    terminal["results"].pop()
    (output / "terminal.json").write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="coverage incomplete"):
        verify_campaign(output)
