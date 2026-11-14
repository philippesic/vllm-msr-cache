import hashlib
import json
from pathlib import Path

import pytest

from tools.benchmarking.feasibility import SOURCE_PATHS, UPSTREAM, verify
from tools.runtime_verify import write_record


def test_receipt_rejects_tampering_and_unsupported_inferences(tmp_path: Path) -> None:
    source = tmp_path / "source/probe.py"
    source.parent.mkdir()
    source.write_text("fixture")
    log = tmp_path / "execution.log"
    log.write_text("fixture log")
    observation = {
        "callbacks_at_return": [],
        "gpu_hit_finish_groups": [],
        "shared_gpu_hit_finish_groups": [],
        "scheduler_gpu_hit_chunks_skipped": 2,
        "request_level_alternative_start": 0,
        "ready_cpu_keys_present": ["01", "02"],
        "positions_available_privately": 2,
        "late_store_readiness_callback": True,
        "observer_identity_evictions": [[], [], [], [], []],
        "stock_identity_evictions": [[], [], [], [], []],
        "callback_trace": [["finish", "return", []]],
    }
    observed = tmp_path / "observed.json"
    observed.write_text(json.dumps({"arc": observation, "lru": observation}))
    admission = {
        "schema": "pr-feasibility-admission-v1",
        "upstream_revision": UPSTREAM,
        "upstream_sha256": dict.fromkeys(SOURCE_PATHS, "fixture"),
        "installed_sha256": dict.fromkeys(SOURCE_PATHS, "fixture"),
        "source_sha256": {"probe.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        "runtime_python": "fixture",
        "gpu_exposure_seconds": 0,
        "scope": "fixture",
    }
    terminal = {
        "schema": "pr-feasibility-terminal-v1",
        "exit_code": 0,
        "wall_seconds": 1,
        "execution_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "observed_sha256": hashlib.sha256(observed.read_bytes()).hexdigest(),
    }
    write_record(tmp_path / "admission.json", admission)
    write_record(tmp_path / "terminal.json", terminal)
    assert verify(tmp_path)["upstream_revision"] == UPSTREAM
    source.write_text("changed")
    with pytest.raises(ValueError, match="source snapshot drift"):
        verify(tmp_path)
    source.write_text("fixture")
    log.write_text("changed")
    with pytest.raises(ValueError, match="execution drift"):
        verify(tmp_path)
    log.write_text("fixture log")
    observed.write_text("{}")
    with pytest.raises(ValueError, match="observation drift"):
        verify(tmp_path)
    observed.write_text(json.dumps({"arc": observation, "lru": observation}))
    for change in ({"exit_code": 1}, {"schema": "unknown"}, {"extra": True}):
        (tmp_path / "terminal.json").write_text(json.dumps(terminal | change))
        with pytest.raises(ValueError):
            verify(tmp_path)
    (tmp_path / "terminal.json").write_text(json.dumps(terminal))
    altered = observation | {"stock_identity_evictions": [["different"]] * 5}
    observed.write_text(json.dumps({"arc": altered, "lru": observation}))
    terminal["observed_sha256"] = hashlib.sha256(observed.read_bytes()).hexdigest()
    (tmp_path / "terminal.json").write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="interface observation changed"):
        verify(tmp_path)
