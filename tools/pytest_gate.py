from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--gate-manifest", required=True)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    expected = set(json.loads(Path(config.getoption("gate_manifest")).read_text()))
    actual = {item.nodeid for item in items}
    if actual != expected:
        raise pytest.UsageError(
            f"cpu.collection: missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    for item in items:
        if any(any(item.iter_markers(name=n)) for n in ("skip", "skipif", "xfail")):
            raise pytest.UsageError(f"cpu.skipped: {item.nodeid}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if report.skipped or hasattr(report, "wasxfail"):
        report.outcome = "failed"
        report.longrepr = f"cpu.skipped: {report.nodeid}"
    return report
