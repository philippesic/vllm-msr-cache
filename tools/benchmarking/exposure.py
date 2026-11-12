from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.benchmarking.probe import digest_file
from tools.runtime_verify import write_record


def source_group(instance: str) -> str:
    return (
        instance.rsplit("-", 1)[0]
        if instance.rsplit("-", 1)[-1].isdigit()
        else instance.split("-")[0]
    )


def admit_exposure(
    root: Path, assets: Path, output: Path, jobs: list[dict[str, Any]]
) -> dict[str, Any]:
    definitions = json.loads((root / "benchmarks/workloads.json").read_text())
    known = {
        source_group(case["instance"])
        for case in definitions["cases"]
        if case["exposure"] in ("feasibility-pilot", "development-validation")
    }
    registry = assets / "exposure"
    registry.mkdir(exist_ok=True)
    prior = {}
    for path in sorted(registry.glob("*.json")):
        record = json.loads(path.read_text())
        known.update(record["source_groups"])
        prior[path.name] = digest_file(path)
    groups = set()
    for job in jobs:
        if job["kind"] != "application":
            continue
        group = source_group(job["instance"])
        groups.add(group)
        job["source_group"] = group
        job["source_group_previously_exposed"] = group in known
    name = hashlib.sha256(str(output.resolve()).encode()).hexdigest() + ".json"
    record = {
        "schema": "benchmark-exposure-v1",
        "campaign": str(output.resolve()),
        "source_groups": sorted(groups),
        "prior_sha256": prior,
        "policy": "all admitted source groups count exposed even if interrupted; repeated starts within this frozen campaign are not new source tasks",
    }
    write_record(registry / name, record)
    write_record(output / "exposure.json", record)
    return {
        "sha256": digest_file(output / "exposure.json"),
        "source_groups": sorted(groups),
    }
