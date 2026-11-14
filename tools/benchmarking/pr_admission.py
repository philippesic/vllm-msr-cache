from __future__ import annotations

import asyncio
import importlib
import json
import random
import time
from pathlib import Path
from typing import Any

from tools.benchmarking.pr_client import _publish
from tools.benchmarking.pr_events import reconcile_events, request_sources
from tools.benchmarking.streaming import request_stream, verify_stream
from tools.runtime_verify import write_record
from wait_hint_offload.evidence_events import read_journal
from wait_hint_offload.evidence_protocol import EvidenceRequest
from wait_hint_offload.policy_protocol import PolicySession


async def admission_requests(url: str, arm: str, directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    aiohttp = importlib.import_module("aiohttp")
    random_tokens = random.Random(11991)
    tokens = tuple(random_tokens.randrange(1000, 30000) for _ in range(4097))
    captures = []
    resets = []
    hints = []
    async with aiohttp.ClientSession(
        headers={"Authorization": "Bearer local-benchmark"},
        timeout=aiohttp.ClientTimeout(total=None),
    ) as client:
        async with client.get(url + "/v1/pr-evidence/groups") as response:
            response.raise_for_status()
            groups = await response.json()
        write_record(directory / "groups.json", groups)
        for generation, name in enumerate(("prime", "resident", "restored"), 1):
            if name in {"prime", "restored"}:
                async with client.post(
                    url + "/v1/pr-evidence/reset",
                    json={"reset_connector": name == "prime"},
                ) as response:
                    response.raise_for_status()
                    reset = await response.json()
                    resets.append({"before": name, **reset})
                    if reset != {"reset": True}:
                        raise ValueError("admission cache reset was refused")
            payload = EvidenceRequest(
                "admission-" + name,
                tokens,
                32,
                PolicySession("pr-evidence", "admission", "11991", generation),
                12000 if arm in {"A3", "A4t", "A5"} else None,
            ).to_dict()
            started = time.monotonic_ns()
            capture = await request_stream(
                client, url, payload, started, started + 120_000_000_000
            )
            write_record(directory / (name + ".json"), capture)
            verify_stream(capture)
            captures.append(capture)
            if capture["outcome"] != "complete":
                break
            if arm in {"A3p", "A4p"}:
                hints.append(
                    await _publish(
                        client,
                        url,
                        payload,
                        12000,
                        time.monotonic_ns() + 10_000_000_000,
                    )
                )
            await asyncio.sleep(0.25)
    result = {
        "schema": "pr-model-admission-client-v1",
        "arm": arm,
        "groups": groups,
        "resets": resets,
        "hints": hints,
        "request_outcomes": [capture["outcome"] for capture in captures],
        "resident_restored_token_identity": len(captures) == 3
        and captures[1]["token_ids"] == captures[2]["token_ids"],
        "fresh_resident_token_identity": len(captures) >= 2
        and captures[0]["token_ids"] == captures[1]["token_ids"],
        "scope": "same prompt and native cached-boundary continuation; boundary and byte evidence verified separately",
    }
    write_record(directory / "report.json", result)
    return result


def verify_model_admission(directory: Path) -> dict[str, Any]:
    from tools.benchmarking.pr_jobs import verify_gpu_job

    receipt = verify_gpu_job(directory)
    if receipt["terminal"]["status"] != "client_complete":
        return {
            "admitted": False,
            "reason": receipt["terminal"]["detail"],
            "receipt": receipt,
        }
    report = json.loads((directory / "client/report.json").read_text())
    events = reconcile_events(directory / "events")
    sources = {}
    for name in ("prime", "resident", "restored"):
        capture = json.loads((directory / "client" / (name + ".json")).read_text())
        verify_stream(capture)
        if capture["outcome"] != "complete":
            raise ValueError("admission generation did not complete")
        identity = events["external_request_ids"]["admission-" + name]
        sources[name] = request_sources(
            events["requests"][identity],
            4096,
            True,
            set(events["completed_jobs"]),
            set(events["failed_jobs"]),
        )
    probes = list((directory / "events").glob("byte-probe-*.jsonl"))
    if len(probes) != 1:
        raise ValueError("missing TP1 restored-byte probe")
    records = read_journal(probes[0])
    stores = [record["data"] for record in records if record["event"] == "store_bytes"]
    loads = [record["data"] for record in records if record["event"] == "load_bytes"]
    byte_identity = (
        bool(stores and loads)
        and all(
            data["accepted"]
            and data["regions"]
            and all(
                region["equal"]
                and region["source_sha256"] == region["destination_sha256"]
                for region in data["regions"]
            )
            for data in stores + loads
        )
        and all(
            region["matches_observed_store"]
            and region["bytes_different_from_poison"] > 0
            for data in loads
            for region in data["regions"]
        )
    )
    restored_groups = {region["group"] for data in loads for region in data["regions"]}
    configured_groups = {group["group"] for group in events["group_config"]["groups"]}
    same_boundary = (
        sources["resident"]["scheduled_boundaries"][0]
        == sources["restored"]["scheduled_boundaries"][0]
        > 0
    )
    passed = (
        byte_identity
        and restored_groups == configured_groups
        and same_boundary
        and report["resident_restored_token_identity"]
        and sources["restored"]["cpu_reused_tokens"] > 0
        and not events["failed_jobs"]
        and not events["store_failures"]
        and not events["store_refusals"]
    )
    return {
        "schema": "pr-model-admission-verification-v1",
        "admitted": passed,
        "byte_identity": byte_identity,
        "same_boundary": same_boundary,
        "restored_groups": sorted(restored_groups),
        "configured_groups": sorted(configured_groups),
        "resident_restored_token_identity": report["resident_restored_token_identity"],
        "fresh_resident_token_identity": report["fresh_resident_token_identity"],
        "sources": sources,
        "gpu_blocks": events["group_config"]["gpu_blocks"],
        "cpu_config": events["cpu_config"],
        "transfer": events["transfer"],
        "store_refusals": events["store_refusals"],
        "store_failures": events["store_failures"],
        "failed_jobs": events["failed_jobs"],
    }
